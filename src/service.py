"""
The triage service.

    uvicorn src.service:app --port 8000      # docs at /docs

Loads whichever model version the registry calls champion. Every prediction
is appended to a log (text, route, confidence, model version) that the drift
monitor reads, and /metrics exposes counters in Prometheus format.
"""
import json
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .data import ROUTES, SENTIMENTS, normalize
from .rag import Retriever, draft_reply

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.getenv("JETWAY_MODELS", ROOT / "models"))
LOG = Path(os.getenv("JETWAY_LOG", ROOT / "logs" / "predictions.jsonl"))
# Below this confidence a message goes to a person instead of a queue.
# Chosen on the test set's coverage curve: see results/experiments.json.
HANDOFF_BELOW = float(os.getenv("JETWAY_HANDOFF_BELOW", "0.5"))


class TweetIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["@airline my bag never arrived in Denver"])


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class Triage(BaseModel):
    sentiment: str
    sentiment_confidence: float
    route: str
    confidence: float
    needs_human: bool
    alternatives: dict[str, float]
    model_version: str


class Engine:
    def __init__(self):
        registry = json.loads((MODELS / "registry.json").read_text())
        self.version = registry["champion"]
        self.models = joblib.load(MODELS / self.version / "model.joblib")
        self.card = json.loads((MODELS / self.version / "card.json").read_text())
        self.retriever = Retriever()
        self.counts = Counter()
        self.latency_ms = []
        self.lock = threading.Lock()
        LOG.parent.mkdir(parents=True, exist_ok=True)

    def triage(self, texts: list[str]) -> list[Triage]:
        t0 = time.perf_counter()
        clean = [normalize(t) for t in texts]
        out = []
        sent = self._proba("sentiment", clean, SENTIMENTS)
        route = self._proba("route", clean, ROUTES)
        for raw, s, r in zip(texts, sent, route):
            best_r = max(r, key=r.get)
            best_s = max(s, key=s.get)
            item = Triage(sentiment=best_s, sentiment_confidence=round(s[best_s], 4), route=best_r,
                          confidence=round(r[best_r], 4), needs_human=r[best_r] < HANDOFF_BELOW,
                          alternatives={k: round(v, 4) for k, v in sorted(r.items(), key=lambda kv: -kv[1])[:3]},
                          model_version=self.version)
            out.append(item)
            self._log(raw, item)
        with self.lock:
            self.latency_ms.append((time.perf_counter() - t0) * 1000)
            self.latency_ms = self.latency_ms[-1000:]
        return out

    def _proba(self, task, clean, labels):
        m = self.models[task]
        order = [list(m.classes_).index(c) for c in labels]
        return [dict(zip(labels, row[order].tolist())) for row in m.predict_proba(clean)]

    def _log(self, text, item: Triage):
        with self.lock:
            self.counts["predictions"] += 1
            self.counts[f"route:{item.route}"] += 1
            self.counts["handoffs"] += item.needs_human
            with LOG.open("a") as f:
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "text": text,
                                    "route": item.route, "confidence": item.confidence,
                                    "needs_human": item.needs_human, "model": self.version}) + "\n")


engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = Engine()
    yield


app = FastAPI(title="Jetway triage", version="1.0", lifespan=lifespan,
              description="Routes airline customer-care messages and drafts grounded replies.")


@app.get("/health")
def health():
    return {"ok": True, "model_version": engine.version, "trained_at": engine.card["trained_at"],
            "handoff_below": HANDOFF_BELOW}


@app.post("/triage", response_model=Triage)
def triage(body: TweetIn):
    return engine.triage([body.text])[0]


@app.post("/triage/batch", response_model=list[Triage])
def triage_batch(body: BatchIn):
    if any(not t.strip() for t in body.texts):
        raise HTTPException(422, "Every text needs at least one character.")
    return engine.triage(body.texts)


@app.post("/draft")
def draft(body: TweetIn):
    """Triage the message, then draft a reply grounded in policy, with citations."""
    t = engine.triage([body.text])[0]
    if t.route == "no_action" and t.sentiment == "positive":
        route = "no_action"
    else:
        route = t.route
    d = draft_reply(body.text, engine.retriever, route=route)
    engine.counts["drafts"] += 1
    engine.counts["drafts_ungrounded"] += not d.grounded
    return {"triage": t.model_dump(), "reply": d.reply, "citations": d.citations, "grounded": d.grounded,
            "problems": d.problems, "generator": d.generator, "sources": d.passages,
            # A person approves every public reply. The draft saves time; it doesn't post.
            "requires_approval": True}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    c, lat = engine.counts, sorted(engine.latency_ms) or [0.0]
    lines = [
        "# TYPE jetway_predictions_total counter", f"jetway_predictions_total {c['predictions']}",
        "# TYPE jetway_handoffs_total counter", f"jetway_handoffs_total {c['handoffs']}",
        "# TYPE jetway_drafts_total counter", f"jetway_drafts_total {c['drafts']}",
        "# TYPE jetway_drafts_ungrounded_total counter", f"jetway_drafts_ungrounded_total {c['drafts_ungrounded']}",
        "# TYPE jetway_route_total counter",
        *[f'jetway_route_total{{route="{r}"}} {c[f"route:{r}"]}' for r in ROUTES],
        "# TYPE jetway_latency_ms summary",
        f'jetway_latency_ms{{quantile="0.5"}} {lat[len(lat) // 2]:.2f}',
        f'jetway_latency_ms{{quantile="0.95"}} {lat[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0]:.2f}',
        f'jetway_model_info{{version="{engine.version}"}} 1',
    ]
    return "\n".join(lines) + "\n"
