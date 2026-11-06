"""
Retrieval-augmented reply drafting.

Given a customer's tweet and the queue it was routed to, find the policy
passages that apply and draft a public reply grounded in them. Every reply is
checked before it's returned: it may only cite passages that were retrieved,
and any number in it (a dollar amount, a count of miles, a time limit) has
to appear in those passages. That's what stops a model from promising $500
the policy never offered.
"""
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

KB = Path(__file__).resolve().parent.parent / "kb"
STOP = set("a an the and or of to in on for with is are was be it this that as at by from we our you your "
           "they their i my me can if not no do does any but so has have had will just get got".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Passage:
    id: str           # doc-slug#n
    doc: str
    title: str
    route: str
    text: str
    tokens: list[str] = field(repr=False, default_factory=list)


def load_kb(path: Path = KB) -> list[Passage]:
    """One passage per paragraph: small enough to cite precisely, big enough to make sense alone."""
    passages = []
    for f in sorted(path.glob("*.md")):
        raw = f.read_text()
        meta, body = raw.split("---", 2)[1], raw.split("---", 2)[2]
        title = re.search(r"title:\s*(.+)", meta).group(1).strip()
        route = re.search(r"route:\s*(.+)", meta).group(1).strip()
        for i, para in enumerate(p.strip() for p in body.strip().split("\n\n") if p.strip()):
            passages.append(Passage(f"{f.stem}#{i + 1}", f.stem, title, route, para, tokenize(title + " " + para)))
    return passages


class BM25:
    """Okapi BM25, written out so every number in the ranking can be explained."""

    def __init__(self, passages: list[Passage], k1: float = 1.4, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        self.avg_len = sum(len(p.tokens) for p in passages) / len(passages)
        df = Counter(t for p in passages for t in set(p.tokens))
        n = len(passages)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(p.tokens) for p in passages]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for p, tf in zip(self.passages, self.tf):
            norm = self.k1 * (1 - self.b + self.b * len(p.tokens) / self.avg_len)
            out.append(sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf))
        return out


class Retriever:
    def __init__(self, passages: list[Passage] | None = None):
        self.passages = passages or load_kb()
        self.bm25 = BM25(self.passages)

    def search(self, query: str, route: str | None = None, k: int = 3, route_boost: float = 1.5) -> list[tuple[Passage, float]]:
        """
        BM25, optionally nudged toward the queue the classifier picked. The
        boost multiplies, so a passage still needs to match the words of the
        question to rank: the route breaks ties, it doesn't override the text.
        """
        scores = self.bm25.scores(query)
        if route:
            scores = [s * (route_boost if p.route == route else 1.0) for s, p in zip(scores, self.passages)]
        ranked = sorted(zip(self.passages, scores), key=lambda t: -t[1])
        return [(p, s) for p, s in ranked[:k] if s > 0]


# ------------------------------------------------------------------ drafting

SYSTEM = """You draft public replies for an airline's social media customer-care team.
Use only the policy passages provided. Every factual claim must come from them.
Cite the passages you used as [id] at the end, for example [baggage-delayed#3].
Keep the reply under 280 characters, friendly and plain. Never ask for confirmation codes,
card numbers or phone numbers in public; ask the customer to send a direct message.
Never promise an amount, a time or a benefit that isn't in the passages.
Reply with JSON only: {"reply": "...", "citations": ["id", ...]}"""

NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
WORD_NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
WORD_NUMBERS.update({"fourteen": "14", "twenty": "20", "thirty": "30", "ninety": "90"})


def spell_out_numbers(text: str) -> str:
    """Policy says "three hours", a reply may say "3 hours". Both should count as grounded."""
    return re.sub(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", lambda m: WORD_NUMBERS[m.group(1).lower()], text,
                  flags=re.IGNORECASE)


@dataclass
class Draft:
    reply: str
    citations: list[str]
    grounded: bool
    problems: list[str]
    passages: list[dict]
    generator: str


def check_grounded(reply: str, citations: list[str], passages: list[Passage]) -> list[str]:
    problems = []
    ids = {p.id for p in passages}
    for c in citations:
        if c not in ids:
            problems.append(f"cites {c}, which wasn't retrieved")
    source = spell_out_numbers(" ".join(p.text for p in passages))
    source_numbers = {n.replace(",", "") for n in NUMBER.findall(source)}
    for n in NUMBER.findall(re.sub(r"\[[^\]]+\]", "", reply)):
        if n.replace(",", "") not in source_numbers:
            problems.append(f"mentions {n}, which isn't in the cited policy")
    if len(re.sub(r"\s*\[[^\]]+\]", "", reply)) > 280:
        problems.append("longer than 280 characters")
    if not citations:
        problems.append("no citations")
    return problems


# Sentences written for staff ("Agents should...") must never be quoted at a customer.
STAFF_ONLY = re.compile(r"\b(agents?|social media agents?)\b", re.IGNORECASE)


def extractive_reply(query: str, passages: list[Passage]) -> tuple[str, list[str]]:
    """
    No model available: quote the most relevant customer-facing sentence of
    policy, plus a next step. Blunt but safe, since every word is policy.
    """
    if passages[0].route == "no_action":
        return ("Thank you for the kind words! We'd love to pass them on to the crew. "
                "Send us a DM with your flight number?"), [passages[0].id]
    q = set(tokenize(query))
    candidates = [(p, s) for p in passages for s in re.split(r"(?<=[.!?])\s+", p.text) if not STAFF_ONLY.search(s)]
    if not candidates:
        return "Sorry about this. Send us a DM with the details and we'll help.", [passages[0].id]
    # Prefer the best-ranked passage; within it, the sentence sharing most words with the tweet.
    rank = {p.id: i for i, p in enumerate(passages)}
    top, best = max(candidates, key=lambda c: (len(q & set(tokenize(c[1]))) - 0.5 * rank[c[0].id], -len(c[1])))
    opener = "Sorry about this."
    tail = " Send us a DM and we'll help."
    room = 280 - len(opener) - len(tail) - 2
    body = best if len(best) <= room else best[: room - 1].rsplit(" ", 1)[0] + "…"
    return f"{opener} {body}{tail}", [top.id]


def draft_reply(query: str, retriever: Retriever, route: str | None = None, k: int = 3) -> Draft:
    hits = retriever.search(query, route=route, k=k)
    passages = [p for p, _ in hits]
    if not passages:
        return Draft("Thanks for reaching out. Send us a DM with the details and we'll take a look.", [], False,
                     ["nothing in the policy base matched; routed to a person"], [], "none")

    reply, citations, generator = None, [], "extractive"
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            reply, citations = _claude(query, passages)
            generator = "claude"
        except Exception as e:  # fall back rather than fail the request
            reply = None
            generator = f"extractive (model error: {type(e).__name__})"
    if reply is None:
        reply, citations = extractive_reply(query, passages)

    problems = check_grounded(reply, citations, passages)
    if problems and generator == "claude":
        # One bad generation shouldn't reach a customer. Use the safe draft instead.
        reply, citations = extractive_reply(query, passages)
        generator = "extractive (model draft failed checks: " + "; ".join(problems) + ")"
        problems = check_grounded(reply, citations, passages)
    return Draft(reply, citations, not problems, problems,
                 [{"id": p.id, "title": p.title, "route": p.route, "text": p.text, "score": round(s, 3)} for p, s in hits],
                 generator)


def _claude(query: str, passages: list[Passage]) -> tuple[str, list[str]]:
    import httpx
    context = "\n\n".join(f"[{p.id}] ({p.title}) {p.text}" for p in passages)
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        json={"model": os.getenv("JETWAY_MODEL", "claude-sonnet-5-5"), "max_tokens": 400, "system": SYSTEM,
              "messages": [{"role": "user", "content": f"Policy passages:\n{context}\n\nCustomer tweet: {query}"}]},
        timeout=20,
    )
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json()["content"])
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return data["reply"].strip(), list(data.get("citations", []))
