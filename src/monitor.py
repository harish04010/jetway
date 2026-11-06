"""
Drift monitoring: is today's traffic still like the data the model learned from?

Labels arrive days late in customer care (if ever), so the monitor watches
things that are available immediately: what the model predicts, how sure it
is, and how much of the vocabulary it has never seen. PSI (population
stability index) compares distributions; above 0.2 is the usual "investigate"
line, above 0.1 a warning.
"""
import json
import re
from collections import Counter

import numpy as np

from .data import ROUTES

PSI_WARN, PSI_ALERT = 0.1, 0.2


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected / expected.sum(), eps, None)
    a = np.clip(actual / actual.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def route_distribution(routes) -> np.ndarray:
    c = Counter(routes)
    return np.array([c.get(r, 0) for r in ROUTES], dtype=float)


def vocabulary(texts) -> set[str]:
    return {w for t in texts for w in re.findall(r"[a-z']+", t.lower())}


def oov_rate(texts, vocab: set[str]) -> float:
    words = [w for t in texts for w in re.findall(r"[a-z']+", t.lower())]
    return float(np.mean([w not in vocab for w in words])) if words else 0.0


def report(reference: dict, window: dict) -> dict:
    """reference/window: {"routes": [...], "confidence": [...], "texts": [...]}"""
    p = psi(route_distribution(reference["routes"]), route_distribution(window["routes"]))
    ref_vocab = reference.get("vocab") or vocabulary(reference["texts"])
    conf_drop = float(np.mean(reference["confidence"]) - np.mean(window["confidence"]))
    oov = oov_rate(window["texts"], ref_vocab)
    status = "alert" if p >= PSI_ALERT else "warn" if p >= PSI_WARN else "ok"
    ref_share = route_distribution(reference["routes"]) / len(reference["routes"])
    win_share = route_distribution(window["routes"]) / len(window["routes"])
    biggest = ROUTES[int(np.argmax(np.abs(win_share - ref_share)))]
    return {"messages": len(window["routes"]), "route_psi": round(p, 4), "status": status,
            "mean_confidence": round(float(np.mean(window["confidence"])), 4), "confidence_drop": round(conf_drop, 4),
            "oov_rate": round(oov, 4), "biggest_shift": biggest,
            "biggest_shift_share": [round(float(ref_share[ROUTES.index(biggest)]), 4),
                                    round(float(win_share[ROUTES.index(biggest)]), 4)]}


def read_log(path) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {"routes": [r["route"] for r in rows], "confidence": [r["confidence"] for r in rows],
            "texts": [r["text"] for r in rows]}
