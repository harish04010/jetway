"""Does retrieval find the right policy? Scored on 30 hand-labeled questions."""
import json
from pathlib import Path

from .rag import Retriever

EVAL = Path(__file__).resolve().parent.parent / "data" / "rag_eval.jsonl"


def score(retriever: Retriever, rows: list[dict], route_of) -> dict:
    hit1 = hit3 = rr = 0.0
    misses = []
    for row in rows:
        docs = [p.doc for p, _ in retriever.search(row["q"], route=route_of(row), k=10)]
        rank = docs.index(row["doc"]) + 1 if row["doc"] in docs else None
        # One document can contribute several passages; count each document once.
        seen, unique = set(), []
        for d in docs:
            if d not in seen:
                seen.add(d), unique.append(d)
        rank = unique.index(row["doc"]) + 1 if row["doc"] in unique else None
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        if rank != 1:
            misses.append({"q": row["q"], "expected": row["doc"], "got": unique[0] if unique else None})
    n = len(rows)
    return {"hit_at_1": hit1 / n, "hit_at_3": hit3 / n, "mrr": rr / n, "misses": misses}


def run(classify=None) -> dict:
    rows = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    r = Retriever()
    out = {"questions": len(rows),
           "bm25": score(r, rows, lambda row: None),
           "bm25_with_true_route": score(r, rows, lambda row: row["route"])}
    if classify:
        out["bm25_with_predicted_route"] = score(r, rows, lambda row: classify(row["q"]))
    return out


if __name__ == "__main__":
    res = run()
    for k in ("bm25", "bm25_with_true_route"):
        print(f"{k:<24} hit@1 {res[k]['hit_at_1']:.2f}  hit@3 {res[k]['hit_at_3']:.2f}  MRR {res[k]['mrr']:.2f}")
    for m in res["bm25"]["misses"]:
        print("  miss:", m["expected"], "<-", m["got"], "|", m["q"][:70])
