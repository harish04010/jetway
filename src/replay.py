"""
Replays the week as if the model were live.

Trains on Feb 16-19 only, then feeds each later day through the monitor as
unseen traffic, and also scores it against the labels that a real desk would
only get days later. That shows whether the cheap, label-free signals (route
PSI, confidence) move when real accuracy moves.

    python -m src.replay
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .data import ROUTES, load_raw, prepare
from .models import baseline
from .monitor import report, vocabulary

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    df = prepare(load_raw())
    day = df["created"].dt.strftime("%Y-%m-%d")
    ref = df[day < "2015-02-20"]
    m = baseline()["route"].fit(ref["clean"], ref["route"])
    order = [list(m.classes_).index(c) for c in ROUTES]

    def predict(frame):
        p = m.predict_proba(frame["clean"])[:, order]
        return [ROUTES[i] for i in p.argmax(1)], p.max(1)

    # The reference is the model's behavior on its own training days.
    ref_routes, ref_conf = predict(ref)
    reference = {"routes": ref_routes, "confidence": ref_conf, "texts": ref["clean"].tolist(),
                 "vocab": vocabulary(ref["clean"])}
    rows = []
    for d, g in df[day >= "2015-02-20"].groupby(day):
        routes, conf = predict(g)
        r = report(reference, {"routes": routes, "confidence": conf, "texts": g["clean"].tolist()})
        r["day"] = d
        r["route_macro_f1"] = round(float(f1_score(g["route"], routes, average="macro")), 4)
        r["true_disruption_share"] = round(float((g["route"] == "disruption").mean()), 4)
        rows.append(r)
        print(f"{d}  {r['messages']:>5} msgs  {r['status']:<5} PSI {r['route_psi']:.3f}  "
              f"conf {r['mean_confidence']:.3f}  OOV {r['oov_rate']:.3f}  true F1 {r['route_macro_f1']:.3f}  "
              f"disruption {r['true_disruption_share']:.0%}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "drift_replay.json").write_text(json.dumps({"trained_on": "2015-02-16 to 2015-02-19", "days": rows}, indent=1))


if __name__ == "__main__":
    main()
