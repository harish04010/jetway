"""
Trains the champion candidate and the challenger, logs both runs, and
promotes a new model version only if it clearly beats the current one.

    python -m src.train                # baseline only (fast, no TensorFlow needed)
    python -m src.train --challenger   # also trains the multi-task network

Every run writes runs/<id>/{params,metrics}.json. That's a deliberately small
stand-in for MLflow: same idea (every experiment recorded, comparable, never
overwritten), no server to run.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data import ROUTES, SENTIMENTS, load_raw, prepare

ROOT = Path(__file__).resolve().parent.parent
RUNS, MODELS, RESULTS = ROOT / "runs", ROOT / "models", ROOT / "results"
REGISTRY = MODELS / "registry.json"

# A challenger replaces the champion only if it's clearly better on routing,
# the task that decides where a customer's message goes, and doesn't get
# much worse for any single queue on the way.
PROMOTE_MIN_GAIN = 0.01
PROMOTE_MAX_CLASS_DROP = 0.05


def task_metrics(y_true, proba, labels) -> dict:
    pred = proba.argmax(1)
    per_class = f1_score(y_true, pred, labels=range(len(labels)), average=None, zero_division=0)
    conf = proba.max(1)
    coverage = []
    for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
        keep = conf >= t
        coverage.append({"threshold": t, "automated": float(keep.mean()),
                         "accuracy": float((pred[keep] == y_true[keep]).mean()) if keep.any() else None})
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "per_class_f1": {l: float(v) for l, v in zip(labels, per_class)},
            "coverage": coverage}


def log_run(name: str, params: dict, metrics: dict) -> str:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps(params, indent=1))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return run_id


def label_ids(df):
    return {"sentiment": df["airline_sentiment"].map(SENTIMENTS.index).values,
            "route": df["route"].map(ROUTES.index).values}


def train_baseline(tr, te):
    from .models import baseline
    models, proba = baseline(), {}
    for task, col, labels in (("sentiment", "airline_sentiment", SENTIMENTS), ("route", "route", ROUTES)):
        models[task].fit(tr["clean"], tr[col])
        order = [list(models[task].classes_).index(c) for c in labels]
        proba[task] = models[task].predict_proba(te["clean"])[:, order]
    return models, proba


def train_network(tr, va, te):
    import tensorflow as tf
    from tensorflow import keras
    from .models import class_weights, multitask_network
    T = lambda s: tf.constant(s.tolist())
    ytr, yva = label_ids(tr), label_ids(va)
    model = multitask_network(tr["clean"].values)
    hist = model.fit(
        T(tr["clean"]), ytr,
        sample_weight={"sentiment": class_weights(ytr["sentiment"], 3), "route": class_weights(ytr["route"], 6)},
        validation_data=(T(va["clean"]), yva), epochs=12, batch_size=64, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_route_loss", mode="min", patience=2,
                                                 restore_best_weights=True)])
    out = model.predict(T(te["clean"]), verbose=0)
    return model, {"sentiment": out["sentiment"], "route": out["route"]}, len(hist.history["loss"])


def evaluate(proba, te) -> dict:
    y = label_ids(te)
    return {"sentiment": task_metrics(y["sentiment"], proba["sentiment"], SENTIMENTS),
            "route": task_metrics(y["route"], proba["route"], ROUTES)}


def should_promote(challenger: dict, champion: dict | None) -> tuple[bool, str]:
    if champion is None:
        return True, "no champion yet"
    gain = challenger["route"]["macro_f1"] - champion["route"]["macro_f1"]
    drops = {k: champion["route"]["per_class_f1"][k] - v for k, v in challenger["route"]["per_class_f1"].items()}
    worst = max(drops, key=drops.get)
    if gain < PROMOTE_MIN_GAIN:
        return False, f"route macro-F1 gain {gain:+.3f} is below the {PROMOTE_MIN_GAIN} bar"
    if drops[worst] > PROMOTE_MAX_CLASS_DROP:
        return False, f"the {worst} queue drops {drops[worst]:.3f} F1"
    return True, f"route macro-F1 {gain:+.3f}, no queue drops more than {PROMOTE_MAX_CLASS_DROP}"


def register(models, metrics, run_id) -> str:
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"champion": None, "versions": []}
    version = f"v{len(registry['versions']) + 1}"
    d = MODELS / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, d / "model.joblib", compress=3)
    card = {"version": version, "run": run_id, "kind": "tfidf_logreg", "trained_at": f"{datetime.now():%Y-%m-%d}",
            "labels": {"sentiment": SENTIMENTS, "route": ROUTES}, "test_metrics": metrics}
    (d / "card.json").write_text(json.dumps(card, indent=1))
    champion_metrics = None
    if registry["champion"]:
        champion_metrics = json.loads((MODELS / registry["champion"] / "card.json").read_text())["test_metrics"]
    promote, why = should_promote(metrics, champion_metrics)
    registry["versions"].append({"version": version, "run": run_id, "promoted": promote, "reason": why})
    if promote:
        registry["champion"] = version
    REGISTRY.write_text(json.dumps(registry, indent=1))
    print(f"Registered {version}: {'promoted' if promote else 'not promoted'} ({why})")
    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", action="store_true", help="also train the multi-task network")
    args = ap.parse_args()

    df = prepare(load_raw())
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}")
    summary = {"data": {"train": len(tr), "val": len(va), "test": len(te), "duplicates_removed": 14640 - len(df)}}

    t0 = time.perf_counter()
    models, proba = train_baseline(pd_concat(tr, va), te)  # the baseline has no early stopping, so it can learn from day 7 too
    m_base = evaluate(proba, te)
    run = log_run("baseline", {"kind": "tfidf_logreg", "train_days": "Feb 16-22"}, m_base)
    summary["baseline"] = {"run": run, "seconds": round(time.perf_counter() - t0, 1), **m_base}
    print(f"baseline   sentiment F1 {m_base['sentiment']['macro_f1']:.3f}  route F1 {m_base['route']['macro_f1']:.3f}")
    register(models, m_base, run)

    if args.challenger:
        t0 = time.perf_counter()
        _, proba_nn, epochs = train_network(tr, va, te)
        m_nn = evaluate(proba_nn, te)
        run = log_run("multitask_bigru", {"kind": "keras_bigru_multitask", "epochs_run": epochs,
                                          "train_days": "Feb 16-21", "early_stopping_on": "Feb 22"}, m_nn)
        summary["network"] = {"run": run, "seconds": round(time.perf_counter() - t0, 1), "epochs": epochs, **m_nn}
        promote, why = should_promote(m_nn, m_base)
        summary["network"]["promotion_decision"] = {"promote": promote, "reason": why}
        print(f"network    sentiment F1 {m_nn['sentiment']['macro_f1']:.3f}  route F1 {m_nn['route']['macro_f1']:.3f}  "
              f"-> {'would promote' if promote else 'stays a challenger'}: {why}")
        ens = {k: 0.3 * proba_nn[k] + 0.7 * proba[k] for k in proba}
        summary["ensemble_30_70"] = evaluate(ens, te)
        print(f"ensemble   sentiment F1 {summary['ensemble_30_70']['sentiment']['macro_f1']:.3f}  "
              f"route F1 {summary['ensemble_30_70']['route']['macro_f1']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "experiments.json").write_text(json.dumps(summary, indent=1))


def pd_concat(a, b):
    import pandas as pd
    return pd.concat([a, b])


if __name__ == "__main__":
    main()
