import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.data import ROUTES, load_raw, normalize, prepare, repair_label_leak
from src.monitor import psi
from src.rag import BM25, Retriever, check_grounded, draft_reply, load_kb
from src.train import should_promote

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def df():
    return prepare(load_raw())


def test_label_leak_is_repaired():
    assert repair_label_leak("is flight 882 Cancelled Flightled?") == "is flight 882 cancelled?"
    assert repair_label_leak("see you Late Flightr") == "see you later"


def test_no_label_strings_survive_cleaning(df):
    assert not df["clean"].str.contains("Cancelled Flight|Late Flight").any()


def test_airline_names_are_hidden():
    assert normalize("@united @AmericanAir thanks @bob") == "@airline @airline thanks @user"


def test_no_duplicate_texts_across_splits(df):
    assert not df["clean"].duplicated().any()


def test_splits_are_in_time_order(df):
    last_train = df.loc[df["split"] == "train", "created"].max()
    first_test = df.loc[df["split"] == "test", "created"].min()
    assert last_train < first_test


def test_routes_are_consistent_with_sentiment(df):
    assert (df.loc[df["airline_sentiment"] != "negative", "route"] == "no_action").all()
    assert set(df["route"]) == set(ROUTES)


def test_psi_is_zero_for_identical_and_grows_with_shift():
    a = np.array([50, 30, 20.0])
    assert psi(a, a) == pytest.approx(0, abs=1e-9)
    assert psi(a, np.array([20, 30, 50.0])) > 0.2


def test_promotion_rule_blocks_regressions():
    champ = {"route": {"macro_f1": 0.60, "per_class_f1": {"baggage": 0.7, "booking": 0.5}}}
    better_but_breaks_a_queue = {"route": {"macro_f1": 0.63, "per_class_f1": {"baggage": 0.9, "booking": 0.36}}}
    clearly_better = {"route": {"macro_f1": 0.63, "per_class_f1": {"baggage": 0.72, "booking": 0.54}}}
    assert should_promote(better_but_breaks_a_queue, champ)[0] is False
    assert should_promote(clearly_better, champ)[0] is True


def test_bm25_prefers_matching_passages():
    r = Retriever()
    top = r.search("my bag was damaged, broken wheel", k=1)[0][0]
    assert top.doc == "baggage-damaged"


def test_grounding_check_catches_invented_numbers():
    passages = [p for p in load_kb() if p.doc == "delay-compensation"]
    ok = check_grounded("Delays of 3 hours or more get a meal voucher. [delay-compensation#2]",
                        ["delay-compensation#2"], passages)
    bad = check_grounded("We'll send you $500 for the delay. [delay-compensation#2]", ["delay-compensation#2"], passages)
    assert ok == [] and any("$500" in p for p in bad)


def test_draft_without_a_model_is_grounded(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    d = draft_reply("@airline you cancelled my flight, do you cover a hotel tonight?", Retriever(), route="disruption")
    assert d.grounded and d.citations and len(d.reply) <= 280


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["JETWAY_LOG"] = str(tmp_path_factory.mktemp("logs") / "p.jsonl")
    from fastapi.testclient import TestClient
    from src.service import app
    with TestClient(app) as c:
        yield c


def test_api_triage_routes_a_baggage_complaint(client):
    r = client.post("/triage", json={"text": "@united my bag never showed up in Denver, lost luggage again"}).json()
    assert r["route"] == "baggage" and r["sentiment"] == "negative" and r["model_version"]


def test_api_rejects_empty_input(client):
    assert client.post("/triage", json={"text": ""}).status_code == 422


def test_api_logs_predictions_and_exposes_metrics(client):
    client.post("/triage/batch", json={"texts": ["@delta thanks, great flight!", "@delta flight delayed 4 hours"]})
    lines = Path(os.environ["JETWAY_LOG"]).read_text().strip().splitlines()
    assert len(lines) >= 2 and "route" in json.loads(lines[-1])
    assert "jetway_predictions_total" in client.get("/metrics").text


def test_api_draft_requires_approval_and_cites_policy(client):
    r = client.post("/draft", json={"text": "@united cancelled my flight and I'm stuck overnight, hotel?"}).json()
    assert r["requires_approval"] is True and r["citations"] and r["grounded"]
