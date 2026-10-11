"""Unit tests for verifier, metrics, preferences, RAG."""

from __future__ import annotations

import json

from sei.compose.synthetic import iter_seed_examples, scenario_to_example, SCENARIOS
from sei.eval_metrics import evaluate_paired
from sei.rag.attack_rag import retrieve
from sei.train.preferences import (
    reject_always_true_positive,
    reject_bad_technique,
    reject_over_severity,
    reject_wrong_disposition,
)
from sei.verify.validator import verify_example


def test_seed_examples_verify():
    rows = iter_seed_examples([4])
    assert len(rows) >= 8
    for row in rows:
        errs = verify_example(row["messages"][1]["content"], row["messages"][2]["content"])
        assert errs == [], (row["id"], errs)


def test_ungrounded_evidence_fails():
    ex = scenario_to_example(SCENARIOS[0], stage=4)
    obj = json.loads(ex["messages"][2]["content"])
    obj["evidence"][0]["value"] = "NOT_IN_TELEMETRY"
    errs = verify_example(ex["messages"][1]["content"], obj)
    assert any("ungrounded_evidence" in e for e in errs)


def test_bad_technique_fails():
    ex = scenario_to_example(SCENARIOS[0], stage=4)
    obj = json.loads(ex["messages"][2]["content"])
    obj["techniques"][0]["id"] = "T9999"
    errs = verify_example(ex["messages"][1]["content"], obj)
    assert any("bad_technique" in e for e in errs)


def test_preferences_change_severity():
    ex = scenario_to_example(SCENARIOS[0], stage=4)
    gold = ex["messages"][2]["content"]
    bad = json.loads(reject_over_severity(gold))
    assert bad["severity"]["level"] != json.loads(gold)["severity"]["level"] or bad["evidence"] == []


def test_reject_bad_technique_id():
    ex = scenario_to_example(SCENARIOS[0], stage=4)
    bad = json.loads(reject_bad_technique(ex["messages"][2]["content"]))
    assert bad["techniques"][0]["id"] == "T9999.999"


def test_reject_wrong_disposition_flips():
    ex = scenario_to_example(SCENARIOS[0], stage=4)
    gold = json.loads(ex["messages"][2]["content"])
    bad = json.loads(reject_wrong_disposition(gold))
    assert bad["incident"]["disposition"] != gold["incident"]["disposition"]


def test_reject_always_true_positive():
    # Pick a non-TP seed if present; otherwise force FP gold
    gold = json.loads(scenario_to_example(SCENARIOS[0], stage=4)["messages"][2]["content"])
    gold["incident"]["disposition"] = "false_positive"
    bad = json.loads(reject_always_true_positive(gold))
    assert bad["incident"]["disposition"] == "true_positive"


def test_rag_retrieves_cmd():
    hits = retrieve('CommandLine":"cmd.exe /c whoami"')
    assert hits
    assert any(h.id.startswith("T1059") for h in hits)


def test_eval_identity():
    rows = [scenario_to_example(s, stage=4) for s in SCENARIOS[:3]]
    golds = [json.loads(r["messages"][2]["content"]) for r in rows]
    report = evaluate_paired(golds, golds)
    assert report["disposition"]["macro_f1"] == 1.0
    assert report["techniques"]["exact_id_f1"] == 1.0
    assert report["evidence"]["f1"] == 1.0
