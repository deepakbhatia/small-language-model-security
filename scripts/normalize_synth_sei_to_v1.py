#!/usr/bin/env python3
"""Normalize Claude/ChatGPT synthetic SEI rows to schemas/sei_v1.json + repo ORPO shape.

Produces per-source train / holdout SFT JSONL and ORPO preference JSONL.

  PYTHONPATH=src:. python scripts/normalize_synth_sei_to_v1.py
  PYTHONPATH=src:. python scripts/normalize_synth_sei_to_v1.py --holdout-frac 0.2 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ATTACK_CARDS = REPO / "data" / "attack" / "v16.1" / "technique_cards.jsonl"
ATTACK_IDS = REPO / "data" / "attack" / "v16.1" / "technique_ids.txt"

CONF_STR = {
    "very_low": 0.2,
    "low": 0.35,
    "medium": 0.55,
    "med": 0.55,
    "high": 0.8,
    "very_high": 0.92,
    "critical": 0.95,
}
SEV_SCORE = {
    "informational": 10,
    "low": 25,
    "medium": 50,
    "high": 75,
    "critical": 90,
}
# Map free-text / action_id hints → SEI recommended_actions.type
ACTION_TYPE_MAP = [
    (re.compile(r"contain|isolate|disable_account|revoke_session|block_source", re.I), "contain"),
    (re.compile(r"eradicate|remove_persistence|wipe|reimage", re.I), "eradicate"),
    (re.compile(r"recover|restore", re.I), "recover"),
    (re.compile(r"tune|close_false|close_benign|detection", re.I), "tune_detection"),
    (re.compile(r"notify|page|ticket|inform", re.I), "notify"),
    (re.compile(r"hunt|collect|request_context|monitor|correlate|query|gather|triage", re.I), "hunt"),
]


def load_attack() -> tuple[set[str], dict[str, dict[str, str]]]:
    ids: set[str] = set()
    if ATTACK_IDS.exists():
        ids = {ln.strip() for ln in ATTACK_IDS.read_text().splitlines() if ln.strip()}
    cards: dict[str, dict[str, str]] = {}
    if ATTACK_CARDS.exists():
        for line in ATTACK_CARDS.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            tid = o.get("id")
            if tid:
                cards[tid] = {"name": o.get("name") or tid, "tactic": o.get("tactic") or "Unknown"}
                ids.add(tid)
    return ids, cards


def _conf(v: Any) -> float:
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1.0:  # accidental 0-100
            x = x / 100.0
        return max(0.0, min(1.0, round(x, 4)))
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "_")
        if s in CONF_STR:
            return CONF_STR[s]
        try:
            return _conf(float(s))
        except ValueError:
            return 0.5
    return 0.5


def _sev_level(v: Any) -> str:
    if isinstance(v, dict):
        lvl = str(v.get("level") or "medium").lower()
    else:
        lvl = str(v or "medium").lower()
    if lvl in SEV_SCORE:
        return lvl
    if "info" in lvl:
        return "informational"
    if "crit" in lvl:
        return "critical"
    if "high" in lvl:
        return "high"
    if "low" in lvl:
        return "low"
    return "medium"


def _action_type(raw: dict[str, Any]) -> str:
    if raw.get("type") in {"contain", "eradicate", "recover", "hunt", "notify", "tune_detection"}:
        return str(raw["type"])
    blob = " ".join(
        str(raw.get(k) or "")
        for k in ("type", "action", "action_id", "description", "d3fend_id")
    )
    for rx, t in ACTION_TYPE_MAP:
        if rx.search(blob):
            return t
    return "hunt"


def _parse_jsonish(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        return json.loads(v)
    raise TypeError(type(v))


def extract_flat_sei(obj: Any) -> dict[str, Any]:
    """Assistant content may be flat SEI, nested sei_v1, or chat wrapper."""
    obj = _parse_jsonish(obj)
    if isinstance(obj, list):
        # Claude ORPO: [{"role":"assistant","content":"..."}]
        for m in obj:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return extract_flat_sei(m.get("content"))
        raise ValueError("no assistant in list")
    if not isinstance(obj, dict):
        raise TypeError("expected dict SEI")
    if "incident" in obj and isinstance(obj["incident"], dict):
        # already nested-ish — flatten for unify
        inc = obj["incident"]
        return {
            "disposition": inc.get("disposition"),
            "category": inc.get("category"),
            "summary": inc.get("summary"),
            "techniques": obj.get("techniques") or [],
            "severity": obj.get("severity"),
            "confidence": obj.get("confidence"),
            "evidence": obj.get("evidence") or [],
            "recommended_actions": obj.get("recommended_actions") or [],
            "rationale": obj.get("rationale") or (obj.get("severity") or {}).get("rationale")
            if isinstance(obj.get("severity"), dict)
            else obj.get("rationale"),
            "missing_context": obj.get("missing_context") or [],
        }
    return obj


def to_sei_v1(
    flat: dict[str, Any],
    *,
    user_text: str,
    attack_ids: set[str],
    cards: dict[str, dict[str, str]],
    require_grounded_evidence: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Convert flat synth SEI → sei_v1.

    For ORPO *rejected* answers, set require_grounded_evidence=False so intentional
    ungrounded quotes are kept (that is the failure mode being taught).
    """
    errors: list[str] = []
    disp = flat.get("disposition")
    if disp not in {
        "true_positive",
        "false_positive",
        "benign_true_positive",
        "needs_more_data",
    }:
        return None, [f"bad_disposition:{disp}"]

    rationale = str(flat.get("rationale") or flat.get("reasoning") or "").strip()
    missing = flat.get("missing_context") or []
    if isinstance(missing, list) and missing and not rationale:
        rationale = "Missing context: " + "; ".join(str(x) for x in missing[:6])
    summary = str(flat.get("summary") or rationale or f"Synthetic triage: {disp}")[:500]
    category = str(flat.get("category") or disp)[:64]

    sev_level = _sev_level(flat.get("severity"))
    if sev_level not in SEV_SCORE:
        sev_level = "medium"
    if isinstance(flat.get("severity"), dict) and flat["severity"].get("score") is not None:
        try:
            score = int(flat["severity"]["score"])
        except (TypeError, ValueError):
            score = SEV_SCORE[sev_level]
    else:
        score = SEV_SCORE[sev_level]
    score = max(0, min(100, score))
    sev_rat = ""
    if isinstance(flat.get("severity"), dict):
        sev_rat = str(flat["severity"].get("rationale") or "")
    if not sev_rat:
        sev_rat = (rationale[:300] if rationale else f"Severity {sev_level} for {disp}")[:300]

    conf = _conf(flat.get("confidence"))
    if disp == "needs_more_data" and conf > 0.45:
        conf = min(conf, 0.4)

    techniques_out: list[dict[str, Any]] = []
    for t in flat.get("techniques") or []:
        if isinstance(t, str):
            tid = t
            name = cards.get(tid, {}).get("name", tid)
            tactic = cards.get(tid, {}).get("tactic", "Unknown")
            tconf = conf
        elif isinstance(t, dict):
            tid = str(t.get("id") or "")
            name = str(t.get("name") or cards.get(tid, {}).get("name") or tid)
            tactic = str(t.get("tactic") or cards.get(tid, {}).get("tactic") or "Unknown")
            tconf = _conf(t.get("confidence", conf))
        else:
            errors.append("bad_technique_row")
            continue
        if tid and tid not in attack_ids:
            errors.append(f"unknown_technique:{tid}")
            continue
        if not tid:
            continue
        techniques_out.append(
            {"id": tid, "name": name[:200], "tactic": tactic[:64], "confidence": tconf}
        )
        if len(techniques_out) >= 8:
            break
    if disp == "needs_more_data":
        techniques_out = []

    evidence_out: list[dict[str, Any]] = []
    for e in flat.get("evidence") or []:
        if not isinstance(e, dict):
            continue
        val = e.get("value")
        if val is None:
            continue
        val = str(val)
        grounded = val in user_text
        if not grounded:
            errors.append("ungrounded_evidence")
            if require_grounded_evidence:
                continue
        idx = int(e.get("telemetry_index") or 0)
        if idx < 0:
            idx = 0
        evidence_out.append(
            {
                "telemetry_index": idx,
                "field": str(e.get("field") or "message")[:128],
                "value": val,
                "why": str(
                    e.get("why")
                    or rationale
                    or ("Grounded quote from telemetry" if grounded else "Ungrounded (rejected)")
                )[:300],
            }
        )
        if len(evidence_out) >= 16:
            break

    actions_out: list[dict[str, Any]] = []
    for i, a in enumerate(flat.get("recommended_actions") or [], start=1):
        if not isinstance(a, dict):
            continue
        atype = _action_type(a)
        if disp in {"needs_more_data", "false_positive", "benign_true_positive"} and atype in {
            "contain",
            "eradicate",
        }:
            atype = "hunt"
        text = str(
            a.get("action")
            or a.get("description")
            or a.get("action_id")
            or atype
        )[:300]
        automatable = bool(a.get("automatable", False))
        if atype in {"contain", "eradicate"}:
            automatable = False
        row: dict[str, Any] = {
            "priority": int(a.get("priority") or min(i, 5)),
            "type": atype,
            "action": text,
            "automatable": automatable,
        }
        if a.get("d3fend_id"):
            row["d3fend_id"] = str(a["d3fend_id"])[:64]
        if a.get("target"):
            row["target"] = str(a["target"])[:128]
        actions_out.append(row)
        if len(actions_out) >= 5:
            break

    sei = {
        "incident": {
            "disposition": disp,
            "category": category,
            "summary": summary,
        },
        "techniques": techniques_out,
        "severity": {
            "level": sev_level,
            "score": score,
            "rationale": sev_rat,
        },
        "confidence": conf,
        "evidence": evidence_out,
        "recommended_actions": actions_out,
    }
    if (
        require_grounded_evidence
        and "ungrounded_evidence" in errors
        and not evidence_out
        and flat.get("evidence")
    ):
        return None, errors
    return sei, errors


def user_from_messages(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def system_from_messages(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content") or ""
    return (
        "You are a SOC triage model. Emit ONLY valid SEI JSON. "
        "Technique IDs must be from ATT&CK v16.1. "
        "Evidence values must be exact substrings of the telemetry. "
        "Use needs_more_data / empty lists when unsupported."
    )


def normalize_sft_row(
    row: dict[str, Any],
    *,
    source: str,
    attack_ids: set[str],
    cards: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = user_from_messages(messages)
    try:
        flat = extract_flat_sei(messages[-1]["content"])
    except Exception as e:  # noqa: BLE001
        return None
    sei, errs = to_sei_v1(flat, user_text=user, attack_ids=attack_ids, cards=cards)
    if sei is None:
        return None
    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "attack_version": meta.get("attack_version") or "16.1",
            "synthetic": True,
            "license": meta.get("license") or "Apache-2.0",
            "generator_source": source,
            "normalized_to": "sei_v1",
            "normalize_warnings": errs,
        }
    )
    if not meta.get("split_group"):
        meta["split_group"] = f"{source}-{meta.get('id', random.randrange(1_000_000))}"
    out_messages = [
        {"role": "system", "content": system_from_messages(messages)},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(sei, ensure_ascii=False, separators=(",", ":"))},
    ]
    return {"messages": out_messages, "meta": meta}


def normalize_orpo_row(
    row: dict[str, Any],
    *,
    source: str,
    attack_ids: set[str],
    cards: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    prompt = row.get("prompt") or row.get("prompt_messages")
    if not isinstance(prompt, list):
        return None
    user = user_from_messages(prompt)
    try:
        chosen_flat = extract_flat_sei(row["chosen"])
        rejected_flat = extract_flat_sei(row["rejected"])
    except Exception:
        return None
    chosen, cerr = to_sei_v1(
        chosen_flat, user_text=user, attack_ids=attack_ids, cards=cards, require_grounded_evidence=True
    )
    rejected, rerr = to_sei_v1(
        rejected_flat,
        user_text=user,
        attack_ids=attack_ids,
        cards=cards,
        require_grounded_evidence=False,
    )
    if chosen is None or rejected is None:
        return None
    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "attack_version": meta.get("attack_version") or "16.1",
            "synthetic": True,
            "license": meta.get("license") or "Apache-2.0",
            "generator_source": source,
            "normalized_to": "sei_v1",
            "normalize_warnings": list(set(cerr + rerr)),
        }
    )
    recipe = row.get("rejection_type") or row.get("recipe") or meta.get("recipe") or "synth_reject"
    # prompt_messages: system+user only
    prompt_messages = [m for m in prompt if m.get("role") in ("system", "user")]
    if not any(m.get("role") == "system" for m in prompt_messages):
        prompt_messages = [{"role": "system", "content": system_from_messages(prompt)}] + [
            m for m in prompt_messages if m.get("role") == "user"
        ]
    return {
        "prompt_messages": prompt_messages,
        "chosen": json.dumps(chosen, ensure_ascii=False, separators=(",", ":")),
        "rejected": json.dumps(rejected, ensure_ascii=False, separators=(",", ":")),
        "recipe": recipe,
        "meta": meta,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_by_group(
    rows: list[dict[str, Any]],
    *,
    holdout_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    groups = sorted({(r.get("meta") or {}).get("split_group") or "nogroup" for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_hold = max(1, int(round(len(groups) * holdout_frac))) if len(groups) > 1 else 0
    hold_set = set(groups[:n_hold])
    train, hold = [], []
    for r in rows:
        g = (r.get("meta") or {}).get("split_group") or "nogroup"
        (hold if g in hold_set else train).append(r)
    return train, hold, sorted(hold_set)


def quality_stats(rows: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    disps: Counter[str] = Counter()
    warn: Counter[str] = Counter()
    n_tech = 0
    n_evid = 0
    for r in rows:
        if kind == "sft":
            sei = json.loads(r["messages"][-1]["content"])
        else:
            sei = json.loads(r["chosen"])
        disps[sei["incident"]["disposition"]] += 1
        n_tech += len(sei.get("techniques") or [])
        n_evid += len(sei.get("evidence") or [])
        for w in (r.get("meta") or {}).get("normalize_warnings") or []:
            warn[w.split(":")[0]] += 1
    return {
        "n": len(rows),
        "disposition": dict(disps),
        "avg_techniques": round(n_tech / max(len(rows), 1), 3),
        "avg_evidence": round(n_evid / max(len(rows), 1), 3),
        "warning_counts": dict(warn),
    }


def process_source(
    *,
    name: str,
    sft_path: Path | None,
    orpo_path: Path | None,
    out_dir: Path,
    holdout_frac: float,
    seed: int,
    attack_ids: set[str],
    cards: dict[str, dict[str, str]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"source": name, "inputs": {}}
    sft_norm: list[dict[str, Any]] = []
    if sft_path and sft_path.exists():
        raw = load_jsonl(sft_path)
        report["inputs"]["sft_raw"] = len(raw)
        drop = 0
        for row in raw:
            n = normalize_sft_row(row, source=name, attack_ids=attack_ids, cards=cards)
            if n is None:
                drop += 1
            else:
                sft_norm.append(n)
        report["sft_dropped"] = drop
    train, hold, hold_groups = split_by_group(sft_norm, holdout_frac=holdout_frac, seed=seed)
    # Mark holdout meta
    for r in hold:
        r["meta"]["split"] = "holdout_synth"
        r["meta"]["blind_held_out"] = False  # synth contrast eval, not sealed blind
    for r in train:
        r["meta"]["split"] = "train"

    write_jsonl(out_dir / f"{name}_sft_train.jsonl", train)
    write_jsonl(out_dir / f"{name}_sft_holdout.jsonl", hold)
    report["sft_train"] = quality_stats(train, kind="sft")
    report["sft_holdout"] = quality_stats(hold, kind="sft")
    report["holdout_split_groups"] = hold_groups
    report["n_holdout_groups"] = len(hold_groups)

    orpo_norm: list[dict[str, Any]] = []
    if orpo_path and orpo_path.exists():
        raw = load_jsonl(orpo_path)
        report["inputs"]["orpo_raw"] = len(raw)
        drop = 0
        hold_group_set = set(hold_groups)
        for row in raw:
            n = normalize_orpo_row(row, source=name, attack_ids=attack_ids, cards=cards)
            if n is None:
                drop += 1
                continue
            g = (n.get("meta") or {}).get("split_group") or "nogroup"
            if g in hold_group_set:
                # keep ORPO out of train if group reserved for holdout
                n["meta"]["split"] = "holdout_synth_excluded_from_train"
                continue
            n["meta"]["split"] = "train"
            orpo_norm.append(n)
        report["orpo_dropped"] = drop
        report["orpo_excluded_holdout_groups"] = report["inputs"]["orpo_raw"] - drop - len(orpo_norm)
    write_jsonl(out_dir / f"{name}_orpo_train.jsonl", orpo_norm)
    report["orpo_train"] = quality_stats(orpo_norm, kind="orpo") if orpo_norm else {"n": 0}
    return report


def recommend_chatgpt(claude_rep: dict[str, Any], chatgpt_rep: dict[str, Any]) -> dict[str, Any]:
    """Heuristic: use ChatGPT train if drop rate low and NMD share reasonable."""
    def drop_rate(rep: dict[str, Any], kind: str) -> float:
        raw = (rep.get("inputs") or {}).get(f"{kind}_raw") or 0
        dropped = rep.get(f"{kind}_dropped") or 0
        return dropped / raw if raw else 1.0

    c_nmd = (chatgpt_rep.get("sft_train") or {}).get("disposition", {}).get("needs_more_data", 0)
    c_n = (chatgpt_rep.get("sft_train") or {}).get("n") or 1
    nmd_frac = c_nmd / c_n
    sft_drop = drop_rate(chatgpt_rep, "sft")
    orpo_drop = drop_rate(chatgpt_rep, "orpo")
    use = sft_drop < 0.15 and orpo_drop < 0.25 and nmd_frac >= 0.2 and (chatgpt_rep.get("sft_train") or {}).get("n", 0) >= 100
    return {
        "use_chatgpt_for_training": use,
        "reasons": {
            "sft_drop_rate": round(sft_drop, 3),
            "orpo_drop_rate": round(orpo_drop, 3),
            "nmd_fraction_train": round(nmd_frac, 3),
            "claude_sft_train_n": (claude_rep.get("sft_train") or {}).get("n"),
            "chatgpt_sft_train_n": (chatgpt_rep.get("sft_train") or {}).get("n"),
            "rule": "use if sft_drop<0.15 and orpo_drop<0.25 and nmd_frac>=0.2 and n>=100",
        },
        "mix_suggestion": (
            "Primary: claude_orpo_train + claude_sft_train; "
            "Secondary (if use): chatgpt_orpo_train + chatgpt_sft_train at <=50% of Claude volume"
            if use
            else "Train on Claude only; keep ChatGPT holdout for analysis only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "processed" / "certainty_synth",
    )
    args = parser.parse_args()

    attack_ids, cards = load_attack()
    print(f"ATT&CK ids loaded: {len(attack_ids)}")

    sources = [
        (
            "claude",
            REPO / "data/processed/claude_sei_synth_certainty_sft.jsonl",
            REPO / "data/processed/claude_sei_synth_certainty_orpo.jsonl",
        ),
        (
            "chatgpt",
            REPO / "data/processed/chatgpt_sei_synthetic_train_500.jsonl",
            REPO / "data/processed/chatgpt_sei_orpo_preferences_500.jsonl",
        ),
    ]
    reports = {}
    for name, sft, orpo in sources:
        print(f"\n=== normalizing {name} ===")
        reports[name] = process_source(
            name=name,
            sft_path=sft,
            orpo_path=orpo,
            out_dir=args.out_dir,
            holdout_frac=args.holdout_frac,
            seed=args.seed + (0 if name == "claude" else 99),
            attack_ids=attack_ids,
            cards=cards,
        )

    rec = recommend_chatgpt(reports["claude"], reports["chatgpt"])
    summary = {
        "out_dir": str(args.out_dir),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "sources": reports,
        "recommendation": rec,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "QUALITY_REPORT.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Merged train pointers (Claude always; ChatGPT if recommended)
    merged_sft = load_jsonl(args.out_dir / "claude_sft_train.jsonl")
    merged_orpo = load_jsonl(args.out_dir / "claude_orpo_train.jsonl")
    if rec["use_chatgpt_for_training"]:
        # Cap ChatGPT SFT at 50% of Claude train size
        cg = load_jsonl(args.out_dir / "chatgpt_sft_train.jsonl")
        cap = max(1, len(merged_sft) // 2)
        merged_sft = merged_sft + cg[:cap]
        cgo = load_jsonl(args.out_dir / "chatgpt_orpo_train.jsonl")
        ocap = max(1, len(merged_orpo) // 2) if merged_orpo else len(cgo)
        merged_orpo = merged_orpo + cgo[:ocap]
    write_jsonl(args.out_dir / "merged_sft_train.jsonl", merged_sft)
    write_jsonl(args.out_dir / "merged_orpo_train.jsonl", merged_orpo)
    # Combined synth holdout for local contrast eval (not sealed blind)
    hold = load_jsonl(args.out_dir / "claude_sft_holdout.jsonl") + load_jsonl(
        args.out_dir / "chatgpt_sft_holdout.jsonl"
    )
    write_jsonl(args.out_dir / "merged_sft_holdout.jsonl", hold)

    summary["merged"] = {
        "sft_train": len(merged_sft),
        "orpo_train": len(merged_orpo),
        "sft_holdout": len(hold),
        "includes_chatgpt": rec["use_chatgpt_for_training"],
    }
    (args.out_dir / "QUALITY_REPORT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"recommendation": rec, "merged": summary["merged"]}, indent=2))
    print(f"wrote report → {args.out_dir / 'QUALITY_REPORT.json'}")


if __name__ == "__main__":
    main()
