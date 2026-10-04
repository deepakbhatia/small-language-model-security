"""Lightweight ATT&CK card retriever (keyword overlap; no heavy embedder required)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CARDS = REPO_ROOT / "data" / "attack" / "v16.1" / "technique_cards.jsonl"


@dataclass
class TechniqueCard:
    id: str
    name: str
    tactic: str
    description: str
    detection: str
    keywords: list[str]

    def render(self) -> str:
        return (
            f"{self.id} | {self.name} | {self.tactic}\n"
            f"{self.description}\nDetection: {self.detection}"
        )


def load_cards(path: Path | None = None) -> list[TechniqueCard]:
    p = path or DEFAULT_CARDS
    cards: list[TechniqueCard] = []
    if not p.exists():
        return cards
    text = p.read_text().strip()
    if not text:
        return cards
    # Support true JSONL or a JSON array
    if text.startswith("["):
        objs = json.loads(text)
    else:
        objs = []
        buf = ""
        depth = 0
        for ch in text:
            buf += ch
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(json.loads(buf))
                    buf = ""
    for obj in objs:
        cards.append(
            TechniqueCard(
                id=obj["id"],
                name=obj["name"],
                tactic=obj["tactic"],
                description=obj.get("description", ""),
                detection=obj.get("detection", ""),
                keywords=[k.lower() for k in obj.get("keywords", [])],
            )
        )
    return cards


def retrieve(telemetry_text: str, *, top_k: int = 3, cards: list[TechniqueCard] | None = None) -> list[TechniqueCard]:
    cards = cards or load_cards()
    text = telemetry_text.lower()
    tokens = set(re.findall(r"[a-z0-9_.\\/-]+", text))
    scored: list[tuple[int, TechniqueCard]] = []
    for card in cards:
        score = 0
        for kw in card.keywords:
            if kw.lower() in text:
                score += 3
            elif any(tok in kw.lower() or kw.lower() in tok for tok in tokens):
                score += 1
        if score:
            scored.append((score, card))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [c for _, c in scored[:top_k]]


def render_rag_block(telemetry_text: str, *, top_k: int = 3) -> str:
    hits = retrieve(telemetry_text, top_k=top_k)
    if not hits:
        return ""
    return "\n---\n".join(c.render() for c in hits)
