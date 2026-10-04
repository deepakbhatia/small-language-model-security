#!/usr/bin/env python3
"""Download MITRE ATT&CK enterprise technique IDs (STIX)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/attack/v16.1/technique_ids.txt"))
    parser.add_argument("--url", default=STIX_URL)
    args = parser.parse_args()

    print(f"fetching {args.url}")
    r = httpx.get(args.url, timeout=120.0, follow_redirects=True)
    r.raise_for_status()
    bundle = r.json()
    ids = set()
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id", "").startswith("T"):
                ids.add(ref["external_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(ids)) + "\n")
    # also dump cards stub enrichment
    print(f"wrote {len(ids)} technique IDs → {args.out}")


if __name__ == "__main__":
    main()
