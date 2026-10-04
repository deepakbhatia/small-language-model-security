# SEI Data Card

## ATT&CK version
Pinned: **16.1** (`data/attack/v16.1/`). Refresh IDs with `python scripts/download_attack.py`.

## Corpora

| Source | License | Role | Script |
|--------|---------|------|--------|
| Synthetic Atomic/Sigma-style scenarios | Apache-2.0 | Seed + CI full chains | `scripts/compose_synthetic.py` |
| Microsoft GUIDE | CDLA-Permissive-2.0 | Disposition / severity / technique weak labels | `scripts/download_guide.py` + `compose_all.py --guide` |
| MITRE ATT&CK STIX | MITRE terms + notice | Technique vocabulary + RAG cards | `download_attack.py` |
| D3FEND action templates | MITRE terms + notice | Recommended action IDs | `data/d3fend/` |
| Sigma / ESCU (optional future) | DRL-1.1 / Apache-2.0 | Weak labels at scale | TBD composers |

## Denied for commercial train
SecEval (NC), VirusTotal public scrapes, Elastic License v2 rule text redistribution — see `configs/license_allowlist.yaml`.

## Schema
Every training row:
- `messages`: system / user / assistant (compact SEI JSON)
- `meta`: `attack_version`, `source`, `license`, `synthetic`, `curriculum_stage`, `split_group`

## Splits
Hold out by `meta.split_group` (scenario family or GUIDE `OrgId`), never random row shuffle alone.

## Provenance
`scripts/compose_all.py` drops rows failing `sei.verify.validator.verify_example`.
