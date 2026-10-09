#!/usr/bin/env python3
"""Download Atomic Red Team + SigmaHQ rule corpora for SEI compose.

  python scripts/download_attack_corpora.py
  # → data/raw/atomic/atomics , data/raw/sigma/rules
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

import httpx

ATOMIC_ZIP = "https://github.com/redcanaryco/atomic-red-team/archive/refs/heads/master.zip"
SIGMA_ZIP = "https://github.com/SigmaHQ/sigma/archive/refs/heads/master.zip"


def _extract_prefix(zf: zipfile.ZipFile, prefix_parts: tuple[str, ...], dest: Path) -> int:
    """Extract zip members whose path contains .../prefix_parts/... into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = Path(info.filename).parts
        # find prefix sequence in parts
        for i in range(len(parts) - len(prefix_parts) + 1):
            if tuple(parts[i : i + len(prefix_parts)]) == prefix_parts:
                rel = Path(*parts[i + len(prefix_parts) :])
                if not rel.parts:
                    break
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
                break
    return n


def download_zip(url: str, timeout: float = 600.0) -> bytes:
    print(f"fetching {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_bytes():
            buf.write(chunk)
        return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--skip-atomic", action="store_true")
    parser.add_argument("--skip-sigma", action="store_true")
    parser.add_argument(
        "--sigma-subdirs",
        default="windows,linux,network,cloud",
        help="Comma list of sigma/rules subdirs to extract",
    )
    args = parser.parse_args()
    root = args.out_root
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_atomic:
        atomic_dest = root / "atomic" / "atomics"
        if atomic_dest.exists() and any(atomic_dest.glob("T*/T*.yaml")):
            print(f"atomic already present → {atomic_dest}")
        else:
            data = download_zip(ATOMIC_ZIP)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                n = _extract_prefix(zf, ("atomics",), atomic_dest)
            print(f"extracted {n} atomic files → {atomic_dest}")

    if not args.skip_sigma:
        sigma_dest = root / "sigma" / "rules"
        subdirs = [s.strip() for s in args.sigma_subdirs.split(",") if s.strip()]
        if sigma_dest.exists() and any(sigma_dest.rglob("*.yml")):
            print(f"sigma already present → {sigma_dest}")
        else:
            data = download_zip(SIGMA_ZIP)
            total = 0
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for sub in subdirs:
                    n = _extract_prefix(zf, ("rules", sub), sigma_dest / sub)
                    print(f"  sigma/{sub}: {n} files")
                    total += n
            print(f"extracted {total} sigma files → {sigma_dest}")

    print("done")


if __name__ == "__main__":
    main()
