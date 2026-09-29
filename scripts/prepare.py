"""Data preparation: split the whole-domain CFD exports into per-block parquet.

Raw exports live under ``data/raw/`` (see the README):

    data/raw/full_2026-09/Case <n>/<time>.csv   whole-domain exports (five blocks)

Usage:
    python scripts/prepare.py full [--cases 1 9] [--overwrite]   # -> data/processed/full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import FULL_DIR, FULL_RAW_DIR  # noqa: E402
from idealaorta_pinn.data.full_export import convert_all, raw_exports  # noqa: E402


def cmd_full(args) -> None:
    exports = raw_exports()
    print(f"[full] {len(exports)} raw exports under {FULL_RAW_DIR}")
    written = convert_all(cases=args.cases, overwrite=args.overwrite)
    print(f"[full] {len(written)} parquet files written to {FULL_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Data preparation pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("full", help="Split the whole-domain exports into per-block parquet.")
    f.add_argument("--cases", type=int, nargs="*", default=None)
    f.add_argument("--overwrite", action="store_true")
    f.set_defaults(func=cmd_full)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
