"""Data preparation: build the case registry and cache raw CFD exports to parquet.

Expects the 12 ``Case *`` folders (+ ``Results on Slices.xlsx``) to already be
in place under ``data/raw/`` (as shipped alongside this repo).

Subcommands:
    python scripts/prepare.py registry             # discover cases -> data/registry.json
    python scripts/prepare.py cache [--cases 1 4 7] [--overwrite]   # parse CFX CSVs -> parquet cache
    python scripts/prepare.py all                  # registry, then cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import RAW_DIR  # noqa: E402
from idealaorta_pinn.data.cache import cache_one  # noqa: E402
from idealaorta_pinn.data.registry import build_registry  # noqa: E402


# --------------------------------------------------------------------------- registry
def cmd_registry(args) -> None:
    records = build_registry(RAW_DIR, save=True)
    print(f"[registry] {RAW_DIR} -> {len(records)} cases written to data/registry.json")
    for r in records:
        beta = f"{r.beta:.2f}" if r.beta is not None else "  - "
        print(f"  {r.case_id:>2}  {r.inlet_diameter_cm:>4} cm  {r.health:<9} "
              f"{r.symmetry:<13} beta={beta}  {r.available()}")


# --------------------------------------------------------------------------- cache
def cmd_cache(args) -> None:
    from idealaorta_pinn.data.registry import load_registry
    records = load_registry()
    if args.cases:
        records = [r for r in records if r.case_id in set(args.cases)]
    n_ok = n_fail = 0
    for rec in records:
        for kind, phases in rec.files.items():
            for phase in phases:
                try:
                    if cache_one(rec, kind, phase, overwrite=args.overwrite) is not None:
                        n_ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  FAIL case{rec.case_id:02d} {kind} {phase}: {e}")
                    n_fail += 1
    print(f"[cache] {n_ok} cached, {n_fail} failed.")


def cmd_all(args) -> None:
    cmd_registry(argparse.Namespace())
    cmd_cache(argparse.Namespace(cases=None, overwrite=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Data preparation pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("registry", help="Build the case registry.")
    r.set_defaults(func=cmd_registry)

    c = sub.add_parser("cache", help="Parse CFX CSVs into the parquet cache.")
    c.add_argument("--cases", type=int, nargs="*", default=None)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_cache)

    a = sub.add_parser("all", help="registry, then cache.")
    a.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
