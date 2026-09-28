"""Data preparation: build the case registry and cache raw CFD exports to parquet.

Raw exports live under ``data/raw/`` (see the README):

    data/raw/full_2026-09/Case <n>/<time>.csv   whole-domain exports (five blocks), canonical

The original per-case folders (``legacy_2025/``) were retired on 2026-09-27 after their
content was verified in its canonical home: ``data/processed/`` (the parquet cache the
original study reads, including the x-WSS polylines), ``data/results_on_slices.csv`` (the
slice workbook) and ``report/figures/cfdpost/`` (CFD-Post images). ``registry`` and ``cache``
therefore only rebuild when that archive is restored under ``data/raw/legacy_2025/``, and
otherwise leave the tracked registry untouched.

Subcommands:
    python scripts/prepare.py registry             # discover legacy cases -> data/registry.json
    python scripts/prepare.py cache [--cases 1 4 7] [--overwrite]   # legacy CSVs -> parquet cache
    python scripts/prepare.py full [--cases 1 9] [--overwrite]      # whole-domain CSVs -> data/processed/full
    python scripts/prepare.py all                  # registry, cache, then full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import FULL_DIR, FULL_RAW_DIR, LEGACY_RAW_DIR  # noqa: E402
from idealaorta_pinn.data.cache import cache_one  # noqa: E402
from idealaorta_pinn.data.full_export import convert_all, raw_exports  # noqa: E402
from idealaorta_pinn.data.registry import build_registry  # noqa: E402


# --------------------------------------------------------------------------- registry
def cmd_registry(args) -> None:
    records = build_registry(LEGACY_RAW_DIR, save=False)
    missing = [r.case_id for r in records if not r.files]
    if len(records) < 12 or missing:
        # the legacy streamline / clip / plane CSVs were retired once verified in the
        # parquet cache; the tracked registry.json and data/processed/ are canonical now
        print(f"[registry] {LEGACY_RAW_DIR} lacks the legacy CSVs (cases without files: {missing}); "
              "keeping the tracked data/registry.json unchanged.")
        return
    records = build_registry(LEGACY_RAW_DIR, save=True)
    print(f"[registry] {LEGACY_RAW_DIR} -> {len(records)} cases written to data/registry.json")
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


# --------------------------------------------------------------------------- full
def cmd_full(args) -> None:
    exports = raw_exports()
    print(f"[full] {len(exports)} raw exports under {FULL_RAW_DIR}")
    written = convert_all(cases=args.cases, overwrite=args.overwrite)
    print(f"[full] {len(written)} parquet files written to {FULL_DIR}")


def cmd_all(args) -> None:
    cmd_registry(argparse.Namespace())
    cmd_cache(argparse.Namespace(cases=None, overwrite=False))
    cmd_full(argparse.Namespace(cases=None, overwrite=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Data preparation pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("registry", help="Build the case registry from the legacy exports.")
    r.set_defaults(func=cmd_registry)

    c = sub.add_parser("cache", help="Parse the legacy CSVs into the parquet cache.")
    c.add_argument("--cases", type=int, nargs="*", default=None)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_cache)

    f = sub.add_parser("full", help="Split the whole-domain exports into per-block parquet.")
    f.add_argument("--cases", type=int, nargs="*", default=None)
    f.add_argument("--overwrite", action="store_true")
    f.set_defaults(func=cmd_full)

    a = sub.add_parser("all", help="registry, cache, then full.")
    a.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
