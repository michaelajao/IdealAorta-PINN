"""Data preparation: migrate raw CFD exports, build the registry, and cache to parquet.

Subcommands:
    python scripts/prepare.py migrate              # dry-run preview of the move into data/raw
    python scripts/prepare.py migrate --apply      # move the 12 case folders + xlsx into data/raw
    python scripts/prepare.py registry             # discover cases -> data/registry.json
    python scripts/prepare.py registry --use-source  # preview discovery from the source (no write)
    python scripts/prepare.py cache [--cases 1 4 7] [--overwrite]   # parse CFX CSVs -> parquet cache
    python scripts/prepare.py all                  # migrate --apply, then registry, then cache
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import DEFAULT_DATA_SOURCE, RAW_DIR  # noqa: E402
from idealaorta_pinn.data.cache import cache_one  # noqa: E402
from idealaorta_pinn.data.registry import build_registry, discover_cases  # noqa: E402


# --------------------------------------------------------------------------- migrate
def _dir_stats(path: Path) -> dict:
    if path.is_file():
        return {"files": 1, "bytes": path.stat().st_size}
    files = total = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return {"files": files, "bytes": total}


def _migrate_items(source: Path, include_xlsx: bool) -> list[Path]:
    items = sorted(p for p in source.iterdir()
                   if p.is_dir() and p.name.lower().startswith("case "))
    if include_xlsx:
        xlsx = source / "Results on Slices.xlsx"
        if xlsx.exists():
            items.append(xlsx)
    return items


def cmd_migrate(args) -> None:
    source: Path = args.source
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    items = _migrate_items(source, include_xlsx=not args.no_xlsx)
    if not items:
        raise SystemExit(f"No 'Case *' folders found under {source}")

    verb = "COPY" if args.copy else "MOVE"
    print(f"=== migrate [{'APPLY' if args.apply else 'DRY-RUN'}] ({verb}) ===")
    print(f"Source: {source}\nDest:   {RAW_DIR}\nItems:  {len(items)}\n")
    manifest = {}
    for item in items:
        st = _dir_stats(item)
        manifest[item.name] = st
        print(f"  {item.name:55s}  {st['files']:>6} files  {st['bytes']/1e6:>9.1f} MB")
    print(f"\n  TOTAL: {sum(v['bytes'] for v in manifest.values())/1e6:.1f} MB")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to perform the migration.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "_migration_manifest.json").write_text(
        json.dumps({"source": str(source), "verb": verb, "items": manifest}, indent=2))
    for item in items:
        dest = RAW_DIR / item.name
        if dest.exists():
            print(f"  SKIP (exists): {item.name}")
            continue
        if args.copy:
            shutil.copy2(item, dest) if item.is_file() else shutil.copytree(item, dest)
        else:
            shutil.move(str(item), str(dest))
        print(f"  {verb}: {item.name}")

    ok = True
    for name, expected in manifest.items():
        dest = RAW_DIR / name
        got = _dir_stats(dest) if dest.exists() else None
        if got != expected:
            print(f"  MISMATCH/MISSING: {name}")
            ok = False
    print("\n" + ("Migration verified." if ok else "Migration completed WITH MISMATCHES."))


# --------------------------------------------------------------------------- registry
def cmd_registry(args) -> None:
    if args.use_source:
        records = discover_cases(DEFAULT_DATA_SOURCE)
        print(f"[preview] {DEFAULT_DATA_SOURCE} -> {len(records)} cases (not written)")
    else:
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
    cmd_migrate(argparse.Namespace(source=args.source, apply=True, copy=False, no_xlsx=False))
    cmd_registry(argparse.Namespace(use_source=False))
    cmd_cache(argparse.Namespace(cases=None, overwrite=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Data preparation pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("migrate", help="Move/copy raw CFD exports into data/raw.")
    m.add_argument("--source", type=Path, default=DEFAULT_DATA_SOURCE)
    m.add_argument("--apply", action="store_true")
    m.add_argument("--copy", action="store_true")
    m.add_argument("--no-xlsx", action="store_true")
    m.set_defaults(func=cmd_migrate)

    r = sub.add_parser("registry", help="Build/preview the case registry.")
    r.add_argument("--use-source", action="store_true")
    r.set_defaults(func=cmd_registry)

    c = sub.add_parser("cache", help="Parse CFX CSVs into the parquet cache.")
    c.add_argument("--cases", type=int, nargs="*", default=None)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_cache)

    a = sub.add_parser("all", help="migrate --apply, then registry, then cache.")
    a.add_argument("--source", type=Path, default=DEFAULT_DATA_SOURCE)
    a.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
