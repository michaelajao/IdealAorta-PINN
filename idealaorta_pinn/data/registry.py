"""Case discovery and registry.

Walks ``data/raw/Case *`` folders, parses each (messy) folder name into
structured metadata, classifies the CFX files inside each ``PINNS/`` subfolder,
and emits a JSON registry. Robust to the dataset's naming inconsistencies
(``Disaesed``, ``C5,`` prefixes, ``Sysstolic``/``Syastolic`` typos) because all
matching is done by regex, never by literal filenames.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import RAW_DIR, REGISTRY_PATH, load_cases, load_constants
from .cfx import detect_data_kind, normalize_phase, parse_time_seconds

_CASE_ID_RE = re.compile(r"case\s+(\d+)", re.IGNORECASE)
_DIAMETER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*c?m?\s*inlet", re.IGNORECASE)
_HEALTHY_RE = re.compile(r"inlet\s+h\b|healthy", re.IGNORECASE)


@dataclass
class DataFile:
    kind: str            # '3D' | 'XY' | 'XZ' | 'WSS'
    phase: str           # 'systolic' | 'diastolic'
    path: str            # absolute path
    time_s: Optional[float]


@dataclass
class CaseRecord:
    case_id: int
    folder: str
    inlet_diameter_cm: float
    health: str          # 'diseased' | 'healthy'
    symmetry: str        # 'axisymmetric' | 'anterior' | 'posterior' | 'healthy'
    beta: Optional[float]
    aneurysm_diameter_cm: float
    # kind -> phase -> DataFile (stored as plain dicts for JSON round-trip)
    files: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    @property
    def disease_flag(self) -> int:
        return 1 if self.health == "diseased" else 0

    def available(self) -> Dict[str, List[str]]:
        return {kind: sorted(phases.keys()) for kind, phases in self.files.items()}

    def get_file(self, kind: str, phase: str) -> Optional[Path]:
        entry = self.files.get(kind, {}).get(phase)
        return Path(entry["path"]) if entry else None


def _classify_symmetry(folder_low: str, health: str) -> str:
    if health == "healthy":
        return "healthy"
    if "upper" in folder_low:
        return "anterior"
    if "lower" in folder_low:
        return "posterior"
    return "axisymmetric"


def _parse_folder(name: str) -> Optional[Dict]:
    """Parse a case folder name into (case_id, diameter, health, symmetry)."""
    low = name.lower()
    m_id = _CASE_ID_RE.search(low)
    m_d = _DIAMETER_RE.search(low)
    if not m_id or not m_d:
        return None
    health = "healthy" if _HEALTHY_RE.search(low) else "diseased"
    return {
        "case_id": int(m_id.group(1)),
        "inlet_diameter_cm": float(m_d.group(1)),
        "health": health,
        "symmetry": _classify_symmetry(low, health),
    }


def _find_pinns_dir(case_dir: Path) -> Optional[Path]:
    for child in case_dir.iterdir():
        if child.is_dir() and child.name.lower() == "pinns":
            return child
    return None


def _discover_files(pinns_dir: Path) -> Dict[str, Dict[str, dict]]:
    """Classify every CSV in a PINNS folder by (kind, phase)."""
    files: Dict[str, Dict[str, dict]] = {}
    for csv in sorted(pinns_dir.glob("*.csv")):
        kind = detect_data_kind(csv.name)
        phase = normalize_phase(csv.name)
        if kind == "unknown" or phase == "unknown":
            continue
        files.setdefault(kind, {})[phase] = asdict(
            DataFile(kind=kind, phase=phase, path=str(csv.resolve()),
                     time_s=parse_time_seconds(csv.name))
        )
    return files


def discover_cases(root: Path = RAW_DIR) -> List[CaseRecord]:
    """Discover all ``Case *`` folders under ``root`` and build their records."""
    constants = load_constants()
    cases_yaml = load_cases()
    beta_by_symmetry = cases_yaml["beta_by_symmetry"]
    aneurysm_cm = float(constants["geometry"]["aneurysm_diameter_cm"])

    # Expected attributes for cross-checking (keyed by case_id).
    expected = {c["case_id"]: c for c in cases_yaml["cases"]}

    records: List[CaseRecord] = []
    if not root.exists():
        return records

    for case_dir in sorted(p for p in root.iterdir()
                           if p.is_dir() and p.name.lower().startswith("case ")):
        parsed = _parse_folder(case_dir.name)
        if parsed is None:
            continue
        pinns = _find_pinns_dir(case_dir)
        files = _discover_files(pinns) if pinns else {}

        rec = CaseRecord(
            case_id=parsed["case_id"],
            folder=case_dir.name,
            inlet_diameter_cm=parsed["inlet_diameter_cm"],
            health=parsed["health"],
            symmetry=parsed["symmetry"],
            beta=beta_by_symmetry.get(parsed["symmetry"]),
            aneurysm_diameter_cm=aneurysm_cm,
            files=files,
        )

        # Cross-check against the authoritative metadata (warn, do not fail).
        exp = expected.get(rec.case_id)
        if exp is not None:
            if abs(exp["inlet_diameter_cm"] - rec.inlet_diameter_cm) > 1e-6:
                print(f"[registry] WARNING: Case {rec.case_id} diameter "
                      f"{rec.inlet_diameter_cm} != expected {exp['inlet_diameter_cm']} "
                      f"(folder: {rec.folder})")
            if exp["health"] != rec.health:
                print(f"[registry] WARNING: Case {rec.case_id} health "
                      f"{rec.health} != expected {exp['health']} (folder: {rec.folder})")

        records.append(rec)

    records.sort(key=lambda r: r.case_id)
    return records


def build_registry(root: Path = RAW_DIR, save: bool = True) -> List[CaseRecord]:
    """Discover cases and optionally write ``data/registry.json``."""
    records = discover_cases(root)
    if save:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in records], f, indent=2)
    return records


def load_registry(path: Path = REGISTRY_PATH) -> List[CaseRecord]:
    """Load a previously built registry from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [CaseRecord(**rec) for rec in raw]


def cases_by_id(records: List[CaseRecord]) -> Dict[int, CaseRecord]:
    return {r.case_id: r for r in records}
