"""process_routes.py

Generates (and caches) two JSON files in the Layouts folder:
  1) A line layout JSON with station instances.
  2) A process-routes JSON containing every combination of station-instance choices
     across the fixed operation sequence.

NEW (added): bottleneck + monthly capacity estimation
-----------------------------------------------------
After generation (or loading), the script can compute:
  - Effective per-station cycle time per variant, accounting for:
      * base processing time from process_times.json
      * parallel instances
      * per-instance time_scale_factor
  - Bottleneck station (by average effective cycle time across variants)
  - Average line cycle time across variants (where line cycle time per variant is the slowest station time)
  - Monthly capacity using MONTH_SECONDS = 576000 seconds

Notes on effective cycle time
-----------------------------
If a station has multiple parallel instances with time_scale_factor s_k, and a variant has base time t
at the base station, then the combined throughput is:
  sum_k 1/(t*s_k)
So the equivalent time per unit for that station is:
  t / sum_k (1/s_k)
This reproduces the example: FUSE1 robot cell 76.4s with 2 instances, both scale=1.0 => 76.4/2.

Fixed operation sequence:
  1: Bottom cover
  2: Drill station
  3: Robot cell
  4: Inspection
  5: Top cover
  6: Packaging

Station naming convention
------------------------
- Instance index 0  -> "Station N: <Name>"        (no suffix)
- Instance index j>0-> "Station N.j: <Name>"      (suffix .j)

Route ID encoding (multi-digit safe, 1-based)
---------------------------------------------
Route ID is:
  p:<s1>.<s2>.<s3>.<s4>.<s5>.<s6>
where si is the chosen station-instance index in HUMAN numbering:
  base instance (no suffix) => 1
  .1 => 2
  .2 => 3
  ...
This supports large counts, e.g. 25 robotcells.

Caching behavior
----------------
Each file is checked individually:
  - If it exists: reused (loaded)
  - If missing: generated

"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAYOUT_DIR = DATA_DIR / "Layouts"  # BOTH outputs are placed here
PROCESS_TIMES_PATH = DATA_DIR / "process_times.json"
MONTH_SECONDS = 576000  # requested


# -----------------------------
# Station definitions
# -----------------------------

STATIONS: List[Tuple[int, str]] = [
    (1, "Bottom cover"),
    (2, "Drill station"),
    (3, "Robot cell"),
    (4, "Inspection"),
    (5, "Top cover"),
    (6, "Packaging"),
]

STATION_NAME_RE = re.compile(r"^\s*Station\s+(\d+)(?:\.(\d+))?\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class GenerationConfig:
    time_scale_base: float = 1.0
    time_scale_step: float = 0.1
    transport_step_s: float = 10.0
    transport_group_size: int = 2

    def transport_seconds(self, instance_index: int) -> float:
        """Default branch transport rule (kept for compatibility)."""
        if instance_index <= 0:
            return 0.0
        return math.ceil(instance_index / self.transport_group_size) * self.transport_step_s

    def time_scale_factor(self, instance_index: int) -> float:
        """Default scaling: 1.0 + 0.1*index."""
        return self.time_scale_base + self.time_scale_step * instance_index


# -----------------------------
# Naming helpers
# -----------------------------

def _station_label(station_no: int, station_name: str) -> str:
    return f"Station {station_no}: {station_name}"


def station_instance_name(station_no: int, station_name: str, instance_index: int) -> str:
    if instance_index == 0:
        return _station_label(station_no, station_name)
    return f"Station {station_no}.{instance_index}: {station_name}"


def layout_filename(counts: List[int]) -> str:
    return f"layout_{'_'.join(map(str, counts))}.json"


def routes_filename(counts: List[int]) -> str:
    return f"routes_{'_'.join(map(str, counts))}.json"


def route_id_from_selection(selection: List[int], counts: List[int]) -> str:
    """1-based, multi-digit safe route ID.

    Internal selection indices are 0-based.
    Route ID uses index+1 for display (matches station naming idea).
    """
    parts = []
    for sel, c in zip(selection, counts):
        sel = max(0, min(int(sel), int(c) - 1))
        parts.append(str(sel + 1))
    return "p:" + ".".join(parts)


# -----------------------------
# File helpers
# -----------------------------

def _load_json_if_exists(path: Path) -> dict | None:
    if path.exists() and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_process_times(path: Path = PROCESS_TIMES_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"process_times.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------
# Bottleneck & capacity analysis
# -----------------------------

def _canonical_base_station_name(station_instance: str) -> str:
    """Convert 'Station 3.2: Robot cell' -> 'Station 3: Robot cell'."""
    m = STATION_NAME_RE.match(station_instance)
    if not m:
        return station_instance.strip()
    stage_no, _copy, label = m.groups()
    return f"Station {int(stage_no)}: {label.strip()}"


def _group_scales_by_base_station(layout_json: dict) -> dict[str, list[float]]:
    """Return mapping base_station_name -> list of time_scale_factors for its instances."""
    groups: dict[str, list[float]] = {}
    for inst in layout_json.get("station_instances", []):
        name = str(inst.get("station_name", "")).strip()
        if not name:
            continue
        base = _canonical_base_station_name(name)
        scale = float(inst.get("time_scale_factor", 1.0))
        if scale <= 0:
            scale = 1.0
        groups.setdefault(base, []).append(scale)
    return groups


def effective_station_time_per_unit(base_time_s: float, scales: list[float]) -> float:
    """Compute equivalent station time per unit with parallel instances and scaling."""
    inv_sum = sum(1.0 / s for s in scales if s > 0)
    if inv_sum <= 0:
        return float('inf')
    return float(base_time_s) / inv_sum


def compute_bottleneck_and_capacity(
    layout_json: dict,
    process_times_json: dict,
    month_seconds: float = MONTH_SECONDS,
    use_average_across_variants: bool = True,
) -> dict[str, Any]:
    """Compute bottleneck station and monthly capacity.

    - Per station, per variant: effective station time = base_time / sum(1/scale)
    - Station average time: average across variants (if use_average_across_variants True)
    - Bottleneck station: station with maximum station average time
    - Line cycle time per variant: max station eff time (slowest station)
    - Average line cycle time: average across variants
    - Monthly capacity (units): month_seconds / average_line_cycle_time

    Returns a dict with details, suitable to embed into JSON or print.
    """
    station_scales = _group_scales_by_base_station(layout_json)

    process_times = process_times_json.get("process_times", {})
    variants = sorted(process_times.keys())
    station_sequence = list(process_times_json.get("station_sequence", []))

    # Use base station names as keys (e.g., 'Station 3: Robot cell')
    base_stations = list(station_scales.keys())

    # Ensure we only consider stations that exist in process_times station_sequence
    if station_sequence:
        allowed = set(station_sequence)
        base_stations = [s for s in base_stations if s in allowed]

    eff_times: dict[str, dict[str, float]] = {st: {} for st in base_stations}

    for variant in variants:
        vtimes = process_times[variant]
        for st in base_stations:
            base_time = float(vtimes.get(st, 0.0))
            scales = station_scales.get(st, [1.0])
            eff_times[st][variant] = effective_station_time_per_unit(base_time, scales)

    # Station-level average (or max if you want worst-case)
    station_score: dict[str, float] = {}
    for st in base_stations:
        vals = [eff_times[st][v] for v in variants]
        station_score[st] = float(sum(vals) / len(vals)) if (use_average_across_variants and vals) else float(max(vals) if vals else 0.0)

    bottleneck_station = max(station_score, key=station_score.get) if station_score else None

    # Line cycle time per variant is the slowest station effective time
    line_cycle_by_variant: dict[str, float] = {}
    for v in variants:
        slowest = max((eff_times[st][v] for st in base_stations), default=0.0)
        line_cycle_by_variant[v] = float(slowest)

    avg_line_cycle = float(sum(line_cycle_by_variant.values()) / len(line_cycle_by_variant)) if line_cycle_by_variant else 0.0

    # Capacity in units per month (using average line cycle)
    capacity_units_month = (month_seconds / avg_line_cycle) if avg_line_cycle > 0 else 0.0

    # Also provide per-station capacity based on its station average
    station_capacity_month: dict[str, float] = {}
    for st, t in station_score.items():
        station_capacity_month[st] = (month_seconds / t) if t > 0 else 0.0

    return {
        "month_seconds": float(month_seconds),
        "variants": variants,
        "bottleneck_station": bottleneck_station,
        "effective_station_time_s_per_unit": eff_times,  # station -> variant -> seconds
        "station_average_time_s_per_unit": station_score,
        "line_cycle_time_s_per_unit_by_variant": line_cycle_by_variant,
        "average_line_cycle_time_s_per_unit": avg_line_cycle,
        "capacity_units_per_month_by_average_line_cycle": capacity_units_month,
        "capacity_units_per_month_by_station_average": station_capacity_month,
    }


# -----------------------------
# Core generators
# -----------------------------

def generate_layout_json(counts: List[int], cfg: GenerationConfig = GenerationConfig()) -> dict:
    if len(counts) != 6:
        raise ValueError("counts must have 6 integers (for stations 1..6)")
    if any(c < 1 for c in counts):
        raise ValueError("each station count must be >= 1")

    instances = []
    for (station_no, station_name), c in zip(STATIONS, counts):
        for idx in range(c):
            instances.append(
                {
                    "station_name": station_instance_name(station_no, station_name, idx),
                    "time_scale_factor": float(cfg.time_scale_factor(idx)),
                    "branch_transport_from_previous_s": float(cfg.transport_seconds(idx)),
                    "branch_transport_to_next_s": float(cfg.transport_seconds(idx)),
                }
            )

    return {
        "layout_name": f"layout_{'_'.join(map(str, counts))}",
        "station_instances": instances,
    }


def generate_routes_json(counts: List[int]) -> dict:
    if len(counts) != 6:
        raise ValueError("counts must have 6 integers (for stations 1..6)")
    if any(c < 1 for c in counts):
        raise ValueError("each station count must be >= 1")

    routes: Dict[str, dict] = {}

    for sel in product(*[range(c) for c in counts]):
        sel = list(sel)
        rid = route_id_from_selection(sel, counts)
        seq = [
            station_instance_name(st_no, st_name, (s if c > 1 else 0))
            for (st_no, st_name), s, c in zip(STATIONS, sel, counts)
        ]
        routes[rid] = {
            "station_sequence": seq,
        }

    return {
        "route_id_format": "p:<s1>.<s2>.<s3>.<s4>.<s5>.<s6> where si = instance_index+1",
        "counts": counts,
        "routes": routes,
    }


def generate_layout_and_routes(
    counts: List[int],
    cfg: GenerationConfig = GenerationConfig(),
    output_dir: Path = LAYOUT_DIR,
    write_files: bool = True,
    compute_bottleneck: bool = True,
) -> tuple[dict, dict, Path, Path, dict[str, str], dict[str, Any] | None]:
    """Generate/reuse both JSON objects and optionally compute bottleneck/capacity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_path = output_dir / layout_filename(counts)
    routes_path = output_dir / routes_filename(counts)

    status = {"layout": "generated", "routes": "generated"}

    layout_json = _load_json_if_exists(layout_path)
    if layout_json is not None:
        status["layout"] = "reused"
    else:
        layout_json = generate_layout_json(counts, cfg)
        if write_files:
            _write_json(layout_path, layout_json)

    routes_json = _load_json_if_exists(routes_path)
    if routes_json is not None:
        status["routes"] = "reused"
    else:
        routes_json = generate_routes_json(counts)
        if write_files:
            _write_json(routes_path, routes_json)

    analysis = None
    if compute_bottleneck:
        pt = _load_process_times(PROCESS_TIMES_PATH)
        analysis = compute_bottleneck_and_capacity(layout_json, pt, month_seconds=MONTH_SECONDS)

    return layout_json, routes_json, layout_path, routes_path, status, analysis


# -----------------------------
# Interactive CLI
# -----------------------------

def _ask_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if v < 1:
                print("Please enter an integer >= 1")
                continue
            return v
        except ValueError:
            print("Please enter a valid integer")


def main() -> None:
    print("\n--- Layout & Process Routes Generator ---\n")
    print("Enter how many parallel machines/instances exist per station.")
    print("Station sequence: 1 Bottom cover -> 2 Drill -> 3 Robot -> 4 Inspection -> 5 Top cover -> 6 Packaging\n")

    counts = []
    for station_no, station_name in STATIONS:
        counts.append(_ask_int(f"How many '{station_name}' machines (Station {station_no})?    >> "))

    cfg = GenerationConfig()

    layout_json, routes_json, layout_path, routes_path, status, analysis = generate_layout_and_routes(
        counts, cfg, compute_bottleneck=True
    )

    print("\nOutputs (Layouts folder):")
    print(f"  Layout file : {layout_path.name}  ({status['layout']})")
    print(f"  Routes file : {routes_path.name}  ({status['routes']})")

    n_routes = len(routes_json.get("routes", {}))
    print(f"  Number of routes (combinations): {n_routes}")

    if n_routes:
        sample_route_id = random.choice(list(routes_json["routes"].keys()))
        print("\nSample route:")
        print(f"  {sample_route_id}: {routes_json['routes'][sample_route_id]['station_sequence']}")

    if analysis:
        print("\n--- Bottleneck & capacity (based on process_times.json + time_scale_factor) ---")
        print(f"Month seconds: {analysis['month_seconds']}")
        print(f"Bottleneck station (avg across variants): {analysis['bottleneck_station']}")
        print(f"Average line cycle time (s/unit): {analysis['average_line_cycle_time_s_per_unit']:.4f}")
        print(f"Capacity per month (units, avg line cycle): {analysis['capacity_units_per_month_by_average_line_cycle']:.2f}")


if __name__ == "__main__":
    main()
