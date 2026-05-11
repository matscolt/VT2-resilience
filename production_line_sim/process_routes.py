"""process_routes.py (fixed)

Fixes included (per request)
----------------------------
1) Prints a RANDOM sample route (not always the first key).
2) Prints only the FILE NAMES (not the full paths).
3) Route IDs support multi-digit instance indices, e.g. 25 robotcells:
     p:10.2.1.4.5.25

This module generates (and caches) TWO JSON files in the Layouts folder:
  - layout_<counts>.json
  - process_routes_<counts>.json

The operation sequence is fixed:
  Bottom cover -> Drill station -> Robot cell -> Inspection -> Top cover -> Packaging

Station naming convention
------------------------
- Instance index 0  -> "Station N: <Name>"        (no suffix)
- Instance index j>0-> "Station N.j: <Name>"      (suffix .j)

Route ID encoding (supports >9 instances)
----------------------------------------
Route ID is:
  p:<s1>.<s2>.<s3>.<s4>.<s5>.<s6>
where si is the chosen instance index for station i.

- For stations with count=1, the index is always 0.
- For stations with count>1, the index is in [0..count-1] and can be multi-digit.

Example:
  ["Station 1.2: Bottom cover", "Station 2: Drill station", "Station 3.1: Robot cell",
   "Station 4: Inspection", "Station 5: Top cover", "Station 6.4: Packaging"]
becomes route_id = "p:2.0.1.0.0.4".

Defaults
--------
Transport time rule (default):
  index 0 -> 0 sec
  indices 1-2 -> 10 sec
  indices 3-4 -> 20 sec
  indices 5-6 -> 30 sec
  ...
Implemented as: transport = ceil(index/2) * 10

Time scale rule (default):
  time_scale_factor = 1.0 + 0.1*index

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
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAYOUT_DIR = DATA_DIR / "Layouts"  # BOTH outputs are placed here


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


@dataclass(frozen=True)
class GenerationConfig:
    time_scale_base: float = 1.0
    time_scale_step: float = 0.1
    transport_step_s: float = 10.0
    transport_group_size: int = 2

    def transport_seconds(self, instance_index: int) -> float:
        if instance_index <= 0:
            return 0.0
        return math.ceil(instance_index / self.transport_group_size) * self.transport_step_s

    def time_scale_factor(self, instance_index: int) -> float:
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
    parts = [str(sel if c > 1 else 0) for sel, c in zip(selection, counts)]
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
        "route_id_format": "p:<s1>.<s2>.<s3>.<s4>.<s5>.<s6> where si is instance index or 0 if count=1",
        "counts": counts,
        "routes": routes,
    }


def generate_layout_and_routes(
    counts: List[int],
    cfg: GenerationConfig = GenerationConfig(),
    output_dir: Path = LAYOUT_DIR,
    write_files: bool = True,
):
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

    return layout_json, routes_json, layout_path, routes_path, status


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


def main():
    print("\n--- Layout & Process Routes Generator ---\n")
    print("Enter how many parallel machines/instances exist per station.")
    print("Station sequence: 1 Bottom cover -> 2 Drill -> 3 Robot -> 4 Inspection -> 5 Top cover -> 6 Packaging\n")

    counts = []
    for station_no, station_name in STATIONS:
        counts.append(_ask_int(f"How many '{station_name}' machines (Station {station_no})?    >> "))

    cfg = GenerationConfig()
    layout_json, routes_json, layout_path, routes_path, status = generate_layout_and_routes(counts, cfg)

    print("\nOutputs (Layouts folder):")
    # 2) print only filename (not full path)
    print(f"  Layout file : {layout_path.name}  ({status['layout']})")
    print(f"  Routes file : {routes_path.name}  ({status['routes']})")

    n_routes = len(routes_json.get("routes", {}))
    print(f"  Number of routes (combinations): {n_routes}")

    # 1) print a random route id
    if n_routes:
        sample_route_id = random.choice(list(routes_json["routes"].keys()))
        print("\nSample route:")
        print(f"  {sample_route_id}: {routes_json['routes'][sample_route_id]['station_sequence']}")


if __name__ == "__main__":
    main()
