"""process_routes.py

Generate (and cache) two JSON files in the Layouts folder:
  1) A line layout JSON with station instances.
  2) A process-routes JSON containing every combination of station-instance choices
     across the fixed operation sequence.

Fixed operation sequence (station numbers stay fixed):
  1: Bottom cover
  2: Drill station
  3: Robot cell
  4: Inspection
  5: Top cover
  6: Packaging

Station naming convention (as requested)
--------------------------------------
- Instance index 0  -> "Station N: <Name>"        (no suffix)
- Instance index j>0-> "Station N.j: <Name>"      (suffix .j)

Route ID encoding
-----------------
"p" + 6 digits, one for each station number 1..6.
For stations without parallel copies (count=1), digit is 0.
For stations with copies (count>1), digit is the chosen instance index.

Example route:
  ["Station 1.2: Bottom cover", "Station 2: Drill station", "Station 3.1: Robot cell",
   "Station 4: Inspection", "Station 5: Top cover", "Station 6.4: Packaging"]
becomes route_id = "p201004".

Defaults
--------
Transport time rule (default, can be changed):
  base instance (index 0): 0 seconds
  instance indices 1-2: +10 s
  instance indices 3-4: +20 s
  instance indices 5-6: +30 s
  ...
Implemented as: transport = ceil(index/2) * 10

Time scale rule (default, can be changed):
  base instance (index 0): 1.0
  each additional instance adds +0.1: 1.1, 1.2, 1.3, ...

Caching behavior (requested)
----------------------------
Both JSON files are written to the Layouts folder:
  - Layout file:        layout_<counts>.json
  - Process routes file: process_routes_<counts>.json

Before generating, the module checks each file individually:
  - If the file already exists, it will be reused (loaded) and NOT regenerated.
  - If it does not exist, it will be generated.

Usage
-----
Run interactively:
    python process_routes.py

Or import:
    from process_routes import generate_layout_and_routes

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
LAYOUT_DIR = DATA_DIR / "Layouts"  # BOTH outputs will be placed here (as requested)


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
    """Config knobs for defaults."""
    time_scale_base: float = 1.0
    time_scale_step: float = 0.1
    transport_step_s: float = 10.0
    transport_group_size: int = 2

    def transport_seconds(self, instance_index: int) -> float:
        """Default transport rule: ceil(index/2) * 10 seconds (index 0 -> 0s)."""
        if instance_index <= 0:
            return 0.0
        groups = math.ceil(instance_index / self.transport_group_size)
        return groups * self.transport_step_s

    def time_scale_factor(self, instance_index: int) -> float:
        """Default scale rule: 1.0 + 0.1*index."""
        return self.time_scale_base + self.time_scale_step * instance_index


# -----------------------------
# Naming helpers
# -----------------------------

def _station_label(station_no: int, station_name: str) -> str:
    return f"Station {station_no}: {station_name}"


def station_instance_name(station_no: int, station_name: str, instance_index: int) -> str:
    """Station naming as requested: index 0 has no suffix."""
    if instance_index == 0:
        return _station_label(station_no, station_name)
    return f"Station {station_no}.{instance_index}: {station_name}"


def layout_filename(counts: List[int]) -> str:
    return f"layout_{'_'.join(map(str, counts))}.json"


def routes_filename(counts: List[int]) -> str:
    return f"routes_{'_'.join(map(str, counts))}.json"


def route_id_from_selection(selection: List[int], counts: List[int]) -> str:
    """selection: chosen instance index per station (len=6).

    For stations with count=1, digit is forced to 0.
    """
    digits = []
    for sel, c in zip(selection, counts):
        digits.append(str(sel if c > 1 else 0))
    return "p" + "".join(digits)


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
    """Generate layout JSON structure with station_instances."""
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
    """Generate all station-sequence combinations for the given counts."""
    if len(counts) != 6:
        raise ValueError("counts must have 6 integers (for stations 1..6)")
    if any(c < 1 for c in counts):
        raise ValueError("each station count must be >= 1")

    selections = product(*[range(c) for c in counts])

    routes: Dict[str, dict] = {}
    for sel in selections:
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
        "route_id_format": "p" + "[s1][s2][s3][s4][s5][s6] where si is instance index or 0 if count=1",
        "counts": counts,
        "routes": routes,
    }


def generate_layout_and_routes(
    counts: List[int],
    cfg: GenerationConfig = GenerationConfig(),
    output_dir: Path = LAYOUT_DIR,
    write_files: bool = True,
) -> Tuple[dict, dict, Path, Path, Dict[str, str]]:
    """Generate both JSON objects and (optionally) write them to disk.

    Returns:
      layout_json, routes_json, layout_path, routes_path, status

    status is a dict:
      {"layout": "generated"|"reused", "routes": "generated"|"reused"}

    Behavior:
      - If a file exists, it is loaded and reused.
      - If missing, it is generated.
      - Each file is checked independently (as requested).
    """
    if len(counts) != 6:
        raise ValueError("counts must have 6 integers (for stations 1..6)")

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
    print(f"  Layout file : {layout_path}  ({status['layout']})")
    print(f"  Routes file : {routes_path}  ({status['routes']})")

    # Show quick summary + one sample route
    n_routes = len(routes_json.get("routes", {}))
    print(f"  Number of routes (combinations): {n_routes}")

    if n_routes:
        sample_key = int(random.uniform(1,n_routes))
        print("\nSample route:")
        print(f"  {sample_key}: {routes_json['routes'][sample_key]['station_sequence']}")


if __name__ == "__main__":
    main()
