import json
import csv
import random
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# ============================================================
# Constants
# ============================================================

PRIO_LOW = 1
PRIO_HIGH = 5

# ============================================================
# CSV writers
# ============================================================

def write_order_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write orders to CSV with the schema expected by the simulation."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "order_id", "due date", "priority",
                "variant0", "quantity0",
                "variant1", "quantity1",
                "variant2", "quantity2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_disruption_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write disruptions to CSV with the schema expected by the simulation."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "disruption_type", "station_id", "start_time", "end_time", "efficiency_percentage",
                "order_id", "due_date", "priority",
                "variant0", "quantity0",
                "variant1", "quantity1",
                "variant2", "quantity2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Helpers
# ============================================================

def round_half_up(x: float) -> int:
    """Round half up for non-negative values."""
    return int(x + 0.5)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def sample_duration(spec: Dict[str, Any], Range: bool = False) -> int:
    """Sample a disruption duration in seconds as an int >= 1.

    Supports two spec formats:
      - v1: keys like 'duration [s]', 'std', and optional 'range'
      - v2: keys like 'mean [s]', 'std [0-1 of mean]' / 'std [% of mean]', and 'range [s]'

    The Range flag keeps backward compatibility: it clamps v1 durations only when Range=True.
    v2 durations are always clamped to 'range [s]' if provided.
    """

    # --- v2 format ---
    if spec.get("mean [s]") is not None:
        mean = float(spec.get("mean [s]", 0))
        std_frac = spec.get("std [0-1 of mean]")
        if std_frac is None:
            std_pct = spec.get("std [% of mean]")
            std_frac = float(std_pct) / 100.0 if std_pct is not None else 0.0
        std = float(std_frac) * mean

        dur = mean if std <= 0 else random.normalvariate(mean, std)

        rng = spec.get("range [s]")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            dur = clamp(dur, float(rng[0]), float(rng[1]))

        return max(1, round_half_up(dur))

    # --- v1 format (original) ---
    mean = float(spec.get("duration [s]", 0))
    std = float(spec.get("std", 0))
    dur = mean if std <= 0 else random.normalvariate(mean, std)

    if Range is True and spec.get("range") is not None:
        rng = spec.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            dur = clamp(dur, float(rng[0]), float(rng[1]))

    return max(1, round_half_up(dur))


def sample_efficiency_percentage(spec: Dict[str, Any], Range: bool = False) -> int:
    """For efficiency loss, sample the resulting efficiency percentage (1..100).

    Reads:
      - mean drop: 'efficiency drop [%]'
      - std: 'efficiency drop std'
      - optional clamp: 'efficiency drop range'

    Returns efficiency_percentage = 100 - drop.
    """
    mean = float(spec.get("efficiency drop [%]", 0))
    std = float(spec.get("efficiency drop std", 0))
    drop = mean if std <= 0 else random.normalvariate(mean, std)

    if Range == True and spec.get("efficiency drop range") is not None:
        rng = spec.get("efficiency drop range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            drop = clamp(drop, float(rng[0]), float(rng[1]))

    eff = 100.0 - clamp(drop, 0.0, 100.0)
    return int(clamp(round_half_up(eff), 1, 100))


def pick_random_start_non_overlapping(
    duration: int,
    occupied: List[Tuple[int, int]],
    sim_time: int,
) -> Optional[int]:
    """Pick a random start time such that [start, start+duration] does not overlap.

    occupied: list of (start,end) intervals (ints) for one station.
    Returns start time (int) or None if not possible.
    """
    if duration > sim_time:
        return None

    occ = sorted(occupied)

    # build free windows [a,b)
    free: List[Tuple[int, int]] = []
    prev_end = 0
    for s, e in occ:
        if s > prev_end:
            free.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < sim_time:
        free.append((prev_end, sim_time))

    # filter windows that can fit duration
    free = [(a, b) for (a, b) in free if (b - a) >= duration]
    if not free:
        return None

    # choose a window weighted by number of possible integer start positions
    # (b - a - duration) is the span of possible starts in continuous; for int starts use +1.
    weights = []
    for a, b in free:
        count_positions = (b - a - duration) + 1
        weights.append(max(1, count_positions))

    win = random.choices(free, weights=weights, k=1)[0]
    a, b = win
    latest_start = b - duration
    return random.randint(a, latest_start)


def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def sample_event_count_from_time_fraction(target_downtime: float, mean_duration: float) -> int:
    """Compute mean event count from downtime and sample an integer count from a normal distribution.

    mean_events = target_downtime / mean_duration
    std_events  = 10% of mean_events

    Returns an int >= 0.
    """
    if mean_duration <= 0 or target_downtime <= 0:
        return 0

    mean_events = target_downtime / mean_duration
    std_events = 0.1 * mean_events

    if std_events <= 0:
        return max(0, round_half_up(mean_events))

    n = random.normalvariate(mean_events, std_events)
    return max(0, round_half_up(n))

def iter_breakdown_specs_v2(station_cfg: Dict[str, Any]):
    """Yield breakdown specs for both v1 and v2 disruption JSON structures.

    v2 format uses key 'machine breakdowns' which is a list of dict specs.
    v1 format uses key 'breakdown' which is a single dict spec.

    Yields tuples: (disruption_type_string, spec_dict, chance_fraction, mean_duration_seconds)
    """

    # v2: list of machine breakdowns
    if isinstance(station_cfg.get("machine breakdowns"), list):
        for spec in station_cfg["machine breakdowns"]:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name", "breakdown")).strip() or "breakdown"
            dtype = f"breakdown:{name}"
            # Use ONLY 'chance of sim time [%]' (percentage, e.g. 2.4 = 2.4%)
            chance = float(spec.get("chance of sim time [%]", 0)) / 100.0

            mean_dur = float(spec.get("mean [s]", 0))
            if mean_dur <= 0 and isinstance(spec.get("range [s]"), (list, tuple)) and len(spec["range [s]"]) == 2:
                a, b = spec["range [s]"]
                mean_dur = (float(a) + float(b)) / 2.0

            yield dtype, spec, max(0.0, chance), max(0.0, mean_dur)

    # v1: single breakdown dict
    if isinstance(station_cfg.get("breakdown"), dict):
        spec = station_cfg["breakdown"]
        dtype = "breakdown"
        chance = float(spec.get("chance of sim time [%]", 0)) / 100.0
        mean_dur = float(spec.get("duration [s]", 0))
        yield dtype, spec, chance, mean_dur



# ============================================================
# JSON create/read
# ============================================================

def create_setting_json(output_path: Path) -> Dict[str, Any]:
    setting = {
        "sim_time [s]": 36000,
        "seed": datetime.now().strftime("%Y%m%d%H%M%S"),
        "random based disruptions": {"enabled": 2},
        "line_layout_file": "line_layout_single_path.json",
        "carriers": {"number of carriers": 8},
        "lowest acceptable standard deviation [std below]": 3,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4)
    return setting


def create_disruption_json(output_path: Path) -> Dict[str, Any]:
    disruption = {
  "Stations": {
    "1": {
      "station_type": "Bottom cover",
      "machine breakdowns": [
        {
          "name": "PLC Failure",
          "chance [%]": 0.02,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            30,
            60
          ],
          "mean [s]": 45,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Bottom Cover Misaligned",
          "chance [%]": 0.01,
          "chance of sim time [%]": 0.55,
          "range [s]": [
            30,
            80
          ],
          "mean [s]": 55,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        },
        {
          "name": "Bottom Cover Stuck",
          "chance [%]": 0.01,
          "chance of sim time [%]": 0.65,
          "range [s]": [
            40,
            90
          ],
          "mean [s]": 65,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Piston Broke",
          "chance [%]": 0.002,
          "chance of sim time [%]": 2.4,
          "range [s]": [
            600,
            1800
          ],
          "mean [s]": 1200,
          "std [% of mean]": 20,
          "std [0-1 of mean]": 0.2
        }
      ],
      "efficiency loss": {
        "chance [%]": 0.076,
        "chance of sim time [%]": 0.5016,
        "range [%]": [
          10,
          90
        ],
        "range [0-1]": [
          0.1,
          0.9
        ]
      }
    },
    "2": {
      "station_type": "Drilling",
      "machine breakdowns": [
        {
          "name": "PLC Failure",
          "chance [%]": 0.02,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            30,
            60
          ],
          "mean [s]": 45,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Drill To Dull",
          "chance [%]": 0.005,
          "chance of sim time [%]": 1.1,
          "range [s]": [
            120,
            320
          ],
          "mean [s]": 220,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        },
        {
          "name": "Drill Broke",
          "chance [%]": 0.01,
          "chance of sim time [%]": 2.5,
          "range [s]": [
            160,
            340
          ],
          "mean [s]": 250,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        }
      ],
      "efficiency loss": {
        "chance [%]": 0.048,
        "chance of sim time [%]": 0.4992,
        "range [%]": [
          10,
          90
        ],
        "range [0-1]": [
          0.1,
          0.9
        ]
      }
    },
    "3": {
      "station_type": "Robot cell",
      "machine breakdowns": [
        {
          "name": "PLC Failure",
          "chance [%]": 0.02,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            30,
            60
          ],
          "mean [s]": 45,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Cover Wrong Orientation",
          "chance [%]": 0.01,
          "chance of sim time [%]": 2.1,
          "range [s]": [
            120,
            300
          ],
          "mean [s]": 210,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        },
        {
          "name": "Arm Movement Misaligned",
          "chance [%]": 0.005,
          "chance of sim time [%]": 4.5,
          "range [s]": [
            600,
            1200
          ],
          "mean [s]": 900,
          "std [% of mean]": 15,
          "std [0-1 of mean]": 0.15
        },
        {
          "name": "Cart Relised To Early",
          "chance [%]": 0.005,
          "chance of sim time [%]": 2.25,
          "range [s]": [
            300,
            600
          ],
          "mean [s]": 450,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        },
        {
          "name": "Misaligned Material",
          "chance [%]": 0.005,
          "chance of sim time [%]": 1.7,
          "range [s]": [
            200,
            480
          ],
          "mean [s]": 340,
          "std [% of mean]": 15,
          "std [0-1 of mean]": 0.15
        },
        {
          "name": "Tool Broke",
          "chance [%]": 0.001,
          "chance of sim time [%]": 1.35,
          "range [s]": [
            900,
            1800
          ],
          "mean [s]": 1350,
          "std [% of mean]": 20,
          "std [0-1 of mean]": 0.2
        },
        {
          "name": "Robot Arm Broke",
          "chance [%]": 0.001,
          "chance of sim time [%]": 5.2,
          "range [s]": [
            3400,
            7000
          ],
          "mean [s]": 5200,
          "std [% of mean]": 25,
          "std [0-1 of mean]": 0.25
        }
      ],
      "efficiency loss": {
        "chance [%]": 0.027,
        "chance of sim time [%]": 1.9962,
        "range [%]": [
          10,
          90
        ],
        "range [0-1]": [
          0.1,
          0.9
        ]
      }
    },
    "4": {
      "station_type": "Inspection",
      "machine breakdowns": [
        {
          "name": "PLC Failure",
          "chance [%]": 0.02,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            30,
            60
          ],
          "mean [s]": 45,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Camera Broken",
          "chance [%]": 0.002,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            300,
            600
          ],
          "mean [s]": 450,
          "std [% of mean]": 15,
          "std [0-1 of mean]": 0.15
        },
        {
          "name": "Camera Dirty",
          "chance [%]": 0.005,
          "chance of sim time [%]": 0.45,
          "range [s]": [
            60,
            120
          ],
          "mean [s]": 90,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        }
      ],
      "efficiency loss": {
        "chance [%]": 0.119,
        "chance of sim time [%]": 0.2499,
        "range [%]": [
          10,
          90
        ],
        "range [0-1]": [
          0.1,
          0.9
        ]
      }
    },
    "5": {
      "station_type": "Top cover",
      "machine breakdowns": [
        {
          "name": "PLC Failure",
          "chance [%]": 0.02,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            30,
            60
          ],
          "mean [s]": 45,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Top Cover Misaligned",
          "chance [%]": 0.01,
          "chance of sim time [%]": 0.55,
          "range [s]": [
            30,
            80
          ],
          "mean [s]": 55,
          "std [% of mean]": 10,
          "std [0-1 of mean]": 0.1
        },
        {
          "name": "Top Cover Stuck",
          "chance [%]": 0.01,
          "chance of sim time [%]": 0.65,
          "range [s]": [
            40,
            90
          ],
          "mean [s]": 65,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        },
        {
          "name": "Piston Broke",
          "chance [%]": 0.002,
          "chance of sim time [%]": 2.4,
          "range [s]": [
            600,
            1800
          ],
          "mean [s]": 1200,
          "std [% of mean]": 20,
          "std [0-1 of mean]": 0.2
        }
      ],
      "efficiency loss": {
        "chance [%]": 0.109,
        "chance of sim time [%]": 0.5014,
        "range [%]": [
          10,
          90
        ],
        "range [0-1]": [
          0.1,
          0.9
        ]
      }
    },
    "6": {
      "station_type": "Packaging",
      "machine breakdowns": [
        {
          "name": "PLC Failure",
          "chance [%]": 0.02,
          "chance of sim time [%]": 0.9,
          "range [s]": [
            30,
            60
          ],
          "mean [s]": 45,
          "std [% of mean]": 5,
          "std [0-1 of mean]": 0.05
        }
      ],
      "inspection failure": {
        "name": "Inspection Failure",
        "effect": "cancel_and_redo_unit",
        "chance [%]": 0.0068,
        "chance of sim time [%]": 1.3507,
        "action": "The unit is cancelled and redone."
      },
      "efficiency loss": {
        "chance [%]": 0.0625,
        "chance of sim time [%]": 0.25,
        "range [%]": [
          10,
          90
        ],
        "range [0-1]": [
          0.1,
          0.9
        ]
      }
    }
  }
    }


    with output_path.open("w", encoding="utf-8") as f:
        json.dump(disruption, f, indent=4)
    return disruption


def read_settings_json(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_disruption_json(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)

#------------------------------
# Create Orderlist
#------------------------------

def generate_orderlist(num_orders, num_units, sim_time, output_path: Path):
    rows = []
    sum = []
    priosum = []
    if num_units < num_orders:
        print(f"Warning: num_units ({num_units}) is less than num_orders ({num_orders}).\nRemoving empty orders and adjusting num_orders to {num_units}.")
        num_orders = num_units

    unitstd =  0.1
    units_left = num_units

    for order_id in range(1, num_orders+1):
        units = max(1, round_half_up(random.normalvariate(units_left / (num_orders - order_id+1),
                                                        (units_left / (num_orders - order_id+1)) * 0.1)))
        if units > units_left:
            units = units_left
        if order_id == num_orders:
            units = units_left
        units_left -= units
        due_date = round_half_up(random.uniform(min(units * 76.4, sim_time), sim_time))
        priority = round_half_up(min(max(random.expovariate(1/1.5), PRIO_LOW), PRIO_HIGH))
        variant0 = "FUSE0"
        quantity0 = round_half_up(max(random.normalvariate(units * 0.33, unitstd), 0))
        variant1 = "FUSE1"
        quantity1 = round_half_up(max(random.normalvariate(units * 0.33, unitstd), 0))
        variant2 = "FUSE2"
        quantity2 = round_half_up(units - quantity0 - quantity1)
        if units <= 2:
            quantity0, quantity1, quantity2 = 0, 0, 0
            for i in range(units):
                n = random.uniform(0, 1)
                if n <= 0.33:
                    quantity0 += 1
                elif n <= 0.66:
                    quantity1 += 1
                else:
                    quantity2 += 1
            
        row = {
            "order_id": order_id,
            "due date": due_date,
            "priority": priority,
            "variant0": variant0,
            "quantity0": quantity0,
            "variant1": variant1,
            "quantity1": quantity1,
            "variant2": variant2,
            "quantity2": quantity2
        }
        rows.append(row)
        sum.append(quantity0 + quantity1 + quantity2)
        priosum.append(priority)
    
    total_sum = 0
    for s in sum:
        total_sum += s
    priority_sum = 0
    for p in priosum:
        priority_sum += p
    
    print(f"Total priority in orders: {priority_sum} with a mean of {priority_sum/num_orders}")
    print(f"Generated {num_orders} orders with a total of {total_sum} units.\n Average units per order: {total_sum/num_orders}")
    print(f"average phone per hour(if possible): {total_sum/sim_time*3600}")
    write_order_csv(rows, output_path)

# -----------------------------
# Disruption generation
# ============================================================

def generate_disruption_list(sim_time: int, output_path: Path, num_orders: int,num_units: int) -> None:
    """Generate disruptions.csv based on settings.json and disruption.json.

    Your requested semantics implemented:
      - chance is treated as fraction of total sim_time (downtime fraction)
      - number of events is sampled from Normal(mean_events, 0.1*mean_events)
      - events have random start time, end_time = start + duration
      - stations may overlap with each other
      - within a station, events cannot overlap
      - output CSV is sorted by start_time
      - prints a per-station summary to terminal
    """

    input_dir = output_path.parent
    settings = read_settings_json(input_dir / "settings.json")
    disruption_file = (input_dir / "disruption_v2.json") if (input_dir / "disruption_v2.json").exists() else (input_dir / "disruption.json")
    disruption_settings = read_disruption_json(disruption_file)

    # Reproducible randomness
    random.seed(settings.get("seed", None))

    rows: List[Dict[str, Any]] = []

    # Summary: station -> disruption_type -> count
    summary: Dict[int, Dict[str, int]] = {}

    stations: Dict[str, Any] = disruption_settings.get("Stations", {})

    for station_id_str, station_cfg in stations.items():
        station_id = int(station_id_str)
        occupied: List[Tuple[int, int]] = []
        summary.setdefault(station_id, {})

        # --- Breakdown (supports multiple breakdown types from disruption_v2.json) ---
        # v2 uses key: 'machine breakdowns' (list) with name/chance/range/mean/std
        # v1 uses key: 'breakdown' (single dict)
        for dtype, spec, chance, mean_dur in iter_breakdown_specs_v2(station_cfg):
            target_downtime = chance * sim_time
            n_events = sample_event_count_from_time_fraction(target_downtime, mean_dur)

            placed = 0
            for _ in range(n_events):
                # v2 durations are clamped automatically; v1 uses Range flag to clamp
                duration = sample_duration(spec, Range=True)
                start = pick_random_start_non_overlapping(duration, occupied, sim_time)
                if start is None:
                    break
                end = min(sim_time, start + duration)

                occupied.append((start, end))
                placed += 1

                rows.append(
                    {
                        "disruption_type": dtype,
                        "station_id": station_id,
                        "start_time": start,
                        "end_time": end,
                        "efficiency_percentage": 0,
                        "order_id": "",
                        "due_date": "",
                        "priority": "",
                        "variant0": "",
                        "quantity0": "",
                        "variant1": "",
                        "quantity1": "",
                        "variant2": "",
                        "quantity2": "",
                    }
                )

            if placed:
                summary[station_id][dtype] = summary[station_id].get(dtype, 0) + placed

        # --- Efficiency loss ---

        if "efficiency loss" in station_cfg:
            spec = station_cfg["efficiency loss"]

            # v2 format: has keys like 'chance [0-1]' and 'range [0-1]' or 'range [%]' (no duration given)
            if spec.get("chance of sim time [%]") is not None:
                # Use ONLY 'chance of sim time [%]' (percentage, e.g. 2.4 = 2.4%)
                chance = float(spec.get("chance of sim time [%]", 0)) / 100.0

                target_downtime = chance * sim_time
                duration = max(1, round_half_up(target_downtime))

                start = pick_random_start_non_overlapping(duration, occupied, sim_time)
                if start is not None:
                    end = min(sim_time, start + duration)
                    occupied.append((start, end))

                    # sample efficiency drop from provided ranges
                    eff = 100
                    if isinstance(spec.get("range [0-1]"), (list, tuple)) and len(spec["range [0-1]"]) == 2:
                        lo, hi = spec["range [0-1]"]
                        drop_frac = random.uniform(float(lo), float(hi))
                        eff = int(clamp(round_half_up(100.0 * (1.0 - drop_frac)), 1, 100))
                    elif isinstance(spec.get("range [%]"), (list, tuple)) and len(spec["range [%]"]) == 2:
                        lo, hi = spec["range [%]"]
                        drop_pct = random.uniform(float(lo), float(hi))
                        eff = int(clamp(round_half_up(100.0 - drop_pct), 1, 100))

                    rows.append(
                        {
                            "disruption_type": "efficiency_loss",
                            "station_id": station_id,
                            "start_time": start,
                            "end_time": end,
                            "efficiency_percentage": eff,
                            "order_id": "",
                            "due_date": "",
                            "priority": "",
                            "variant0": "",
                            "quantity0": "",
                            "variant1": "",
                            "quantity1": "",
                            "variant2": "",
                            "quantity2": "",
                        }
                    )

                    summary[station_id]["efficiency_loss"] = summary[station_id].get("efficiency_loss", 0) + 1

            # v1 format: has duration and std for efficiency loss
            else:
                chance = float(spec.get("chance of sim time [%]", 0)) / 100.0
                target_downtime = chance * sim_time
                mean_dur = float(spec.get("duration [s]", 0))

                n_events = sample_event_count_from_time_fraction(target_downtime, mean_dur)

                placed = 0
                for _ in range(n_events):
                    duration = sample_duration(spec, Range=True)
                    start = pick_random_start_non_overlapping(duration, occupied, sim_time)
                    if start is None:
                        break
                    end = min(sim_time, start + duration)

                    occupied.append((start, end))
                    placed += 1

                    eff = sample_efficiency_percentage(spec, Range=True)

                    rows.append(
                        {
                            "disruption_type": "efficiency_loss",
                            "station_id": station_id,
                            "start_time": start,
                            "end_time": end,
                            "efficiency_percentage": eff,
                            "order_id": "",
                            "due_date": "",
                            "priority": "",
                            "variant0": "",
                            "quantity0": "",
                            "variant1": "",
                            "quantity1": "",
                            "variant2": "",
                            "quantity2": "",
                        }
                    )

                if placed:
                    summary[station_id]["efficiency_loss"] = summary[station_id].get("efficiency_loss", 0) + placed


        # NOTE: "failed inspection" in your current JSON is probability-based without duration,
        # so it does not fit the downtime-% approach. If you add a duration spec, you can generate it similarly.

    #emergancy orders
    eorders = round_half_up(random.normalvariate(num_orders*0.1, num_orders * 0.01))
    eunits = round_half_up(random.normalvariate(num_units*0.1, num_units * 0.01))
    if eunits < eorders:
        eorders = eunits
    print(f"Generating {eorders} emergency orders with {eunits} units (10% of total orders with some variance).")
    eunits_left = eunits
    for i in range(eorders):
        order_id = num_orders + i + 1
        eunits_per_order = max(1, round_half_up(random.normalvariate((eunits_left / (eorders - i)),
                                                        (eunits_left / (eorders - i)*0.1))))
        if eunits_per_order > eunits_left:
            eunits_per_order = eunits_left
        if i == eorders-1:
            eunits_per_order = eunits_left
        eunits_left -= eunits_per_order
        start_time = round_half_up(random.uniform(0, max(sim_time-eunits_per_order*76.4, 0)))
        due_date = round_half_up(random.uniform(min(start_time+eunits_per_order*76.4, sim_time), sim_time))
        
        x50 = 0.5 # 50th percentile of the distribution (median)
        x90 = 2.0 # 90th percentile of the distribution (chosen to create a long tail for emergency orders)

        k = math.log(math.log(10)/math.log(2)) / math.log(x90/x50) # shape parameter for weibull distribution (k)
        lam = x50 / (math.log(2)**(1.0/k)) # scale parameter for weibull distribution (lambda)

        due_date = min(round_half_up(start_time+eunits_per_order*76.4*(1+random.weibullvariate(lam, k))), sim_time)
        
        priority = PRIO_HIGH  # Emergency orders get highest priority
        variant0 = "FUSE0"
        quantity0 = max(0, round_half_up(random.normalvariate(eunits_per_order*0.33, eunits_per_order * 0.033)))
        variant1 = "FUSE1"
        quantity1 = max(0, round_half_up(random.normalvariate(eunits_per_order*0.33, eunits_per_order * 0.033)))
        variant2 = "FUSE2"
        quantity2 = max(0, eunits_per_order - quantity0 - quantity1)
        if eunits_per_order <= 2:
            quantity0, quantity1, quantity2 = 0, 0, 0
            for i in range(eunits_per_order):
                n = random.uniform(0, 1)
                if n <= 0.33:
                    quantity0 += 1
                elif n <= 0.66:
                    quantity1 += 1
                else:
                    quantity2 += 1

            
        rows.append(
            {
                "disruption_type": "emergency_order",
                "station_id": "",
                "start_time": start_time,
                "end_time": "",
                "efficiency_percentage": "",
                "order_id": order_id,
                "due_date": due_date,
                "priority": priority,
                "variant0": variant0,
                "quantity0": quantity0,
                "variant1": variant1,
                "quantity1": quantity1,
                "variant2": variant2,
                "quantity2": quantity2,
            }
        )
    

    # Sort by start_time (then station_id for stable ordering)
    rows.sort(key=lambda r: (int(r["start_time"])))

    write_disruption_csv(rows, output_path)

    # Terminal summary
    print("\nDisruption generation summary")
    print("============================")
    for station_id in sorted(summary.keys()):
        types = summary[station_id]
        if not types:
            print(f"Station {station_id}: 0 events")
            continue
        parts = [f"{t}={types[t]}" for t in sorted(types.keys())]
        total = sum(types.values())
        print(f"Station {station_id}: {total} events (" + ", ".join(parts) + ")")

    print(f"\nTotal events written: {len(rows)}")

# ============================================================
# Gantt plot for disruptions
# ============================================================

def plot_disruption_gantt(order_dir: Path,
    disruptions_csv: str | Path,
    sim_time: int | None = None,
    title: str = "Disruptions Gantt Chart",
    figsize=(14, 6),
    lane_height: float = 0.8,
    sort_stations: bool = True,
    show: bool = False,
    ax=None,
):
    """
    Plot a Gantt chart of disruptions per station from a disruptions.csv file.

    Expected CSV fields:
      - disruption_type: e.g. 'breakdown' or 'efficiency_loss' (or similar)
      - station_id
      - start_time
      - end_time

    Colors:
      - breakdown -> green
      - efficiency loss / efficiency reduction -> red
    """

    disruptions_csv = Path(disruptions_csv)

    # --- Read disruptions CSV ---
    events = []
    with disruptions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                station = int(row["station_id"])
                start = float(row["start_time"])
                end = float(row["end_time"])
                dtype = (row["disruption_type"] or "").strip().lower()
            except (KeyError, ValueError, TypeError):
                continue

            if end <= start:
                continue

            events.append({"station": station, "start": start, "end": end, "type": dtype})

    if not events:
        raise ValueError(f"No valid disruptions found in: {disruptions_csv}")

    # --- Determine station ordering ---
    stations = sorted({e["station"] for e in events}) if sort_stations else list({e["station"] for e in events})
    station_to_y = {st: i for i, st in enumerate(stations)}

    # --- If sim_time not provided, infer from max end_time ---
    if sim_time is None:
        sim_time = int(max(e["end"] for e in events))

    # --- Create axes if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- Color mapping ---
    def color_for(dtype: str) -> str:
        # accept multiple naming conventions
        if "break" in dtype:
            return "green"
        if "eff" in dtype or "reduc" in dtype or "loss" in dtype:
            return "red"
        return "gray"  # fallback

    # --- Plot as broken_barh per station ---
    # broken_barh expects [(xmin, width), ...] per lane
    by_station = {st: [] for st in stations}
    by_station_color = {st: [] for st in stations}

    # Sort by start time for nicer rendering
    events.sort(key=lambda e: (e["start"], e["station"]))

    for e in events:
        st = e["station"]
        start = e["start"]
        width = e["end"] - e["start"]
        by_station[st].append((start, width))
        by_station_color[st].append(color_for(e["type"]))

    # Draw each event as its own broken_barh to allow different colors in same lane
    for st in stations:
        y = station_to_y[st]
        y0 = y - lane_height / 2
        for (start, width), c in zip(by_station[st], by_station_color[st]):
            ax.broken_barh([(start, width)], (y0, lane_height), facecolors=c, edgecolors="black", linewidth=0.3)

    # --- Formatting ---
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Station")

    ax.set_yticks([station_to_y[st] for st in stations])
    ax.set_yticklabels([str(st) for st in stations])
    ax.invert_yaxis()

    ax.set_xlim(0, sim_time)

    # Legend
    legend_items = [
        Patch(facecolor="green", edgecolor="black", label="Breakdown"),
        Patch(facecolor="red", edgecolor="black", label="Efficiency reduction"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{order_dir / title}.png", dpi=300)

    if show:
        plt.show()

    return ax
# ============================================================
# Main
# ============================================================

def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    input_dir.mkdir(exist_ok=True)

    n=1
    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    orderfoldername = f"orders_{timestamp}_{n}"
    
    while (input_dir / orderfoldername).exists():
        n=n+1
        orderfoldername = f"orders_{timestamp}_{n}"
    
    ordername_csv = f"order_list_{timestamp}_{n}.csv"
        
    order_dir = input_dir / orderfoldername
    order_dir.mkdir(parents=True, exist_ok=True)

    # Generate paths
    output_path_ordercsv = order_dir / ordername_csv
    output_path_disruptioncsv = order_dir / f"disruption_list_{timestamp}_{n}.csv"
    output_path_settingsjson = order_dir / "settings.json"
    output_path_disruptionjson = order_dir / "disruption.json"

    # Generate settings and disruption json files
    create_setting_json(output_path_settingsjson)
    create_disruption_json(output_path_disruptionjson)

    num_orders = int(input("Enter amount of orders: "))
    num_units = int(input("Enter amount of units: "))
   
    # read settings json file

    settings = read_settings_json(output_path_settingsjson)
    sim_time = int(settings.get("sim_time [s]", 36000))
    seed = settings["seed"]
    random.seed(seed)

    # Generate orderlist

    # should be in format: order_id, due_date, priority, variant0, quantity, variant1, quantity, variant2, quantity
    generate_orderlist(num_orders, num_units, sim_time, output_path_ordercsv)
    if settings["random based disruptions"]["enabled"] == 2:
        print("Generating event based disruptions")
        generate_disruption_list(sim_time, output_path_disruptioncsv, num_orders, num_units)
        plot_disruption_gantt(order_dir, output_path_disruptioncsv, sim_time=sim_time, title=f"Disruptions Gantt Chart for {orderfoldername}", show=False)

    print(f"--- Created order file: {orderfoldername} ----")


if __name__ == "__main__":
    main()
