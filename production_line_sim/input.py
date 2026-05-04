import json
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------
# CSV writers
# -----------------------------

# 🔷 Write CSV
def write_order_csv(rows, output_path: Path):
    """Write orders to CSV with a fixed schema expected by the simulation."""
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


def write_disruption_csv(rows, output_path: Path):
    """Write disruptions to CSV with a fixed schema expected by the simulation."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "disruption_type", "station_id", "start_time", "end_time", "efficiency_percentage",
                "order_id", "Order_time", "priority",
                "variant0", "quantity0",
                "variant1", "quantity1",
                "variant2", "quantity2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Small helpers
# -----------------------------

def round_half_up(n: float) -> int:
    """Round half up (1.5 -> 2) for non-negative values."""
    return int(n + 0.5)


def _normalize_chance(value: float) -> float:
    """Accept either fraction (0.05) or percent (5) and return fraction."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(v, 1.0))


def _sample_duration(spec: Dict[str, Any]) -> float:
    """Sample a disruption duration from spec. Uses normalvariate + clamp to range if present."""
    mean = float(spec.get("duration [s]", 0))
    std = float(spec.get("std", 0))
    dur = mean if std <= 0 else random.normalvariate(mean, std)

    # Clamp to range if present
    rng = spec.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        dur = max(lo, min(dur, hi))

    return max(dur, 1.0)


def _sample_efficiency_drop(spec: Dict[str, Any]) -> float:
    """Sample an efficiency drop percentage from spec (e.g., mean 20)."""
    mean = float(spec.get("efficiency drop [%]", 0))
    std = float(spec.get("efficiency drop std", 0))
    drop = mean if std <= 0 else random.normalvariate(mean, std)

    rng = spec.get("efficiency drop range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        drop = max(lo, min(drop, hi))

    return max(0.0, min(drop, 100.0))


def _pick_start_in_free_window(
    duration: float,
    intervals: List[Tuple[float, float]],
    sim_time: float,
) -> Optional[float]:
    """Pick a random start time such that [start, start+duration] does not overlap existing intervals.

    Returns None if no feasible slot exists.
    """
    if duration > sim_time:
        return None

    # Sort existing intervals and build free windows.
    intervals_sorted = sorted(intervals)
    free: List[Tuple[float, float]] = []

    prev_end = 0.0
    for s, e in intervals_sorted:
        if s > prev_end:
            free.append((prev_end, s))
        prev_end = max(prev_end, e)

    if prev_end < sim_time:
        free.append((prev_end, sim_time))

    # Filter windows that can fit duration
    free = [(a, b) for a, b in free if (b - a) >= duration]
    if not free:
        return None

    # Choose a window weighted by available placement length (b-a-duration)
    weights = [(b - a - duration) for a, b in free]
    total = sum(weights)

    # If total == 0, all windows are exactly duration long; choose uniformly among them
    if total <= 0:
        a, b = random.choice(free)
        return a

    r = random.random() * total
    acc = 0.0
    chosen = free[-1]
    for w, win in zip(weights, free):
        acc += w
        if r <= acc:
            chosen = win
            break

    a, b = chosen
    latest_start = b - duration
    return random.uniform(a, latest_start)


# -----------------------------
# JSON create/read
# -----------------------------

def create_setting_json(output_path: Path) -> Dict[str, Any]:
    setting = {
        "sim_time [s]": 36000,
        "seed": datetime.now().strftime("%Y%m%d%H%M%S"),
        "random based disruptions": {"enabled": 2},
        "line_layout_file": "line_layout_single_path.json",
        "carriers": {"number of carriers": 8},
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4)
    return setting


def create_disruption_json(output_path: Path) -> Dict[str, Any]:
    disruption = {
        "Stations": {
            "1": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std": 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
            },
            "2": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std": 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
            },
            "3": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std": 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
            },
            "4": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std": 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
            },
            "5": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std": 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
            },
            "6": {
                "failed inspection": {"wrong assembly chance": 0.005},
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std": 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std": 10,
                },
            },
        },
        "Material": {
            "Broken material chance [%]": 0.15,
            "ran out of material chance [%]": 0.00,
        },
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
    ordermean = num_units/num_orders
    orderstd = ordermean * 0.1
    unitstd =  0.1

    earliest_due_date = ordermean * 76.4
    for order_id in range(1, num_orders):
        units = round_half_up(max(random.normalvariate(ordermean, orderstd), 1))
        due_date = round_half_up(random.uniform(earliest_due_date, sim_time))
        priority = round_half_up(min(max(random.expovariate(1/1.5), 1), 5))
        variant0 = "FUSE0"
        quantity0 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
        variant1 = "FUSE1"
        quantity1 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
        variant2 = "FUSE2"
        quantity2 = round_half_up(units - quantity0 - quantity1)
        while quantity2 < 0:
            quantity0 = quantity0 - 1
            quantity1 = quantity1 - 1
            quantity2 = units - quantity0 - quantity1
            
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
    print(f"Total units in orders: {total_sum}")
    units = num_units-total_sum
    due_date = round_half_up(random.uniform(earliest_due_date, sim_time))
    priority = round_half_up(max(random.normalvariate(2.5, 1), 1))
    variant0 = "FUSE0"
    quantity0 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
    variant1 = "FUSE1"
    quantity1 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
    variant2 = "FUSE2"
    quantity2 = round_half_up(units - quantity0 - quantity1)
    while quantity2 < 0:
        quantity0 = quantity0 - 1
        quantity1 = quantity1 - 1
        quantity2 = units - quantity0 - quantity1

    row = {
        "order_id": order_id+1,
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
    total_sum += quantity0 + quantity1 + quantity2
    priority_sum += priority

    print(f"Total priority in orders: {priority_sum} with a mean of {priority_sum/num_orders}")
    print(f"Generated {num_orders} orders with a total of {total_sum} units.\n Average units per order: {total_sum/num_orders}")
    print(f"average phone per hour: {total_sum/sim_time*3600}")
    write_order_csv(rows, output_path)

# -----------------------------
# Disruption generation
# -----------------------------

def generate_disruption_list(sim_time: int, output_path: Path):
    """Generate a disruption CSV based on the disruption.json.

    Semantics implemented (as you requested):
    - Each disruption chance is treated as a fraction of total sim_time.
      Example: 0.05 means ~5% of sim_time should be affected by that disruption type.
    - Disruptions get a random start time and end_time = start_time + duration.
    - Different stations may overlap in time.
    - Within a station, disruptions are not allowed to overlap (no double disruptions).
    """

    input_dir = output_path.parent
    settings = read_settings_json(input_dir / "settings.json")
    disruption_settings = read_disruption_json(input_dir / "disruption.json")

    # Seed randomness for reproducibility (string seed is fine for random.seed)
    random.seed(settings.get("seed", None))

    rows: List[Dict[str, Any]] = []

    stations: Dict[str, Any] = disruption_settings.get("Stations", {})

    for station_id_str, station_cfg in stations.items():
        station_id = int(station_id_str)

        # Track existing disruptions for this station to prevent overlaps
        intervals: List[Tuple[float, float]] = []

        # ---- Breakdown ----
        if "breakdown" in station_cfg:
            bcfg = station_cfg["breakdown"]
            chance = _normalize_chance(bcfg.get("Machine breakdown chance [%]", 0))
            target_downtime = chance * sim_time

            mean_dur = float(bcfg.get("duration [s]", 0))
            if mean_dur > 0 and target_downtime > 0:
                n_events = round_half_up(target_downtime / mean_dur)

                for _ in range(n_events):
                    duration = _sample_duration(bcfg)
                    start = _pick_start_in_free_window(duration, intervals, sim_time)
                    if start is None:
                        break
                    end = start + duration
                    start_i = round_half_up(start)
                    end_i = min(round_half_up(end), sim_time)
                    # Store rounded intervals so the no-overlap rule also holds in the written CSV
                    intervals.append((start_i, end_i))

                    rows.append(
                        {
                            "disruption_type": "breakdown",
                            "station_id": station_id,
                            "start_time": start_i,
                            "end_time": end_i,
                            "efficiency_percentage": 0,
                            "order_id": "",
                            "Order_time": "",
                            "priority": "",
                            "variant0": "",
                            "quantity0": "",
                            "variant1": "",
                            "quantity1": "",
                            "variant2": "",
                            "quantity2": "",
                        }
                    )

        # ---- Efficiency loss ----
        if "efficiency loss" in station_cfg:
            ecfg = station_cfg["efficiency loss"]
            chance = _normalize_chance(ecfg.get("efficiency drop chance [%]", 0))
            target_downtime = chance * sim_time

            mean_dur = float(ecfg.get("duration [s]", 0))
            if mean_dur > 0 and target_downtime > 0:
                n_events = round_half_up(target_downtime / mean_dur)

                for _ in range(n_events):
                    duration = _sample_duration(ecfg)
                    start = _pick_start_in_free_window(duration, intervals, sim_time)
                    if start is None:
                        break
                    end = start + duration
                    start_i = round_half_up(start)
                    end_i = min(round_half_up(end), sim_time)
                    intervals.append((start_i, end_i))

                    drop = _sample_efficiency_drop(ecfg)
                    eff = max(1.0, min(100.0, 100.0 - drop))

                    rows.append(
                        {
                            "disruption_type": "efficiency_loss",
                            "station_id": station_id,
                            "start_time": start_i,
                            "end_time": end_i,
                            "efficiency_percentage": round_half_up(eff),
                            "order_id": "",
                            "Order_time": "",
                            "priority": "",
                            "variant0": "",
                            "quantity0": "",
                            "variant1": "",
                            "quantity1": "",
                            "variant2": "",
                            "quantity2": "",
                        }
                    )

        # NOTE: "failed inspection" in your JSON is not time-based and has no duration.
        # Because you requested time-percentage-driven disruptions, we do not emit it to the CSV here.
        # If you later add a duration spec to "failed inspection", you can generate it the same way.

    # Optional: sort for readability
    rows.sort(key=lambda r: (int(r["station_id"]), int(r["start_time"])))

    write_disruption_csv(rows, output_path)

# -----------------------------
# Main
# -----------------------------

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
    disruption_settings = read_disruption_json(output_path_disruptionjson)

    # Generate orderlist

    # should be in format: order_id, Order_time, priority, variant0, quantity, variant1, quantity, variant2, quantity
    generate_orderlist(num_orders, num_units, sim_time, output_path_ordercsv)
    if settings["random based disruptions"]["enabled"] == 2:
        print("Generating event based disruptions")
        generate_disruption_list(sim_time, output_path_disruptioncsv)
    print(f"--- Created order file: {orderfoldername} ----")


if __name__ == "__main__":
    main()
