
from __future__ import annotations
import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESS_TIMES_PATH = DATA_DIR / "process_times.json"
OUTPUT_DIR = DATA_DIR / "Layouts"

MONTH_SECONDS = 576000

# -----------------------------
# Helpers
# -----------------------------
def load_process_times():
    with open(PROCESS_TIMES_PATH, "r") as f:
        return json.load(f)


def ask_int(prompt, default=None):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "" and default is not None:
        return default
    return int(raw)


def ask_float(prompt, default=None):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "" and default is not None:
        return default
    return float(raw)


# -----------------------------
# Core logic
# -----------------------------
def compute_average_times(process_times_json):
    variants = process_times_json["process_times"]

    station_avg = {}

    for variant in variants.values():
        for station, time in variant.items():
            station_avg.setdefault(station, []).append(time)

    # average per station
    return {
        station: sum(times) / len(times)
        for station, times in station_avg.items()
    }


import re

def get_base_station(name: str) -> str:
    """
    Converts:
    - 'Station 1: Bottom cover'
    - 'Station 1.1: Bottom cover'
    - 'Station 1.23: Bottom cover'
    into:
    - 'Station 1: Bottom cover'
    """
    match = re.match(r"^(Station\s+\d+)(?:\.\d+)?(:\s+.+)$", name)
    if match:
        return match.group(1) + match.group(2)
    return name  # fallback


def compute_capacity(station_avg_times, layout_instances, buffer_factor):
    station_rates = {}

    # group scales by station
    for inst in layout_instances:
        name = inst["station_name"]
        base = get_base_station(name)

        station_rates.setdefault(base, []).append(inst["time_scale_factor"])

    # compute rate per station
    station_capacity = {}

    for station, scales in station_rates.items():
        avg_time = station_avg_times[station]

        total_rate = sum(1 / (avg_time * s) for s in scales)

        station_capacity[station] = total_rate

    # bottleneck = lowest rate
    bottleneck = min(station_capacity, key=station_capacity.get)
    bottleneck_rate = station_capacity[bottleneck]

    monthly_capacity = bottleneck_rate * MONTH_SECONDS
    scaled_capacity = monthly_capacity * buffer_factor

    return bottleneck, monthly_capacity, scaled_capacity


# -----------------------------
# Layout generation
# -----------------------------
def generate_layout():
    process_data = load_process_times()
    stations = process_data["station_sequence"]

    print("\n--- INPUT ---\n")

    buffer_pct = ask_float("Buffer (%)", 10.0)
    buffer_factor = 1 - buffer_pct / 100

    layout_instances = []
    counts = []

    for i, station in enumerate(stations, start=1):
        count = ask_int(f"{station} instances", 2)
        counts.append(count)

        for j in range(count):
            if j == 0:
                name = station
            else:
                name = station.replace(":", f".{j}:")

            scale = ask_float(f"  {name} scale", 1.0)

            layout_instances.append({
                "station_name": name,
                "time_scale_factor": scale,
                "branch_transport_from_previous_s": 0.0 if j == 0 else 10.0,
                "branch_transport_to_next_s": 0.0 if j == 0 else 10.0
            })
    # compute
    station_avg = compute_average_times(process_data)
    bottleneck, monthly, scaled = compute_capacity(
        station_avg, layout_instances, buffer_factor
    )


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- RESULT ---")
    print(f"Bottleneck: {bottleneck}")
    print(f"Monthly capacity: {monthly:.1f}")
    print(f"Scaled capacity: {scaled:.1f}")
    carriers = ask_int("\nhow many carriers should this line have? ",40)

    # output
    layout = {
        "bottleneck station": bottleneck,
        "monthly_capacity": round(monthly, 1),
        "scaled_monthly_capacity": round(scaled, 1),
        "capacity_percentage": f"{buffer_factor*100:.1f}%",
        "layout_name": f"layout_{'_'.join(map(str, counts))}",
        "carriers": carriers,
        "station_instances": layout_instances
    }
    file = OUTPUT_DIR / f"{layout['layout_name']}.json"

    with open(file, "w") as f:
        json.dump(layout, f, indent=2)

    print(f"Saved to: {file}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # we try again
    generate_layout()
