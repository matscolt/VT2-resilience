from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import json
import os
import shutil
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOTDIR = Path(__file__).parent
RESULTSDIR = ROOTDIR / "RESULTS"
SIM_DAY_SECONDS = 8 * 60 * 60
THROUGHPUT_RATE_WINDOW_S = 7200
THROUGHPUT_RATE_SAMPLE_S = 600
THROUGHPUT_RATE_INTERVAL_S = 1800

def display_station_name(name: str) -> str:
    if name in (None, ""):
        return name
    s = str(name).strip()
    m = re.match(r'^(Station\s+6(?:\.\d+)?)(?:\s*:\s*|\s+)(Packaging)$', s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}: Unloading"
    return s


def _seconds_to_sim_days(seconds_value: float) -> float:
    return float(seconds_value) / float(SIM_DAY_SECONDS)


def _parse_main_folder_name(name: str, assumed_year: int) -> Optional[Tuple[datetime, int]]:
    """Parse main folder name: main_DD-MM_HH-MM_n
    Returns (timestamp_dt, counter_n) or None if not parseable.
    """
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "main":
        return None
    try:
        timestamp_str = parts[1] + "_" + parts[2]  # "DD-MM_HH-MM"
        ts = datetime.strptime(timestamp_str, "%d-%m_%H-%M").replace(year=assumed_year)
        n = int(parts[3])
        return ts, n
    except Exception:
        return None


def list_output_runs(output_dir: Path) -> List[Path]:
    """List main output folders (main_DD-MM_HH-MM_n) sorted newest first."""
    output_dir = Path(output_dir)
    assumed_year = datetime.now().year
    folders: List[Tuple[datetime, int, Path]] = []

    for name in os.listdir(output_dir):
        full_path = output_dir / name
        if not full_path.is_dir():
            continue
        parsed = _parse_main_folder_name(name, assumed_year)
        if not parsed:
            continue
        ts, n = parsed
        folders.append((ts, n, full_path))

    folders.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for (_, _, p) in folders]


def prompt_for_data_folder(output_dir: Path, outputs: List[Path]) -> Path:
    """Interactive selection of a main output folder."""
    if not outputs:
        raise FileNotFoundError(f"No output folders found in: {output_dir}")

    print("Available main folders in ./output (newest first):")
    for i, p in enumerate(outputs, start=1):
        print(f" {i:2d}) {p.name}")

    default = 1
    while True:
        s = input(f"Choose main folder number (Enter = {default}): ").strip()
        if s == "":
            return outputs[default - 1]
        if s.isdigit() and 1 <= int(s) <= len(outputs):
            return outputs[int(s) - 1]
        candidate = output_dir / s
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")


def list_run_folders(main_dir: Path) -> List[Path]:
    """Return run folders inside a main folder, sorted by run number."""
    main_dir = Path(main_dir)
    runs = [p for p in main_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]

    def run_num(path: Path) -> int:
        m = re.match(r"run_(\d+)$", path.name)
        return int(m.group(1)) if m else 10**9

    runs.sort(key=run_num)
    return runs


def prompt_for_run_folder(main_dir: Path, runs: List[Path]) -> Path:
    """Interactive selection of a run_* folder inside a main folder."""
    if not runs:
        raise FileNotFoundError(f"No run folders found in: {main_dir}")

    print(f"Available run folders in {Path(main_dir).name} (lowest run number first):")
    for i, p in enumerate(runs, start=1):
        print(f" {i:2d}) {p.name}")

    default = 1
    while True:
        s = input(f"Choose run folder number (Enter = {default}): ").strip()
        if s == "":
            return runs[default - 1]
        if s.isdigit() and 1 <= int(s) <= len(runs):
            return runs[int(s) - 1]
        candidate = Path(main_dir) / s
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")


def prompt_for_compare_main_folders(output_dir: Path, count: int = 5) -> List[Path]:
    """Interactively select multiple distinct main folders for comparison."""
    outputs = list_output_runs(output_dir)
    if len(outputs) < count:
        raise FileNotFoundError(f"Need at least {count} main folders in: {output_dir}")

    selected: List[Path] = []
    available = outputs[:]
    for idx in range(count):
        print(f"Select compare main folder {idx + 1}/{count}:")
        chosen = prompt_for_data_folder(output_dir, available)
        selected.append(chosen)
        available = [p for p in available if p != chosen]
    return selected


def _run_sort_key(run_name: str):
    m = re.match(r"run_(\d+)$", str(run_name))
    if m:
        return int(m.group(1))
    return str(run_name)


def find_results_folder(
    output_dir: Path,
    target_timestamp: Optional[str] = None,
    prompt: bool = True,
) -> Tuple[Path, Path]:
    """Pick a main folder (optionally closest to target_timestamp), then pick run_*, then return run/results.

    Returns
    -------
    (Path, Path):
        (<main_folder>, <run_folder>/results)
    """
    output_dir = Path(output_dir)
    assumed_year = datetime.now().year

    candidates: List[Tuple[datetime, int, Path]] = []
    for name in os.listdir(output_dir):
        full_path = output_dir / name
        if not full_path.is_dir():
            continue
        parsed = _parse_main_folder_name(name, assumed_year)
        if not parsed:
            continue
        ts, n = parsed
        candidates.append((ts, n, full_path))

    if not candidates:
        raise FileNotFoundError(f"No valid output folders found in {output_dir}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if target_timestamp:
        target_dt = datetime.strptime(target_timestamp, "%d-%m_%H-%M").replace(year=assumed_year)
        chosen_main = min(candidates, key=lambda x: abs(x[0] - target_dt))[2]
    else:
        if prompt:
            outputs = [p for (_, _, p) in candidates]
            chosen_main = prompt_for_data_folder(output_dir, outputs)
        else:
            chosen_main = candidates[0][2]

    runs = list_run_folders(chosen_main)
    return chosen_main, runs


def clear_folder(folder: Path):
    folder_path = folder / "graphs"
    if not folder_path.exists():
        print(f">> Folder does not exist: {folder_path}")
        return

    for item in folder_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=False)
            else:
                item.unlink()
        except PermissionError:
            print(f">> Could not delete (locked): {item}")

    folder_name = str(folder).split("\\")[-1].split("__")[0]
    print(f"cleaned out {folder_name}/graph")


def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif ext == ".csv":
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def to_float(data, keys):
    for row in data:
        for key in keys:
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
    return data


def load_all_data(data_folder):
    station_schedule = load_data(os.path.join(data_folder, "station_schedule.csv"))
    station_summary = load_data(os.path.join(data_folder, "station_summary.csv"))
    transport_data = load_data(os.path.join(data_folder, "transport_schedule.csv"))
    unit_data = load_data(os.path.join(data_folder, "unit_summary.csv"))
    material_data = load_data(os.path.join(data_folder, "unit_summary.csv"))
    order_data = load_data(os.path.join(data_folder, "order_summary.csv"))

    station_schedule = to_float(station_schedule, [
        "start_time_s", "finish_time_s", "process_time_s"
    ])
    station_summary = to_float(station_summary, [
        "busy_time_s", "first_start_time_s", "last_finish_time_s",
        "max_queue_length", "average_queue_length", "average_wait_time_s",
        "total_wait_time_s", "utilization_overall", "utilization_active_window"
    ])
    transport_data = to_float(transport_data, [
        "start_time_s", "finish_time_s", "transport_time_s"
    ])
    unit_data = to_float(unit_data, [
        "completion_time_s", "flow_time_s", "active_flow_time_s"
    ])
    order_data = to_float(order_data, [
        "due date", "start_time", "finish_time", "through_put_time",
        "lateness", "fitness", "priority", "planned_week", "finished_week", "planned_day", "finished_day"
    ])

    for row in station_schedule:
        if "station_name" in row:
            row["station_name"] = display_station_name(row["station_name"])
    for row in station_summary:
        if "station_name" in row:
            row["station_name"] = display_station_name(row["station_name"])
    for row in transport_data:
        if "from_station" in row:
            row["from_station"] = display_station_name(row["from_station"])
        if "to_station" in row:
            row["to_station"] = display_station_name(row["to_station"])

    return station_schedule, station_summary, transport_data, unit_data, material_data, order_data


# ---------------------------
# PLOTS
# ---------------------------

def plot_gantt(station_data, transport_data, graphfolder_dir, starttime_gantt, endtime_gantt):
    print(">> Generating Gantt charts!")

    station_data_window = [
        row for row in station_data
        if float(row["finish_time_s"]) > starttime_gantt and float(row["start_time_s"]) < endtime_gantt
    ]
    transport_data_window = [
        row for row in transport_data
        if float(row["finish_time_s"]) > starttime_gantt and float(row["start_time_s"]) < endtime_gantt
    ]

    if not station_data_window and not transport_data_window:
        print(f">> No station/transport activity in time window [{starttime_gantt}, {endtime_gantt}]")
        return

    stations = sorted(set(
        (int(row["station_index"]), row["station_name"])
        for row in station_data_window
    ))
    transports = sorted(set(
        (int(row["transport_index"]), row["transport_name"])
        for row in transport_data_window
    ))

    y_labels_full = []
    for i in range(len(stations)):
        y_labels_full.append(stations[i][1])
        if i < len(transports):
            y_labels_full.append(transports[i][1])
    y_pos_full = {label: i for i, label in enumerate(y_labels_full)}

    y_labels_station = [s[1] for s in stations]
    y_pos_station = {label: i for i, label in enumerate(y_labels_station)}

    units = list(set(
        [row["unit_id"] for row in station_data_window] +
        [row["unit_id"] for row in transport_data_window]
    ))
    colors = {u: i for i, u in enumerate(units)}

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    graphname = f"Gantt_chart_stations_{int(starttime_gantt)}_{int(endtime_gantt)}.png"

    for row in station_data_window:
        start = max(float(row["start_time_s"]), starttime_gantt)
        finish = min(float(row["finish_time_s"]), endtime_gantt)
        duration = finish - start
        if duration <= 0:
            continue

        y = y_pos_station[row["station_name"]]
        unit = row["unit_id"]
        ax1.barh(
            y=y,
            width=duration,
            left=start,
            color=plt.cm.tab20(colors[unit] % 20),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75
        )

        if duration > 50:
            ax1.text(
                start + duration / 2,
                y,
                unit,
                ha="center",
                va="center",
                fontsize=7
            )

    ax1.set_yticks(range(len(y_labels_station)))
    ax1.set_yticklabels(y_labels_station)
    ax1.set_xlabel("Time [s]")
    ax1.set_title(f"Gantt Chart (Stations only) [{starttime_gantt}, {endtime_gantt}]")
    ax1.set_xlim(starttime_gantt, endtime_gantt)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(graphfolder_dir / graphname, dpi=200)
    plt.close(fig1)
    print(f">> Generated {graphname}")

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    graphname2 = f"Gantt_chart_with_transport_{int(starttime_gantt)}_{int(endtime_gantt)}.png"

    for row in station_data_window:
        start = max(float(row["start_time_s"]), starttime_gantt)
        finish = min(float(row["finish_time_s"]), endtime_gantt)
        duration = finish - start
        if duration <= 0:
            continue

        y = y_pos_full[row["station_name"]]
        unit = row["unit_id"]
        ax2.barh(
            y=y,
            width=duration,
            left=start,
            color=plt.cm.tab20(colors[unit] % 20),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75
        )

        if duration > 50:
            ax2.text(
                start + duration / 2,
                y,
                unit,
                ha="center",
                va="center",
                fontsize=7
            )

    for row in transport_data_window:
        start = max(float(row["start_time_s"]), starttime_gantt)
        finish = min(float(row["finish_time_s"]), endtime_gantt)
        duration = finish - start
        if duration <= 0:
            continue

        y = y_pos_full[row["transport_name"]]
        unit = row["unit_id"]
        ax2.barh(
            y=y,
            width=duration,
            left=start,
            color=plt.cm.tab20(colors[unit] % 20),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75
        )

        if duration > 50:
            ax2.text(
                start + duration / 2,
                y,
                unit,
                ha="center",
                va="center",
                fontsize=6
            )

    ax2.set_yticks(range(len(y_labels_full)))
    ax2.set_yticklabels(y_labels_full)
    ax2.set_xlabel("Time [s]")
    ax2.set_title(f"Gantt Chart (Stations + Transport) [{starttime_gantt}, {endtime_gantt}]")
    ax2.set_xlim(starttime_gantt, endtime_gantt)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(graphfolder_dir / graphname2, dpi=200)
    plt.close(fig2)
    print(f">> Generated {graphname2}")



def _safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')


def plot_cumulative_completed_units_by_station(station_data, graphfolder):
    """
    Create one cumulative completed-units-over-time plot per station.

    Uses station_schedule.csv because it contains the station_name and the exact
    finish_time_s for each unit at each station. That is the most direct measure
    of when a unit was completed at a station.
    """
    print(">> Generating cumulative completed units plots per station!")

    station_rows = {}
    for row in station_data:
        station_name = row.get("station_name", "")
        finish_val = row.get("finish_time_s", "")
        if station_name in ("", None) or finish_val in ("", None):
            continue
        try:
            finish_time = float(finish_val)
        except ValueError:
            continue
        station_rows.setdefault(station_name, []).append(finish_time)

    if not station_rows:
        print(">> No valid station completion data found. Skipping station cumulative plots.")
        return

    station_folder = graphfolder / "cumulative_completed_units_by_station"
    station_folder.mkdir(parents=True, exist_ok=True)

    for station_name, finish_times in sorted(station_rows.items()):
        station_name = display_station_name(station_name)
        if not finish_times:
            continue

        finish_times.sort()
        cumulative_units = list(range(1, len(finish_times) + 1))

        graphname = f"cumulative_completed_units_{_safe_filename(station_name)}.png"

        plt.figure(figsize=(12, 6))
        finish_days = [_seconds_to_sim_days(t) for t in finish_times]
        plt.step(finish_days, cumulative_units, where="post", linewidth=2, color="#1f77b4")
        plt.xlabel("Time [days]")
        plt.ylabel("Completed units [-]")
        plt.title(f"Cumulative completed units over time\n{station_name}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(station_folder / graphname, dpi=200, bbox_inches="tight")
        plt.close()

    print(f">> Generated cumulative station plots in {station_folder}")

def plot_throughput_times(unit_data, graphfolder):
    print(">> Generating throughput time plots!")
    units = [row["unit_id"] for row in unit_data]
    throughput_time = [float(row["active_flow_time_s"]) for row in unit_data]
    graphname = "throughput_times.png"
    avg_throughput_time = sum(throughput_time) / len(throughput_time)

    plt.figure()
    plt.bar(units, throughput_time)

    plt.axhline(
        y=avg_throughput_time,
        linestyle=":",
        linewidth=1,
        color="black",
        label=f"Average = {avg_throughput_time:.2f} s"
    )

    plt.legend(loc="upper left")

    max_labels = 30
    n_units = len(units)
    step = max(1, n_units // max_labels)

    plt.xticks([])
    plt.ylabel("Throughput time [s]")
    plt.title("Throughput time per unit")

    plt.ylim(0,2000)

    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")



def plot_cumulative_completed_units(unit_data, graphfolder):
    print(">> Generating cumulative completed units plot!")

    completion_times = []
    for row in unit_data:
        value = row.get("completion_time_s", row.get("completion_time", ""))
        if value in ("", None):
            continue
        try:
            completion_times.append(float(value))
        except ValueError:
            pass

    if not completion_times:
        print(">> No valid completion times found. Skipping cumulative completed units plot.")
        return

    completion_times.sort()
    cumulative_units = list(range(1, len(completion_times) + 1))

    graphname = "cumulative_completed_units.png"

    plt.figure(figsize=(12, 6))
    completion_days = [_seconds_to_sim_days(t) for t in completion_times]
    plt.step(completion_days, cumulative_units, where="post", linewidth=2, color="#1f77b4")
    plt.xlabel("Time [days]")
    plt.ylabel("Completed units [-]")
    plt.title("Cumulative completed units over time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()

    print(f">> Generated {graphname}")


def plot_throughput_rate_moving(unit_data, graphfolder):
    print(">> Generating moving throughput rate plot!")

    completion_times = []
    for row in unit_data:
        value = row.get("completion_time_s", row.get("completion_time", ""))
        if value in ("", None):
            continue
        try:
            completion_times.append(float(value))
        except ValueError:
            pass

    if not completion_times:
        print(">> No valid completion times found. Skipping moving throughput rate plot.")
        return

    completion_times.sort()
    window_s = float(THROUGHPUT_RATE_WINDOW_S)
    sample_s = float(THROUGHPUT_RATE_SAMPLE_S)
    max_time = max(completion_times)
    n_steps = int(max_time // sample_s) + 1
    sample_times = [i * sample_s for i in range(n_steps + 1)]

    rates_per_hour = []
    left = 0
    right = 0
    n = len(completion_times)

    for t in sample_times:
        while left < n and completion_times[left] < t - window_s:
            left += 1
        while right < n and completion_times[right] <= t:
            right += 1
        count_in_window = right - left
        rates_per_hour.append(count_in_window * 3600.0 / window_s)

    sample_days = [_seconds_to_sim_days(t) for t in sample_times]
    graphname = "throughput_rate_moving.png"

    plt.figure(figsize=(12, 6))
    plt.plot(sample_days, rates_per_hour, linewidth=0.8, color="#1f77b4")
    plt.xlabel("Time [days]")
    plt.ylabel("Throughput rate [units/hour]")
    plt.title(f"Moving throughput rate over time (trailing window = {int(window_s)} s, sampling = {int(sample_s)} s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()

    print(f">> Generated {graphname}")



def plot_throughput_rate_interval(unit_data, graphfolder):
    print(">> Generating interval throughput rate plot!")

    completion_times = []
    for row in unit_data:
        value = row.get("completion_time_s", row.get("completion_time", ""))
        if value in ("", None):
            continue
        try:
            completion_times.append(float(value))
        except ValueError:
            pass

    if not completion_times:
        print(">> No valid completion times found. Skipping interval throughput rate plot.")
        return

    completion_times.sort()
    interval_s = float(THROUGHPUT_RATE_INTERVAL_S)
    max_time = max(completion_times)
    n_bins = int(max_time // interval_s) + 1
    bin_starts = [i * interval_s for i in range(n_bins)]
    bin_rates = [0.0 for _ in range(n_bins)]

    for t in completion_times:
        idx = min(int(t // interval_s), n_bins - 1)
        bin_rates[idx] += 3600.0 / interval_s

    bin_days = [_seconds_to_sim_days(t) for t in bin_starts]
    graphname = "throughput_rate_interval.png"

    plt.figure(figsize=(12, 6))
    plt.step(bin_days, bin_rates, where="post", linewidth=2, color="#ff7f0e")
    plt.xlabel("Time [days]")
    plt.ylabel("Throughput rate [units/hour]")
    plt.title(f"Interval throughput rate over time (interval = {int(interval_s)} s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()

    print(f">> Generated {graphname}")


def plot_order_lateness_boxplot(order_data, graphfolder):
    print(">> Generating lateness box plot!")

    lateness_values = []
    for row in order_data:
        value = row.get("lateness", "")
        if value in ("", None):
            continue
        try:
            lateness_values.append(float(value) / 3600.0)
        except ValueError:
            pass

    if not lateness_values:
        print(">> No valid lateness data found. Skipping lateness box plot.")
        return

    graphname = "lateness_boxplot.png"

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        lateness_values,
        patch_artist=True,
        boxprops=dict(facecolor="#9ecae1", color="black"),
        medianprops=dict(color="red", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="#1f77b4", markersize=4, markeredgecolor="black")
    )
    plt.axhline(y=0, linestyle="--", linewidth=1, color="black", label="Due date")
    plt.xticks([1], ["orders"])
    plt.ylabel("lateness [h]")
    plt.title("distribution of order lateness")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()

    print(f">> Generated {graphname}")

def _priority_color(priority_value):
    """Map priority 1..5 to colors.
    1 = green, 2 = yellow-green, 3 = yellow, 4 = orange, 5 = red.
    Unknown/missing priority = gray.
    """
    try:
        p = int(float(priority_value))
    except (TypeError, ValueError):
        return "#7f7f7f"

    mapping = {
        1: "#2ca02c",  # green
        2: "#9ACD32",  # yellow-green
        3: "#fceb31",  # yellow
        4: "#ff7f0e",  # orange
        5: "#d62728",  # red
    }
    return mapping.get(p)


def _plot_order_lateness_variant(valid_rows, graphfolder, graphname, title, show_order_ids=True):
    if not valid_rows:
        print(f">> No valid lateness data found. Skipping {graphname}.")
        return

    orders = [row["order_id"] for row in valid_rows]
    lateness = [row["lateness"] / 3600.0 for row in valid_rows]
    priorities = [row.get("priority") for row in valid_rows]
    avg_lateness = sum(lateness) / len(lateness)
    colors = [_priority_color(p) for p in priorities]

    plt.figure(figsize=(12, 6))
    plt.bar(orders, lateness, color=colors)

    plt.axhline(
        y=0,
        linestyle="-",
        linewidth=1,
        color="black",
        label="Due date"
    )

    plt.axhline(
        y=avg_lateness,
        linestyle=":",
        linewidth=1,
        color="blue",
        label=f"Average lateness = {avg_lateness:.2f} h"
    )

    legend_handles = [
        mpatches.Patch(color="#2ca02c", label="Priority 1"),
        mpatches.Patch(color="#9ACD32", label="Priority 2"),
        mpatches.Patch(color="#fceb31", label="Priority 3"),
        mpatches.Patch(color="#ff7f0e", label="Priority 4"),
        mpatches.Patch(color="#d62728", label="Priority 5"),
    ]
    plt.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1, label="Due date"),
        plt.Line2D([0], [0], color="blue", linestyle=":", linewidth=1, label=f"Average lateness = {avg_lateness:.2f} h"),
    ], loc="lower left")

    max_labels = 30
    n_orders = len(orders)
    step = max(1, n_orders // max_labels)

    plt.xticks([])

    plt.ylabel("Lateness [h]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")


def plot_order_lateness(order_data, graphfolder):
    print(">> Generating order lateness plots!")

    valid_rows = []
    for row in order_data:
        lateness_val = row.get("lateness", "")
        order_id_val = row.get("order_id", "")
        if lateness_val in ("", None) or order_id_val in ("", None):
            continue

        try:
            record = {
                "order_id": str(order_id_val),
                "lateness": float(lateness_val),
                "priority": row.get("priority", None),
                "due date": row.get("due date", None),
                "finish_time": row.get("finish_time", None),
            }

            due_val = row.get("due date", None)
            if due_val not in ("", None):
                record["due date"] = float(due_val)
            else:
                record["due date"] = None

            finish_val = row.get("finish_time", None)
            if finish_val not in ("", None):
                record["finish_time"] = float(finish_val)
            else:
                record["finish_time"] = None

            valid_rows.append(record)
        except ValueError:
            pass

    if not valid_rows:
        print(">> No valid lateness data found. Skipping order lateness plots.")
        return

    def sort_by_order_id(row):
        try:
            return int(row["order_id"])
        except ValueError:
            return row["order_id"]

    def sort_by_due_date(row):
        due_date = row.get("due date")
        if due_date is None:
            return float("inf")
        return due_date

    def sort_by_lateness(row):
        return row["lateness"]

    def sort_by_finish_time(row):
        finish_time = row.get("finish_time")
        if finish_time is None:
            return float("inf")
        return finish_time

    by_order_id = sorted(valid_rows, key=sort_by_order_id)
    by_due_date = sorted(valid_rows, key=sort_by_due_date)
    by_lateness = sorted(valid_rows, key=sort_by_lateness, reverse=True)
    by_finish_time = sorted(valid_rows, key=sort_by_finish_time)

    _plot_order_lateness_variant(
        by_order_id,
        graphfolder,
        graphname="order_lateness.png",
        title="Lateness per order (sorted by order id)",
        show_order_ids=False
    )

    _plot_order_lateness_variant(
        by_due_date,
        graphfolder,
        graphname="order_lateness_due_date.png",
        title="Lateness per order (sorted by due date)"
    )

    _plot_order_lateness_variant(
        by_lateness,
        graphfolder,
        graphname="order_lateness_lateness.png",
        title="Lateness per order (sorted by lateness)"
    )

    _plot_order_lateness_variant(
        by_finish_time,
        graphfolder,
        graphname="order_lateness_finish_time.png",
        title="Lateness per order (sorted by finish time)"
    )


def _apply_fitness_symlog(values):
    """Apply a symmetric log y-scale so fitness plots can handle negative, zero, and positive values."""
    nonzero = [abs(float(v)) for v in values if float(v) != 0.0]
    linthresh = min(nonzero) / 10.0 if nonzero else 1.0
    plt.yscale("symlog", linthresh=linthresh)

def _plot_order_fitness_variant(valid_rows, graphfolder, graphname, title, show_order_ids=True):
    if not valid_rows:
        print(f">> No valid fitness data found. Skipping {graphname}.")
        return

    orders = [row["order_id"] for row in valid_rows]
    fitness_values = [row["fitness"] for row in valid_rows]
    priorities = [row.get("priority") for row in valid_rows]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    colors = [_priority_color(p) for p in priorities]

    plt.figure(figsize=(12, 6))
    plt.bar(orders, fitness_values, color=colors)

    plt.axhline(
        y=0,
        linestyle="-",
        linewidth=1,
        color="black",
        label="Zero fitness"
    )

    plt.axhline(
        y=avg_fitness,
        linestyle=":",
        linewidth=1,
        color="blue",
        label=f"Average fitness = {avg_fitness:.2f}"
    )

    legend_handles = [
        mpatches.Patch(color="#2ca02c", label="Priority 1"),
        mpatches.Patch(color="#9ACD32", label="Priority 2"),
        mpatches.Patch(color="#fceb31", label="Priority 3"),
        mpatches.Patch(color="#ff7f0e", label="Priority 4"),
        mpatches.Patch(color="#d62728", label="Priority 5"),
    ]
    plt.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1, label="Zero fitness"),
        plt.Line2D([0], [0], color="blue", linestyle=":", linewidth=1, label=f"Average fitness = {avg_fitness:.2f}"),
    ], loc="lower left")

    max_labels = 30
    n_orders = len(orders)
    step = max(1, n_orders // max_labels)

    plt.xticks([])
    _apply_fitness_symlog(fitness_values)
    plt.ylabel("Fitness [-] (log)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")


def plot_order_fitness(order_data, graphfolder):
    print(">> Generating order fitness plots!")
    valid_rows = []
    for row in order_data:
        fitness_val = row.get("fitness", "")
        order_id_val = row.get("order_id", "")
        if fitness_val in ("", None) or order_id_val in ("", None):
            continue
        try:
            record = {
                "order_id": str(order_id_val),
                "fitness": float(fitness_val),
                "priority": row.get("priority", None),
                "due date": row.get("due date", None),
                "finish_time": row.get("finish_time", None),
            }
            due_val = row.get("due date", None)
            if due_val not in ("", None):
                record["due date"] = float(due_val)
            else:
                record["due date"] = None
            finish_val = row.get("finish_time", None)
            if finish_val not in ("", None):
                record["finish_time"] = float(finish_val)
            else:
                record["finish_time"] = None
            valid_rows.append(record)
        except ValueError:
            pass

    if not valid_rows:
        print(">> No valid fitness data found. Skipping order fitness plots.")
        return

    def sort_by_order_id(row):
        try:
            return int(row["order_id"])
        except ValueError:
            return row["order_id"]

    def sort_by_due_date(row):
        due_date = row.get("due date")
        if due_date is None:
            return float("inf")
        return due_date

    def sort_by_fitness(row):
        return row["fitness"]

    def sort_by_finish_time(row):
        finish_time = row.get("finish_time")
        if finish_time is None:
            return float("inf")
        return finish_time

    by_order_id = sorted(valid_rows, key=sort_by_order_id)
    by_due_date = sorted(valid_rows, key=sort_by_due_date)
    by_fitness = sorted(valid_rows, key=sort_by_fitness, reverse=True)
    by_finish_time = sorted(valid_rows, key=sort_by_finish_time)

    _plot_order_fitness_variant(
        by_order_id,
        graphfolder,
        graphname="order_fitness.png",
        title="Fitness per order (sorted by order id)",
        show_order_ids=False
    )
    _plot_order_fitness_variant(
        by_due_date,
        graphfolder,
        graphname="order_fitness_due_date.png",
        title="Fitness per order (sorted by due date)"
    )
    _plot_order_fitness_variant(
        by_fitness,
        graphfolder,
        graphname="order_fitness_fitness.png",
        title="Fitness per order (sorted by fitness)"
    )
    _plot_order_fitness_variant(
        by_finish_time,
        graphfolder,
        graphname="order_fitness_finish_time.png",
        title="Fitness per order (sorted by finish time)"
    )


def plot_order_fitness_boxplot(order_data, graphfolder):
    print(">> Generating fitness box plot!")
    fitness_values = []
    for row in order_data:
        value = row.get("fitness", "")
        if value in ("", None):
            continue
        try:
            fitness_values.append(float(value))
        except ValueError:
            pass
    if not fitness_values:
        print(">> No valid fitness data found. Skipping fitness box plot.")
        return

    graphname = "fitness_boxplot.png"
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        fitness_values,
        patch_artist=True,
        boxprops=dict(facecolor="#c7e9c0", color="black"),
        medianprops=dict(color="red", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="#31a354", markersize=4, markeredgecolor="black")
    )
    plt.axhline(y=0, linestyle="--", linewidth=1, color="black", label="Zero fitness")
    _apply_fitness_symlog(fitness_values)
    plt.xticks([1], ["orders"])
    plt.ylabel("Fitness [-] (log)")
    plt.title("Distribution of order fitness")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")

def plot_station_waiting_time(station_data, graphfolder):
    print(">> Generating station waiting time!")
    graphname = "Station_waiting_time.png"

    stations = [row["station_name"] for row in station_data]
    times = [float(row["average_wait_time_s"]) for row in station_data]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(stations))))
    bars = ax.barh(stations, times, color="#1f77b4")

    ax.set_xlabel("Average wait time [s]")
    ax.set_title("Station waiting time")

    try:
        ax.bar_label(bars, labels=[f"{t:.1f} s" for t in times], padding=3)
    except AttributeError:
        for bar, val in zip(bars, times):
            ax.text(
                val + max(times) * 0.01 if times else 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} s",
                va="center", ha="left", fontsize=9
            )

    right = max(times) if times else 1
    ax.set_xlim(0, right * 1.15 if right > 0 else 1)

    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">> Generated {graphname}")


def plot_station_queue_size(station_data, graphfolder):
    print(">> Generating station queue size!")
    graphname = "Station_queue_size.png"

    stations = [row["station_name"] for row in station_data]
    times = [float(row["average_queue_length"]) for row in station_data]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(stations))))
    bars = ax.barh(stations, times, color="#9467bd")

    ax.set_xlabel("Average queue size [-]")
    ax.set_title("Station queue size")

    try:
        ax.bar_label(bars, labels=[f"{t:.2f}" for t in times], padding=3)
    except AttributeError:
        for bar, val in zip(bars, times):
            ax.text(
                val + max(times) * 0.01 if times else 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left", fontsize=9
            )

    right = max(times) if times else 1
    ax.set_xlim(0, right * 1.15 if right > 0 else 1)

    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">> Generated {graphname}")


def plot_station_utilization(station_data, graphfolder):
    print(">> Generating station utilization!")
    graphname = "Station_utilization.png"

    stations = [row["station_name"] for row in station_data]
    times = [float(row["utilization_active_window"]) * 100 for row in station_data]

    lower = 40
    higher = 80

    def color_for(u):
        if u < lower:
            return "#2ca02c"
        elif u < higher:
            return "#fceb31"
        else:
            return "#d62728"

    colors = [color_for(u) for u in times]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(stations))))
    bars = ax.barh(stations, times, color=colors)

    ax.set_xlabel("Utilization [%]")
    ax.set_title("Station utilization")

    try:
        ax.bar_label(bars, labels=[f"{t:.1f}%" for t in times], padding=3)
    except AttributeError:
        for bar, val in zip(bars, times):
            ax.text(
                val + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left", fontsize=9
            )

    right = max(times) if times else 1
    ax.set_xlim(0, right * 1.15)

    legend_handles = [
        mpatches.Patch(color="#2ca02c", label=f"Low (<{lower}%)"),
        mpatches.Patch(color="#fceb31", label=f"Medium ({lower}–{higher}%)"),
        mpatches.Patch(color="#d62728", label=f"High (≥{higher}%)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">> Generated {graphname}")



def plot_station_availability(station_data, graphfolder):
    print(">> Generating station availability!")
    graphname = "Station_availability.png"

    stations = [row["station_name"] for row in station_data]
    times = [float(row["availability"]) * 100 for row in station_data]

    lower = 40
    higher = 80

    def color_for(u):
        if u < lower:
            return "#d62728"
        elif u < higher:
            return "#fceb31"
        else:
            return "#2ca02c"

    colors = [color_for(u) for u in times]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(stations))))
    bars = ax.barh(stations, times, color=colors)

    ax.set_xlabel("Availability [%]")
    ax.set_title("Station availability")

    try:
        ax.bar_label(bars, labels=[f"{t:.1f}%" for t in times], padding=3)
    except AttributeError:
        for bar, val in zip(bars, times):
            ax.text(
                val + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left", fontsize=9
            )

    right = max(times) if times else 1
    ax.set_xlim(0, right * 1.15)

    legend_handles = [
        mpatches.Patch(color="#d62728", label=f"Low (<{lower}%)"),
        mpatches.Patch(color="#fceb31", label=f"Medium ({lower}–{higher}%)"),
        mpatches.Patch(color="#2ca02c", label=f"High (≥{higher}%)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">> Generated {graphname}")

def _extract_completion_times(unit_data):
    completion_times = []
    for row in unit_data:
        value = row.get("completion_time_s", row.get("completion_time", ""))
        if value in ("", None):
            continue
        try:
            completion_times.append(float(value))
        except ValueError:
            pass
    completion_times.sort()
    return completion_times


def _calc_cumulative_completed_series(unit_data):
    completion_times = _extract_completion_times(unit_data)
    if not completion_times:
        return [], []
    completion_days = [_seconds_to_sim_days(t) for t in completion_times]
    cumulative_units = list(range(1, len(completion_times) + 1))
    return completion_days, cumulative_units


def _calc_throughput_rate_moving_series(unit_data):
    completion_times = _extract_completion_times(unit_data)
    if not completion_times:
        return [], []

    window_s = float(THROUGHPUT_RATE_WINDOW_S)
    sample_s = float(THROUGHPUT_RATE_SAMPLE_S)
    max_time = max(completion_times)
    n_steps = int(max_time // sample_s) + 1
    sample_times = [i * sample_s for i in range(n_steps + 1)]

    rates_per_hour = []
    left = 0
    right = 0
    n = len(completion_times)

    for t in sample_times:
        while left < n and completion_times[left] < t - window_s:
            left += 1
        while right < n and completion_times[right] <= t:
            right += 1
        count_in_window = right - left
        rates_per_hour.append(count_in_window * 3600.0 / window_s)

    sample_days = [_seconds_to_sim_days(t) for t in sample_times]
    return sample_days, rates_per_hour


def _calc_throughput_rate_interval_series(unit_data):
    completion_times = _extract_completion_times(unit_data)
    if not completion_times:
        return [], []

    interval_s = float(THROUGHPUT_RATE_INTERVAL_S)
    max_time = max(completion_times)
    n_bins = int(max_time // interval_s) + 1
    bin_starts = [i * interval_s for i in range(n_bins)]
    bin_rates = [0.0 for _ in range(n_bins)]

    for t in completion_times:
        idx = min(int(t // interval_s), n_bins - 1)
        bin_rates[idx] += 3600.0 / interval_s

    bin_days = [_seconds_to_sim_days(t) for t in bin_starts]
    return bin_days, bin_rates


def _load_throughput_rate_per_hour(resultfolder: Path):
    candidates = [
        Path(resultfolder) / "kpi_summary.csv",
        Path(resultfolder) / "kpi_summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "throughput_rate_per_hour" in data:
                    return float(data["throughput_rate_per_hour"])
                continue

            with path.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for row in rows:
                if "throughput_rate_per_hour" in row and row["throughput_rate_per_hour"] not in ("", None):
                    return float(row["throughput_rate_per_hour"])
                lower_map = {str(k).strip().lower(): v for k, v in row.items()}
                for key_name in ("kpi", "metric", "name"):
                    if key_name in lower_map and str(lower_map[key_name]).strip().lower() == "throughput_rate_per_hour":
                        for value_name in ("value", "val", "result"):
                            if value_name in lower_map and lower_map[value_name] not in ("", None):
                                return float(lower_map[value_name])

            raw = path.read_text(encoding="utf-8-sig")
            for line in raw.splitlines():
                if "throughput_rate_per_hour" in line:
                    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                    for part in reversed(parts):
                        try:
                            return float(part)
                        except ValueError:
                            pass
        except Exception:
            continue
    return None




def _calc_average_rate_from_unit_data_within_limits(unit_data, x_min_days=None, x_max_days=None):
    completion_times = _extract_completion_times(unit_data)
    if not completion_times:
        return None

    x_min_s = None if x_min_days is None else float(x_min_days) * float(SIM_DAY_SECONDS)
    x_max_s = None if x_max_days is None else float(x_max_days) * float(SIM_DAY_SECONDS)

    filtered_completion_times = []
    for completion_time in completion_times:
        if x_min_s is not None and completion_time < x_min_s:
            continue
        if x_max_s is not None and completion_time > x_max_s:
            continue
        filtered_completion_times.append(float(completion_time))

    if len(filtered_completion_times) < 2:
        return None

    earliest_finish = min(filtered_completion_times)
    latest_finish = max(filtered_completion_times)
    if latest_finish <= earliest_finish:
        return None

    elapsed_time_s = latest_finish - earliest_finish
    units_finished = len(filtered_completion_times)
    return units_finished * 3600.0 / elapsed_time_s



def _combine_compare_average_rate(compare_entries, x_min_days=None, x_max_days=None):
    avg_rates = []
    for entry in compare_entries[:2]:
        unit_data = entry.get("unit_data")
        if not unit_data:
            continue
        avg_rate = _calc_average_rate_from_unit_data_within_limits(unit_data, x_min_days=x_min_days, x_max_days=x_max_days)
        if avg_rate is not None:
            avg_rates.append(avg_rate)
    if not avg_rates:
        return None
    return float(sum(avg_rates)) / float(len(avg_rates))


def _calc_signed_rl_t_line(x_values, y_values, average_rate, x_min=None, x_max=None):
    if average_rate is None or len(x_values) < 2:
        return 0.0
    area_day_units_per_hour = 0.0
    for i in range(len(x_values) - 1):
        x0 = float(x_values[i])
        x1 = float(x_values[i + 1])
        y0 = float(y_values[i])
        y1 = float(y_values[i + 1])
        if x1 <= x0:
            continue
        seg_left = x0 if x_min is None else max(x0, x_min)
        seg_right = x1 if x_max is None else min(x1, x_max)
        if seg_right <= seg_left:
            continue
        dx = x1 - x0
        left_ratio = (seg_left - x0) / dx
        right_ratio = (seg_right - x0) / dx
        yl = y0 + (y1 - y0) * left_ratio
        yr = y0 + (y1 - y0) * right_ratio
        dl = average_rate - yl
        dr = average_rate - yr
        width_days = seg_right - seg_left
        area_day_units_per_hour += (dl + dr) * 0.5 * width_days
    return area_day_units_per_hour * (SIM_DAY_SECONDS / 3600.0)



def _calc_signed_rl_t_step(x_values, y_values, average_rate, x_min=None, x_max=None):
    if average_rate is None or len(x_values) < 2:
        return 0.0
    area_day_units_per_hour = 0.0
    for i in range(len(x_values) - 1):
        left = float(x_values[i])
        right = float(x_values[i + 1])
        if right <= left:
            continue
        seg_left = left if x_min is None else max(left, x_min)
        seg_right = right if x_max is None else min(right, x_max)
        if seg_right <= seg_left:
            continue
        gap = average_rate - float(y_values[i])
        area_day_units_per_hour += gap * (seg_right - seg_left)
    return area_day_units_per_hour * (SIM_DAY_SECONDS / 3600.0)


def plot_compare_throughput_rate_moving(compare_entries, graphfolder, run_name):
    print(f">> Generating compare moving throughput rate plot for {run_name}!")
    graphname = "compare_throughput_rate_moving.png"

    plt.figure(figsize=(12, 6))
    plotted = False
    x_min = _seconds_to_sim_days(7200)
    x_max = None
    average_rate = _combine_compare_average_rate(compare_entries, x_min_days=x_min, x_max_days=x_max)
    rl_entries = []

    if average_rate is not None:
        plt.axhline(y=average_rate, linewidth=1.8, linestyle=":", color="black", label="Average")
        plotted = True

    color_offset = 0
    for idx, entry in enumerate(compare_entries[2:]):
        x_days, y_values = _calc_throughput_rate_moving_series(entry["unit_data"])
        if not x_days:
            continue
        line, = plt.plot(x_days, y_values, linewidth=0.8, label=entry["main_name"])
        where_mask = [((x >= x_min) and (x_max is None or x <= x_max)) for x in x_days] if average_rate is not None else None
        if average_rate is not None:
            plt.fill_between(
                x_days,
                y_values,
                [average_rate] * len(x_days),
                where=where_mask,
                interpolate=True,
                hatch='///',
                facecolor='none',
                edgecolor=line.get_color(),
                linewidth=0.0,
                zorder=1.3 - 0.1 * idx,
            )
            rl_value = _calc_signed_rl_t_line(x_days, y_values, average_rate, x_min=x_min, x_max=x_max)
            rl_entries.append((entry["main_name"], rl_value, line.get_color()))
        color_offset += len(entry.get("disruptions", []))
        plotted = True

    if not plotted:
        plt.close()
        print(f">> No valid moving throughput data found for {run_name}. Skipping compare moving throughput plot.")
        return

    plt.xlabel("Time [days]")
    plt.ylabel("Throughput rate [units/hour]")
    plt.title(f"Moving throughput rate comparison ({run_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    legend = plt.legend(loc="lower right")
    if rl_entries:
        legend_handles = legend.legend_handles if hasattr(legend, 'legend_handles') else legend.legendHandles
        legend_labels = [text.get_text() for text in legend.get_texts()]
        for main_name, rl_value, color in rl_entries:
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=color, hatch='///', label=f"{main_name} RL(t) = {rl_value:.2f} units lost"))
            legend_labels.append(f"{main_name} RL(t) = {rl_value:.2f} units lost")
        plt.legend(legend_handles, legend_labels, loc="lower right")
    plt.xlim(left=x_min, right=x_max)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")


def plot_compare_throughput_rate_interval(compare_entries, graphfolder, run_name):
    print(f">> Generating compare interval throughput rate plot for {run_name}!")
    graphname = "compare_throughput_rate_interval.png"

    plt.figure(figsize=(12, 6))
    plotted = False
    x_min = _seconds_to_sim_days(7200)
    x_max = None
    average_rate = _combine_compare_average_rate(compare_entries, x_min_days=x_min, x_max_days=x_max)
    rl_entries = []

    if average_rate is not None:
        plt.axhline(y=average_rate, linewidth=1.8, linestyle=":", color="black", label="Average")
        plotted = True

    color_offset = 0
    for idx, entry in enumerate(compare_entries[2:]):
        x_days, y_values = _calc_throughput_rate_interval_series(entry["unit_data"])
        if not x_days:
            continue
        line = plt.step(x_days, y_values, where="post", linewidth=2, label=entry["main_name"])[0]
        where_mask = [((x >= x_min) and (x_max is None or x <= x_max)) for x in x_days] if average_rate is not None else None
        if average_rate is not None:
            plt.fill_between(
                x_days,
                y_values,
                [average_rate] * len(x_days),
                where=where_mask,
                step='post',
                hatch='///',
                facecolor='none',
                edgecolor=line.get_color(),
                linewidth=0.0,
                zorder=1.3 - 0.1 * idx,
            )
            rl_value = _calc_signed_rl_t_step(x_days, y_values, average_rate, x_min=x_min, x_max=x_max)
            rl_entries.append((entry["main_name"], rl_value, line.get_color()))
        color_offset += len(entry.get("disruptions", []))
        plotted = True

    if not plotted:
        plt.close()
        print(f">> No valid interval throughput data found for {run_name}. Skipping compare interval throughput plot.")
        return

    plt.xlabel("Time [days]")
    plt.ylabel("Throughput rate [units/hour]")
    plt.title(f"Interval throughput rate comparison ({run_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    legend = plt.legend(loc="lower right")
    if rl_entries:
        legend_handles = legend.legend_handles if hasattr(legend, 'legend_handles') else legend.legendHandles
        legend_labels = [text.get_text() for text in legend.get_texts()]
        for main_name, rl_value, color in rl_entries:
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=color, hatch='///', label=f"{main_name} RL(t) = {rl_value:.2f} units lost"))
            legend_labels.append(f"{main_name} RL(t) = {rl_value:.2f} units lost")
        plt.legend(legend_handles, legend_labels, loc="lower right")
    plt.xlim(left=x_min, right=x_max)
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")


def _normalize_disruption_text(value) -> str:
    return str(value or "").strip().lower()



def _first_present(row, keys, default=""):
    for key in keys:
        if key in row and row.get(key) not in ("", None):
            return row.get(key)
    return default



def _parse_positive_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed > 0:
        return parsed
    return None



def _extract_emergency_order_duration_s(row):
    count_keys = [
        "units_in_order",
        "amount_of_units",
        "amount_units",
        "unit_count",
        "units",
        "n_units",
        "num_units",
        "number_of_units",
        "qty",
        "quantity",
        "order_size",
        "amount",
    ]
    for key in count_keys:
        parsed = _parse_positive_float(row.get(key))
        if parsed is not None:
            return parsed * 720.0

    list_keys = [
        "unit_ids",
        "units_included",
        "order_units",
        "unit_list",
    ]
    for key in list_keys:
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        tokens = [token.strip() for token in re.split(r'[;,|]+', str(raw)) if token.strip() != ""]
        if tokens:
            return float(len(tokens)) * 720.0

    return 720.0



def _load_station_type_names(disruption_config_path: Path | None = None):
    if disruption_config_path is None:
        disruption_config_path = ROOTDIR / "data" / "disruption_v2.json"
    path = Path(disruption_config_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    station_types = {}
    stations = data.get("Stations", {}) if isinstance(data, dict) else {}
    for station_id, station_info in stations.items():
        try:
            station_key = int(float(station_id))
        except (TypeError, ValueError):
            continue
        if isinstance(station_info, dict):
            station_type = station_info.get("station_type", "")
            if station_type not in ("", None):
                station_types[station_key] = str(station_type).strip()
    return station_types



def _format_station_instance_label(station_instance_value, station_type_names=None):
    s = str(station_instance_value or "").strip()
    if s == "":
        return s
    if s.lower().startswith("station "):
        return display_station_name(s)
    try:
        numeric_value = float(s)
    except ValueError:
        return s
    base_station = int(numeric_value)
    if station_type_names is None:
        station_type_names = {}
    station_type = station_type_names.get(base_station, "")
    if station_type:
        return f"Station {s}: {station_type}"
    return f"Station {s}"



def _station_instance_sort_key(value):
    s = str(value or "").strip()
    if s.lower().startswith("station "):
        s = s[8:]
        s = s.split(":", 1)[0].strip()
    try:
        parts = s.split(".")
        major = int(parts[0]) if parts[0] != "" else 10**9
        minor = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
        return (major, minor, s)
    except Exception:
        return (10**9, 0, s)


def _load_machine_breakdown_names(disruption_config_path: Path | None = None):
    if disruption_config_path is None:
        disruption_config_path = ROOTDIR / "data" / "disruption_v2.json"
    path = Path(disruption_config_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    station_breakdowns = {}
    stations = data.get("Stations", {}) if isinstance(data, dict) else {}
    for station_id, station_info in stations.items():
        try:
            station_key = int(station_id)
        except (TypeError, ValueError):
            continue
        breakdowns = station_info.get("machine breakdowns", []) if isinstance(station_info, dict) else []
        names = set()
        for entry in breakdowns:
            if not isinstance(entry, dict):
                continue
            normalized_name = _normalize_disruption_text(entry.get("name", ""))
            if normalized_name:
                names.add(normalized_name)
        station_breakdowns[station_key] = names
    return station_breakdowns


def _find_disruptions_used_csv(resultfolder: Path):
    candidates = [
        Path(resultfolder) / "disruptions_used.csv",
        Path(resultfolder) / "disruption_used.csv",
        Path(resultfolder).parent / "disruptions_used.csv",
        Path(resultfolder).parent / "disruption_used.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None



def plot_compare_disruption_gantt(graphfolder, disruptions_csv: str | Path, main_name: str, run_name: str):
    disruptions_csv = Path(disruptions_csv)
    machine_breakdown_names = _load_machine_breakdown_names()
    station_type_names = _load_station_type_names()
    events = []
    emergency_lane_key = "__emergency_orders__"
    emergency_lane_label = "Emergency orders"

    with disruptions_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dtype_raw = _first_present(row, ["disruption_type", "type", "disruption"], "")
            label_raw = _first_present(row, ["name", "disruption_name", "event_name"], dtype_raw)
            dtype = _normalize_disruption_text(dtype_raw)
            label = _normalize_disruption_text(label_raw)

            is_emergency_order = dtype == "emergency_order" or label == "emergency_order"
            is_inspection_failure = dtype == "inspection_failure" or label == "inspection_failure"

            start_raw = _first_present(row, ["start_time", "start_time_s"], "")
            end_raw = _first_present(row, ["end_time", "end_time_s"], "")
            if start_raw in ("", None):
                continue
            try:
                start_time = float(start_raw)
            except (TypeError, ValueError):
                continue

            if is_emergency_order:
                lane_key = emergency_lane_key
                lane_label = emergency_lane_label
                base_station_id = None
                duration_s = _extract_emergency_order_duration_s(row)
                end_time = start_time + duration_s
            else:
                station_id_raw = _first_present(row, ["station_id", "station"], "")
                station_instance_raw = _first_present(
                    row,
                    ["station_index", "station_instance", "station_instance_index"],
                    station_id_raw,
                )
                if station_instance_raw in ("", None):
                    continue
                lane_key = str(station_instance_raw).strip()
                if lane_key == "":
                    continue
                lane_label = _format_station_instance_label(lane_key, station_type_names)
                try:
                    base_station_id = int(float(station_id_raw)) if station_id_raw not in ("", None) else int(float(lane_key))
                except (TypeError, ValueError):
                    continue

                if is_inspection_failure:
                    end_time = start_time + 500.0
                else:
                    if end_raw in ("", None):
                        continue
                    try:
                        end_time = float(end_raw)
                    except (TypeError, ValueError):
                        continue

            if end_time <= start_time:
                continue

            events.append({
                "lane_key": lane_key,
                "lane_label": lane_label,
                "base_station_id": base_station_id,
                "start": start_time,
                "end": end_time,
                "type": dtype,
                "label": label,
                "is_emergency_order": is_emergency_order,
                "is_inspection_failure": is_inspection_failure,
            })

    if not events:
        print(f">> No valid disruptions found in {disruptions_csv}. Skipping compare disruption Gantt chart.")
        return

    lane_label_map = {event["lane_key"]: event["lane_label"] for event in events}
    normal_lanes = sorted(
        {e["lane_key"] for e in events if e["lane_key"] != emergency_lane_key},
        key=_station_instance_sort_key,
    )
    lanes = normal_lanes + ([emergency_lane_key] if emergency_lane_key in lane_label_map else [])
    lane_to_y = {lane: i for i, lane in enumerate(lanes)}
    plan_time = max(e["end"] for e in events)

    def color_for(event) -> str:
        if event["is_emergency_order"]:
            return "#9467bd"
        if event["is_inspection_failure"]:
            return "#1f77b4"
        station_breakdowns = machine_breakdown_names.get(event["base_station_id"], set())
        if event["label"] in station_breakdowns:
            return "green"
        if "eff" in event["type"] or "reduc" in event["type"] or "loss" in event["type"] or "eff" in event["label"] or "reduc" in event["label"] or "loss" in event["label"]:
            return "red"
        return "gray"

    events.sort(key=lambda e: (e["start"], lane_to_y[e["lane_key"]]))
    fig, ax = plt.subplots(figsize=(14, 6))
    lane_height = 0.8
    for event in events:
        y = lane_to_y[event["lane_key"]]
        y0 = y - lane_height / 2
        start_day = _seconds_to_sim_days(event["start"])
        duration_days = _seconds_to_sim_days(event["end"] - event["start"])
        ax.broken_barh(
            [(start_day, duration_days)],
            (y0, lane_height),
            facecolors=color_for(event),
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_title(f"Disruptions Gantt Chart ({main_name}, {run_name})")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Station")
    ax.set_yticks([lane_to_y[lane] for lane in lanes])
    ax.set_yticklabels([lane_label_map.get(lane, lane) for lane in lanes])
    ax.invert_yaxis()
    ax.set_xlim(0, _seconds_to_sim_days(plan_time))
    legend_items = [
        mpatches.Patch(facecolor="green", edgecolor="black", label="Machine breakdown"),
        mpatches.Patch(facecolor="red", edgecolor="black", label="Efficiency reduction"),
        mpatches.Patch(facecolor="#1f77b4", edgecolor="black", label="Inspection failure"),
        mpatches.Patch(facecolor="#9467bd", edgecolor="black", label="Emergency order"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    graphname = "compare_disruptions_gantt.png"
    fig.savefig(graphfolder / graphname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f">> Generated {graphname}")

def plot_compare_cumulative_completed_units(unit_data_by_main, graphfolder, run_name):
    print(f">> Generating compare cumulative completed units plot for {run_name}!")
    graphname = "compare_cumulative_completed_units.png"

    plt.figure(figsize=(12, 6))
    plotted = False
    for main_name, unit_data in unit_data_by_main:
        x_days, y_values = _calc_cumulative_completed_series(unit_data)
        if not x_days:
            continue
        plt.step(x_days, y_values, where="post", linewidth=2, label=main_name)
        plotted = True

    if not plotted:
        plt.close()
        print(f">> No valid cumulative data found for {run_name}. Skipping compare cumulative plot.")
        return

    plt.xlabel("Time [days]")
    plt.ylabel("Completed units [-]")
    plt.title(f"Cumulative completed units comparison ({run_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Generated {graphname}")



def generate_compare_graphs(mainfolders, post_processing_folder, starttime):
    compare_root = post_processing_folder / "COMPARE"
    compare_root.mkdir(parents=True, exist_ok=True)

    run_maps = []
    common_runs = None
    for mainfolder in mainfolders:
        runs = list_run_folders(mainfolder)
        run_map = {run.name: run / "results" for run in runs if (run / "results").exists()}
        run_maps.append((mainfolder, run_map))
        run_names = set(run_map.keys())
        common_runs = run_names if common_runs is None else (common_runs & run_names)

    if not common_runs:
        print(">> No common runs found across the selected main folders. Skipping compare plots.")
        return

    for run_name in sorted(common_runs, key=_run_sort_key):
        print(f"----creating compare graphs for {run_name}----")
        compare_folder = compare_root / run_name
        compare_folder.mkdir(parents=True, exist_ok=True)
        clear_folder(compare_folder)
        graph_folder = compare_folder / "graphs"
        graph_folder.mkdir(exist_ok=True)

        compare_entries = []
        cumulative_entries = []
        first_dynamic_disruptions_csv = None
        first_dynamic_main_name = None
        for idx, (mainfolder, run_map) in enumerate(run_maps):
            resultfolder = run_map[run_name]
            station_schedule, station_summary, transport_data, unit_data, material_data, order_data = load_all_data(resultfolder)
            compare_entries.append({
                "main_name": mainfolder.name,
                "unit_data": unit_data,
                "avg_rate": _load_throughput_rate_per_hour(resultfolder),
            })
            cumulative_entries.append((mainfolder.name, unit_data))
            if idx == 2:
                first_dynamic_disruptions_csv = _find_disruptions_used_csv(resultfolder)
                first_dynamic_main_name = mainfolder.name

        plot_compare_throughput_rate_moving(compare_entries, graph_folder, run_name)
        print("Time spent: " + str(time.perf_counter() - starttime))
        plot_compare_throughput_rate_interval(compare_entries, graph_folder, run_name)
        print("Time spent: " + str(time.perf_counter() - starttime))
        plot_compare_cumulative_completed_units(cumulative_entries, graph_folder, run_name)
        print("Time spent: " + str(time.perf_counter() - starttime))
        if first_dynamic_disruptions_csv is None:
            print(f"first_dynamic_disruptions_csv is none :(")
        if first_dynamic_disruptions_csv is not None:
            plot_compare_disruption_gantt(graph_folder, first_dynamic_disruptions_csv, first_dynamic_main_name, run_name)
            print("Time spent: " + str(time.perf_counter() - starttime))



def main(starttime=time.perf_counter()):
    output_dir = RESULTSDIR / "output"
    post_processing_folder = RESULTSDIR / "post_processing"
    post_processing_folder.mkdir(parents=True, exist_ok=True)

    choice = input("Do you want to plot data from one main? [y/N] >> ").strip().lower()
    if choice in ("y", "yes"):
        mainfolder, runs = find_results_folder(output_dir)
        mainfoldername = str(mainfolder).split("\\")[-1]
        for chosenrun in runs:
            resultfolder = chosenrun / "results"
            chosenrun = chosenrun.name
            print(f"----creating graphs for {chosenrun}----")
            ppfolder = post_processing_folder / mainfoldername / chosenrun
            ppfolder.mkdir(parents=True, exist_ok=True)

            print(f"placing graphs and so on inside {mainfoldername}")
            clear_folder(ppfolder)

            station_schedule, station_summary, transport_data, unit_data, material_data, order_data = load_all_data(resultfolder)

            graph_folder = ppfolder / "graphs"
            graph_folder.mkdir(exist_ok=True)

            starttime_gantt =0 #float(input("where do you want your gantt chart to start from? >>"))
            endtime_gantt = 0  #float(input("where do you want your gantt chart to end from? >>"))
            if endtime_gantt - starttime_gantt <= 0:
                print("time invalid therefore skipping")
            else:
                plot_gantt(station_schedule, transport_data, graph_folder, starttime_gantt, endtime_gantt)
                print("Time spent: " + str(time.perf_counter() - starttime))

            #plot_throughput_times(unit_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_throughput_rate_moving(unit_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_throughput_rate_interval(unit_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_cumulative_completed_units(unit_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_cumulative_completed_units_by_station(station_schedule, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))

            plot_order_lateness(order_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))

            plot_order_lateness_boxplot(order_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))

            plot_order_fitness(order_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_order_fitness_boxplot(order_data, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_station_waiting_time(station_summary, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_station_queue_size(station_summary, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_station_utilization(station_summary, graph_folder)
            print("Time spent: " + str(time.perf_counter() - starttime))
            plot_station_availability(station_summary, graph_folder)

    compare_choice = input("Do you want to compare five mains (2 average lines + 3 dynamic mains)? [y/N] >> ").strip().lower()
    if compare_choice in ("y", "yes"):
        compare_mainfolders = prompt_for_compare_main_folders(output_dir, count=5)
        generate_compare_graphs(compare_mainfolders, post_processing_folder, starttime)


if __name__ == "__main__":
    starttime = time.perf_counter()
    main(starttime)
    endtime = time.perf_counter()
    print(f"Total graph generation time: {endtime - starttime:.6f} seconds")
