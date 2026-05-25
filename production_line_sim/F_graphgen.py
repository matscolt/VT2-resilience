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
        if not finish_times:
            continue

        finish_times.sort()
        cumulative_units = list(range(1, len(finish_times) + 1))

        graphname = f"cumulative_completed_units_{_safe_filename(station_name)}.png"

        plt.figure(figsize=(12, 6))
        plt.step(finish_times, cumulative_units, where="post", linewidth=2, color="#1f77b4")
        plt.xlabel("time [s]")
        plt.ylabel("completed units [-]")
        plt.title(f"cumulative completed units over time\n{station_name}")
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

    plt.xticks(
        ticks=range(0, n_units, step),
        labels=units[::step],
        rotation=90
    )
    plt.ylabel("throughput time [s]")
    plt.title("throughput time per unit")

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
    plt.step(completion_times, cumulative_units, where="post", linewidth=2, color="#1f77b4")
    plt.xlabel("time [s]")
    plt.ylabel("completed units [-]")
    plt.title("cumulative completed units over time")
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


def _plot_order_lateness_variant(valid_rows, graphfolder, graphname, title):
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

    plt.xticks(
        ticks=range(0, n_orders, step),
        labels=orders[::step],
        rotation=90
    )

    plt.ylabel("lateness [h]")
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
            }

            due_val = row.get("due date", None)
            if due_val not in ("", None):
                record["due date"] = float(due_val)
            else:
                record["due date"] = None

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

    by_order_id = sorted(valid_rows, key=sort_by_order_id)
    by_due_date = sorted(valid_rows, key=sort_by_due_date)
    by_lateness = sorted(valid_rows, key=sort_by_lateness, reverse=True)

    _plot_order_lateness_variant(
        by_order_id,
        graphfolder,
        graphname="order_lateness.png",
        title="lateness per order (sorted by order id)"
    )

    _plot_order_lateness_variant(
        by_due_date,
        graphfolder,
        graphname="order_lateness_due_date.png",
        title="lateness per order (sorted by due date)"
    )

    _plot_order_lateness_variant(
        by_lateness,
        graphfolder,
        graphname="order_lateness_lateness.png",
        title="lateness per order (sorted by lateness)"
    )



def _plot_order_fitness_variant(valid_rows, graphfolder, graphname, title):
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

    plt.xticks(
        ticks=range(0, n_orders, step),
        labels=orders[::step],
        rotation=90
    )
    plt.ylabel("fitness [-]")
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
            }
            due_val = row.get("due date", None)
            if due_val not in ("", None):
                record["due date"] = float(due_val)
            else:
                record["due date"] = None
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

    by_order_id = sorted(valid_rows, key=sort_by_order_id)
    by_due_date = sorted(valid_rows, key=sort_by_due_date)
    by_fitness = sorted(valid_rows, key=sort_by_fitness, reverse=True)

    _plot_order_fitness_variant(
        by_order_id,
        graphfolder,
        graphname="order_fitness.png",
        title="fitness per order (sorted by order id)"
    )
    _plot_order_fitness_variant(
        by_due_date,
        graphfolder,
        graphname="order_fitness_due_date.png",
        title="fitness per order (sorted by due date)"
    )
    _plot_order_fitness_variant(
        by_fitness,
        graphfolder,
        graphname="order_fitness_fitness.png",
        title="fitness per order (sorted by fitness)"
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
    plt.xticks([1], ["orders"])
    plt.ylabel("fitness [-]")
    plt.title("distribution of order fitness")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    plt.close()
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

def main(starttime=time.perf_counter()):
    output_dir = RESULTSDIR / "output"
    mainfolder, runs = find_results_folder(output_dir)

    post_processing_folder = RESULTSDIR / "post_processing"
    post_processing_folder.mkdir(parents=True, exist_ok=True)
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

        plot_throughput_times(unit_data, graph_folder)
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
        plot_station_utilization(station_summary, graph_folder)
        print("Time spent: " + str(time.perf_counter() - starttime))
        plot_station_availability(station_summary, graph_folder)


if __name__ == "__main__":
    starttime = time.perf_counter()
    main(starttime)
    endtime = time.perf_counter()
    print(f"Total graph generation time: {endtime - starttime:.6f} seconds")
