import matplotlib.pyplot as plt
import csv
import json
import os
from datetime import datetime
from pathlib import Path
#  +-----------------------------+
#  | Creates plots from the data |
#  +-----------------------------+

ROOTDIR = Path(__file__).parent

#find the correct order
def find_output_folder(base_path=None, target_timestamp=None):
    if base_path is None:
        base_path = ROOTDIR / "output"

    folders = []

    for name in os.listdir(base_path):
        full_path = base_path / name   

        if not full_path.is_dir():
            continue

        try:
            timestamp_str = name.split("__")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            folders.append((timestamp, full_path))

        except Exception:
            continue

    if not folders:
        raise FileNotFoundError(f"No valid output folders found in {base_path}")

    folders.sort(key=lambda x: x[0])

    if target_timestamp:
        target_dt = datetime.strptime(target_timestamp, "%Y%m%d_%H%M%S")
        closest = min(folders, key=lambda x: abs(x[0] - target_dt))
        return closest[1]

    return folders[-1][1]

#import data function
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
            if key in row:
                row[key] = float(row[key])
    return data

def load_all_data(output_folder):
    station_data = load_data(os.path.join(output_folder, "station_schedule.csv"))
    transport_data = load_data(os.path.join(output_folder, "transport_schedule.csv"))
    unit_data = load_data(os.path.join(output_folder, "unit_summary.csv"))

    # convert relevant columns
    station_data = to_float(station_data, [
        "start_time_s", "finish_time_s", "process_time_s"
    ])

    transport_data = to_float(transport_data, [
        "start_time_s", "finish_time_s", "transport_time_s"
    ])

    unit_data = to_float(unit_data, [
        "completion_time_s", "flow_time_s"
    ])

    return station_data, transport_data, unit_data

def plot_gantt_colored(station_data):
    fig, ax = plt.subplots(figsize=(12, 6))

    units = list(set(row["unit_id"] for row in station_data))
    colors = {u: i for i, u in enumerate(units)}

    for row in station_data:
        start = row["start_time_s"]
        duration = row["finish_time_s"] - start
        station = row["station_name"]
        unit = row["unit_id"]

        ax.barh(
            y=station,
            width=duration,
            left=start,
            color=plt.cm.tab20(colors[unit] % 20),
            alpha = 0.75
        )

    ax.set_xlabel("Time [s]")
    ax.set_title("Gantt Chart (colored by unit)")

    plt.tight_layout()
    plt.show()

def plot_flow_times(unit_data):
    units = [row["unit_id"] for row in unit_data]
    flow = [row["flow_time_s"] for row in unit_data]

    plt.figure()
    plt.bar(units, flow)
    plt.xticks(rotation=90)
    plt.ylabel("Flow time [s]")
    plt.title("Flow time per unit")

    plt.tight_layout()
    plt.show()

def plot_station_load(station_data):
    load = {}

    for row in station_data:
        station = row["station_name"]
        load.setdefault(station, 0)
        load[station] += row["process_time_s"]

    stations = list(load.keys())
    times = list(load.values())

    plt.figure()
    plt.barh(stations, times)
    plt.xlabel("Total processing time [s]")
    plt.title("Station workload")

    plt.tight_layout()
    plt.show()

#main
def main():
    specific_folder = "20260420_142405__orders_20_04_13_15_1_5xFUSE0_10xFUSE1_5xFUSE2"#enter the wanted foldername in the output folder
    specific_folder_choice = 0 #yes = 1, no = 0
    if specific_folder_choice == 1:
        folder = find_output_folder(target_timestamp=specific_folder)
    else:
        folder = find_output_folder()
    print(f"{folder}")
    station_data, transport_data, unit_data = load_all_data(folder)

    plot_gantt_colored(station_data)
    plot_flow_times(unit_data)
    plot_station_load(station_data)

if __name__ == "__main__":
    main()


"""
def _unit_color_map(unit_ids: list[str]) -> dict[str, Any]:
    if plt is None:
        return {}
    cmap = plt.get_cmap("tab20")
    sorted_unit_ids = sorted(unit_ids)
    return {unit_id: cmap(i % cmap.N) for i, unit_id in enumerate(sorted_unit_ids)}


def create_gantt_chart(
    operations: list[OperationRecord],
    transport_records: list[TransportRecord],
    output_path: Path,
) -> bool:
    if plt is None:
        return False

    if not operations:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Production line Gantt chart")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Stations / Transport")
        ax.text(0.5, 0.5, "No units were produced.", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True

    station_entries: list[tuple[int, int, str]] = []
    seen_station_names: set[str] = set()
    for op in sorted(operations, key=lambda x: x.station_index):
        if op.station_name in seen_station_names:
            continue
        stage_number, copy_number, _ = _extract_station_name_parts(op.station_name)
        station_entries.append((stage_number or op.station_index, copy_number or 0, op.station_name))
        seen_station_names.add(op.station_name)

    transport_entries: dict[int, str] = {}
    for tr in sorted(transport_records, key=lambda x: x.transport_index):
        transport_stage_number = _transport_stage_number_from_name(tr.transport_name)
        if transport_stage_number is None:
            transport_stage_number = tr.transport_index
        transport_entries.setdefault(transport_stage_number, tr.transport_name)

    stations_by_stage: defaultdict[int, list[tuple[int, str]]] = defaultdict(list)
    for stage_number, copy_number, station_name in station_entries:
        stations_by_stage[stage_number].append((copy_number, station_name))

    row_names: list[str] = []
    all_stage_numbers = sorted(stations_by_stage.keys())
    for stage_number in all_stage_numbers:
        for _, station_name in sorted(stations_by_stage[stage_number], key=lambda item: item[0]):
            row_names.append(station_name)
        if stage_number in transport_entries:
            row_names.append(transport_entries[stage_number])

    row_to_y = {name: idx for idx, name in enumerate(row_names)}
    unit_colors = _unit_color_map([op.unit_id for op in operations])

    fig, ax = plt.subplots(figsize=(18, 9))

    for op in operations:
        y = row_to_y[op.station_name]
        duration = op.finish_time_s - op.start_time_s
        color = unit_colors.get(op.unit_id)
        ax.barh(
            y,
            duration,
            left=op.start_time_s,
            height=0.62,
            color=color,
            edgecolor="black",
            linewidth=0.25,
        )
        if duration > 1.0:
            ax.text(
                op.start_time_s + duration / 2,
                y,
                f"{op.unit_id}-{op.variant}",
                ha="center",
                va="center",
                fontsize=5,
            )

    for tr in transport_records:
        y = row_to_y[tr.transport_name]
        duration = tr.finish_time_s - tr.start_time_s
        color = unit_colors.get(tr.unit_id)
        ax.barh(
            y,
            duration,
            left=tr.start_time_s,
            height=0.42,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.65,
        )
        if duration > 1.0:
            ax.text(
                tr.start_time_s + duration / 2,
                y,
                f"{tr.unit_id}-{tr.variant}",
                ha="center",
                va="center",
                fontsize=4.5,
            )

    ax.set_yticks(list(row_to_y.values()))
    ax.set_yticklabels(list(row_to_y.keys()))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stations / Transport")
    ax.set_title("Production line Gantt chart")
    ax.grid(True, axis="x", alpha=0.3)
    # Intentionally do not invert the y-axis so the chart keeps the original bottom-to-top orientation.
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_gantt_chart_no_transport(
    operations: list[OperationRecord],
    transport_records: list[TransportRecord],
    output_path: Path,
) -> bool:
    if plt is None:
        return False

    if not operations:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Production line Gantt chart")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Stations / Transport")
        ax.text(0.5, 0.5, "No units were produced.", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True

    station_names = []
    seen_stations = set()
    for op in sorted(operations, key=lambda x: x.station_index):
        if op.station_name not in seen_stations:
            station_names.append(op.station_name)
            seen_stations.add(op.station_name)

    row_names: list[str] = []
    for idx, station_name in enumerate(station_names):
        row_names.append(station_name)

    row_to_y = {name: idx for idx, name in enumerate(row_names)}
    unit_colors = _unit_color_map([op.unit_id for op in operations])

    fig, ax = plt.subplots(figsize=(18, 9))

    for op in operations:
        y = row_to_y[op.station_name]
        duration = op.finish_time_s - op.start_time_s
        color = unit_colors.get(op.unit_id)
        ax.barh(
            y,
            duration,
            left=op.start_time_s,
            height=0.62,
            color=color,
            edgecolor="black",
            linewidth=0.25,
        )
        if duration > 1.0:
            ax.text(
                op.start_time_s + duration / 2,
                y,
                f"{op.unit_id}-{op.variant}",
                ha="center",
                va="center",
                fontsize=5,
            )

    ax.set_yticks(list(row_to_y.values()))
    ax.set_yticklabels(list(row_to_y.keys()))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stations / Transport")
    ax.set_title("Production line Gantt chart")
    ax.grid(True, axis="x", alpha=0.3)
    # Intentionally do not invert the y-axis so the chart keeps the original bottom-to-top orientation.
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_throughput_chart(unit_summaries: list[UnitSummary], output_path: Path) -> bool:
    if plt is None:
        return False

    completion_times = sorted(u.completion_time_s for u in unit_summaries)
    x = [0.0] + completion_times
    y = [0] + list(range(1, len(completion_times) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(x, y, where="post")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Completed units")
    ax.set_title("Cumulative completed phones over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True

chart_notes: list[str] = []
if not create_gantt_chart(operations, transport_records, run_output_dir / "gantt_chart.png"):
    chart_notes.append("gantt_chart.png not created because matplotlib is not installed.")
if not create_gantt_chart_no_transport(operations, transport_records, run_output_dir / "gantt_chart_no_transport.png"):
    chart_notes.append("gantt_chart_no_transport.png not created because matplotlib is not installed.")
if not create_throughput_chart(unit_summaries, run_output_dir / "throughput_chart.png"):
    chart_notes.append("throughput_chart.png not created because matplotlib is not installed.")
if chart_notes:
    (run_output_dir / "charts_skipped.txt").write_text("\n".join(chart_notes), encoding="utf-8")
"""