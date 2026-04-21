import matplotlib.pyplot as plt
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
#  +-----------------------------+
#  | Creates plots from the data |
#  +-----------------------------+

ROOTDIR = Path(__file__).parent

#find the correct order
def find_output_folder(output_path=None, target_timestamp=None):
    if output_path is None:
        output_path = ROOTDIR / "output"

    folders = []

    for name in os.listdir(output_path):
        full_path = output_path / name   

        if not full_path.is_dir():
            continue

        try:
            timestamp_str = name.split("__")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            folders.append((timestamp, full_path))

        except Exception:
            continue

    if not folders:
        raise FileNotFoundError(f"No valid output folders found in {output_path}")

    folders.sort(key=lambda x: x[0])

    if target_timestamp:
        target_dt = datetime.strptime(target_timestamp, "%Y%m%d_%H%M%S")
        closest = min(folders, key=lambda x: abs(x[0] - target_dt))
        return closest[1]

    return folders[-1][1]


def clear_folder(folder):
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
    folder = str(folder).split("\\")[-1].split("__")[0]
    print(f"cleaned out {folder}/graph")


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

#convert the data to a float 
def to_float(data, keys):
    for row in data:
        for key in keys:
            if key in row:
                row[key] = float(row[key])
    return data

def load_all_data(data_folder):
    station_data = load_data(os.path.join(data_folder, "station_schedule.csv"))
    transport_data = load_data(os.path.join(data_folder, "transport_schedule.csv"))
    unit_data = load_data(os.path.join(data_folder, "unit_summary.csv"))
    material_data = load_data(os.path.join(data_folder, "unit_summary.csv"))

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

    return station_data, transport_data, unit_data,material_data

# ---------------------------
# PLOTS
# ---------------------------



def plot_gantt(station_data, transport_data, graphfolder_dir):

    # ---------------------------
    # Build ordered y-axis
    # ---------------------------
    stations = sorted(set(
        (int(row["station_index"]), row["station_name"])
        for row in station_data
    ))

    transports = sorted(set(
        (int(row["transport_index"]), row["transport_name"])
        for row in transport_data
    ))

    # Interleave: S1, T1, S2, T2, ...
    y_labels_full = []
    for i in range(len(stations)):
        y_labels_full.append(stations[i][1])
        if i < len(transports):
            y_labels_full.append(transports[i][1])

    y_pos_full = {label: i for i, label in enumerate(y_labels_full)}

    # Station-only
    y_labels_station = [s[1] for s in stations]
    y_pos_station = {label: i for i, label in enumerate(y_labels_station)}

    # Colors
    units = list(set(row["unit_id"] for row in station_data))
    colors = {u: i for i, u in enumerate(units)}

    # ===========================
    # Stations only
    # ===========================
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    graphname = "Gantt_chart_stations.png"
    for row in station_data:
        start = row["start_time_s"]
        duration = row["finish_time_s"] - start
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
    ax1.set_title("Gantt Chart (Stations only)")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    fig1.tight_layout()
    fig1.savefig(graphfolder_dir / graphname, dpi=200)
    plt.close(fig1)

    print(f">> Generated {graphname}")

    # ===========================
    # Stations + Transport
    # ===========================
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    graphname2 = "Gantt_chart_with_transport.png"
    # --- Stations ---
    for row in station_data:
        start = row["start_time_s"]
        duration = row["finish_time_s"] - start
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

        ax2.text(
            start + duration / 2,
            y,
            unit,
            ha="center",
            va="center",
            fontsize=7
        )

    # --- Transport ---
    for row in transport_data:
        start = row["start_time_s"]
        duration = row["finish_time_s"] - start
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
    ax2.set_title("Gantt Chart (Stations + Transport)")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    fig2.tight_layout()
    fig2.savefig(graphfolder_dir / graphname2, dpi=200)
    plt.close(fig2)

    print(f">> Generated {graphname2}")
    
def plot_flow_times(unit_data,graphfolder):
    units = [row["unit_id"] for row in unit_data]
    flow = [float(row["active_flow_time_s"]) for row in unit_data]
    graphname = "Flow_times.png"
    avg_flow = sum(flow) / len(flow)

    plt.figure()
    plt.bar(units, flow)

    plt.axhline(
        y=avg_flow,
        linestyle=":",
        linewidth=1,
        color="black",
        label=f"Average = {avg_flow:.2f} s"
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
    plt.ylabel("Flow time [s]")
    plt.title("Flow time per unit")

    plt.tight_layout()
    plt.savefig(graphfolder/graphname, dpi=200, bbox_inches="tight")
    print(f">> Generated {graphname}")

def plot_station_load(station_data,graphfolder):
    load = {}
    graphname = "Station_operation_time.png"

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
    plt.savefig(graphfolder/graphname, dpi=200, bbox_inches="tight")
    print(f">> Generated {graphname}")

#main
def main():
    specific_folder = "20260420_131845"#enter the wanted foldername(only the first timestamp) in the output folder
    
    specific_folder_choice = 0 #yes = 1, no = 0
    
    if specific_folder_choice == 1:
        try:
            folder = find_output_folder(target_timestamp=specific_folder)
            print(f">> Using selected folder: {folder}")
        except Exception as e:
            print(f">> Warning: Could not find requested folder ({e})")
            print(">> Falling back to newest folder instead.")
            folder = find_output_folder()
    else:
        folder = find_output_folder()
    foldername = str(folder).split("\\")[-1].split("__")[0]
    print(f"using folder: {foldername}")

    clear_folder(folder)
    station_data, transport_data, unit_data, material_data = load_all_data(folder)

    graph_folder = folder / "graphs"
    graph_folder.mkdir(exist_ok=True)
    
    plot_gantt(station_data,transport_data,graph_folder)
    plot_flow_times(unit_data,graph_folder)
    plot_station_load(station_data,graph_folder)

if __name__ == "__main__":
    starttime = time.perf_counter()
    main()
    endtime = time.perf_counter()
    print(f"Total graph generation time: {endtime - starttime:.6f} seconds")


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