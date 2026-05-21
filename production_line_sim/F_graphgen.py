import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

import os
from datetime import datetime
from pathlib import Path

def find_output_folder(output_path=None, target_timestamp=None, prompt=True):
    if output_path is None:
        output_path = ROOTDIR / "output"
    else:
        output_path = Path(output_path)

    folders = []
    current_year = datetime.now().year

    for name in os.listdir(output_path):
        full_path = output_path / name

        if not full_path.is_dir():
            continue

        try:
            # Expect: main_DD-MM_HH-MM_n  -> ["main", "DD-MM", "HH-MM", "n"]
            parts = name.split("_")
            if len(parts) < 4 or parts[0] != "main":
                continue

            timestamp_str = parts[1] + "_" + parts[2]      # "DD-MM_HH-MM"
            ts = datetime.strptime(timestamp_str, "%d-%m_%H-%M").replace(year=current_year)

            # Counter (n) can be used as a tie-breaker if same timestamp
            n = int(parts[3])

            folders.append((ts, n, full_path))

        except Exception:
            continue

    if not folders:
        raise FileNotFoundError(f"No valid output folders found in {output_path}")

    # Newest first: sort by timestamp then counter
    folders.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # If target timestamp is given, pick closest
    if target_timestamp:
        # Expect "DD-MM_HH-MM" (same as folder naming, no year)
        target_dt = datetime.strptime(target_timestamp, "%d-%m_%H-%M").replace(year=current_year)
        closest = min(folders, key=lambda x: abs(x[0] - target_dt))
        return closest[2]

    # If user wants to choose interactively
    if prompt:
        outputs = [p for (_, _, p) in folders]
        return prompt_for_data_folder(output_path, outputs)

    # Default: newest
    return folders[0][2]


def prompt_for_data_folder(output_dir: Path, outputs: list[Path]) -> Path:
    if not outputs:
        raise FileNotFoundError(f"No output folders found in: {output_dir}")

    print("\nAvailable run folders in ./output (newest first):")
    for i, p in enumerate(outputs, start=1):
        print(f"  {i:2d}) {p.name}")

    default = 1
    while True:
        s = input(f"Choose output folder number (Enter = {default}): ").strip()
        if s == "":
            return outputs[default - 1]
        if s.isdigit() and 1 <= int(s) <= len(outputs):
            return outputs[int(s) - 1]
        candidate = output_dir / s
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")

def prompt_for_run_folder(main_dir: Path, runs: list[Path]) -> Path:
    if not runs:
        raise FileNotFoundError(f"No output folders found in: {main_dir}")

    print("\nAvailable run folders in ./output (newest first):")
    for i, p in enumerate(runs, start=1):
        print(f"  {i:2d}) {p.name}")

    default = 1
    while True:
        s = input(f"Choose output folder number (Enter = {default}): ").strip()
        if s == "":
            return runs[default - 1]
        if s.isdigit() and 1 <= int(s) <= len(runs):
            return runs[int(s) - 1]
        candidate = main_dir / s
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")


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
    station_schedule = load_data(os.path.join(data_folder, "station_schedule.csv"))
    station_summary = load_data(os.path.join(data_folder, "station_summary.csv"))
    transport_data = load_data(os.path.join(data_folder, "transport_schedule.csv"))
    unit_data = load_data(os.path.join(data_folder, "unit_summary.csv"))
    material_data = load_data(os.path.join(data_folder, "unit_summary.csv"))

    # convert relevant columns
    station_schedule = to_float(station_schedule, [
        "start_time_s", "finish_time_s", "process_time_s"
    ])
   
    station_summary = to_float(station_summary, [
        "busy_time_s","first_start_time_s","last_finish_time_s",
        "max_queue_length","average_queue_length","average_wait_time_s",
        "total_wait_time_s","utilization_overall","utilization_active_window"
    ]) 
    transport_data = to_float(transport_data, [
        "start_time_s", "finish_time_s", "transport_time_s"
    ])

    unit_data = to_float(unit_data, [
        "completion_time_s", "flow_time_s"
    ])

    return station_schedule, station_summary, transport_data, unit_data,material_data

# ---------------------------
# PLOTS
# ---------------------------



def plot_gantt(station_data, transport_data, graphfolder_dir):
    print(">> Generating Gantt charts!")
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
    print(">> Generating throughput time plots!")
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

def plot_station_utilization(station_data, graphfolder):
    print(">> Generating station utilization!")
    graphname = "Station_utilization.png"

    # 1) Data (én række per station i din CSV)
    stations = [row["station_name"] for row in station_data]
    times = [float(row["utilization_active_window"]) * 100 for row in station_data]

    # 2) Farvelogik (justér thresholds efter behov)
    lower = 40
    higher = 80

    def color_for(u):
        if u < lower:
            return "#2ca02c"   # grøn
        elif u < higher:
            return "#fceb31"   # gul
        else:
            return "#d62728"   # rød

    colors = [color_for(u) for u in times]

    # 3) Plot
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(stations))))
    bars = ax.barh(stations, times, color=colors)

    ax.set_xlabel("Utilization [%]")
    ax.set_title("Station utilization")

    # 4) Skriv værdien ved enden af hver bar
    # (bar_label kræver Matplotlib >= 3.4; fallback nedenfor hvis du får fejl)
    try:
        ax.bar_label(bars, labels=[f"{t:.1f}%" for t in times], padding=3)
    except AttributeError:
        for bar, val in zip(bars, times):
            ax.text(val + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center", ha="left", fontsize=9)

    # 5) Giv luft i højre side så labels ikke klippes
    right = max(times) if times else 1
    ax.set_xlim(0, right * 1.15)

    # 6) Legend (forklar farverne)
    legend_handles = [
        mpatches.Patch(color="#2ca02c", label=f"Low (<{lower}%)"),
        mpatches.Patch(color="#fceb31", label=f"Medium ({lower}–{higher}%)"),
        mpatches.Patch(color="#d62728", label=f"High (≥{higher}%)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(graphfolder / graphname, dpi=200, bbox_inches="tight")
    print(f">> Generated {graphname}")

#main
def main(starttime = time.perf_counter()):
    specific_folder = "20260420_131845"#enter the wanted foldername(only the first timestamp) in the output folder
    
    specific_folder_choice = 0 #yes = 1, no = 0
    
    if specific_folder_choice == 1:
        try:
            datafolder = find_output_folder(target_timestamp=specific_folder)
            print(f">> Using selected folder: {datafolder}")
        except Exception as e:
            print(f">> Warning: Could not find requested folder ({e})")
            print(">> Falling back to newest folder instead.")
            datafolder = find_output_folder()
    else:
        datafolder = find_output_folder()
    mainfoldername = str(datafolder).split("\\")[-1]
    runfolder = datafolder / prompt_for_run_folder() / "results"

    print(f"using datafolder: {mainfoldername} run folder: {runfolder}")

    post_processing_folder = ROOTDIR / "post_processing"
    ppfolder = post_processing_folder / mainfoldername
    print(f"placing graphs and so on inside {mainfoldername}")
    clear_folder(ppfolder)
    station_schedule, station_summary, transport_data, unit_data, material_data = load_all_data(runfolder)

    graph_folder = ppfolder / "graphs"
    graph_folder.mkdir(exist_ok=True)

    #plot_gantt(station_schedule,transport_data,graph_folder)
    print("Time spent: "+str(time.perf_counter()-starttime))
    plot_flow_times(unit_data,graph_folder)
    print("Time spent: "+str(time.perf_counter()-starttime))
    plot_station_utilization(station_summary,graph_folder)

if __name__ == "__main__":
    starttime = time.perf_counter()
    main(starttime)
    endtime = time.perf_counter()
    print(f"Total graph generation time: {endtime - starttime:.6f} seconds")

