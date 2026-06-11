# this is the script that calls all the functions from other scripts and runs them correctly

"""
1. A_input.py: creates the input file for the production planning.
2. B_production_planning.py: creates the production plan based on the input file.
3. C_GA_Scheduling.py / C_SA_Scheduling.py: creates the schedule.
4. D_production_line_sim.py: simulates the actual production process based on the schedule and disruptions.
5. F_graphgen.py: creates the graphs based on the output from the production line simulation.
6. G_after_movie.py: creates the movie based on the output from the production line simulation.

Wants and wishes
- display capacity and actual units in queue
"""

import csv
import json
import time as ti
from pathlib import Path
from itertools import product
from datetime import datetime

import A_input
import B_production_planning
import C_GA_Scheduling
import D_production_line_sim
import F_graphgen
import G_after_movie

# Try both possible capitalizations for the SA file.
# Use C_SA_Scheduler in the code below.
try:
    import C_SA_Scheduling as C_SA_Scheduler
except ImportError:
    import C_SA_scheduling as C_SA_Scheduler


# ================================================================================
# constants, paths and global variables
# ================================================================================

HOURS_PER_DAY = 8
WORKDAYS_PER_WEEK = 5
SECONDS_PER_WEEK = WORKDAYS_PER_WEEK * HOURS_PER_DAY * 3600

BASE_DIR = Path(__file__).parent
data_dir = BASE_DIR / "data"
layout_dir = data_dir / "Layouts"


# ================================================================================
# settings helper
# ================================================================================

def create_setting_json(
    output_path: Path,
    run_idx,
    layout_file,
    p_val,
    r_val,
    a_name,
    seed,
    label,
    pathlist
):
    """Create/update run-specific main_settings.json."""

    settings = {
        "run number": run_idx,
        "settings": {
            "Scenarios": layout_file,
            "pressure_of_capacity": p_val,
            "order_units_ratio": r_val,
            "algorithms": a_name,
            "weightage": None,
            "seed": seed,
        },
        "label": label,
        "pathlist": {k: str(v) for k, v in pathlist.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    return settings


# ================================================================================
# event time helper
# ================================================================================

def find_all_event_times(main_settings_json: Path):
    """
    Build sorted event times from disruptions CSV, plus t=0 and simulation horizon.

    This drives the event-based loop:
    - scheduler is called at event/segment boundaries
    - simulation is advanced until the next event time
    """

    main_settings_json = Path(main_settings_json)
    main_settings = A_input.read_settings_json(main_settings_json)

    pathlist = main_settings.get("pathlist", {})
    run_settings = main_settings.get("settings", {})

    disruptions_csv = (
        Path(pathlist["input_disruptions_csv"])
        if pathlist.get("input_disruptions_csv")
        else None
    )

    if disruptions_csv is None or not disruptions_csv.exists():
        raise FileNotFoundError(f"Disruptions CSV not found: {disruptions_csv}")

    event_times = [0]

    with disruptions_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            for col in ("start_time", "end_time"):
                raw = row.get(col)

                if raw is None:
                    continue

                raw = str(raw).strip()

                if raw == "" or raw.lower() == "nan":
                    continue

                try:
                    event_times.append(int(float(raw)))
                except ValueError:
                    continue

    try:
        base_settings = A_input.read_settings_json(data_dir / "base_settings.json")
    except Exception:
        base_settings = {}

    horizon = (
        run_settings.get("sim_time [s]")
        or base_settings.get("sim_time [s]")
        or base_settings.get("plan_time [s]")
        or SECONDS_PER_WEEK
    )

    try:
        horizon = int(float(horizon))
        event_times.append(horizon)
    except Exception:
        pass

    return sorted(set(int(t) for t in event_times if int(t) >= 0))


# ================================================================================
# terminal print helpers
# ================================================================================

def terminal_print(algo, react, dis, seg):
    print(
        "You are now running the simulation with the following settings!\n"
        f"Algorithm: {algo}\n"
        f"Reactive routing: {react}\n"
        f"Disruptions: {dis}"
    )

    if seg is not None:
        print(f"Segments: {seg}")
    else:
        print()


def terminal_print_after(algo, react, dis, seg):
    print(
        "You ran the simulation with the following settings!\n"
        f"Algorithm: {algo}\n"
        f"Reactive routing: {react}\n"
        f"Disruptions: {dis}"
    )

    if seg is not None:
        print(f"Segments: {seg}")
    else:
        print()


# ================================================================================
# scheduler selection
# ================================================================================

def normalize_scheduler_name(algorithm_name):
    """
    Converts different allowed names into either 'GA' or 'SA'.
    """

    name = str(algorithm_name).strip().lower()
    name = name.replace("-", "_").replace(" ", "_")

    if name in [
        "ga",
        "genetic_algorithm",
        "genetic_algortihm",   # keeps old misspelled setting from breaking
        "genetic_algorithm_",
    ]:
        return "GA"

    if name in [
        "sa",
        "simulated_annealing",
    ]:
        return "SA"

    raise ValueError(
        f"Unknown scheduling algorithm: {algorithm_name}. "
        "Allowed algorithms are 'GA' and 'SA'."
    )


def run_scheduling_algorithm(
    algorithm_name,
    main_settings_dir,
    t_start,
    t_stop,
    seed,
    rescheduling_enabled
):
    """
    Runs the selected scheduling algorithm using the same interface.
    """

    scheduler = normalize_scheduler_name(algorithm_name)

    if scheduler == "GA":
        print(f"----MAIN.py: running GA at t={t_start}")

        C_GA_Scheduling.main(
            main_settings_dir,
            t_start,
            t_stop,
            seed,
            rescheduling_enabled
        )

    elif scheduler == "SA":
        print(f"----MAIN.py: running SA at t={t_start}")

        C_SA_Scheduler.main(
            main_settings_dir,
            t_start,
            t_stop,
            seed,
            rescheduling_enabled
        )


# ================================================================================
# main
# ================================================================================

def main():
    main_start_time = ti.perf_counter()

    # --------------------------------------------------------------------------
    # create/check folders
    # --------------------------------------------------------------------------

    input_dir = BASE_DIR / "input"
    input_dir.mkdir(exist_ok=True)

    on_going_dir = BASE_DIR / "on_going"
    on_going_dir.mkdir(exist_ok=True)

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    post_processing_dir = BASE_DIR / "post_processing"
    post_processing_dir.mkdir(exist_ok=True)

    dirs = [input_dir, on_going_dir, output_dir, post_processing_dir]

    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    n = 0
    main_loop_name = f"main_{timestamp}_{n}"

    while (dirs[0] / main_loop_name).exists():
        n += 1
        main_loop_name = f"main_{timestamp}_{n}"

    for folder in dirs:
        subfolder = folder / main_loop_name
        subfolder.mkdir(exist_ok=True)

    dirs = [folder / main_loop_name for folder in dirs]

    simulation_performance_path = dirs[2] / "simulation_performance.csv"

    if not simulation_performance_path.exists():
        with simulation_performance_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "start of run",
                "run number",
                "schedule_calculation",
                "simulation_runtime"
            ])

    # --------------------------------------------------------------------------
    # create input subfolders
    # --------------------------------------------------------------------------

    disruption_dir = dirs[0] / "disruptions"
    disruption_dir.mkdir(exist_ok=True)

    orders_dir = dirs[0] / "orders"
    orders_dir.mkdir(exist_ok=True)

    runs_dir = dirs[0] / "runs"
    runs_dir.mkdir(exist_ok=True)

    dirs[0] = runs_dir

    # --------------------------------------------------------------------------
    # read settings
    # --------------------------------------------------------------------------

    base_settings = A_input.read_settings_json(data_dir / "base_settings.json")

    scenarios = list(base_settings["Scenarios"].items())
    pressures = list(base_settings["pressure_of_capacity"].items())
    ratios = list(base_settings["order_units_ratio"].items())
    algos = list(base_settings["algorithms"].items())
    seeds = list(base_settings["seeds"].items())

    sc_idx = {k: i + 1 for i, (k, _) in enumerate(scenarios)}
    p_idx = {k: i + 1 for i, (k, _) in enumerate(pressures)}
    r_idx = {k: i + 1 for i, (k, _) in enumerate(ratios)}
    s_idx = {k: i + 1 for i, (k, _) in enumerate(seeds)}

    # --------------------------------------------------------------------------
    # print settings/warnings
    # --------------------------------------------------------------------------

    warning = None
    seg = None

    algorithm_text = ", ".join(str(a_name) for _, a_name in algos)

    if int(base_settings["rescheduling_enabled"]) == 1:
        algo = algorithm_text
    else:
        algo = "earliest due date [production plan]"

        if int(base_settings["fast_GA"]) == 1:
            warning = (
                "Fast GA is enabled while there is no rescheduling. "
                "This may break the earliest due date sorting."
            )

    if int(base_settings["reaction_enabled"]) == 1:
        react = "Enabled"
    else:
        react = "Disabled"

    if int(base_settings["random based disruptions"]["enabled"]) == 2:
        dis = "Enabled"

        if int(base_settings["segment_time"]) == 1:
            warning = "segment_time is enabled while disruptions also are enabled"

    elif int(base_settings["random based disruptions"]["enabled"]) == 0:
        dis = "Disabled"

        if int(base_settings["segment_time"]) == 1:
            seg = (
                f"Enabled\n"
                f"Segment interval: {base_settings['segment_interval']}"
            )

        if int(base_settings["rescheduling_enabled"]) == 1:
            if int(base_settings["segment_time"]) == 0:
                warning = (
                    "With segments disabled you may have problems with scheduling "
                    "while there are no disruptions."
                )

        if int(base_settings["rescheduling_enabled"]) == 0:
            if int(base_settings["segment_time"]) == 1:
                seg_set = base_settings["segment_interval"]
                seg = f"Enabled\nSegment interval: {seg_set}"

    else:
        print(
            "Something is wrong in base_settings.\n"
            "--> 'random based disruptions.enabled' needs to be either 2 or 0."
        )
        return

    if warning is None:
        terminal_print(algo, react, dis, seg)
    else:
        print(f"WARNING!\n{warning}\n")
        return

    input("ARE THESE SETTINGS CORRECT? - press [ENTER] if you wish to run the program")

    plan_time = base_settings["plan_time [s]"]

    # --------------------------------------------------------------------------
    # generate order and disruption lists
    # --------------------------------------------------------------------------

    A_input.create_disruption_json(disruption_dir / "disruption.json")

    number = 0
    next_pct = 0
    max_number = len(scenarios) * len(pressures) * len(ratios) * len(seeds)

    action = "generating: "

    print("\n --- Generating the order and disruption lists ---\n")

    for (sc_name, layout_file), (p_name, p_val), (r_name, r_val), (s_id, seed) in product(
        scenarios,
        pressures,
        ratios,
        seeds
    ):
        number += 1

        layout_settings = A_input.read_settings_json(layout_dir / layout_file)

        num_units = int(layout_settings["scaled_monthly_capacity"] * p_val)
        num_orders = int(num_units / r_val)

        label = f"{sc_idx[sc_name]}_{p_idx[p_name]}_{r_idx[r_name]}_{s_idx[s_id]}"

        orderpath = orders_dir / f"unsorted_orders_{label}.csv"
        disruptionpath = disruption_dir / f"disruptions_{label}.csv"

        print(f"units: {num_units} and orders: {num_orders}")

        A_input.generate_orderlist(
            seed,
            plan_time,
            orderpath,
            num_orders,
            num_units
        )

        if int(base_settings["random based disruptions"]["enabled"]) == 2:
            A_input.generate_disruption_list(
                seed,
                plan_time,
                disruptionpath,
                num_orders,
                num_units,
                layout_dir / layout_file
            )

        elif int(base_settings["random based disruptions"]["enabled"]) == 0:
            A_input.generate_empty_disruption_list(disruptionpath)

        else:
            print(
                "Something is wrong in base_settings.\n"
                "--> 'random based disruptions.enabled' needs to be either 2 or 0."
            )
            return

        if int(base_settings["random based disruptions"]["enabled"]) == 2:
            A_input.plot_disruption_gantt(disruption_dir, disruptionpath)

        next_pct = G_after_movie.progress_update(
            number,
            max_number,
            next_pct,
            action=action
        )

    # --------------------------------------------------------------------------
    # run simulation combinations
    # --------------------------------------------------------------------------

    run_idx = 0
    next_pct = 0

    max_idx = (
        len(scenarios)
        * len(pressures)
        * len(ratios)
        * len(algos)
        * len(seeds)
    )

    print("\n --- Running the different combinations of scenarios ---\n")

    action = "Running simulations: "

    for (sc_name, layout_file), (p_name, p_val), (r_name, r_val), (a_id, a_name), (s_id, seed) in product(
        scenarios,
        pressures,
        ratios,
        algos,
        seeds
    ):
        run_idx += 1
        run_time_start = ti.perf_counter()

        # Create run folders
        for folder in dirs[0:3]:
            subfolder = folder / f"run_{run_idx}"
            subfolder.mkdir(exist_ok=True)

        label = f"{sc_idx[sc_name]}_{p_idx[p_name]}_{r_idx[r_name]}_{s_idx[s_id]}"

        order_csv_path = orders_dir / f"unsorted_orders_{label}.csv"
        on_going_run_path = dirs[1] / f"run_{run_idx}"

        input_runs_run = dirs[0] / f"run_{run_idx}"
        on_going_run = dirs[1] / f"run_{run_idx}"
        output_run = dirs[2] / f"run_{run_idx}"

        pathlist = {
            "input": dirs[0].parent,
            "input_disruptions": disruption_dir,
            "input_disruptions_csv": disruption_dir / f"disruptions_{label}.csv",
            "input_disruptions_json": disruption_dir / f"disruption.json",
            "input_orders": orders_dir,
            "input_orders_csv": orders_dir / f"unsorted_orders_{label}.csv",
            "input_runs": dirs[0],
            "input_runs_run": input_runs_run,
            "on_going": dirs[1],
            "on_going_run": on_going_run,
            "on_going_run_current_schedule": on_going_run / "current_schedule.csv",
            "on_going_run_dis_his": on_going_run / "disruption_his.csv",
            "on_going_run_production_plan": on_going_run / f"production_plan_{label}.csv",
            "on_going_run_unit_summary": on_going_run / "unit_summary.csv",
            "output": dirs[2],
            "output_run": output_run,
            "output_run_results": output_run / "results",
            "post_processing": dirs[3],
        }

        main_settings_dir = dirs[0] / f"run_{run_idx}" / "main_settings.json"

        create_setting_json(
            main_settings_dir,
            run_idx,
            layout_file,
            p_val,
            r_val,
            a_name,
            seed,
            label,
            pathlist
        )

        B_production_planning.create_production_plan(
            order_csv_path,
            on_going_run_path,
            layout_file,
            label,
            SECONDS_PER_WEEK
        )

        # Calculate num_units and num_orders again for correct run summary
        layout_settings = A_input.read_settings_json(layout_dir / layout_file)
        num_units = int(layout_settings["scaled_monthly_capacity"] * p_val)
        num_orders = int(num_units / r_val)

        # ----------------------------------------------------------------------
        # event times
        # ----------------------------------------------------------------------

        if int(base_settings["segment_time"]) == 0:
            event_times = find_all_event_times(main_settings_dir)

        elif int(base_settings["segment_time"]) == 1:
            segment_interval = base_settings["segment_interval"]
            segment_length = int(segment_interval * 60 * 60 * HOURS_PER_DAY)

            event_times = [0]

            t = segment_length

            while t < plan_time:
                event_times.append(t)
                t += segment_length

            event_times.append(plan_time)
            event_times.append(base_settings["sim_time [s]"])

            event_times = sorted(set(int(x) for x in event_times))

            print(f"event_times: {event_times}")

        else:
            print(
                "Something is wrong in base_settings.\n"
                "--> 'segment_time' needs to be either 1 or 0."
            )
            return

        rescheduling_enabled = int(base_settings["rescheduling_enabled"])
        reaction_enabled = int(base_settings["reaction_enabled"])

        # ----------------------------------------------------------------------
        # segment loop
        # ----------------------------------------------------------------------

        for i in range(max(0, len(event_times) - 1)):
            t_start = event_times[i]
            t_stop = event_times[i + 1]

            if t_stop <= t_start:
                continue

            start = ti.perf_counter()

            print(
                f"[MAIN] Segment {i + 1}/{len(event_times) - 1}: "
                f"t={t_start} -> {t_stop}\n"
                f"day {t_start / (3600 * 8):.4f} to {t_stop / (3600 * 8):.4f}"
            )

            if i == 0 or reaction_enabled:
                print(
                    f"----MAIN.py: running {a_name} "
                    f"in run {run_idx} at t={t_start}"
                )

                schedule_start = ti.perf_counter()

                run_scheduling_algorithm(
                    a_name,
                    main_settings_dir,
                    t_start,
                    t_stop,
                    seed,
                    rescheduling_enabled
                )

                schedule_end = ti.perf_counter()
                schedule_calctime = schedule_end - schedule_start

            else:
                print(
                    f"----MAIN.py: keeping existing schedule "
                    f"in run {run_idx} at t={t_start}"
                )

                schedule_calctime = 0

            print(
                f"----MAIN.py: running main sim from {t_start} until {t_stop}\n"
                f"day {t_start / (3600 * 8):.4f} to {t_stop / (3600 * 8):.4f}"
            )

            simstart = ti.perf_counter()

            D_production_line_sim.main(
                main_settings_dir,
                t_stop,
                t_start
            )

            simend = ti.perf_counter()
            sim_run_time = simend - simstart

            runtime = start - run_time_start

            with simulation_performance_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    runtime,
                    run_idx,
                    schedule_calctime,
                    sim_run_time
                ])

            print(f"[MAIN] segment wall time: {simend - start}\n\n\n - - - - - \n")
            print(f"the simulation has run for {int(simend - run_time_start)} seconds")

            ctotal_time = int(simend - main_start_time)
            ch = ctotal_time // 3600
            cm = (ctotal_time % 3600) // 60
            cs = ctotal_time % 60

            cclock_time = (
                f"{ch:d}:{cm:02d}:{cs:02d}"
                if ch > 0
                else f"{cm:02d}:{cs:02d}"
            )

            print(f"Time since starting program: {cclock_time}")

        # ----------------------------------------------------------------------
        # run summary
        # ----------------------------------------------------------------------

        run_time_end = ti.perf_counter()

        print(f"\n\n--- RUN {run_idx} ---")
        print(f"Run time {run_time_end - run_time_start}")
        print(f"Number of units: {num_units}")
        print(f"Number of orders: {num_orders}")
        print(f"Scenario = {sc_name} layout = {layout_file}, seed = {seed}")
        print(
            f"Pressure = {p_name} ({p_val})  "
            f"Ratio = {r_name} ({r_val})  "
            f"Algo = {a_name}"
        )

        print("----------------------------------------------------------------------------------------------------")

        next_pct = G_after_movie.progress_update(
            run_idx,
            max_idx,
            next_pct,
            action=action
        )

    # --------------------------------------------------------------------------
    # total summary
    # --------------------------------------------------------------------------

    main_stop_time = ti.perf_counter()

    total_time = int(main_stop_time - main_start_time)

    h = total_time // 3600
    m = (total_time % 3600) // 60
    s = total_time % 60

    clock_time = (
        f"{h:d}:{m:02d}:{s:02d}"
        if h > 0
        else f"{m:02d}:{s:02d}"
    )

    print(f"\nTOTAL TIME RUNNING MAIN\n{clock_time}")

    terminal_print_after(algo, react, dis, seg)


if __name__ == "__main__":
    main()