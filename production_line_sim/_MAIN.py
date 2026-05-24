#this is the script that calls all the function from other script and runs it correctly


"""
1. input.py: creates the input file for the production planning.
2. production_planning.py: creates the production plan based on the input file.
3. IPPS.py: creates the IPPS based on the process plans and scheduling. 
   By using algorithms from algo.py and the sim without disruptions.
4. production_line_sim.py: simulates the actual production process based on the IPPS and the disruptions.
5. graphgen.py: creates the graphs based on the output from the production line simulation.
6. after_movie.py: creates the movie based on the output from the production line simulation.

Wants and wishes
- display a capacity and actual units in queue 

"""
import A_input, B_production_planning, C_GA_Scheduling, D_production_line_sim, F_graphgen, G_after_movie
from pathlib import Path
from itertools import product
from copy import deepcopy
from datetime import datetime
import json
import time as ti
import csv

# ================================================================================
# constands, paths and global variables
# ================================================================================
HOURS_PER_DAY = 8
WORKDAYS_PER_WEEK = 5
SECONDS_PER_WEEK = WORKDAYS_PER_WEEK * HOURS_PER_DAY * 3600

BASE_DIR = Path(__file__).parent
data_dir = BASE_DIR / "data"
layout_dir = data_dir / "Layouts"

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
    """Create/update run-specific main_settings.json (single source of truth for paths)."""

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

def find_all_event_times(main_settings_json: Path):
    """Build sorted event times from disruptions CSV, plus t=0 and the planning/simulation horizon.

    This drives the event-based loop: GA is called at every start/end time already reached,
    then the simulation is advanced only until the next event time. Future events are not
    written into the on-going history before their timestamp is reached.
    """
    main_settings_json = Path(main_settings_json)
    main_settings = A_input.read_settings_json(main_settings_json)
    pathlist = main_settings.get("pathlist", {})
    run_settings = main_settings.get("settings", {})

    disruptions_csv = Path(pathlist["input_disruptions_csv"]) if pathlist.get("input_disruptions_csv") else None
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

    # Add final horizon so the final segment runs after the last disruption event.
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

def terminal_print(algo,react,dis,seg):
    print("You are now running the simulation with the following settings!\n"+
              f"Algorithm: {algo}\n" + 
              f"Reactive routing: {react}\n"+
              f"Disruptions: {dis}")
    print(f"Segments: {seg}") if seg is not None else print()

def terminal_print_after(algo,react,dis,seg):
    print("You ran the simulation with the following settings!\n"+
              f"Algorithm: {algo}\n" + 
              f"Reactive routing: {react}\n"+
              f"Disruptions: {dis}")
    print(f"Segments: {seg}") if seg is not None else print()


def main():
    main_start_time = ti.perf_counter()
    #create folders/check they are there
    input_dir = BASE_DIR / "input"
    input_dir.mkdir(exist_ok=True)
    on_going_dir = BASE_DIR / "on_going"
    on_going_dir.mkdir(exist_ok=True)
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    post_processing_dir = BASE_DIR / "post_processing"
    post_processing_dir.mkdir(exist_ok=True)

    #create subfolders
    dirs = [input_dir,on_going_dir,output_dir,post_processing_dir]

    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    n=0
    main_loop_name = f"main_{timestamp}_{n}"
    while (dirs[0] / main_loop_name).exists():
            n=n+1
            main_loop_name = f"main_{timestamp}_{n}"
    for dir in dirs:
        subfolder = dir / main_loop_name
        subfolder.mkdir(exist_ok=True)
    
    dirs = [dir / main_loop_name for dir in dirs]
    
    #create input subfolders
    disruption_dir = dirs[0] / "disruptions"
    disruption_dir.mkdir(exist_ok=True)
    orders_dir = dirs[0] / "orders"
    orders_dir.mkdir(exist_ok=True)
    runs_dir = dirs[0] / "runs"
    runs_dir.mkdir(exist_ok=True)

    dirs[0] = runs_dir
    
    #read settings
    base_settings = A_input.read_settings_json(data_dir / "base_settings.json")
    scenarios = list(base_settings["Scenarios"].items())
    pressures = list(base_settings["pressure_of_capacity"].items())
    ratios = list(base_settings["order_units_ratio"].items())
    algos = list(base_settings["algorithms"].items())
    seeds = list(base_settings["seeds"].items())
    
    # build index maps for the keys
    sc_idx = {k: i+1 for i, (k, _) in enumerate(scenarios)}
    p_idx  = {k: i+1 for i, (k, _) in enumerate(pressures)}
    r_idx  = {k: i+1 for i, (k, _) in enumerate(ratios)}
    s_idx  = {k: i+1 for i, (k, _) in enumerate(seeds)}

    #print which version you are running with
    warning = None
    seg = None
    if base_settings["rescheduling_enabled"] == 1:
        algo = "Genetic algorithm"
        if base_settings["fast_GA"] == 1:
            algo = "Fast genetic algotihm"
    elif base_settings["rescheduling_enabled"] == 0:
        algo = "earliest due date [production plan]"
        if base_settings["fast_GA"] == 1:
            warning = "Fast ga is enabled while there is no rescheduling which will break the earliest due date sorting!"

    if base_settings["reaction_enabled"] == 1:
        react = "Enabled"
    elif base_settings["reaction_enabled"] == 0:
        react = "Disabled"
    
    if base_settings["random based disruptions"]["enabled"]==2:
        dis ="enabled"
        if base_settings["segment_time"]==1:
            warning= "Segment_time is enabled while disruptions also are enabled"
    elif base_settings["random based disruptions"]["enabled"]==0:
        dis ="Disabled"
        if base_settings["segment_time"]==1:
            seg = f"Enabled \nSegment interval: {base_settings['segment_interval']}"
        if base_settings["rescheduling_enabled"] == 1:
            if base_settings["segment_time"]==0:
                warning = "With segments disabled you will have problems with the GA while there are no disruptions"
        if base_settings["rescheduling_enabled"] == 0:
            if base_settings["segment_time"]==1:
                seg_set = base_settings['segment_interval']
                seg = f"Enabled \nSegment interval: {seg_set}"

    if warning is None:
        terminal_print(algo,react,dis,seg)
    else:
        print(f"WARNING!\n{warning}\n")
        return
        
    input("ARE THESE SETTING CORRECT? - press [ENTER] if you wish to run the program")
    plan_time = base_settings["plan_time [s]"]
    #generate the order lists and the disruption lists
    A_input.create_disruption_json(disruption_dir / "disruption.json")

    number = 0
    next_pct = 0
    max_number = len(scenarios) * len(pressures) * len(ratios) * len(seeds)
    action = "generating: "
    print("\n --- Generating the order and disruption lists ---\n")
    for (sc_name, layout_file), (p_name, p_val), (r_name, r_val), (s_id, seed) in product(
        scenarios, pressures, ratios, seeds
    ):
        number += 1
        layout_settings = A_input.read_settings_json(layout_dir / layout_file)
        num_units = int(layout_settings["scaled_monthly_capacity"]*p_val)
        num_orders = int(num_units/r_val)

        label = f"{sc_idx[sc_name]}_{p_idx[p_name]}_{r_idx[r_name]}_{s_idx[s_id]}"

        orderpath = orders_dir / f"unsorted_orders_{label}.csv"
        disruptionpath = disruption_dir/f"disruptions_{label}.csv"
        print(f"units: {num_units} and orders: {num_orders}")
        A_input.generate_orderlist(seed,plan_time,orderpath,num_orders, num_units)
        if base_settings["random based disruptions"]["enabled"] == 2:
            A_input.generate_disruption_list(seed,plan_time,disruptionpath,num_orders, num_units,layout_dir /layout_file)
        elif base_settings["random based disruptions"]["enabled"] == 0:
            A_input.generate_empty_disruption_list(disruptionpath)
        else:
            print("something is wrong in the base_settings \n--> the \"random based disruptions\" needs to be either 2 or 0 for enabled or disabled")
            return

        # takes wayyy too long to generate a gantt chart for each one
        if base_settings["random based disruptions"]["enabled"]==2:
            A_input.plot_disruption_gantt(disruption_dir,disruptionpath)
        next_pct = G_after_movie.progress_update(number, max_number, next_pct,action=action)
        #break
    #input("change the disruptions file")
    run_idx = 0
    next_pct = 0
    max_idx = len(scenarios) * len(pressures) * len(ratios)* len(algos) * len(seeds)
    print("\n --- Running the different combinations of scenarios ---\n")
    action = "Running simulations: "
    for (sc_name, layout_file), (p_name, p_val), (r_name, r_val), (a_id, a_name), (s_id, seed) in product(
        scenarios, pressures, ratios, algos, seeds
    ):
        run_idx += 1
        run_time_start = ti.perf_counter()
        #creating the run dirs
        for dir in dirs[0:3]:
            subfolder = dir / f"run_{run_idx}"
            subfolder.mkdir(exist_ok=True)
         

        # creating the production plan
        label = f"{sc_idx[sc_name]}_{p_idx[p_name]}_{r_idx[r_name]}_{s_idx[s_id]}"
        order_csv_path = orders_dir / f"unsorted_orders_{label}.csv"
        on_going_run_path = dirs[1] / f"run_{run_idx}"

        # Path list for the folders is generated here
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
            "on_going_run_current_schedule": on_going_run / f"current_schedule.csv",
            "on_going_run_dis_his": on_going_run / f"disruption_his.csv",
            "on_going_run_production_plan": on_going_run / f"production_plan_{label}.csv",
            "on_going_run_unit_summary": on_going_run / f"unit_summary.csv",
            "output": dirs[2],
            "output_run": output_run,
            "output_run_results":output_run / "results",
            "post_processing": dirs[3]
        }
        

        # creating the selected settings json for the run
        main_settings_dir = dirs[0]/ f"run_{run_idx}"/"main_settings.json"
        create_setting_json(main_settings_dir, run_idx, layout_file, p_val, r_val, a_name, seed,
                            label,pathlist)

        
        B_production_planning.create_production_plan(order_csv_path,on_going_run_path,layout_file,label,SECONDS_PER_WEEK)

        #finds the event_times based on the disruption file for the run
        if base_settings["segment_time"] == 0:
            event_times = find_all_event_times(main_settings_dir)
        elif base_settings["segment_time"] == 1:
            
            segment_interval = base_settings["segment_interval"]
            segment_length = int(segment_interval * 60*60*HOURS_PER_DAY)

            event_times = [0]
            t = segment_length
            while t < plan_time:
                    event_times.append(t)
                    t += segment_length

            event_times.append(plan_time)
            event_times.append(base_settings["sim_time [s]"])
            print(f"event_times: {event_times}")
        else:
            print("something is wrong in the base_settings \n--> the \"segment_time\" needs to be either 1 or 0 for enabled or disabled")
        # Run simulation in event-driven segments: [t_i, t_{i+1}).
        # GA is called at t=0 to create the first routed schedule. If the selected
        # algorithm name contains GA, it is also called again at every disruption
        # start/end timestamp so it can react only to already-known events.
        # Run GA at every segment boundary so it can reschedule from the current snapshot.
        rescheduling_enabled = base_settings["rescheduling_enabled"]
        reaction_enabled = base_settings["reaction_enabled"]

                
        for i in range(max(0, len(event_times) - 1)):
            t_start = event_times[i]
            t_stop = event_times[i + 1]
            if t_stop <= t_start:
                continue

            start = ti.perf_counter()
            print(f"[MAIN] Segment {i+1}/{len(event_times)-1}: t={t_start} -> {t_stop}\n day {t_start/(3600*8):.4f} to {t_stop/(3600*8):.4f}")

            if i == 0 or reaction_enabled:
                print(f"----MAIN.py: running GA in run {run_idx} at t={t_start}")
                C_GA_Scheduling.main(main_settings_dir, t_start,t_stop,seed,rescheduling_enabled)
            else:
                print(f"----MAIN.py: keeping existing schedule in run {run_idx} at t={t_start}")

            print(f"----MAIN.py: running main sim from {t_start} until {t_stop}\n day {t_start/(3600*8):.4f} to {t_stop/(3600*8):.4f}")
            D_production_line_sim.main(main_settings_dir, t_stop, t_start)
            end = ti.perf_counter()
            print(f"[MAIN] segment wall time: {end - start}\n\n\n - - - - - \n")
            print(f"the simulation has run for {int(end - run_time_start)} seconds")
            ctotal_time = int(end-main_start_time)
            ch = ctotal_time // 3600
            cm = (ctotal_time % 3600) // 60
            cs = ctotal_time % 60
            cclock_time = f"{ch:d}:{cm:02d}:{cs:02d}"if ch > 0 else f"{cm:02d}:{cs:02d}"
            print(f"Time since starting program: {cclock_time}")
            #input("Press [ENTER] to continue the loop")
        run_time_end = ti.perf_counter()
        print(f"\n\n--- RUN {run_idx} ---")
        print(f"Run time {run_time_end-run_time_start}")
        print(f"Number of units: {num_units}")
        print(f"Number of orders: {num_orders}")
        print(f"Scenario = {sc_name} layout = {layout_file}, seed = {seed}")
        print(f"Pressure = {p_name} ({p_val})  Ratio = {r_name} ({r_val})  Algo = {a_name}")

        #pipeline(settings, num_orders=num_orders, num_units=num_units, algo_choice=algo_choice)
        print("----------------------------------------------------------------------------------------------------")
        next_pct = G_after_movie.progress_update(run_idx, max_idx, next_pct,action=action)
        #stop the loop
    main_stop_time = ti.perf_counter()
    total_time = int(main_stop_time-main_start_time)
    h = total_time // 3600
    m = (total_time % 3600) // 60
    s = total_time % 60
    clock_time = f"{h:d}:{m:02d}:{s:02d}"if h > 0 else f"{m:02d}:{s:02d}"
    print(f"\nTOTAL TIME RUNNING MAIN\n{clock_time}")
    terminal_print_after(algo,react,dis,seg)

if __name__ == "__main__":
   main()
