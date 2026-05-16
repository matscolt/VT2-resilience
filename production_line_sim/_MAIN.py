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
import A_input, B_production_planning, C_IPPS, D_algo,MFI_GA_Scheduling_V3_3 , E_production_line_sim, F_graphgen, G_after_movie
from pathlib import Path
from itertools import product
from copy import deepcopy
from datetime import datetime
import json

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
    settings = {
        "run number": run_idx,
        "settings": {
            "Scenarios": layout_file,
            "pressure_of_capacity": p_val,
            "order_units_ratio": r_val,
            "algorithms": a_name,
            "weightage": None,
            "seed": seed
        },
        "label": label,
        "pathlist": {k: str(v) for k, v in pathlist.items()}
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)


def find_all_event_times(main_settings_json):
   print("here we find all the timestamps for when an 'event start' or 'event ends' happens")
   main_settings_json_read = A_input.read_settings_json(main_settings_json)

def main():
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
   
   print("reading settings")
   #read settings
   mainsettings = A_input.read_settings_json(data_dir / "main_setting.json")
   base_settings = A_input.read_settings_json(data_dir / "settings.json")
   scenarios = list(mainsettings["Scenarios"].items())
   pressures = list(mainsettings["pressure_of_capacity"].items())
   ratios = list(mainsettings["order_units_ratio"].items())
   algos = list(mainsettings["algorithms"].items())
   seeds = list(mainsettings["seeds"].items())
   
   # build index maps for the keys
   sc_idx = {k: i+1 for i, (k, _) in enumerate(scenarios)}
   p_idx  = {k: i+1 for i, (k, _) in enumerate(pressures)}
   r_idx  = {k: i+1 for i, (k, _) in enumerate(ratios)}
   s_idx  = {k: i+1 for i, (k, _) in enumerate(seeds)}


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
      A_input.generate_disruption_list(seed,plan_time,disruptionpath,num_orders, num_units,layout_dir /layout_file)
      # takes wayyy too long to generate a gantt chart for each one
      # A_input.plot_disruption_gantt(disruption_dir,disruptionpath)
      #next_pct = G_after_movie.progress_update(number, max_number, next_pct,action=action)
      break

   run_idx = 0
   next_pct = 0
   max_idx = len(scenarios) * len(pressures) * len(ratios)* len(algos) * len(seeds)
   print("\n --- Running the different combinations of scenarios ---\n")
   action = "Running simulations: "
   for (sc_name, layout_file), (p_name, p_val), (r_name, r_val), (a_id, a_name), (s_id, seed) in product(
        scenarios, pressures, ratios, algos, seeds
    ):
      run_idx += 1
      settings = deepcopy(base_settings)
      #creating the run dirs
      for dir in dirs[0:3]:
         subfolder = dir / f"run_{run_idx}"
         subfolder.mkdir(exist_ok=True)

      # creating the production plan
      label = f"{sc_idx[sc_name]}_{p_idx[p_name]}_{r_idx[r_name]}_{s_idx[s_id]}"
      order_csv_path = orders_dir / f"unsorted_orders_{label}.csv"
      on_going_run_path = dirs[1] / f"run_{run_idx}"
      label = f"{sc_idx[sc_name]}_{p_idx[p_name]}_{r_idx[r_name]}_{s_idx[s_id]}"

      # Path list for the folders is generated here
      input_runs_run = dirs[0] / f"run_{run_idx}"
      on_going_run = dirs[1] / f"run_{run_idx}"
      output_run = dirs[2] / f"run_{run_idx}"

      pathlist = {
         "input": dirs[0].parent,
         "input_disruptions": disruption_dir,
         "input_disruptions_csv": disruption_dir / f"disruption_{label}.csv",
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
      #event_times = find_all_event_times(main_settings_dir)
      event_times = [1,1,1,1,1]

      # runs a loop of the breaks for the main sim

      for time in event_times:
         MFI_GA_Scheduling_V3_3.main(main_settings_dir)
         E_production_line_sim.main(main_settings_dir)


      print(f"\n\n--- RUN {run_idx} ---")
      print(f"Number of units: {num_units}")
      print(f"Number of orders: {num_orders}")
      print(f"Scenario = {sc_name} layout = {layout_file}, seed = {seed}")
      print(f"Pressure = {p_name} ({p_val})  Ratio = {r_name} ({r_val})  Algo = {a_name}")

      #pipeline(settings, num_orders=num_orders, num_units=num_units, algo_choice=algo_choice)
      print("----------------------------------------------------------------------------------------------------")
      next_pct = G_after_movie.progress_update(run_idx, max_idx, next_pct,action=action)
      return #stop the loop



if __name__ == "__main__":
   main()
