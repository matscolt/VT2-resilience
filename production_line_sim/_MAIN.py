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
import A_input, B_production_planning, C_IPPS, D_algo, E_production_line_sim, F_graphgen, G_after_movie
from pathlib import Path
from itertools import product
from copy import deepcopy

# ================================================================================
# constands, paths and global variables
# ================================================================================
HOURS_PER_DAY = 8
WORKDAYS_PER_WEEK = 5
SECONDS_PER_WEEK = WORKDAYS_PER_WEEK * HOURS_PER_DAY * 3600

base_dir = Path(__file__).parent
data_dir = base_dir / "data"
layout_dir = data_dir / "Layouts"

#a single run of the disruption sim
def pipeline(settings):
   print(f"Based on the current plan_time no more than {A_input.round_half_up(settings['plan_time [s]']/A_input.AVE_CYCLE_TIME_PER_UNIT)} units should be selected")
   num_orders = int(input("Enter amount of orders: "))
   num_units = int(input("Enter amount of units: "))
   order_dir = A_input.main(num_orders, num_units)
   B_production_planning.main(order_dir,SECONDS_PER_WEEK)
   #sim with disruption loop
   # the sim should pause when a disruption happens and then 
   # the IPPS should come up with a new solution to the disrupted line and continue the sim with disruptions
   C_IPPS.main(order_dir) #creates new plan
   #E_production_line_sim.run_simulation() #with disruptions

   #F_graphgen.main()
   #G_after_movie.main()

# -  main function  -
def main():
   mainsettings = A_input.read_settings_json(data_dir / "main_setting.json")
   settings = A_input.read_settings_json(data_dir / "settings.json")

   pipeline(settings)
   


def main2():
    mainsettings = A_input.read_settings_json(data_dir / "main_setting.json")
    base_settings = A_input.read_settings_json(data_dir / "settings.json")

    scenarios = list(mainsettings["Scenarios"].items())
    pressures = list(mainsettings["pressure_of_capacity"].items())
    ratios = list(mainsettings["order_units_ratio"].items())
    algos = list(mainsettings["algorithms"].items())
    


    run_idx = 0
    for (sc_name, layout_file), (p_name, p_val), (r_name, r_val), (a_id, a_name) in product(
        scenarios, pressures, ratios, algos
    ):
        run_idx += 1
        settings = deepcopy(base_settings)

        # Apply scenario -> layout
        settings["line_layout_file"] = layout_file
        layout_settings = A_input.read_settings_json(layout_dir / layout_file)
        # Apply pressure_of_capacity (your code needs to define what this means)
        # Example: scale plan_time[s] or simulation_time[s]
        # settings["plan_time [s]"] = settings["plan_time [s]"] * (p_val/100)

        # Apply order_units_ratio (again: define your mapping)
        # Example: increase/decrease units relative to the base
        num_units = int(layout_settings["scaled_monthly_capacity"]*p_val)
        num_orders = int(num_units/r_val)


        # Algorithm choice (pass into IPPS when you support it)
        algo_choice = {"algorithm_id": a_id, "algorithm_name": a_name}

        print(f"\n--- RUN {run_idx} ---")
        print(f"number of units: {num_units}")
        print(f"number of orders: {num_orders}")
        print(f"Scenario={sc_name} layout={layout_file}")
        print(f"Pressure={p_name} ({p_val})  Ratio={r_name} ({r_val})  Algo={a_name}")

        #pipeline(settings, num_orders=num_orders, num_units=num_units, algo_choice=algo_choice)
        print("----------------------------------------------------------------------------------------------------")



if __name__ == "__main__":
   loop = True
   if loop == True:
      while loop == True:
         user = input("old main(o) or new main(n)  (o/n)\n>> ").lower()
         if user == "o":
            main()
            loop = False
         if user == "n":
            main2()
            loop = False
         elif user != "n" or user != "o":
            print("\n--- please select between 'o' or 'n' ---")
   else:
      main2()