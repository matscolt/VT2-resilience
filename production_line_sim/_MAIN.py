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
import A_input, B_production_planning
from pathlib import Path

# ================================================================================
# constands, paths and global variables
# ================================================================================
HOURS_PER_DAY = 8
WORKDAYS_PER_WEEK = 5
SECONDS_PER_WEEK = WORKDAYS_PER_WEEK * HOURS_PER_DAY * 3600


base_dir = Path(__file__).parent
data_dir = base_dir / "data"


# -  main function  -
def main():
   settings = A_input.read_settings_json(data_dir / "settings.json")
   print(f"Based on the current plan_time no more than {A_input.round_half_up(settings['plan_time [s]']/A_input.AVE_FLOW_TIME_PER_UNIT)} units should be selected")
   num_orders = int(input("Enter amount of orders: "))
   num_units = int(input("Enter amount of units: "))
   order_dir = A_input.main(num_orders, num_units)
   B_production_planning.main(order_dir,SECONDS_PER_WEEK)



if __name__ == "__main__":
   main()