#This is our IPPS system 
#This script creates the IPPS based on an optimized selection of the process plans and scheduling.
# It uses the algorithms from algo.py and the sim without disruptions to create the the optimal IPPS.
# it needs to generate a starting guess in order to start the optimization process.

# -  INPUT  -
# the process plans for the different variants based on the layout selected in the settings.json
# the production plan for the line in form of a csv file created in the production_planning.py

# - INTERNAL  -
# make a random starting guess for the schedule and process route
# [MAYBE] sort the orders for the 5 days in order of due date for a better starting point
# randomize the rest

# takes the production plan and creates a schedule for the production line as an iteration by using algo.py
# the schedule is then used to run a simulation without disruptions (via production_line_sim.py) to evaluate the schedule 
# create a new schedule based on the result of the simulation and the optimization process in algo.py
# loops the internal until the optimization process is complete and the best schedule is found (convergence or max iterations)

# -  OUTPUT  -
# a sorted schedule (running 5 days ahead) for the production sim to run (with disruptions) 
# based on the optimized IPPS in form of a csv file


def main():
   print("this is the IPPS")


if __name__ == "__main__":
   main()