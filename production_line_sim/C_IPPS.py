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

import D_algo, E_production_line_sim, process_routes

   


def main():
   print("this is the IPPS")
   print("this uses the algo script, the simulation script and the process planning script")

   # define convergence demand
   # max iter or better solution is not found within x
   iter = 0
   max_iter = 1000
   best_iter = 100


   record_score = None
   current_score = None

    #Rough sort of the plan in order assign days to the orders we want in the schedule
    

   while convergence == False:
    iter += 1

    #feed last score to generate a new iter
    #if last score is = None then its a starting guess

    guess = D_algo.test_algo()

    #generate schedule
    generate_schedule(guess)

    E_production_line_sim.run_simulation()
    E_production_line_sim.calculate_kpis()

    current_score = kpi_calc_to_score()
    
        
    if record_score > current_score or record_score is None:
       record_score = current_score
       record_iter = iter
    #convergence update

    #stop the loop
    if max_iter == iter or iter == record_iter + best_iter:
        convergence = True



   # run while loop until convergence 





if __name__ == "__main__":
   main()