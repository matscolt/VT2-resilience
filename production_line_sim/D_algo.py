# This script holds our algorithms for the optimization process in the IPPS. 
# It is used in the C_IPPS.py script to create the optimal schedule for the production line simulation. 
# The algorithms are based on the principles of genetic algorithms and simulated annealing (maybe adding complex), 
# but we have adapted them to fit our specific problem and constraints. 
# The main goal of these algorithms is to find the best schedule that minimizes the objective function,
# which is a combination of KPIS weighted in a manner that reflects the priorities of the production line


# -  INPUT  -
# the script itself does not take any direct input, 
# but it uses the production plan created in B_production_planning.py and 
# the process plans for the different variants based on the layout selected in the settings.json

# -  OUTPUT  -
# provides functions for the optimization process in the IPPS, 
# which is used to create the optimal schedule for the production line simulation.



# -  ALGORITHMS  -

# 1. Genetic Algorithm: This algorithm is inspired by the process of natural selection 
# and is used to find optimal solutions to complex problems. 
# It works by creating a population of potential solutions (schedules), 
# evaluating their fitness based on the objective function, and then using 
# selection, crossover, and mutation operations to create new generations of solutions. 
# The algorithm iteratively improves the population until it converges to an optimal solution 
# or reaches a maximum number of iterations.



# 2. Simulated Annealing: This algorithm is inspired by the annealing process in metallurgy 
# and is used to find a good approximation of the global optimum in a large search space. 
# It works by starting with an initial solution (schedule) and then iteratively making small changes to 
# the solution. The algorithm accepts changes that improve the objective function, 
# but it also accepts worse solutions with a certain probability that decreases over time (the "temperature"). 
# This allows the algorithm to escape local optima and explore the search space more effectively.




# 3. [MAYBE] Complex optimazation
# This is the box complex method 
