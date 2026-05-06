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

def main():
   num_orders = int(input("Enter amount of orders: "))
   num_units = int(input("Enter amount of units: "))
   order_dir = A_input.main(num_orders, num_units)
   B_production_planning.main(order_dir)


if __name__ == "__main__":
   main()