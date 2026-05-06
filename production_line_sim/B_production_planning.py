#this is our production planning for the line
#it needs to quickly sort the orders and create the production plan for the line 
#based on the input file created in input.py
# this is orderbased and weekly planning for a month ahead (4 weeks)

#  -  INPUT  -
# some layout of the line (from settings.json) and the order file created in input.py
# based on the layout a capacity should be calculated for the line
# a unsorted order csv file for the entire month (4 weeks)

# -  OUTPUT  -
# a rough sorted production plan csv file for the 4 weeks (unsorted within the week)
# production plan for the line in form of a csv file 
import csv
from pathlib import Path
from A_input import read_settings_json




def main(order_dir):
    print("reading settings and order file...")
    # read settings json file
    order_dir = Path(order_dir)
    settings = read_settings_json(order_dir / "settings.json")
    # read order csv file
    order_csv_path = order_dir / "orders.csv"



    

if __name__ == "__main__":
   main()