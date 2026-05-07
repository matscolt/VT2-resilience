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

# ================================================================================
# constands, paths and global variables
# ================================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LAYOUT_DIR = DATA_DIR / "Layouts"

def create_production_plan(order_dir, settings, SECONDS_PER_WEEK):
    # read order csv file
    order_csv_path = order_dir / "unsorted_orders.csv"
    
    orders = []
    with open(order_csv_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            orders.append(row)

    # create production plan based on the orders and the settings
    # for now we will just sort the orders based on the due date and the variant
    # we can later add more complex sorting algorithms based on the capacity of the line and the processing times of the variants

    orders.sort(key=lambda x: int(x['due date']))

    #after sort now we need to calculate the capacity for each week and assign the orderline a week number
    # we can calculate the capacity based on the settings and the layout of the line

    #capacity for the week is 
    layout_name = settings["line_layout_file"]
    layout_settings = read_settings_json(LAYOUT_DIR / layout_name) # this is the time it takes to process one unit at the bottleneck station, we can later calculate this based on the layout and the processing times of the variants
    process_times_json = read_settings_json(DATA_DIR / "process_times.json")
    process_times = process_times_json["process_times"]

    bottleneck_station = layout_settings["bottleneck_station"]
    

    cycle_time_common = {}
    for variant, stations in process_times.items():
        if bottleneck_station not in stations:
            raise KeyError(f"Station '{bottleneck_station}' not found for variant '{variant}'")
        cycle_time_common[variant] = float(stations[bottleneck_station])
        return cycle_time_common
    print("hello")
    print(f"Cycle times at bottleneck station '{bottleneck_station}': {cycle_time_common}")

    def order_work_slow_seconds(order):
        return sum(order.get(v, 0) * cycle_time_common[v] for v in cycle_time_common)

    def assign_completion_week_pooled(orders, start_week=1, slow_cap_s=5000, fast_cap_s=5000, fast_speed=2.0):
        pooled_cap = slow_cap_s * 1.0 + fast_cap_s * fast_speed  # in "slow-equivalent seconds"
        week = start_week
        remaining = pooled_cap

        out = []
        for o in orders:
            work = order_work_slow_seconds(o)

            while work > remaining:
                week += 1
                remaining = pooled_cap

            remaining -= work

            o2 = dict(o)
            o2["work_slow_seconds"] = work
            o2["planned_week"] = week      # completion week
            out.append(o2)

        return out
    orders = assign_completion_week_pooled(orders, start_week=1, slow_cap_s=SECONDS_PER_WEEK, fast_cap_s=SECONDS_PER_WEEK, fast_speed=2.0)

    # write production plan to csv file
    production_plan_csv_path = order_dir / "production_plan.csv"
    with open(production_plan_csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['order_id', 'due date', 'priority', 'variant0', 'quantity0', 'variant1', 'quantity1', 'variant2', 'quantity2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for order in orders:
            writer.writerow(order)



def main(order_dir,SECONDS_PER_WEEK):
    print("Reading settings and order file...")
    # read settings json file
    order_dir = Path(order_dir)
    settings = read_settings_json(order_dir / "settings.json")
    # read order csv file
    order_csv_path = order_dir / "unsorted_orders.csv"
    if not order_csv_path.exists():
        print(f"Error: Order file {order_csv_path} does not exist.")
        return
    create_production_plan(order_dir, settings,SECONDS_PER_WEEK)
    print(f"Production plan created for orders in {order_dir.name}.")



    

if __name__ == "__main__":
   #THIS CANT RUN ON ITS OWN AS ORDER_DIR IS MISSING, IT NEEDS TO BE CALLED FROM THE MAIN SCRIPT
   custom_order_dir = r"C:\Users\mathi\OneDrive\Dokumenter\1.UNI\8. semester\Projekt\Github\VT2-resilience\production_line_sim\input\orders_07-05_10-51_1"
   main(custom_order_dir, 144000)