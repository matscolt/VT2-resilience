import json
import csv
import random
from datetime import datetime
from pathlib import Path

# 🔷 Write CSV
def write_order_csv(rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order_id", "due date", "priority", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2"]
        )
        writer.writeheader()
        writer.writerows(rows)

def write_disruption_csv(rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["disruption_type", "station_id", "start_time", "end_time", "efficiency_percentage", "order_id", "Order_time", "priority", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2"]
        )
        writer.writeheader()
        writer.writerows(rows)

#output csv file with disruptions

def round_half_up(n):
    n = n+0.5
    return int(n)

def generate_orderlist(num_orders, num_units, sim_time, output_path: Path):
    rows = []
    sum = []
    priosum = []
    ordermean = num_units/num_orders
    orderstd = ordermean * 0.1
    unitstd =  0.1

    earliest_due_date = ordermean * 76.4
    for order_id in range(1, num_orders):
        units = round_half_up(max(random.normalvariate(ordermean, orderstd), 1))
        due_date = round_half_up(random.uniform(earliest_due_date, sim_time))
        priority = round_half_up(min(max(random.expovariate(1/1.5), 1), 5))
        variant0 = "FUSE0"
        quantity0 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
        variant1 = "FUSE1"
        quantity1 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
        variant2 = "FUSE2"
        quantity2 = round_half_up(units - quantity0 - quantity1)
        while quantity2 < 0:
            quantity0 = quantity0 - 1
            quantity1 = quantity1 - 1
            quantity2 = units - quantity0 - quantity1
            
        row = {
            "order_id": order_id,
            "due date": due_date,
            "priority": priority,
            "variant0": variant0,
            "quantity0": quantity0,
            "variant1": variant1,
            "quantity1": quantity1,
            "variant2": variant2,
            "quantity2": quantity2
        }
        rows.append(row)
        sum.append(quantity0 + quantity1 + quantity2)
        priosum.append(priority)
    
    total_sum = 0
    for s in sum:
        total_sum += s
    priority_sum = 0
    for p in priosum:
        priority_sum += p
    print(f"Total units in orders: {total_sum}")
    units = num_units-total_sum
    due_date = round_half_up(random.uniform(earliest_due_date, sim_time))
    priority = round_half_up(max(random.normalvariate(2.5, 1), 1))
    variant0 = "FUSE0"
    quantity0 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
    variant1 = "FUSE1"
    quantity1 = round_half_up(units * max(random.normalvariate(0.33, unitstd), 0))
    variant2 = "FUSE2"
    quantity2 = round_half_up(units - quantity0 - quantity1)
    while quantity2 < 0:
        quantity0 = quantity0 - 1
        quantity1 = quantity1 - 1
        quantity2 = units - quantity0 - quantity1

    row = {
        "order_id": order_id+1,
        "due date": due_date,
        "priority": priority,
        "variant0": variant0,
        "quantity0": quantity0,
        "variant1": variant1,
        "quantity1": quantity1,
        "variant2": variant2,
        "quantity2": quantity2
    }
    rows.append(row)
    total_sum += quantity0 + quantity1 + quantity2
    priority_sum += priority

    print(f"Total priority in orders: {priority_sum} with a mean of {priority_sum/num_orders}")
    print(f"Generated {num_orders} orders with a total of {total_sum} units.\n Average units per order: {total_sum/num_orders}")
    print(f"average phone per hour: {total_sum/sim_time*3600}")
    write_order_csv(rows, output_path)

def generate_disruption_list(sim_time: int, output_path: Path):
    #read settings json file
    settings = read_settings_json(output_path.parent / "settings.json")
    #read disruption json file
    disruption_settings = read_disruption_json(output_path.parent / "disruption.json")
    #generate disruptions csv based on settings and disruption json files
    #   it should calculate the chance of each disruption on each station
    #   Use the data to generate a list of disruptions
    rows = []
    def generate_single_station_disruption(station_id, sim_time):
        breakdown_chance = disruption_settings["Stations"][str(station_id)]["breakdown"]["Machine breakdown chance [%]"]
        breakdown_mean = disruption_settings["Stations"][str(station_id)]["breakdown"]["duration [s]"]
        
        efficiency_loss_chance = disruption_settings["Stations"][str(station_id)]["efficiency loss"]["efficiency drop chance [%]"]
        efficiency_loss_percentage = disruption_settings["Stations"][str(station_id)]["efficiency loss"]["efficiency drop [%]"]
        efficiency_loss_duration = disruption_settings["Stations"][str(station_id)]["efficiency loss"]["duration [s]"]
        
        failed_inspection_chance = disruption_settings["Stations"][str(station_id)].get("failed inspection", {}).get("wrong assembly chance", 0)

        breakdown_amount = breakdown_chance * sim_time / breakdown_mean
        efficiency_loss_amount = efficiency_loss_chance * sim_time / efficiency_loss_duration
        failed_inspection_amount = round_half_up(failed_inspection_chance * sim_time / 76.4) #assuming that the inspection station is the bottleneck and that the mean order time is 76.4 seconds
        print(f"Station {station_id}: Breakdowns: {breakdown_amount}, Efficiency Loss: {efficiency_loss_amount}, Failed Inspections: {failed_inspection_amount}")
        for disruption in range(round_half_up(breakdown_amount)):
            
            row = {
            "disruption_type": disruption_type,
            "station_id": station_id,
            "start_time": start_time,
            "end_time": end_time,
            "efficiency_percentage": efficiency_percentage,
            "order_id": None,
            "order_time": None,
            "priority": None,
            "variant0": None,
            "quantity0": None,
            "variant1": None,
            "quantity1": None,
            "variant2": None,
            "quantity2": None}

        


    

    #stations kunne tælles igennem layout filen for at finde ud af hvor mange der er isteden for at hardcode det
    #plus man ville kunne tælle hvor mange der er af hver type
    stations = 6 if settings["line_layout_file"] == "line_layout_single_path.json" else 9 if settings["line_layout_file"] == "line_layout_multi_path.json" else 0
    for station_id in range(1,stations+1):
        generate_single_station_disruption(station_id,sim_time)

    #generate emergenct orders
    # This would involve generating orders that are generated based on the total time of the simulation
    for i in range(round_half_up(sim_time/3600*10)): #assuming that there are 10 emergenct orders per hour
        order_time = round_half_up(random.uniform(0, sim_time))
        priority = 5
        variant0 = "FUSE0"
        quantity0 = round_half_up(random.uniform(1, 5))
        variant1 = "FUSE1"
        quantity1 = round_half_up(random.uniform(1, 5))
        variant2 = "FUSE2"
        quantity2 = round_half_up(random.uniform(1, 5))
        row = {
            "disruption_type": "emergency_order",
            "station_id": None,
            "start_time": order_time,
            "end_time": None,
            "efficiency_percentage": None,
            "order_id": i+1,
            "order_time": order_time,
            "priority": priority,
            "variant0": variant0,
            "quantity0": quantity0,
            "variant1": variant1,
            "quantity1": quantity1,
            "variant2": variant2,
            "quantity2": quantity2}
        rows.append(row)


    write_disruption_csv(rows, output_path)




# output json file with settings and disruptions

def create_setting_json(output_path: Path):
    Setting = {
        "sim_time [s]" :    36000,
        "seed": datetime.now().strftime("%Y%m%d%H%M%S"),
        "random based disruptions": {
            "enabled" : 2
        },
        "line_layout_file": "line_layout_single_path.json",
        "carriers": {
            "number of carriers": 8
        }
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(Setting, f, indent=4) 

#Create disruption JSON file
def create_disruption_json(output_path: Path):
    setting = {   
        "Stations": {
            "1": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10},
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std" : 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                }
            },
            "2": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10},
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std" : 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                }
            },
            "3": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std" : 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                }
            },
            "4": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std" : 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                }
            },
            "5": {
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std" : 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                }
            },
            "6": {
                "failed inspection":{
                "wrong assembly chance": 0.005
                },
                "breakdown": {
                    "Machine breakdown chance [%]": 0.05,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                },
                "efficiency loss": {
                    "efficiency drop chance [%]": 0.05,
                    "efficiency drop [%]": 20,
                    "efficiency drop range": [30, 90],
                    "efficiency drop std" : 10,
                    "duration [s]": 60,
                    "range": [30, 90],
                    "std" : 10
                }
            }
        },
        "Material": {
            "Broken material chance [%]": 0.15,
            "ran out of material chance [%]": 0.00
        }
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4)

def read_settings_json(input_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        settings = json.load(f)
    return settings

def read_disruption_json(input_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        disruption_settings = json.load(f)
    return disruption_settings

def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    input_dir.mkdir(exist_ok=True)

    n=1
    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    orderfoldername = f"orders_{timestamp}_{n}"
    
    while (input_dir / orderfoldername).exists():
        n=n+1
        orderfoldername = f"orders_{timestamp}_{n}"
    
    ordername_csv = f"order_list_{timestamp}_{n}.csv"
        
    order_dir = input_dir / orderfoldername
    order_dir.mkdir(parents=True, exist_ok=True)

    # Generate paths
    output_path_ordercsv = order_dir / ordername_csv
    output_path_disruptioncsv = order_dir / f"disruption_list_{timestamp}_{n}.csv"
    output_path_settingsjson = order_dir / "settings.json"
    output_path_disruptionjson = order_dir / "disruption.json"

    # Generate settings and disruption json files
    create_setting_json(output_path_settingsjson)
    create_disruption_json(output_path_disruptionjson)

    num_orders = int(input("Enter amount of orders: "))
    num_units = int(input("Enter amount of units: "))
   
    # read settings json file

    settings = read_settings_json(output_path_settingsjson)
    sim_time = settings["sim_time [s]"]
    seed = settings["seed"]
    random.seed(seed)
    disruption_settings = read_disruption_json(output_path_disruptionjson)

    # Generate orderlist

    # should be in format: order_id, Order_time, priority, variant0, quantity, variant1, quantity, variant2, quantity
    generate_orderlist(num_orders, num_units, sim_time, output_path_ordercsv)
    if settings["random based disruptions"]["enabled"] == 2:
        print("Generating event based disruptions")
        generate_disruption_list(sim_time, output_path_disruptioncsv)
    print(f"--- Created order file: {orderfoldername} ----")


if __name__ == "__main__":
    main()


