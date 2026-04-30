import json
import csv
from datetime import datetime
from pathlib import Path

# 🔷 Write CSV
def write_csv(rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order_id", "due date", "priority", "variant0", "quantity", "variant1", "quantity", "variant2", "quantity"]
        )
        writer.writeheader()
        writer.writerows(rows)

#output csv file with disruptions

def generate_orderlist(num_orders, num_units, output_path: Path):
    rows = []
    for order_id in range(1, num_orders + 1):
        due_date = 0
        priority = 1
        variant0 = f"FUSE{order_id % 3}"
        quantity0 = num_units
        variant1 = f"FUSE{(order_id + 1) % 3}"
        quantity1 = num_units
        variant2 = f"FUSE{(order_id + 2) % 3}"
        quantity2 = num_units

        row = {
            "order_id": order_id,
            "due date": due_date,
            "priority": priority,
            "variant0": variant0,
            "quantity": quantity0,
            "variant1": variant1,
            "quantity": quantity1,
            "variant2": variant2,
            "quantity": quantity2
        }
        rows.append(row)

    write_csv(rows, output_path)

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
    output_path_csv = order_dir / ordername_csv
    output_path_settingsjson = order_dir / "settings.json"
    output_path_disruptionjson = order_dir / "disruption.json"

    # Generate settings and disruption json files
    create_setting_json(output_path_settingsjson)
    create_disruption_json(output_path_disruptionjson)

    num_orders = int(input("Enter amount of orders: "))
    num_units = int(input("Enter amount of units: "))
    # Generate orderlist

    # should be in format: order_id, Order_time, priority, variant0, quantity, variant1, quantity, variant2, quantity
    generate_orderlist(num_orders, num_units, output_path_csv)
    print(f"--- Created order file: {orderfoldername} ----")


if __name__ == "__main__":
    main()


