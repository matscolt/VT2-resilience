from pathlib import Path
import csv
import random
import re
import json
from datetime import datetime


# 🔷 Generate orders
def generate_order():
    return [{
        "order_id": 1,
        "Order_time": 0,
        "priority": 1,
        "variant0": "FUSE0",
        "variant1": "FUSE1",   
        "variant2": "FUSE2",
        "quantity": 0
        }]

#Create settings JSON file
def create_setting_json(output_path: Path):
    Setting = {
        "Sim_time [s]" :    3600
        }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(Setting, f, indent=4) 

#Create disruption JSON file
def create_disruption_json(output_path: Path):
    setting = {
        "SEED": datetime.now().strftime("%Y%m%d%H%M%S"),
        #Stations
        "Stations": {
            "1": {
                "Machine breakdown chance [%]": 0,
                "Machine breakdown duration [s]": 0
            },
            "2": {
                "Machine breakdown chance [%]": 0,
                "Machine breakdown duration [s]": 0
            },
            "3": {
                "Machine breakdown chance [%]": 0,
                "Machine breakdown duration [s]": 0
            },
            "4": {
                "Machine breakdown chance [%]": 0,
                "Machine breakdown duration [s]": 0
            },
            "5": {
                "Machine breakdown chance [%]": 0,
                "Machine breakdown duration [s]": 0
            },
            "6": {
                "Machine breakdown chance [%]": 0,
                "Machine breakdown duration [s]": 0
            }
        },
        "Material": {
            "Broken material chance [%]": 0
        }
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4)

# 🔷 Write CSV
def write_csv(rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order_id", "Order_time", "priority", "variant0", "quantity", "variant1", "quantity", "variant2", "quantity"]
        )
        writer.writeheader()
        writer.writerows(rows)


# 🔷 MAIN
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

    ordername_csv = f"_orders_{timestamp}_{n}.csv"
        
    order_dir = input_dir / orderfoldername
    order_dir.mkdir(parents=True, exist_ok=True)

    output_path_csv = order_dir / ordername_csv
    output_path_settingsjson = order_dir / "settings.json"
    output_path_disruptionjson = order_dir / "disruption.json"

    rows = generate_order()
    write_csv(rows, output_path_csv)
    create_setting_json(output_path_settingsjson)
    create_disruption_json(output_path_disruptionjson)
    print(f"--- Created order file: {orderfoldername} ----")


if __name__ == "__main__":
    main()


