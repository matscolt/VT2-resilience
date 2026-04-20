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
        "Sim_time [s]" :    3600,
        "seed": datetime.now().strftime("%Y%m%d%H%M%S"),
        "line_layout_file": "line_layout_single_path",
        "carriers": {
            "number of carriers": 8,
        }
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(Setting, f, indent=4) 

#Create disruption JSON file
def create_disruption_json(output_path: Path):
    setting = {   
        "Disruptions": {
            "Enabled" : 1
        },
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
            "ran out of material chance [%]": 0.01
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


