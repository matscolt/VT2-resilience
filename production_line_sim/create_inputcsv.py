from pathlib import Path
import csv
import random
import re
from datetime import datetime

# 🔷 Generate orders
def generate_order():
    return [{
        "order_id": 1,
        "Order_time": 0,
        "priority": 1,
        "variant0": "FUSE0",   
        "quantity": 0,
        "variant1": "FUSE1",   
        "quantity": 0,
        "variant2": "FUSE2",
        "quantity": 0
        }]

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

    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    ordername = f"orders_{timestamp}_1.csv"
    while (input_dir / ordername).exists():
        ordername = f"orders_{timestamp}_{int(ordername.split('_')[-1].removesuffix('.csv')) + 1}.csv"
    output_path = input_dir / ordername

    rows = generate_order()
    write_csv(rows, output_path)
    print(f"--- Created order file: {ordername} ----")


if __name__ == "__main__":
    main()


