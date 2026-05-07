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
import json
import re
from prettytable import PrettyTable
from copy import deepcopy
from pathlib import Path
from A_input import read_settings_json

# ================================================================================
# constands, paths and global variables
# ================================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LAYOUT_DIR = DATA_DIR / "Layouts"



# -----------------------------
# Helpers for layout + times
# -----------------------------
def canonical_station_name(station_name: str) -> str:
    """
    Map 'Station 3.2: Robot cell' -> 'Station 3: Robot cell'
    Keep other stations unchanged.
    """
    return re.sub(r"(Station\s+\d+)\.\d+:", r"\1:", station_name)


def load_process_times(process_times_path: Path) -> dict:
    with open(process_times_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bottleneck_instances(layout_json: dict, bottleneck_base: str) -> list[dict]:
    """
    Extract all station instances whose canonical name equals bottleneck_base.
    Uses time_scale_factor if present, otherwise defaults to 1.0.
    Transport fields are ignored (per your wish).
    """
    instances = []
    for inst in layout_json.get("station_instances", []):
        name = inst.get("station_name", "")
        base = canonical_station_name(name)
        if base == bottleneck_base:
            instances.append(
                {
                    "instance_name": name,
                    "time_scale_factor": float(inst.get("time_scale_factor", 1.0)),
                    "week": 1,
                    "remaining_s": None,  # set later
                }
            )
    return instances


def iter_variant_qty(order: dict, max_pairs: int = 3):
    """
    Yields (variant, qty_int) from fields: variant0/quantity0, variant1/quantity1, variant2/quantity2
    """
    for i in range(max_pairs):
        v = (order.get(f"variant{i}") or "").strip()
        q_raw = order.get(f"quantity{i}")
        if not v or q_raw in (None, "", "0"):
            continue
        # handle "10" or "10.0"
        qty = int(float(q_raw))
        if qty > 0:
            yield v, qty


# -----------------------------
# Core planning (split across instances & weeks)
# -----------------------------
def plan_orders_across_robotcells(
    orders: list[dict],
    process_times_json: dict,
    layout_json: dict,
    seconds_per_week: float,
    bottleneck_base: str = "Station 3: Robot cell",
    variant_key_normalizer=str.upper,   # set to None if you don't want normalizing
    max_variant_pairs: int = 3,
):
    """
    Plans orders across all bottleneck station instances found in layout_json.
    - Uses bottleneck processing times from process_times_json at bottleneck_base.
    - Applies time_scale_factor per instance (multiplicative).
    - Ignores transport time completely (your wish).
    - Splits quantities freely across instances and weeks.
    Returns: (orders_out, schedule)
      schedule[week][instance_name][variant] = qty produced
    """
    process_times = process_times_json["process_times"]

    instances = build_bottleneck_instances(layout_json, bottleneck_base=bottleneck_base)
    if not instances:
        raise ValueError(f"No station instances found matching bottleneck '{bottleneck_base}'")

    # init remaining capacity per instance
    for inst in instances:
        inst["remaining_s"] = float(seconds_per_week)

    schedule = {}  # schedule[week][instance][variant] = qty

    def ensure_bucket(week: int, instance_name: str):
        schedule.setdefault(week, {})
        schedule[week].setdefault(instance_name, {})
        return schedule[week][instance_name]

    def eff_cycle_time(variant: str, inst: dict) -> float:
        # base cycle time for the bottleneck station comes from process_times.json
        base_t = float(process_times[variant][bottleneck_base])
        return base_t * inst["time_scale_factor"]

    orders_out = deepcopy(orders)

    for o in orders_out:
        # Remaining quantities to allocate for this order
        remaining = {}
        for v, q in iter_variant_qty(o, max_pairs=max_variant_pairs):
            v_norm = variant_key_normalizer(v) if variant_key_normalizer else v
            remaining[v_norm] = remaining.get(v_norm, 0) + q

        # If order has no content, just set week 1
        if not remaining:
            o["planned_week"] = 1
            continue

        completion_week = 1

        # Allocate until all variants completed
        while any(q > 0 for q in remaining.values()):
            progressed = False

            # allocate variant by variant. (You can change ordering if you want priority rules.)
            for variant, q_left in list(remaining.items()):
                if q_left <= 0:
                    continue
                if variant not in process_times:
                    raise KeyError(f"Variant '{variant}' is not defined in process_times.json")

                # rank instances by fastest effective cycle time for this variant
                ranked = sorted(instances, key=lambda inst: eff_cycle_time(variant, inst))

                for inst in ranked:
                    t = eff_cycle_time(variant, inst)
                    can_make = int(inst["remaining_s"] // t)
                    if can_make <= 0:
                        continue

                    make = min(q_left, can_make)
                    inst["remaining_s"] -= make * t
                    remaining[variant] -= make
                    q_left -= make
                    progressed = True

                    bucket = ensure_bucket(inst["week"], inst["instance_name"])
                    bucket[variant] = bucket.get(variant, 0) + make

                    completion_week = max(completion_week, inst["week"])

                    if q_left <= 0:
                        break

            if not progressed:
                # nothing fits anywhere -> advance the instance with the smallest remaining time
                # (simple way to open next-week capacity)
                stuck = min(instances, key=lambda inst: inst["remaining_s"])
                stuck["week"] += 1
                stuck["remaining_s"] = float(seconds_per_week)
                completion_week = max(completion_week, stuck["week"])

        o["planned_week"] = completion_week

    return orders_out, schedule


# -----------------------------
# Your main function, combined
# -----------------------------
def create_production_plan(order_dir, settings, SECONDS_PER_WEEK):
    order_dir = Path(order_dir)
    order_csv_path = order_dir / "unsorted_orders.csv"

    # read orders
    orders = []
    with open(order_csv_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            orders.append(row)

    # sort orders by due date (numeric)
    # if due date can be empty, add a fallback
    def due_key(x):
        v = (x.get("due date") or "").strip()
        return int(v) if v else 10**9

    orders.sort(key=due_key)

    # load layout + process times
    layout_name = settings["line_layout_file"]
    layout_json = read_settings_json(LAYOUT_DIR / layout_name)
    process_times_json = read_settings_json(DATA_DIR / "process_times.json")

    # Determine bottleneck station from your settings/layout
    # If you already have this in layout_settings, use it.
    bottleneck_base = layout_json["bottleneck_station"] if layout_json.get("bottleneck_station") is not None else "Station 3: Robot cell" 

    # plan across robotcell instances, ignoring transport time by design
    planned_orders, schedule = plan_orders_across_robotcells(
        orders=orders,
        process_times_json=process_times_json,
        layout_json=layout_json,
        seconds_per_week=SECONDS_PER_WEEK,
        bottleneck_base=bottleneck_base,
        variant_key_normalizer=str.upper,  # helps if CSV uses "fuse0" etc.
        max_variant_pairs=3,
    )

    # write production plan to csv
    production_plan_csv_path = order_dir / "production_plan.csv"
    fieldnames = [
        "order_id", "due date", "priority",
        "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2",
        "planned_week",
    ]

    with open(production_plan_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for order in planned_orders:
            writer.writerow(order)

import pandas as pd

def build_schedule_summary_df(
    schedule: dict,
    variants_order=None,
    weeks_order=None,
    station_name_fn=None,
    show_zero_variant_rows=True,
):
    """
    Build a summary DataFrame from:
      schedule[week][instance][variant] = qty

    Returns DataFrame:
      rows: Total + variants + station totals + station variant subtotals
      cols: week 1, week 2, ...

    Parameters
    ----------
    schedule : dict
        Nested dict schedule[week][station][variant] = qty.
    variants_order : list[str] | None
        Desired ordering of variants (e.g., ["FUSE0","FUSE1","FUSE2"]).
        If None, inferred from schedule.
    weeks_order : list[int] | None
        Which weeks to show and ordering (e.g., [1,2,3,4]).
        If None, inferred from schedule.
    station_name_fn : callable | None
        Optional formatting fn for station names (e.g., lambda s: s.split(":")[0]).
    show_zero_variant_rows : bool
        If True, show variant rows even when 0 under a station.
    """
    # Flatten schedule to records
    records = []
    for week, inst_map in schedule.items():
        for inst, var_map in inst_map.items():
            for variant, qty in var_map.items():
                records.append({
                    "week": int(week),
                    "station": inst,
                    "variant": str(variant),
                    "qty": int(qty),
                })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Optional station formatting
    if station_name_fn:
        df["station"] = df["station"].map(station_name_fn)

    # Infer ordering
    if variants_order is None:
        variants_order = sorted(df["variant"].unique().tolist())
    if weeks_order is None:
        weeks_order = sorted(df["week"].unique().tolist())

    col_labels = [f"week {w}" for w in weeks_order]

    # Totals per week
    total_by_week = (
        df.groupby("week")["qty"].sum()
          .reindex(weeks_order, fill_value=0)
    )

    # Totals per variant per week
    total_by_variant_week = (
        df.groupby(["variant", "week"])["qty"].sum()
          .reset_index()
          .pivot_table(index="variant", columns="week", values="qty", fill_value=0, aggfunc="sum")
    )
    total_by_variant_week = (
        total_by_variant_week
        .reindex(index=variants_order, fill_value=0)
        .reindex(columns=weeks_order, fill_value=0)
    )
    total_by_variant_week.columns = col_labels

    # Station totals per week
    station_total = (
        df.groupby(["station", "week"])["qty"].sum()
          .reset_index()
          .pivot_table(index="station", columns="week", values="qty", fill_value=0, aggfunc="sum")
    )
    station_total = station_total.reindex(columns=weeks_order, fill_value=0)
    station_total.columns = col_labels

    # Station + variant breakdown per week
    station_variant = (
        df.groupby(["station", "variant", "week"])["qty"].sum()
          .reset_index()
          .pivot_table(index=["station", "variant"], columns="week", values="qty", fill_value=0, aggfunc="sum")
    )
    station_variant = station_variant.reindex(columns=weeks_order, fill_value=0)
    station_variant.columns = col_labels

    # Build final table
    rows = []
    values = []

    # Total row
    rows.append("Total")
    values.append([int(x) for x in total_by_week.values])

    # Total by variant
    for v in variants_order:
        rows.append(f" - {v.lower()}")
        values.append([int(x) for x in total_by_variant_week.loc[v].values])

    # Station rows
    for station in sorted(df["station"].unique().tolist()):
        rows.append(station)
        values.append([int(x) for x in station_total.loc[station].values])

        for v in variants_order:
            idx = (station, v)
            if idx in station_variant.index:
                rows.append(f" - {v.lower()}")
                values.append([int(x) for x in station_variant.loc[idx].values])
            elif show_zero_variant_rows:
                rows.append(f" - {v.lower()}")
                values.append([0 for _ in weeks_order])

    out = pd.DataFrame(values, index=rows, columns=col_labels)
    return out

def print_df_prettytable(df, title=None, left_align_first_col=True):
    """
    Print a pandas DataFrame using PrettyTable.
    """
    if df is None or df.empty:
        print("Nothing to display (empty table).")
        return

    pt = PrettyTable()
    pt.field_names = [""] + list(df.columns)

    if title:
        print(title)

    # Alignment
    if left_align_first_col:
        pt.align[""] = "l"
    for c in df.columns:
        pt.align[c] = "r"

    # Add rows
    for idx, row in df.iterrows():
        pt.add_row([idx] + [int(v) if float(v).is_integer() else v for v in row.values])

    print(pt)

def print_schedule_summary_prettytable(
    schedule: dict,
    variants_order=None,
    weeks_order=None,
    station_name_fn=None,
    title="Weekly capacity allocation (units)",
):
    df = build_schedule_summary_df(
        schedule=schedule,
        variants_order=variants_order,
        weeks_order=weeks_order,
        station_name_fn=station_name_fn,
    )
    print_df_prettytable(df, title=title)


def main(order_dir,SECONDS_PER_WEEK):
    print("\n \n--- Starting production planning ---")
    # read settings json file
    order_dir = Path(order_dir)
    settings = read_settings_json(order_dir / "settings.json")
    # read order csv file
    order_csv_path = order_dir / "unsorted_orders.csv"
    if not order_csv_path.exists():
        print(f"Error: Order file {order_csv_path} does not exist.")
        return
    create_production_plan(order_dir, settings,SECONDS_PER_WEEK)
    print_df_prettytable()
    print(f"Production plan created for orders in {order_dir.name}.")



    

if __name__ == "__main__":
   #THIS CANT RUN ON ITS OWN AS ORDER_DIR IS MISSING, IT NEEDS TO BE CALLED FROM THE MAIN SCRIPT
   ordername = r"\orders_07-05_10-51_1"
   custom_order_dir = r"C:\Users\mathi\OneDrive\Dokumenter\1.UNI\8. semester\Projekt\Github\VT2-resilience\production_line_sim\input" + ordername
   main(custom_order_dir, 144000)