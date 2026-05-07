# Production planning for the line
# - Reads unsorted orders (CSV)
# - Sorts by due date
# - Plans weekly capacity on the bottleneck station (Robot cell)
# - Supports multiple station instances from a layout JSON
# - Applies per-instance time scaling via `time_scale_factor`
# - Ignores all transport times (per requirement)
# - Allows orders to split freely across station instances and weeks
# - Writes production_plan.csv
# - Prints:
#     1) Weekly summary table (Total + per variant + per station instance) using pandas + PrettyTable
#     2) Order IDs produced in each planning week

import csv
import json
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd
from prettytable import PrettyTable

from A_input import read_settings_json

# ==============================================================================
# constants, paths
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAYOUT_DIR = DATA_DIR / "Layouts"


# ==============================================================================
# helpers: layout + times
# ==============================================================================

def canonical_station_name(station_name: str) -> str:
    """Map 'Station 3.2: Robot cell' -> 'Station 3: Robot cell'."""
    return re.sub(r"(Station\s+\d+)\.\d+:", r"\1:", station_name)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bottleneck_instances(layout_json: dict, bottleneck_base: str) -> list[dict]:
    """Extract all station instances whose canonical name equals bottleneck_base.

    NOTE: Any transport fields in layout are ignored.
    """
    instances: list[dict] = []
    for inst in layout_json.get("station_instances", []):
        name = inst.get("station_name", "")
        base = canonical_station_name(name)
        if base == bottleneck_base:
            instances.append(
                {
                    "instance_name": name,
                    "time_scale_factor": float(inst.get("time_scale_factor", 1.0)),
                }
            )
    return instances


def all_station_instance_names(layout_json: dict) -> list[str]:
    """Return all station instance names from the layout (Station 1.., Station 3.1.., etc.)."""
    return [inst.get("station_name", "") for inst in layout_json.get("station_instances", []) if inst.get("station_name")]


def iter_variant_qty(order: dict, max_pairs: int = 3):
    """Yield (variant, qty_int) from fields variant0/quantity0, variant1/quantity1, ..."""
    for i in range(max_pairs):
        v = (order.get(f"variant{i}") or "").strip()
        q_raw = order.get(f"quantity{i}")
        if not v or q_raw in (None, "", "0"):
            continue
        qty = int(float(q_raw))  # accepts strings like "10" or "10.0"
        if qty > 0:
            yield v, qty


# ==============================================================================
# core planning: split across station instances & weeks
# ==============================================================================

def plan_orders_across_bottleneck_instances(
    orders: list[dict],
    process_times_json: dict,
    layout_json: dict,
    seconds_per_week: float,
    bottleneck_base: str = "Station 3: Robot cell",
    order_id_key: str = "order_id",
    variant_key_normalizer=str.upper,
    max_variant_pairs: int = 3,
    start_week: int = 1,
):
    """Plan orders across ALL station instances that belong to `bottleneck_base`.

    - base cycle time comes from process_times.json for `bottleneck_base`
    - effective cycle time per instance = base * time_scale_factor
    - transport times ignored
    - quantities can split across instances and weeks

    Returns:
      planned_orders, schedule, orders_by_week

    schedule format:
      schedule[week][instance_name][variant] = qty

    orders_by_week format:
      orders_by_week[week] = sorted list of order_ids that had ANY production in that week
    """
    process_times = process_times_json["process_times"]

    instances = build_bottleneck_instances(layout_json, bottleneck_base=bottleneck_base)
    if not instances:
        raise ValueError(f"No station instances found matching bottleneck '{bottleneck_base}'")

    current_week = int(start_week)
    remaining_s = {inst["instance_name"]: float(seconds_per_week) for inst in instances}

    schedule: dict[int, dict[str, dict[str, int]]] = {}
    orders_by_week: dict[int, set] = {}

    def ensure_sched_bucket(week: int, instance_name: str) -> dict:
        schedule.setdefault(week, {})
        schedule[week].setdefault(instance_name, {})
        return schedule[week][instance_name]

    def mark_order_in_week(week: int, oid: str):
        orders_by_week.setdefault(week, set()).add(oid)

    def eff_cycle_time(variant: str, inst: dict) -> float:
        base_t = float(process_times[variant][bottleneck_base])
        return base_t * float(inst["time_scale_factor"])

    planned_orders = deepcopy(orders)

    for o in planned_orders:
        oid = str(o.get(order_id_key, "")).strip() or "(missing_order_id)"

        rem_qty: dict[str, int] = {}
        for v, q in iter_variant_qty(o, max_pairs=max_variant_pairs):
            v_norm = variant_key_normalizer(v) if variant_key_normalizer else v
            rem_qty[v_norm] = rem_qty.get(v_norm, 0) + int(q)

        if not rem_qty:
            o["planned_week"] = current_week
            continue

        completion_week = current_week

        while any(q > 0 for q in rem_qty.values()):
            progressed = False

            for variant, q_left in list(rem_qty.items()):
                if q_left <= 0:
                    continue
                if variant not in process_times:
                    raise KeyError(f"Variant '{variant}' is not defined in process_times.json")
                if bottleneck_base not in process_times[variant]:
                    raise KeyError(f"Station '{bottleneck_base}' not defined for variant '{variant}'")

                ranked = sorted(instances, key=lambda inst: eff_cycle_time(variant, inst))

                for inst in ranked:
                    name = inst["instance_name"]
                    t = eff_cycle_time(variant, inst)

                    can_make = int(remaining_s[name] // t)
                    if can_make <= 0:
                        continue

                    make = min(q_left, can_make)

                    remaining_s[name] -= make * t
                    rem_qty[variant] -= make
                    q_left -= make
                    progressed = True

                    bucket = ensure_sched_bucket(current_week, name)
                    bucket[variant] = bucket.get(variant, 0) + int(make)
                    mark_order_in_week(current_week, oid)

                    completion_week = max(completion_week, current_week)

                    if q_left <= 0:
                        break

            if not progressed:
                current_week += 1
                remaining_s = {inst["instance_name"]: float(seconds_per_week) for inst in instances}
                completion_week = max(completion_week, current_week)

        o["planned_week"] = completion_week

    # convert sets to sorted lists for stable printing
    orders_by_week_sorted = {w: sorted(list(ids)) for w, ids in orders_by_week.items()}

    return planned_orders, schedule, orders_by_week_sorted


# ==============================================================================
# reporting: pandas dataframe + PrettyTable
# ==============================================================================

def build_schedule_summary_df(
    schedule: dict,
    variants_order=None,
    weeks_order=None,
    station_name_fn=None,
    all_stations=None,
    show_zero_variant_rows: bool = True,
) -> pd.DataFrame:
    """Build a summary DataFrame from schedule[week][instance][variant] = qty.

    If `all_stations` is provided, rows for stations with zero production will still be included.
    """
    records = []
    for week, inst_map in schedule.items():
        for inst, var_map in inst_map.items():
            for variant, qty in var_map.items():
                records.append({
                    "week": int(week),
                    "station": str(inst),
                    "variant": str(variant),
                    "qty": int(qty),
                })

    if not records:
        # If schedule is empty but user still wants stations/variants/weeks, return a zero table
        if all_stations and variants_order and weeks_order:
            col_labels = [f"week {w}" for w in weeks_order]
            rows = ["Total"] + [f" - {v.lower()}" for v in variants_order]
            for st in all_stations:
                st_name = station_name_fn(st) if station_name_fn else st
                rows.append(st_name)
                rows.extend([f" - {v.lower()}" for v in variants_order])
            return pd.DataFrame(0, index=rows, columns=col_labels)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if station_name_fn:
        df["station"] = df["station"].map(station_name_fn)

    if variants_order is None:
        variants_order = sorted(df["variant"].unique().tolist())
    if weeks_order is None:
        weeks_order = sorted(df["week"].unique().tolist())

    col_labels = [f"week {w}" for w in weeks_order]

    # Determine station list
    inferred_stations = sorted(df["station"].unique().tolist())
    if all_stations:
        # include all provided stations (after optional formatting)
        formatted = [station_name_fn(s) if station_name_fn else s for s in all_stations]
        # preserve layout order but remove dups
        seen = set()
        stations_list = []
        for s in formatted:
            if s not in seen:
                stations_list.append(s)
                seen.add(s)
        # also include any stations that appear in df but weren't in all_stations
        for s in inferred_stations:
            if s not in seen:
                stations_list.append(s)
                seen.add(s)
    else:
        stations_list = inferred_stations

    # compute totals
    total_by_week = df.groupby("week")["qty"].sum().reindex(weeks_order, fill_value=0)

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

    station_total = (
        df.groupby(["station", "week"])["qty"].sum()
          .reset_index()
          .pivot_table(index="station", columns="week", values="qty", fill_value=0, aggfunc="sum")
    )
    station_total = station_total.reindex(columns=weeks_order, fill_value=0)
    station_total.columns = col_labels

    station_variant = (
        df.groupby(["station", "variant", "week"])["qty"].sum()
          .reset_index()
          .pivot_table(index=["station", "variant"], columns="week", values="qty", fill_value=0, aggfunc="sum")
    )
    station_variant = station_variant.reindex(columns=weeks_order, fill_value=0)
    station_variant.columns = col_labels

    rows = []
    values = []

    rows.append("Total")
    values.append([int(x) for x in total_by_week.values])

    for v in variants_order:
        rows.append(f" - {v.lower()}")
        values.append([int(x) for x in total_by_variant_week.loc[v].values])
    """
    for station in stations_list:
        rows.append(station)
        if station in station_total.index:
            values.append([int(x) for x in station_total.loc[station].values])
        else:
            values.append([0 for _ in weeks_order])

        for v in variants_order:
            idx = (station, v)
            if idx in station_variant.index:
                rows.append(f" - {v.lower()}")
                values.append([int(x) for x in station_variant.loc[idx].values])
            elif show_zero_variant_rows:
                rows.append(f" - {v.lower()}")
                values.append([0 for _ in weeks_order])
    """
    return pd.DataFrame(values, index=rows, columns=col_labels)


def print_df_prettytable(df: pd.DataFrame, title: str | None = None):
    """Print a pandas DataFrame using PrettyTable."""
    if df is None or df.empty:
        print("Nothing to display (empty table).")
        return

    pt = PrettyTable()
    pt.field_names = [""] + list(df.columns)

    pt.align[""] = "l"
    for c in df.columns:
        pt.align[c] = "r"

    for idx, row in df.iterrows():
        safe_vals = []
        for v in row.values:
            try:
                fv = float(v)
                safe_vals.append(int(fv) if fv.is_integer() else fv)
            except Exception:
                safe_vals.append(v)
        pt.add_row([idx] + safe_vals)

    if title:
        print(title)
    print(pt)


def print_schedule_summary_prettytable(
    schedule: dict,
    variants_order=None,
    weeks_order=None,
    station_name_fn=None,
    all_stations=None,
    title: str = "Weekly capacity allocation (units)",
):
    df = build_schedule_summary_df(
        schedule=schedule,
        variants_order=variants_order,
        weeks_order=weeks_order,
        station_name_fn=station_name_fn,
        all_stations=all_stations,
    )
    print_df_prettytable(df, title=title)


def print_orders_by_week(orders_by_week: dict, weeks_order=None):
    """Print order_ids produced in each week."""
    if not orders_by_week:
        print("\nNo orders were produced (orders_by_week is empty).")
        return

    if weeks_order is None:
        weeks_order = sorted(orders_by_week.keys())

    print("\norder_id produced in each week")
    for w in weeks_order:
        ids = orders_by_week.get(w, [])
        if ids:
            print(f"week{w}: {', '.join(ids)}")
        else:
            print(f"week{w}: (none)")


# ==============================================================================
# main production planning pipeline
# ==============================================================================

def create_production_plan(order_dir, settings, SECONDS_PER_WEEK):
    """Create a production plan CSV with planned_week and print weekly summaries."""

    order_dir = Path(order_dir)
    order_csv_path = order_dir / "unsorted_orders.csv"

    # Read orders
    orders: list[dict] = []
    with open(order_csv_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            orders.append(row)

    # Sort by due date numerically
    def due_key(x):
        v = (x.get("due date") or "").strip()
        return int(v) if v else 10**9

    orders.sort(key=due_key)

    # Load layout + process times
    layout_name = settings.get("line_layout_file")
    if not layout_name:
        raise KeyError("settings.json must include 'line_layout_file'")

    layout_json = read_settings_json(LAYOUT_DIR / layout_name)
    process_times_json = read_settings_json(DATA_DIR / "process_times.json")

    bottleneck_base = layout_json.get("bottleneck_station", "Station 3: Robot cell")

    planned_orders, schedule, orders_by_week = plan_orders_across_bottleneck_instances(
        orders=orders,
        process_times_json=process_times_json,
        layout_json=layout_json,
        seconds_per_week=SECONDS_PER_WEEK,
        bottleneck_base=bottleneck_base,
        order_id_key="order_id",
        variant_key_normalizer=str.upper,
        max_variant_pairs=3,
        start_week=1,
    )

    # Write production plan
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

    # Print schedule summary (include all instances from layout even if 0)
    all_instances = all_station_instance_names(layout_json)

    print_schedule_summary_prettytable(
        schedule,
        variants_order=["FUSE0", "FUSE1", "FUSE2"],
        weeks_order=[1, 2, 3, 4],
        station_name_fn=lambda s: s.split(":")[0],
        all_stations=all_instances,
        title="\nWeekly plan (units) - totals and per station instance",
    )

    # Print order IDs produced in each week
    print_orders_by_week(orders_by_week, weeks_order=[1, 2, 3, 4])


def main(order_dir: str, SECONDS_PER_WEEK: float):
    print("\n\n--- Starting production planning ---")
    order_dir = Path(order_dir)

    settings_path = order_dir / "settings.json"
    if not settings_path.exists():
        print(f"Error: settings.json not found in {order_dir}")
        return

    settings = read_settings_json(settings_path)

    order_csv_path = order_dir / "unsorted_orders.csv"
    if not order_csv_path.exists():
        print(f"Error: Order file does not exist: {order_csv_path}")
        return

    create_production_plan(order_dir, settings, SECONDS_PER_WEEK)
    print(f"\nProduction plan created\n")


if __name__ == "__main__":
    # Example manual run (adjust path + seconds/week)
    ordername = r"\\orders_07-05_10-51_1"
    custom_order_dir = r"C:\\Users\\mathi\\OneDrive\\Dokumenter\\1.UNI\\8. semester\\Projekt\\Github\\VT2-resilience\\production_line_sim\\input" + ordername
    main(custom_order_dir, 144000)
