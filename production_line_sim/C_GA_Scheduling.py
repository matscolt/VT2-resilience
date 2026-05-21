import random
import copy
import contextlib
import io
import math
import shutil
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
import pandas as pd
import D_production_line_sim as simulator,A_input
# OPTION A: if makespan > horizon_end_s for 1-day horizon, rerun GA with 5 days (no further fallback)
### PATCH: NO BEST-SUMMARY FOLDER / ON_GOING ONLY
# This file is auto-patched to ensure the GA only overwrites current_schedule.csv in ON_GOING_RUN_DIR.
# All best-schedule summary folder creation/moves/cleanup are disabled.



# ============================================================
# PATHS / MAIN SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

# These globals are configured at runtime from main_settings.json.
MAIN_SETTINGS_PATH = None
MAIN_SETTINGS = None
ON_GOING_RUN_DIR = None
OUTPUT_RUN_DIR = None
PRODUCTION_PLAN_PATH = None
DISRUPTION_HISTORY_PATH = None
UNIT_SUMMARY_PATH = None
CLEAN_TEMP_OUTPUTS = True
# DISABLED: KEEP_ONLY_BEST_SUMMARY = True


def load_main_settings(main_settings_path) -> dict:
    """Load main_settings.json supplied by Main_script."""

    path = Path(main_settings_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"main_settings.json not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        settings = json.load(file)

    return settings


def configure_paths_from_main_settings(main_settings_path):
    """Configure all GA paths from main_settings.json.

    Expected main_settings fields:
    - label
    - pathlist.on_going_run
    - pathlist.output_run (optional, but recommended)

    The production plan is expected at:
    on_going_run / f"production_plan_{label}.csv"

    The simulator state files are expected in the same on_going_run folder:
    - disruption_hist.csv
    - unit_summary.csv
    If they do not exist, the run is treated as t = 0 / no completed units.
    """

    global MAIN_SETTINGS_PATH, MAIN_SETTINGS, ON_GOING_RUN_DIR, OUTPUT_RUN_DIR
    global PRODUCTION_PLAN_PATH, DISRUPTION_HISTORY_PATH, UNIT_SUMMARY_PATH

    MAIN_SETTINGS_PATH = Path(main_settings_path).expanduser().resolve()
    MAIN_SETTINGS = load_main_settings(MAIN_SETTINGS_PATH)

    label = MAIN_SETTINGS.get("label")
    pathlist = MAIN_SETTINGS.get("pathlist", {})

    if not label:
        raise KeyError("Missing 'label' in main_settings.json")

    if "on_going_run" not in pathlist:
        raise KeyError("Missing 'pathlist.on_going_run' in main_settings.json")

    ON_GOING_RUN_DIR = Path(pathlist["on_going_run"])

    # Event-driven state files
    global DISRUPTION_HISTORY_PATH, UNIT_SUMMARY_PATH
    DISRUPTION_HISTORY_PATH = Path(pathlist.get("on_going_run_dis_his", ON_GOING_RUN_DIR / "disruption_his.csv"))
    UNIT_SUMMARY_PATH = Path(pathlist.get("on_going_run_unit_summary", ON_GOING_RUN_DIR / "unit_summary.csv")).expanduser().resolve()
    OUTPUT_RUN_DIR = Path(pathlist.get("output_run", ON_GOING_RUN_DIR)).expanduser().resolve()

    if not ON_GOING_RUN_DIR.exists():
        raise FileNotFoundError(
            f"on_going_run folder not found: {ON_GOING_RUN_DIR}"
        )

    PRODUCTION_PLAN_PATH = Path(pathlist.get("on_going_run_production_plan", ON_GOING_RUN_DIR / f"production_plan_{label}.csv")).expanduser().resolve()
    DISRUPTION_HISTORY_PATH = Path(pathlist.get("on_going_run_dis_his", ON_GOING_RUN_DIR / "disruption_his.csv")).expanduser().resolve()
    UNIT_SUMMARY_PATH = Path(pathlist.get("on_going_run_unit_summary", ON_GOING_RUN_DIR / "unit_summary.csv")).expanduser().resolve()

    if not PRODUCTION_PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Production plan not found: {PRODUCTION_PLAN_PATH}"
        )

    return MAIN_SETTINGS

def get_completed_unit_ids_from_unit_summary(current_time_s: float) -> Set[str]:
    """Return completed unit IDs from unit_summary.csv.

    A unit is considered completed if completion_time_s <= current_time_s.
    If unit_summary.csv does not exist, no units are completed.
    """

    if UNIT_SUMMARY_PATH is None or not UNIT_SUMMARY_PATH.exists():
        return set()

    unit_df = pd.read_csv(UNIT_SUMMARY_PATH)

    if "completion_time_s" not in unit_df.columns:
        raise KeyError(
            f"Missing column 'completion_time_s' in {UNIT_SUMMARY_PATH}"
        )

    unit_id_columns = ["unit_id", "unitID", "unitId", "UnitID", "unit"]
    unit_id_column = next((c for c in unit_id_columns if c in unit_df.columns), None)

    completed_mask = (
        pd.to_numeric(unit_df["completion_time_s"], errors="coerce")
        <= current_time_s
    )

    if unit_id_column is None:
        raise KeyError(
            f"Missing unit ID column in {UNIT_SUMMARY_PATH}. "
            "Expected one of: unit_id, unitID, unitId, UnitID, unit"
        )

    return set(
        unit_df.loc[completed_mask, unit_id_column]
        .dropna()
        .astype(str)
    )

# ============================================================
# GA SETTINGS / ROLLING HORIZON SETTINGS
# ============================================================

# One production day is currently defined as 8 hours.
SECONDS_PER_PRODUCTION_DAY = 8 * 60 * 60

# Set these values here while testing.
DEFAULT_LOOKAHEAD_DAYS = 1

# Rolling horizon options:
# 1 day  -> schedule the rest of current day only
# 3 days -> schedule rest of current day + 2 full days
# 5 days -> schedule rest of current day + 4 full days
BASE_SETTINGS = A_input.read_settings_json(ROOT / "data" / "base_settings.json")
allowed_map = BASE_SETTINGS["ALLOWED_LOOKAHEAD_DAYS"]
ALLOWED_LOOKAHEAD_DAYS = sorted(int(v) for v in allowed_map.values())
GENERATION_LIMIT = BASE_SETTINGS["GENERATION_LIMIT"]
CROSSOVER_RATE = BASE_SETTINGS["CROSSOVER_RATE"]
MUTATION_RATE = BASE_SETTINGS["MUTATION_RATE"]
SWAPS_SCALE = BASE_SETTINGS["swaps_scale"]
SWAPS_CONSTANT = BASE_SETTINGS["swaps_constant"]
SWAPS_MIN = BASE_SETTINGS["swaps_min"]
POPULATION_SCALE = BASE_SETTINGS["population_scale"]
POPULATION_CONSTANT = BASE_SETTINGS["population_constant"]
POPULATION_MIN = BASE_SETTINGS["population_min"]
ELITE_SCALE = BASE_SETTINGS["elite_scale"]
ELITE_CONSTANT = BASE_SETTINGS["elite_constant"]
ELITE_MIN = BASE_SETTINGS["elite_min"]
TOURNAMENT_SCALE = BASE_SETTINGS["tournament_scale"]
TOURNAMENT_CONSTANT = BASE_SETTINGS["tournament_constant"]
TOURNAMENT_MIN = BASE_SETTINGS["tournament_min"]

# ============================================================
# FITNESS WEIGHTS
# ============================================================

# Fitness model:
#   1) Weighted exponential penalty for each delayed order
#   2) Weighted extra exponential penalty for the worst delayed order
#   3) Very small linear earliness reward as a tie-breaker
# Notes:
# Lower fitness is better.
# Times from the simulator are in seconds, so tardiness/earliness are
# converted to days before being used in the fitness function.

ALPHA = BASE_SETTINGS["ALPHA"]
BETA = BASE_SETTINGS["BETA"]
GAMMA = BASE_SETTINGS["GAMMA"] # exponent for priority weighting (w_i = priority^gamma)
DELTA = BASE_SETTINGS["DELTA"] 


TIME_SCALE = 60 * 60  # 1 hours in seconds

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Unit:
    unit_id: str
    order_id: int
    variant: str
    due_date: float
    priority: int


@dataclass
class Order:
    order_id: int
    due_date: float
    priority: int
    total_units: int
    planned_week: int = 1
    planned_day: int = 1


# ============================================================
# LOAD FILES
# ============================================================

def load_production_plan(production_plan_path: Path) -> pd.DataFrame:

    path = Path(production_plan_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Production plan not found: {path}"
        )

    print(f"Loading production plan from: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported production plan file type: {path.suffix}"
    )


# ============================================================
# LOAD ORDERS AND UNITS
# ============================================================

def load_orders_and_units_from_file(df: pd.DataFrame):

    orders = []
    units = []
    order_units = {}

    global_unit_counter = 1

    print("\n================================================")
    print("LOADING ORDERS AND UNITS FROM FILE")
    print("================================================")

    for _, row in df.sort_values("order_id").iterrows():

        order_id = int(row["order_id"])
        due_date = float(row["due date"])
        priority = int(row["priority"])

        order_units[order_id] = []

        total_units = 0

        """print(
            f"\nORDER {order_id} | "
            f"Due={due_date} | "
            f"Priority={priority} | "
            f"Planned day={int(row['planned_day'])}"
        )"""

        for i in range(3):

            variant = str(row[f"variant{i}"])
            qty = int(row[f"quantity{i}"])

            if qty <= 0:
                continue

            for _ in range(qty):

                internal_unit_id = f"U{global_unit_counter:03d}"

                unit = Unit(
                    unit_id=internal_unit_id,
                    order_id=order_id,
                    variant=variant,
                    due_date=due_date,
                    priority=priority
                )

                units.append(unit)
                order_units[order_id].append(internal_unit_id)

                global_unit_counter += 1
                total_units += 1

        order = Order(
            order_id=order_id,
            due_date=due_date,
            priority=priority,
            total_units=total_units,
            planned_week=int(row["planned_week"]),
            planned_day=int(row["planned_day"])
        )

        orders.append(order)

        #print(f"  Total units for order {order_id}: {total_units}")

    print("\n================================================")
    print(f"TOTAL ORDERS LOADED: {len(orders)}")
    print(f"TOTAL UNITS LOADED: {len(units)}")
    print("================================================\n")

    return orders, units, order_units

# ============================================================
# ROLLING HORIZON / SIMULATION STATE HELPERS
# ============================================================

def get_current_planned_day(current_time_s: float) -> int:
    """Return the production-plan day number containing current_time_s.

    Day numbering follows the production_plan column planned_day:
    current_time_s in [0, 28800) is planned_day 1,
    current_time_s in [28800, 57600) is planned_day 2, etc.
    """

    return int(current_time_s // SECONDS_PER_PRODUCTION_DAY) + 1


def get_planned_day_window(
    current_time_s: float,
    lookahead_days: int
) -> Tuple[int, int]:
    """
    Calculate the planned_day interval for the rolling horizon.

    lookahead_days=1 keeps only the current planned_day.
    lookahead_days=3 keeps current planned_day + 2 following planned days.
    lookahead_days=5 keeps current planned_day + 4 following planned days.
    """

    first_day = get_current_planned_day(current_time_s)
    last_day = first_day + lookahead_days - 1

    return first_day, last_day


def normalize_completed_unit_ids(completed_unit_ids=None) -> Set[str]:
    """
    Normalise completed unit IDs from the main simulation.

    Expected format is the scheduler internal unit_id format, e.g. '12.3'.
    For now this can be left empty until the main simulation supplies it.
    """

    if completed_unit_ids is None:
        return set()

    return {str(unit_id) for unit_id in completed_unit_ids}


def filter_orders_and_units_for_rolling_horizon(
    orders: List[Order],
    units: List[Unit],
    order_units: dict,
    current_time_s: float = 0.0,
    lookahead_days: int = ALLOWED_LOOKAHEAD_DAYS[0],
    completed_unit_ids=None
) -> Tuple[List[Order], List[Unit], dict, Dict[str, float]]:
    """
    Prepare a reduced scheduling problem for the GA.

    The filter does two things:
    1) Removes units that have already been completed by the main simulation.
    2) Keeps all unfinished carry-over orders from earlier planned days plus
       the orders up to the rolling look-ahead window end.

    Important: If disruptions delay production, unfinished orders from earlier
    planned days are carried into the next horizon instead of being discarded.
    If an order has partly completed units, the remaining quantity is scheduled
    as a reduced order with the same order_id/due_date/priority.
    """

    completed = normalize_completed_unit_ids(completed_unit_ids)

    first_planned_day, last_planned_day = get_planned_day_window(
        current_time_s=current_time_s,
        lookahead_days=lookahead_days
    )

    units_by_id = {
        unit.unit_id: unit
        for unit in units
    }

    horizon_orders = []
    horizon_units = []
    horizon_order_units = {}

    for order in orders:

        # Carry-over logic:
        # If an order was planned for an earlier day but did not finish because
        # disruptions pushed the schedule late, it must stay in the scheduling
        # problem until it is completed. Therefore the GA includes every
        # unfinished order up to the end of the current look-ahead window,
        # not only orders whose planned_day is inside [first_day, last_day].
        if int(order.planned_day) > int(last_planned_day):
            continue

        remaining_unit_ids = [
            unit_id
            for unit_id in order_units[order.order_id]
            if unit_id not in completed
        ]

        if not remaining_unit_ids:
            continue

        horizon_order_units[order.order_id] = remaining_unit_ids

        horizon_orders.append(
            Order(
                order_id=order.order_id,
                due_date=order.due_date,
                priority=order.priority,
                total_units=len(remaining_unit_ids),
                planned_week=order.planned_week,
                planned_day=order.planned_day
            )
        )

        horizon_units.extend(
            units_by_id[unit_id]
            for unit_id in remaining_unit_ids
        )

    # Important:
    # Horizon start is the actual current simulation time,
    # not the start of the production day.
    horizon_start_s = current_time_s

    # Horizon end is the end of the last planned day included in the window.
    horizon_end_s = last_planned_day * SECONDS_PER_PRODUCTION_DAY

    horizon_info = {
        "first_planned_day": first_planned_day,
        "last_planned_day": last_planned_day,
        "horizon_start_s": horizon_start_s,
        "horizon_end_s": horizon_end_s,
    }

    return horizon_orders, horizon_units, horizon_order_units, horizon_info

def is_schedule_feasible_within_horizon(simulation_result: dict, horizon_info: dict) -> bool:
    """True if simulated makespan finishes within horizon end."""
    if not simulation_result or not horizon_info:
        if not simulation_result:
            return True
    makespan = simulation_result.get('makespan')
    horizon_end = horizon_info.get('horizon_end_s')
    if  horizon_end is None:
        print("horizon is None!!")
        return False
    if makespan is None:
        print("makespan is None!")
        return False
    try:
        return float(makespan) <= float(horizon_end)
    except (TypeError, ValueError):
        return False


def merge_schedule_with_previous_tail(
    new_schedule_df: pd.DataFrame,
    previous_schedule_df: pd.DataFrame,
    planned_day_by_order: dict,
    cut_day: int,
    completed_unit_ids: Set[str]
) -> pd.DataFrame:
    """Merge the newly optimized horizon with any unfinished previous schedule rows.

    This is the carry-over protection that prevents units/orders from being
    discarded when disruptions push the makespan past the current horizon.

    Rules:
    - Start with the newly optimized schedule.
    - Append every row from the previous current_schedule.csv that is not
      completed and not already present in the new schedule.
    - This keeps untouched future units, emergency/unknown-plan units, and any
      older unfinished units that somehow were not included by the new horizon.
    - Renumber unit_seq from 1..N.
    """
    if previous_schedule_df is None or previous_schedule_df.empty:
        merged = new_schedule_df.copy()
        merged['unit_seq'] = range(1, len(merged) + 1)
        return merged

    prev = previous_schedule_df.copy()

    if 'unit_id' not in prev.columns:
        merged = new_schedule_df.copy()
        merged['unit_seq'] = range(1, len(merged) + 1)
        return merged

    # Remove completed units from the previous schedule snapshot.
    if completed_unit_ids:
        prev = prev[~prev['unit_id'].astype(str).isin({str(u) for u in completed_unit_ids})]

    # Remove units already covered by the newly optimized horizon.
    new_unit_ids = set(new_schedule_df['unit_id'].astype(str).tolist()) if 'unit_id' in new_schedule_df.columns else set()
    if new_unit_ids:
        prev = prev[~prev['unit_id'].astype(str).isin(new_unit_ids)]

    merged = pd.concat([new_schedule_df, prev], ignore_index=True)

    # Drop any accidental duplicate unit rows while preserving first occurrence.
    if 'unit_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['unit_id'], keep='first')

    # Renumber unit_seq.
    merged['unit_seq'] = range(1, len(merged) + 1)

    # Keep expected column order if possible.
    cols = ['unit_seq', 'order_id', 'unit_id', 'variant', 'route_id']
    merged = merged[[c for c in cols if c in merged.columns]]

    return merged




# ============================================================
# INITIAL SEED - PURE EDD
# ============================================================

def create_initial_order(orders: List[Order]):

    order_scores = []

    for order in orders:
        order_scores.append({
            "order_id": order.order_id,
            "due_date": order.due_date,
            "priority": order.priority,
            "total_units": order.total_units,
            "planned_day": order.planned_day
        })

    order_scores.sort(
        key=lambda x: x["due_date"]
    )

    initial = [
        row["order_id"]
        for row in order_scores
    ]

    print("\n================================================")
    print("INITIAL ORDER USING PURE EDD")
    print("================================================")

    for row in order_scores:
        print(
            f"Order {row['order_id']} | "
            f"Due={row['due_date']:.1f} | "
            f"Priority={row['priority']} | "
            f"Units={row['total_units']} | "
            f"Planned day={row['planned_day']}"
        )

    print("================================================\n")

    return initial


# ============================================================
# POPULATION
# ============================================================

def create_order_population(orders: List[Order],swaps,population_size):

    if not orders:
        return []

    initial = create_initial_order(orders)

    population = [initial]

    print("\n================================================")
    print("CREATING INITIAL POPULATION")
    print("================================================")

    print("\nChromosome 1 (initial):")
    print(initial)

    while len(population) < population_size:

        chrom = copy.deepcopy(initial)

        for _ in range(swaps):

            i = random.randint(0, len(chrom) - 1)
            j = random.randint(0, len(chrom) - 1)

            chrom[i], chrom[j] = chrom[j], chrom[i]

        population.append(chrom)

        print(f"\nChromosome {len(population)}:")
        print(chrom)

    print("\n================================================")
    print(f"TOTAL POPULATION CREATED: {len(population)}")
    print("================================================\n")

    return population


# ============================================================
# CONVERSIONS
# ============================================================

def order_chromosome_to_unit_sequence(
    order_chromosome,
    order_units
):

    unit_sequence = []

    for order_id in order_chromosome:
        unit_sequence.extend(order_units[order_id])

    return unit_sequence

def chromosome_to_unit_dataframe(
    chromosome,
    order_units,
    units_lookup,
    route_id=0,
    route_id_by_unit_id=None,
):

    unit_sequence = order_chromosome_to_unit_sequence(
        chromosome,
        order_units
    )

    rows = []

    for unit_seq, internal_unit_id in enumerate(
        unit_sequence,
        start=1
    ):

        unit = units_lookup[internal_unit_id]

        rows.append({
            "unit_seq": unit_seq,
            "order_id": unit.order_id,
            "unit_id": unit.unit_id,
            "variant": unit.variant,
            "route_id": (route_id_by_unit_id or {}).get(str(unit.unit_id), route_id)
        })

    return pd.DataFrame(rows)

# ============================================================
# EXPORT SCHEDULE
# ============================================================


def export_schedule(
    chromosome,
    order_units,
    units_lookup,
    filename="current_schedule.csv",
    verbose=False,
    route_id_by_unit_id=None,
):
    """Export ONLY the current schedule CSV into the on-going run folder.

    The GA's sole filesystem side-effect should be overwriting:
        ON_GOING_RUN_DIR / "current_schedule.csv"
    """
    unit_df = chromosome_to_unit_dataframe(
        chromosome=chromosome,
        order_units=order_units,
        units_lookup=units_lookup,
        route_id=0,
        route_id_by_unit_id=route_id_by_unit_id,
    )

    if ON_GOING_RUN_DIR is None:
        raise RuntimeError(
            "ON_GOING_RUN_DIR is not configured. Call configure_paths_from_main_settings() first."
        )

    output_path = Path(ON_GOING_RUN_DIR) / "current_schedule.csv"
    unit_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Schedule saved to {output_path}")

    return output_path



def safe_delete_file(path: Path):
    """NO-OP (disabled)."""
    return



def safe_delete_folder(path: Path):
    """NO-OP (disabled)."""
    return

def find_summary_folder(
    generation,
    chromosome_index
):
    """Find the newest simulator summary folder/file for the evaluated schedule.

    Preferred location is on_going_run/unit_summary.csv because the simulator
    is expected to work from the same main_settings.json paths. If the
    simulator writes summaries into output_run instead, the newest folder
    containing unit_summary.csv is used.
    """

    candidates = []

    if UNIT_SUMMARY_PATH is not None and UNIT_SUMMARY_PATH.exists():
        candidates.append(ON_GOING_RUN_DIR)

    if OUTPUT_RUN_DIR is not None and OUTPUT_RUN_DIR.exists():
        if (OUTPUT_RUN_DIR / "unit_summary.csv").exists():
            candidates.append(OUTPUT_RUN_DIR)

        candidates.extend(
            folder
            for folder in OUTPUT_RUN_DIR.iterdir()
            if folder.is_dir() and (folder / "unit_summary.csv").exists()
        )

    if not candidates:
        raise FileNotFoundError(
            f"No unit_summary.csv found in {ON_GOING_RUN_DIR} or {OUTPUT_RUN_DIR}"
        )

    return max(
        candidates,
        key=lambda folder: (folder / "unit_summary.csv").stat().st_mtime
    )


def read_simulation_result_from_unit_summary(
    summary_folder: Path
):

    unit_summary_path = summary_folder / "unit_summary.csv"

    if not unit_summary_path.exists():
        raise FileNotFoundError(
            f"unit_summary.csv not found in: {summary_folder}"
        )

    unit_df = pd.read_csv(unit_summary_path)

    required_columns = [
        "orderID",
        "completion_time_s"
    ]

    for column in required_columns:
        if column not in unit_df.columns:
            raise KeyError(
                f"Missing column '{column}' in {unit_summary_path}"
            )

    order_completion_times = (
        unit_df
        .groupby("orderID")["completion_time_s"]
        .max()
        .to_dict()
    )

    makespan = (
        unit_df["completion_time_s"]
        .max()
    )

    return {
        "order_completion_times": order_completion_times,
        "makespan": makespan
    }


# ============================================================
# SIMULATOR WRAPPER
# ============================================================
def evaluate_schedule_with_simulator(
    chromosome,
    order_units,
    units_lookup,
    chromosome_index,
    generation
, current_time_s: float = 0.0):

    filename = "current_schedule.csv"

    schedule_path = export_schedule(
        chromosome=chromosome,
        order_units=order_units,
        units_lookup=units_lookup,
        filename=filename,
        verbose=False
    )

    with contextlib.redirect_stdout(io.StringIO()):
        simulation_result = simulator.simulate_for_ga(str(MAIN_SETTINGS_PATH), current_time_s)

    # No summary folder is produced when using simulate_for_ga (in-memory evaluation).
    summary_folder = None
    return simulation_result, summary_folder


# ============================================================
# FITNESS
# ============================================================


def calculate_fitness(
    simulation_result,
    orders
):
    """Calculate GA fitness for a simulated schedule.

    Lower fitness is better.

    Implements the hybrid objective that prioritizes avoiding tardiness while giving
    only a very small reward for earliness (tie-breaker):

        F = sum_i w_i * (exp(k * T_i) - 1)  -  ε * sum_i E_i

    Where:
      - T_i = tardiness in days (max(0, completion - due) / TIME_SCALE)
      - E_i = earliness in days (max(0, due - completion) / TIME_SCALE)
      - w_i = priority_i ** PRIORITY_GAMMA
      - k   = ALPHA_TARDINESS
      - ε   = GAMMA_EARLINESS

    The simulator returns completion times in seconds, and due dates are assumed to
    use the same unit.
    """

    if not simulation_result or "order_completion_times" not in simulation_result:
        return {
            "fitness": float("inf"),
            "weighted_exp_tardiness": float("inf"),
            "weighted_earliness_reward": 0.0,
            "raw_exp_tardiness": float("inf"),
            "raw_weighted_exp_tardiness": float("inf"),
            "max_tardiness_days": None,
            "raw_earliness_days": 0.0,
            "late_orders": 0,
        }

    order_info = {
        str(o.order_id): {
            "due_date": float(o.due_date),
            "priority": int(getattr(o, 'priority', 1) or 1),
        }
        for o in orders
    }

    k = float(ALPHA)
    eps = float(BETA)

    raw_exp_tardiness = 0.0
    raw_weighted_exp_tardiness = 0.0
    raw_earliness_days = 0.0
    late_orders = 0

    for order_id, completion in simulation_result["order_completion_times"].items():

        order_id_key = str(order_id)
        if order_id_key not in order_info:
            continue

        due = float(order_info[order_id_key]["due_date"])
        priority = max(1, int(order_info[order_id_key].get("priority", 1)))
        w = float(priority) ** float(GAMMA)

        completion = float(completion)
        lateness_s = completion - due

        tardiness_s = max(0.0, lateness_s)
        earliness_s = max(0.0, -lateness_s)

        # Convert to days (or whatever TIME_SCALE represents)
        T_hours = tardiness_s / float(TIME_SCALE)
        E_hours = earliness_s / float(TIME_SCALE)

        if tardiness_s > 0.0:
            late_orders += 1

        exp_term = math.exp(k * T_hours) - 1.0
        raw_weighted_exp_tardiness += float(DELTA)*w * exp_term

        raw_earliness_days += w * E_hours

    weighted_exp_tardiness = raw_weighted_exp_tardiness
    weighted_earliness_reward = eps * raw_earliness_days

    fitness = weighted_exp_tardiness - weighted_earliness_reward

    return {
        "fitness": fitness,
        "weighted_exp_tardiness": weighted_exp_tardiness,
        "weighted_earliness_reward": weighted_earliness_reward,
        "raw_exp_tardiness": raw_exp_tardiness,
        "raw_weighted_exp_tardiness": raw_weighted_exp_tardiness,
        "raw_earliness_days": raw_earliness_days,
        "late_orders": late_orders,
        "k_tardiness": k,
        "priority_weight": float(GAMMA),
        "epsilon_earliness": eps,
    }


# ============================================================
# GA OPERATORS
# ============================================================

def tournament_selection(
    population,
    fitnesses,
    tournament_size
):
    sampled = random.sample(
        list(zip(population, fitnesses)),
        min(tournament_size, len(population))
    )

    sampled.sort(key=lambda x: x[1])

    return copy.deepcopy(sampled[0][0])


def order_crossover(p1, p2):

    size = len(p1)

    a = random.randint(0, size - 1)
    b = random.randint(a, size - 1)

    child = [-1] * size

    child[a:b + 1] = p1[a:b + 1]

    fill = [
        g for g in p2
        if g not in child
    ]

    ptr = 0

    for i in range(size):

        if child[i] == -1:

            child[i] = fill[ptr]
            ptr += 1

    return child


def insert_mutation(chrom):

    chrom = copy.deepcopy(chrom)

    i = random.randint(0, len(chrom) - 1)
    j = random.randint(0, len(chrom) - 1)

    gene = chrom.pop(i)

    chrom.insert(j, gene)

    return chrom

def best_generation_percentage(generation_number,best_fitness,counted_best_fitness):
    percentage_needed = 0.002*generation_number**2+0.01*generation_number
    return counted_best_fitness*(1-percentage_needed)>best_fitness

# ============================================================
# GA LOOP
# ============================================================

def run_ga(
    orders,
    units,
    order_units
, current_time_s: float = 0.0):

    swaps =int(max(len(orders)*SWAPS_SCALE+SWAPS_CONSTANT,SWAPS_MIN))
    population_size = int(max(len(orders)*POPULATION_SCALE+POPULATION_CONSTANT,POPULATION_MIN))
    elite_size = int(max(population_size*ELITE_SCALE+ELITE_CONSTANT,ELITE_MIN))
    tournament_size = int(max(population_size*TOURNAMENT_SCALE+TOURNAMENT_CONSTANT,TOURNAMENT_MIN))

    print(f"swaps: {swaps}\npopulation: {population_size}\nelite: {elite_size}\ntournament: {tournament_size}")
    



    units_lookup = {
        u.unit_id: u
        for u in units
    }

    population = create_order_population(
        orders,
        swaps,
        population_size
    )

    if not population:
        print("No orders inside the selected rolling horizon.")
        return (None, [], None, float("inf"), None)

    best_order_solution = None
    best_unit_sequence = None
    best_simulation_result = None
    # DISABLED: best_summary_folder = None

    best_fitness = float("inf")
    counted_best_fitness = None

    generation_number = 0
    best_generation = 0

    while best_generation + GENERATION_LIMIT > generation_number:

        generation_number = generation_number + 1

        print("\n================================================")
        print(f"GENERATION {generation_number}")
        print("================================================")

        fitnesses = []

        generation_best_fitness = float("inf")
        generation_best_chromosome = None
        generation_best_order_sequence = None

        for chromosome_index, order_chromosome in enumerate(
            population,
            start=1
        ):

            unit_sequence = order_chromosome_to_unit_sequence(
                order_chromosome,
                order_units
            )

            simulation_result, summary_folder = evaluate_schedule_with_simulator(
                chromosome=order_chromosome,
                order_units=order_units,
                units_lookup=units_lookup,
                chromosome_index=chromosome_index,
                generation=generation_number,
                current_time_s=current_time_s,
            )

            fitness_result = calculate_fitness(
                simulation_result,
                orders
            )

            fitness = fitness_result["fitness"]

            fitnesses.append(fitness)

            if fitness < generation_best_fitness:

                generation_best_fitness = fitness
                generation_best_chromosome = chromosome_index
                generation_best_order_sequence = copy.deepcopy(
                    order_chromosome
                )

            is_new_global_best = fitness < best_fitness

            if is_new_global_best:

                # DISABLED: previous_best_summary_folder = best_summary_folder

                best_fitness = fitness

                best_order_solution = copy.deepcopy(
                    order_chromosome
                )

                best_unit_sequence = copy.deepcopy(
                    unit_sequence
                )

                best_simulation_result = copy.deepcopy(
                    simulation_result
                )
                if counted_best_fitness is None:
                    counted_best_fitness = best_fitness
                    best_generation = generation_number
                    print("COUNTED BEST FITNESS IS NONE!!! FACK!!")
                elif best_generation_percentage(generation_number, best_fitness,counted_best_fitness):
                    best_generation = generation_number
                    counted_best_fitness = best_fitness
                else:
                    print(f"Gen {generation_number:02d} | "f"Chrom {chromosome_index:02d}""\nThis chromosome was better but not good enough for our limit!")
                # DISABLED: best_summary_folder = summary_folder

            best_marker = " <-- NEW BEST" if is_new_global_best else ""

            print(
                f"Gen {generation_number:02d} | "
                f"Chrom {chromosome_index:02d} | "
                f"Fitness {fitness:12.6f} | "
                f"Late {fitness_result['late_orders']:2d} | "
                f"Tardiness {fitness_result['weighted_exp_tardiness']:8.4f} | "
                f"-Earliness {-fitness_result['weighted_earliness_reward']:8.4f} | "
                f"{best_marker}"
            )

        print(
            f"\n>>> Generation {generation_number} done | "
            f"Generation best chromosome: {generation_best_chromosome} | "
            f"Generation best fitness: {generation_best_fitness:.4f} | "
            f"Global best fitness: {best_fitness:.4f}"
        )

        print(
            f">>>Best order sequence in generation {generation_number}: "
            f"{generation_best_order_sequence}"
        )
        print(f"Generations since last best {generation_number-best_generation}. Stopping after {GENERATION_LIMIT} without new best.")

        ranked = sorted(
            zip(population, fitnesses),
            key=lambda x: x[1]
        )

        new_population = [
            copy.deepcopy(ranked[i][0])
            for i in range(min(elite_size, len(ranked)))
        ]

        while len(new_population) < population_size:

            p1 = tournament_selection(
                population,
                fitnesses,
                tournament_size
            )

            p2 = tournament_selection(
                population,
                fitnesses,
                tournament_size
            )

            if random.random() < CROSSOVER_RATE:

                child = order_crossover(
                    p1,
                    p2
                )

            else:

                child = copy.deepcopy(p1)

            if random.random() < MUTATION_RATE:

                child = insert_mutation(
                    child
                )

            new_population.append(child)

        population = new_population
    print("STOPPED DUE TO GENERATION LIMIT")
    return (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness,
        # DISABLED: best_summary_folder
    )



# ===============================
# Schedule existence & coverage check
# ===============================
def _schedule_exists_and_has_content(schedule_path):
    if schedule_path is None or not schedule_path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(schedule_path)
        return len(df) > 0
    except Exception:
        return False


def _schedule_has_less_than_one_day(current_time_s,segment_end_time_s, schedule_path, production_plan_path):
    """Return True if the existing schedule covers less than 1 production day beyond current time.

    The schedule CSV does NOT include planned_day, so we infer coverage by:
      1) Reading distinct order_id values from current_schedule.csv
      2) Looking up each order_id in production_plan.csv to get planned_week/planned_day
      3) Computing the maximum absolute planned day covered by the schedule
      4) Comparing with the current absolute planned day derived from current_time_s

    If any required file/column is missing, this returns True (forcing a full-horizon schedule).
    """
    print("Checking if the schedule has less than a day")
    if schedule_path is None or production_plan_path is None:
        return True

    try:
        schedule_df = pd.read_csv(schedule_path)
    except Exception:
        return True

    if schedule_df is None or len(schedule_df) == 0 or 'order_id' not in schedule_df.columns:
        return True

    # Extract unique order IDs from schedule
    order_ids = (
        pd.to_numeric(schedule_df['order_id'], errors='coerce')
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not order_ids:
        return True

    try:
        plan_df = pd.read_csv(production_plan_path)
    except Exception:
        return True

    required_cols = {'order_id', 'planned_day'}
    if plan_df is None or len(plan_df) == 0 or not required_cols.issubset(set(plan_df.columns)):
        return True

    # Normalize plan columns
    plan_df = plan_df.copy()
    plan_df['order_id'] = pd.to_numeric(plan_df['order_id'], errors='coerce')
    plan_df['planned_day'] = pd.to_numeric(plan_df['planned_day'], errors='coerce')
    plan_df = plan_df.dropna(subset=['order_id', 'planned_day'])

    if len(plan_df) == 0:
        return True

    covered = plan_df[plan_df['order_id'].astype(int).isin([int(x) for x in order_ids])]
    if len(covered) == 0:
        return True

    max_abs_day = int(covered['planned_day'].max())
    segment_end_day = int(float(segment_end_time_s) // SECONDS_PER_PRODUCTION_DAY) + 1

    
    # Current absolute day from current_time_s (each production day = 8h)
    try:
        current_abs_day = int(float(current_time_s) // SECONDS_PER_PRODUCTION_DAY) + 1
    except Exception:
        current_abs_day = 1

    # If the schedule does not extend into at least the next day, treat as < 1 day left.
    print(f"max_abs_day: {max_abs_day} | current_abs_day {current_abs_day}")
    return (max_abs_day - current_abs_day) < 1, max_abs_day < segment_end_day, segment_end_day - current_abs_day+1



def main(
    main_settings_path,
    current_time_s,
    segment_end_time_s,
    seed,
    lookahead_days: int = ALLOWED_LOOKAHEAD_DAYS[0],
    ):
    random.seed(seed)
    # Configure paths before the schedule pre-check, otherwise ON_GOING_RUN_DIR is still None.
    configure_paths_from_main_settings(main_settings_path)

    # ===============================
    # NEW: schedule pre-check
    # ===============================
    schedule_path = None
    if ON_GOING_RUN_DIR is not None:
        schedule_path = ON_GOING_RUN_DIR / "current_schedule.csv"

    need_full_horizon = False

    if not _schedule_exists_and_has_content(schedule_path):
        print("\nschedule is missing\n")
        need_full_horizon = True
    elif _schedule_has_less_than_one_day(current_time_s,segment_end_time_s, schedule_path, PRODUCTION_PLAN_PATH)[0]:
        print("\nhorizon less than a day\n")
        need_full_horizon = True
    elif _schedule_has_less_than_one_day(current_time_s,segment_end_time_s, schedule_path, PRODUCTION_PLAN_PATH)[1]:
        lookahead_days = _schedule_has_less_than_one_day(current_time_s,segment_end_time_s, schedule_path, PRODUCTION_PLAN_PATH)[2]
        print(f"The segment is longer than the horizon!\nnew lookahead_days: {lookahead_days}")
    
    if need_full_horizon:
        print("!!!NEEDED A FULL HORIZON!!!")
        lookahead_days = max(ALLOWED_LOOKAHEAD_DAYS)

    # Snapshot the previous schedule BEFORE GA evaluations overwrite current_schedule.csv.
    previous_schedule_df = None
    prev_schedule_path = Path(ON_GOING_RUN_DIR) / 'current_schedule.csv'
    if prev_schedule_path.exists():
        try:
            previous_schedule_df = pd.read_csv(prev_schedule_path)
        except Exception:
            previous_schedule_df = None

    completed_units = get_completed_unit_ids_from_unit_summary(
        current_time_s=current_time_s
    )

    production_df = load_production_plan(PRODUCTION_PLAN_PATH)

    # Build order_id -> planned_day mapping for merge logic.
    planned_day_by_order = (
        production_df
        .set_index('order_id')['planned_day']
        .astype(int)
        .to_dict()
    )

    all_orders, all_units, all_order_units = load_orders_and_units_from_file(production_df)

    def _run_once(lookahead_days_local: int):
        orders, units, order_units, horizon_info = filter_orders_and_units_for_rolling_horizon(
            orders=all_orders,
            units=all_units,
            order_units=all_order_units,
            current_time_s=current_time_s,
            lookahead_days=lookahead_days_local,
            completed_unit_ids=completed_units
        )

        if not orders:
            return None, None, float('inf'), None, horizon_info, units, order_units

        best_order_solution, best_unit_sequence, best_simulation_result, best_fitness, *_ = run_ga(
            orders=orders,
            units=units,
            order_units=order_units,
            current_time_s=current_time_s,
        )

        return best_order_solution, best_simulation_result, best_fitness, order_units, horizon_info, units, order_units

    # --- Try 1 day first (or the requested lookahead) ---
    best_order_solution, best_simulation_result, best_fitness, order_units, horizon_info, units, _ou = _run_once(lookahead_days)

    feasible = is_schedule_feasible_within_horizon(best_simulation_result, horizon_info)
    def horizon_print():
        print(
            f"\n[GA] Horizon feasibility check: \n"
            f"-makespan={best_simulation_result.get('makespan') if best_simulation_result else None}, \n"
            f"-horizon_window_s={(horizon_info.get('horizon_window_s') if horizon_info and horizon_info.get('horizon_window_s') is not None else ( (float(horizon_info.get('horizon_end_s')) - float(horizon_info.get('horizon_start_s'))) if horizon_info and horizon_info.get('horizon_end_s') is not None and horizon_info.get('horizon_start_s') is not None else None ))}, \n"
            f"-horizon_start_s={horizon_info.get('horizon_start_s') if horizon_info else None}, \n"
            f"-horizon_end_s={horizon_info.get('horizon_end_s') if horizon_info else None}, \n"
            f"-feasible={feasible}\n"
        )
    horizon_print()
    
    if lookahead_days < max(ALLOWED_LOOKAHEAD_DAYS) and (not feasible):
        best_order_solution, best_simulation_result, best_fitness, order_units, horizon_info, units, _ou = _run_once(max(ALLOWED_LOOKAHEAD_DAYS))
        print(f"had to expand our horizon - running max: {max(ALLOWED_LOOKAHEAD_DAYS)}")
        feasible = is_schedule_feasible_within_horizon(best_simulation_result, horizon_info)
        
        horizon_print()

    # --- Export logic ---
    if best_order_solution is None:
        return

    # Build the new schedule dataframe
    units_lookup = {u.unit_id: u for u in units}
    best_route_map = {}
    if isinstance(best_simulation_result, dict):
        best_route_map = dict(best_simulation_result.get("route_id_by_unit_id", {}))

    new_schedule_df = chromosome_to_unit_dataframe(
        chromosome=best_order_solution,
        order_units=order_units,
        units_lookup=units_lookup,
        route_id=0,
        route_id_by_unit_id=best_route_map,
    )

    # Always merge against the previous schedule when it exists.
    # This prevents uncompleted units from being lost when disruptions make the
    # chosen horizon finish late. Completed units are removed inside the merge.
    if previous_schedule_df is not None:
        cut_day = int(horizon_info.get('last_planned_day', get_current_planned_day(current_time_s)))
        final_schedule_df = merge_schedule_with_previous_tail(
            new_schedule_df=new_schedule_df,
            previous_schedule_df=previous_schedule_df,
            planned_day_by_order=planned_day_by_order,
            cut_day=cut_day,
            completed_unit_ids=completed_units
        )
    else:
        final_schedule_df = new_schedule_df.copy()
        final_schedule_df['unit_seq'] = range(1, len(final_schedule_df) + 1)

    if "route_id" not in final_schedule_df.columns:
        final_schedule_df["route_id"] = "0"
    if best_route_map:
        final_schedule_df["route_id"] = final_schedule_df.apply(
            lambda row: best_route_map.get(str(row.get("unit_id")), row.get("route_id", "0")),
            axis=1,
        )

    # Write final schedule (overwrite)
    output_path = Path(ON_GOING_RUN_DIR) / 'current_schedule.csv'
    final_schedule_df.to_csv(output_path, index=False)

    return


if __name__ == "__main__": #Kan slettes når koden kun skal køres af MAIN
    print("\n lmao u idiot dont run it from here\n")