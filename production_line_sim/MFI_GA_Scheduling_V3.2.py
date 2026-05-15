import random
import copy
import json
import re
import contextlib
import io
import math
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict

import pandas as pd

import E_production_line_sim as simulator


# ============================================================
# PATHS
# ============================================================

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MAIN_RUN = "main_14-05_21-59_1"
RUN_ID = 1

INPUT_DIR = (
    ROOT
    / "on_going"
    / MAIN_RUN
    / f"run_{RUN_ID}"
)

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Selected run folder not found: {INPUT_DIR}"
    )

LAYOUT_PATH = (
    ROOT / "data" / "Layouts" /
    "line_layout_robotcell_6staggered_example.json"
)

PROCESS_TIMES_PATH = ROOT / "data" / "process_times.json"
TRANSPORT_TIMES_PATH = ROOT / "data" / "transport_times.json"

CLEAN_TEMP_OUTPUTS = True
KEEP_ONLY_BEST_SUMMARY = True
# ============================================================
# GA SETTINGS / ROLLING HORIZON SETTINGS
# ============================================================

# One production day is currently defined as 8 hours.
SECONDS_PER_PRODUCTION_DAY = 8 * 60 * 60

# Set these values here while testing.
# Later, CURRENT_TIME_S and completed units should come from the main simulation.
CURRENT_TIME_S = 0
DEFAULT_LOOKAHEAD_DAYS = 3

# Temporary test input:
# Number of units already produced before the rolling-horizon scheduler starts.
# The code will remove the first N units from the full production plan order.
# Later, replace this with exact completed unit IDs from the main simulation.
COMPLETED_UNITS_COUNT = 0

# Rolling horizon options:
# 1 day  -> schedule the rest of current day only
# 3 days -> schedule rest of current day + 2 full days
# 5 days -> schedule rest of current day + 4 full days
ALLOWED_LOOKAHEAD_DAYS = {1, 3, 5}

SWAPS = 3
POPULATION_SIZE = 6
GENERATIONS = 4
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.4

# ============================================================
# FITNESS WEIGHTS
# ============================================================

# Fitness model:
#   1) Weighted exponential penalty for each delayed order
#   2) Weighted extra exponential penalty for the worst delayed order
#   3) Very small linear earliness reward as a tie-breaker
#
# Lower fitness is better.
#
# Times from the simulator are in seconds, so tardiness/earliness are
# converted to days before being used in the fitness function.

ALPHA_TARDINESS = 1.0
BETA_MAX_TARDINESS = 1.5
GAMMA_EARLINESS = 0.001

TIME_SCALE = 24 * 60 * 60  # 1 calendar day in seconds


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

def load_json(path: Path):

    print(f"Loading JSON: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_production_plan(input_dir: Path) -> pd.DataFrame:

    production_files = list(input_dir.glob("production_plan*"))

    if not production_files:
        raise FileNotFoundError(
            f"No production_plan file found in: {input_dir}"
        )

    path = production_files[0]

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

        print(
            f"\nORDER {order_id} | "
            f"Due={due_date} | "
            f"Priority={priority} | "
            f"Planned day={int(row['planned_day'])}"
        )

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

        print(f"  Total units for order {order_id}: {total_units}")

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

    if lookahead_days not in ALLOWED_LOOKAHEAD_DAYS:
        raise ValueError(
            f"lookahead_days must be one of {sorted(ALLOWED_LOOKAHEAD_DAYS)}, "
            f"got {lookahead_days}"
        )

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


def get_completed_unit_ids_from_count(
    units: List[Unit],
    completed_units_count: int = 0
) -> Set[str]:
    """
    Temporary helper for testing before main_sim is connected.

    If completed_units_count = N, the scheduler treats the first N units in the
    loaded production-plan order as already completed.

    Later, main_sim should preferably provide exact completed unit IDs instead
    of only a count, because exact IDs are more robust when the actual produced
    sequence differs from the original production-plan order.
    """

    if completed_units_count is None:
        completed_units_count = 0

    completed_units_count = int(completed_units_count)

    if completed_units_count < 0:
        raise ValueError(
            f"completed_units_count must be >= 0, got {completed_units_count}"
        )

    completed_units_count = min(completed_units_count, len(units))

    return {
        unit.unit_id
        for unit in units[:completed_units_count]
    }


def filter_orders_and_units_for_rolling_horizon(
    orders: List[Order],
    units: List[Unit],
    order_units: dict,
    current_time_s: float = 0.0,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    completed_unit_ids=None
) -> Tuple[List[Order], List[Unit], dict, Dict[str, float]]:
    """
    Prepare a reduced scheduling problem for the GA.

    The filter does two things:
    1) Removes units that have already been completed by the main simulation.
    2) Keeps only orders where production_plan.planned_day is inside the
       rolling look-ahead window.

    Important: If an order has partly completed units, the remaining quantity is
    scheduled as a reduced order with the same order_id/due_date/priority.
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

        if not (first_planned_day <= order.planned_day <= last_planned_day):
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


# ============================================================
# INITIAL SEED - PURE EDD
# ============================================================

def create_order_seed(orders: List[Order]):

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

    seed = [
        row["order_id"]
        for row in order_scores
    ]

    print("\n================================================")
    print("ORDER SEED USING PURE EDD")
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

    return seed


# ============================================================
# POPULATION
# ============================================================

def create_order_population(orders: List[Order]):

    if not orders:
        return []

    seed = create_order_seed(orders)

    population = [seed]

    print("\n================================================")
    print("CREATING INITIAL POPULATION")
    print("================================================")

    print("\nChromosome 1 (Seed):")
    print(seed)

    while len(population) < POPULATION_SIZE:

        chrom = copy.deepcopy(seed)

        for _ in range(SWAPS):

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
    route_id=0
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
            "route_id": route_id
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
    verbose=False
):
    unit_df = chromosome_to_unit_dataframe(
        chromosome=chromosome,
        order_units=order_units,
        units_lookup=units_lookup,
        route_id=0
    )

    output_path = INPUT_DIR / filename

    unit_df.to_csv(
        output_path,
        index=False
    )

    if verbose:
        print(f"Saved schedule: {output_path}")

    return output_path


# ============================================================
# OUTPUT CLEANUP
# ============================================================

def safe_delete_file(path: Path):
    if path is not None and path.exists() and path.is_file():
        try:
            path.unlink()
        except PermissionError as exc:
            print(f"WARNING: Could not delete file because it is locked: {path}")
            print(f"         {exc}")


def safe_delete_folder(path: Path):
    if path is not None and path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
        except PermissionError as exc:
            print(f"WARNING: Could not delete folder because it is locked: {path}")
            print(f"         {exc}")


def clear_old_summary_output_folder():
    """Remove old simulator summary folders for this input batch."""

    output_root = ROOT / "output" / INPUT_DIR.name

    if not output_root.exists():
        return

    for folder in output_root.iterdir():
        if folder.is_dir():
            safe_delete_folder(folder)


def cleanup_non_best_summary(summary_folder: Path, best_summary_folder: Path):
    if not CLEAN_TEMP_OUTPUTS:
        return

    if summary_folder is None:
        return

    if KEEP_ONLY_BEST_SUMMARY and summary_folder != best_summary_folder:
        safe_delete_folder(summary_folder)


def cleanup_previous_best_summary(previous_best_folder: Path, new_best_folder: Path):
    if not CLEAN_TEMP_OUTPUTS:
        return

    if not KEEP_ONLY_BEST_SUMMARY:
        return

    if previous_best_folder is not None and previous_best_folder != new_best_folder:
        safe_delete_folder(previous_best_folder)


# ============================================================
# SUMMARY READING
# ============================================================

def find_summary_folder(
    generation,
    chromosome_index
):

    if (INPUT_DIR / "unit_summary.csv").exists():
        return INPUT_DIR

    raise FileNotFoundError(
        f"unit_summary.csv not found in {INPUT_DIR}"
    )

    candidates = []

    if (INPUT_DIR / "unit_summary.csv").exists():
        candidates.append(INPUT_DIR)

    output_root = ROOT / "output" / INPUT_DIR.name

    if output_root.exists():
        candidates.extend(
            folder
            for folder in output_root.iterdir()
            if folder.is_dir() and (folder / "unit_summary.csv").exists()
        )

    if not candidates:
        raise FileNotFoundError(
            f"No folder with unit_summary.csv found in {INPUT_DIR} or {output_root}"
        )

    return max(
        candidates,
        key=lambda folder: (folder / "unit_summary.csv").stat().st_mtime
    )

    output_root = ROOT / "output" / INPUT_DIR.name

    base_name = "current_schedule_summary"

    exact_folder = output_root / base_name

    if exact_folder.exists():
        return exact_folder

    matching_folders = [
        folder
        for folder in output_root.glob(f"{base_name}*")
        if folder.is_dir()
    ]

    if not matching_folders:
        raise FileNotFoundError(
            f"No summary folder found for {base_name} in {output_root}"
        )

    return max(
        matching_folders,
        key=lambda folder: folder.stat().st_mtime
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
):

    filename = "current_schedule.csv"

    schedule_path = export_schedule(
        chromosome=chromosome,
        order_units=order_units,
        units_lookup=units_lookup,
        filename=filename,
        verbose=False
    )

    with contextlib.redirect_stdout(io.StringIO()):
        simulator.main()

    summary_folder = find_summary_folder(
        generation=generation,
        chromosome_index=chromosome_index
    )

    simulation_result = read_simulation_result_from_unit_summary(
        summary_folder
    )

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

    Fitness consists of:
    - Weighted sum of exponential tardiness penalty for all delayed orders.
    - Weighted extra exponential penalty for the worst delayed order.
    - Small linear earliness reward as a tie-breaker.

    The simulator returns completion times in seconds. Due dates are assumed
    to use the same unit. Tardiness and earliness are converted to days before
    being used in the fitness function.
    """

    order_info = {
        o.order_id: {
            "due_date": o.due_date
        }
        for o in orders
    }

    raw_exp_tardiness = 0.0
    max_tardiness_days = 0.0
    raw_earliness_days = 0.0
    late_orders = 0

    for order_id, completion in (
        simulation_result["order_completion_times"].items()
    ):

        order_id = int(order_id)

        if order_id not in order_info:
            continue

        due = order_info[order_id]["due_date"]

        lateness = completion - due

        tardiness = max(
            0.0,
            lateness
        )

        earliness = max(
            0.0,
            -lateness
        )

        tardiness_days = (
            tardiness / TIME_SCALE
        )

        earliness_days = (
            earliness / TIME_SCALE
        )

        if tardiness > 0:
            late_orders += 1

        raw_exp_tardiness += (
            math.exp(tardiness_days) - 1
        )

        max_tardiness_days = max(
            max_tardiness_days,
            tardiness_days
        )

        raw_earliness_days += earliness_days

    raw_max_exp_tardiness = (
        math.exp(max_tardiness_days) - 1
    )

    weighted_exp_tardiness = (
        ALPHA_TARDINESS * raw_exp_tardiness
    )

    weighted_max_exp_tardiness = (
        BETA_MAX_TARDINESS * raw_max_exp_tardiness
    )

    weighted_earliness_reward = (
        GAMMA_EARLINESS * raw_earliness_days
    )

    fitness = (
        weighted_exp_tardiness
        + weighted_max_exp_tardiness
        - weighted_earliness_reward
    )

    return {
        "fitness": fitness,
        "weighted_exp_tardiness": weighted_exp_tardiness,
        "weighted_max_exp_tardiness": weighted_max_exp_tardiness,
        "weighted_earliness_reward": weighted_earliness_reward,
        "raw_exp_tardiness": raw_exp_tardiness,
        "raw_max_exp_tardiness": raw_max_exp_tardiness,
        "max_tardiness_days": max_tardiness_days,
        "raw_earliness_days": raw_earliness_days,
        "late_orders": late_orders
    }


# ============================================================
# GA OPERATORS
# ============================================================

def tournament_selection(
    population,
    fitnesses
):

    sampled = random.sample(
        list(zip(population, fitnesses)),
        min(TOURNAMENT_SIZE, len(population))
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


# ============================================================
# GA LOOP
# ============================================================

def run_ga(
    orders,
    units,
    order_units
):

    units_lookup = {
        u.unit_id: u
        for u in units
    }

    if CLEAN_TEMP_OUTPUTS:
        clear_old_summary_output_folder()

    population = create_order_population(
        orders
    )

    if not population:
        print("No orders inside the selected rolling horizon.")
        return (None, [], None, float("inf"), None)

    best_order_solution = None
    best_unit_sequence = None
    best_simulation_result = None
    best_summary_folder = None

    best_fitness = float("inf")

    for generation in range(GENERATIONS):

        generation_number = generation + 1

        print("\n================================================")
        print(f"GENERATION {generation_number}/{GENERATIONS}")
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
                generation=generation_number
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

                previous_best_summary_folder = best_summary_folder

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

                best_summary_folder = summary_folder

                cleanup_previous_best_summary(
                    previous_best_folder=previous_best_summary_folder,
                    new_best_folder=best_summary_folder
                )

            else:

                cleanup_non_best_summary(
                    summary_folder=summary_folder,
                    best_summary_folder=best_summary_folder
                )

            best_marker = " <-- NEW BEST" if is_new_global_best else ""

            print(
                f"Gen {generation_number:02d} | "
                f"Chrom {chromosome_index:02d} | "
                f"Fitness {fitness:12.4f} | "
                f"Late {fitness_result['late_orders']:2d} | "
                f"A*Exp {fitness_result['weighted_exp_tardiness']:8.4f} | "
                f"B*Max {fitness_result['weighted_max_exp_tardiness']:8.4f} | "
                f"-G*Early {-fitness_result['weighted_earliness_reward']:8.4f} | "
                f"MaxDay {fitness_result['max_tardiness_days']:6.2f} | "
                f"{best_marker}"
            )

        print(
            f"\n>>> Generation {generation_number} done | "
            f"Generation best chromosome: {generation_best_chromosome} | "
            f"Generation best fitness: {generation_best_fitness:.4f} | "
            f"Global best fitness: {best_fitness:.4f}"
        )

        print(
            f"Best order sequence in generation {generation_number}: "
            f"{generation_best_order_sequence}"
        )

        ranked = sorted(
            zip(population, fitnesses),
            key=lambda x: x[1]
        )

        new_population = [
            copy.deepcopy(ranked[i][0])
            for i in range(min(ELITE_SIZE, len(ranked)))
        ]

        while len(new_population) < POPULATION_SIZE:

            p1 = tournament_selection(
                population,
                fitnesses
            )

            p2 = tournament_selection(
                population,
                fitnesses
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

    if KEEP_ONLY_BEST_SUMMARY and best_summary_folder is not None:
        final_summary_folder = ROOT / "output" / INPUT_DIR.name / "best_schedule_summary"

        if final_summary_folder.exists() and final_summary_folder != best_summary_folder:
            safe_delete_folder(final_summary_folder)

        if best_summary_folder.exists() and best_summary_folder != final_summary_folder:
            try:
                best_summary_folder.rename(final_summary_folder)
                best_summary_folder = final_summary_folder
            except PermissionError as exc:
                print(f"WARNING: Could not rename best summary folder: {best_summary_folder}")
                print(f"         {exc}")

    return (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness,
        best_summary_folder
    )


# ============================================================
# MAIN
# ============================================================

def main(
    current_time_s: float = CURRENT_TIME_S,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    completed_units_count: int = COMPLETED_UNITS_COUNT,
    completed_unit_ids=None
):

    print("\n================================================")
    print("STARTING GA SCHEDULER")
    print("================================================")

    print("\nROOT:")
    print(ROOT)

    print("\nInput folder:")
    print(INPUT_DIR)

    print("\nRolling horizon setup:")
    print(f"Current simulation time [s]: {current_time_s}")
    print(f"Lookahead days: {lookahead_days}")
    print(f"Completed units count: {completed_units_count}")

    print("\nLoading files...")

    production_df = load_production_plan(
        INPUT_DIR
    )

    load_json(
        LAYOUT_PATH
    )

    load_json(
        PROCESS_TIMES_PATH
    )

    load_json(
        TRANSPORT_TIMES_PATH
    )

    print("\nFiles loaded successfully.")

    all_orders, all_units, all_order_units = load_orders_and_units_from_file(
        production_df
    )

    completed_from_count = get_completed_unit_ids_from_count(
        units=all_units,
        completed_units_count=completed_units_count
    )

    completed_from_ids = normalize_completed_unit_ids(
        completed_unit_ids
    )

    completed_units = completed_from_count.union(completed_from_ids)

    print(f"Completed unit IDs used for filtering: {len(completed_units)}")

    orders, units, order_units, horizon_info = filter_orders_and_units_for_rolling_horizon(
        orders=all_orders,
        units=all_units,
        order_units=all_order_units,
        current_time_s=current_time_s,
        lookahead_days=lookahead_days,
        completed_unit_ids=completed_units
    )

    print("\nRolling horizon result:")
    print(
        f"Planned day window: {horizon_info['first_planned_day']} "
        f"-> {horizon_info['last_planned_day']}"
    )
    print(f"Horizon start [s]: {horizon_info['horizon_start_s']}")
    print(f"Horizon end [s]: {horizon_info['horizon_end_s']}")
    print(f"Orders inside horizon: {len(orders)} / {len(all_orders)}")
    print(f"Remaining units inside horizon: {len(units)} / {len(all_units)}")

    if not orders:
        print("\nNo remaining orders/units inside selected horizon. Nothing to schedule.")
        return None

    print("\nRunning GA...")

    (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness,
        best_summary_folder
    ) = run_ga(
        orders=orders,
        units=units,
        order_units=order_units
    )

    print("\n================================================")
    print("GA FINISHED")
    print("================================================")

    print(f"\nBest fitness: {best_fitness:.4f}")

    print("\nBest order solution:")
    print(best_order_solution)

    if best_simulation_result is not None:

        print(
            f"\nBest makespan: "
            f"{best_simulation_result.get('makespan', 'N/A')}"
        )

    if best_summary_folder is not None:

        print("\nBest summary folder:")
        print(best_summary_folder)

    if best_order_solution is not None:
        units_lookup = {
            u.unit_id: u
            for u in units
        }

        export_schedule(
            chromosome=best_order_solution,
            order_units=order_units,
            units_lookup=units_lookup,
            filename="current_schedule.csv",
            verbose=True
        )

    print("\nFinished.")


if __name__ == "__main__":

    main(
        current_time_s=CURRENT_TIME_S,
        lookahead_days=DEFAULT_LOOKAHEAD_DAYS,
        completed_units_count=COMPLETED_UNITS_COUNT,
        completed_unit_ids=[]
    )
