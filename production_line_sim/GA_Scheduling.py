import random
import copy
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

import E_production_line_sim_schedule_input_fastest_parallel as simulator


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_BATCH_NAME_RE = re.compile(
    r"^orders_(\d{2})-(\d{2})_(\d{2})-(\d{2})_(\d+)$"
)


def _parse_input_batch_sort_key(name: str):
    match = INPUT_BATCH_NAME_RE.match(name)

    if not match:
        raise ValueError(f"Invalid input batch name: {name}")

    day, month, hour, minute, batch_number = map(int, match.groups())

    return month, day, hour, minute, batch_number


def find_newest_input_batch_dir(input_root: Path) -> Path:

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    candidate_dirs = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and INPUT_BATCH_NAME_RE.match(path.name)
    ]

    if not candidate_dirs:
        raise FileNotFoundError(
            f"No generated input folders were found in {input_root}"
        )

    return max(
        candidate_dirs,
        key=lambda path: _parse_input_batch_sort_key(path.name)
    )


INPUT_DIR = find_newest_input_batch_dir(ROOT / "input")

LAYOUT_PATH = (
    ROOT / "data" / "Layouts" /
    "line_layout_robotcell_6staggered_example.json"
)

PROCESS_TIMES_PATH = ROOT / "data" / "process_times.json"

TRANSPORT_TIMES_PATH = ROOT / "data" / "transport_times.json"


# ============================================================
# CONSTANTS
# ============================================================

SWAPS = 3


# ============================================================
# GA SETTINGS
# ============================================================

POPULATION_SIZE = 10
GENERATIONS = 50
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.4


# ============================================================
# FITNESS WEIGHTS
# ============================================================

ALPHA_TARDINESS = 1.0
BETA_LATE_ORDERS = 5000
GAMMA_EARLINESS = 0.05


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

    print("\n================================================")
    print("LOADING ORDERS AND UNITS FROM FILE")
    print("================================================")

    for _, row in df.iterrows():

        order_id = int(row["order_id"])
        due_date = float(row["due date"])
        priority = int(row["priority"])

        order_units[order_id] = []

        unit_counter = 1
        total_units = 0

        print(
            f"\nORDER {order_id} | "
            f"Due={due_date} | "
            f"Priority={priority}"
        )

        for i in range(3):

            variant = str(row[f"variant{i}"])
            qty = int(row[f"quantity{i}"])

            if qty <= 0:
                continue

            print(f"  {variant} -> Qty={qty}")

            for _ in range(qty):

                internal_unit_id = f"{order_id}.{unit_counter}"

                unit = Unit(
                    unit_id=internal_unit_id,
                    order_id=order_id,
                    variant=variant,
                    due_date=due_date,
                    priority=priority
                )

                units.append(unit)

                order_units[order_id].append(
                    internal_unit_id
                )

                print(
                    f"    Created Unit: "
                    f"{internal_unit_id}"
                )

                unit_counter += 1
                total_units += 1

        order = Order(
            order_id=order_id,
            due_date=due_date,
            priority=priority,
            total_units=total_units
        )

        orders.append(order)

        print(
            f"  Total units for order "
            f"{order_id}: {total_units}"
        )

    print("\n================================================")
    print(f"TOTAL ORDERS LOADED: {len(orders)}")
    print(f"TOTAL UNITS LOADED: {len(units)}")
    print("================================================\n")

    return orders, units, order_units


# ============================================================
# INITIAL SEED
# ============================================================

def create_order_seed(orders: List[Order]):

    """
    Initial chromosome:
    1. Lowest due date first
    2. Highest priority first
    """

    order_scores = []

    for order in orders:

        order_scores.append({
            "order_id": order.order_id,
            "due_date": order.due_date,
            "priority": order.priority,
            "total_units": order.total_units
        })

    order_scores.sort(
        key=lambda x: (
            x["due_date"],
            -x["priority"]
        )
    )

    seed = [
        row["order_id"]
        for row in order_scores
    ]

    print("\n================================================")
    print("ORDER SEED USING DUE DATE + PRIORITY")
    print("================================================")

    for row in order_scores:

        print(
            f"Order {row['order_id']} | "
            f"Due={row['due_date']:.1f} | "
            f"Priority={row['priority']} | "
            f"Units={row['total_units']}"
        )

    print("================================================\n")

    return seed


# ============================================================
# POPULATION
# ============================================================

def create_order_population(orders: List[Order]):

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
# CONVERT ORDER CHROMOSOME
# ============================================================

def order_chromosome_to_unit_sequence(
    order_chromosome,
    order_units
):

    unit_sequence = []

    for order_id in order_chromosome:
        unit_sequence.extend(order_units[order_id])

    return unit_sequence


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

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

        local_unit_id = int(
            unit.unit_id.split(".")[1]
        )

        rows.append({
            "unit_seq": unit_seq,
            "order_id": unit.order_id,
            "unit_id": local_unit_id,
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
    filename
):

    unit_df = chromosome_to_unit_dataframe(
        chromosome=chromosome,
        order_units=order_units,
        units_lookup=units_lookup,
        route_id=0
    )

    output_dir = INPUT_DIR / "output_schedules"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / filename

    unit_df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved schedule: {output_path}")

    return output_path


# ============================================================
# FITNESS
# ============================================================

def calculate_fitness(
    simulation_result,
    orders
):

    order_info = {
        o.order_id: {
            "due_date": o.due_date,
            "priority": o.priority
        }
        for o in orders
    }

    total_tardiness = 0.0
    late_orders = 0
    total_earliness = 0.0

    for order_id, completion in (
        simulation_result["order_completion_times"].items()
    ):

        order_id = int(order_id)

        due = order_info[order_id]["due_date"]
        priority = order_info[order_id]["priority"]

        tardiness = max(
            0.0,
            completion - due
        )

        earliness = max(
            0.0,
            due - completion
        )

        if tardiness > 0:
            late_orders += 1

        total_tardiness += (
            tardiness * (priority + 1)
        )

        total_earliness += earliness

    fitness = (
        ALPHA_TARDINESS * total_tardiness
        + BETA_LATE_ORDERS * late_orders
        - GAMMA_EARLINESS * total_earliness
    )

    return fitness


# ============================================================
# GA OPERATORS
# ============================================================

def tournament_selection(
    population,
    fitnesses
):

    sampled = random.sample(
        list(zip(population, fitnesses)),
        TOURNAMENT_SIZE
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
# SIMULATOR WRAPPER
# ============================================================

def evaluate_schedule_with_simulator(
    chromosome,
    order_units,
    units_lookup
):

    export_schedule(
        chromosome=chromosome,
        order_units=order_units,
        units_lookup=units_lookup,
        filename="current_schedule.csv"
    )

    print("Running simulator...")

    simulation_result = simulator.main()

    print("Simulator finished.")

    return simulation_result


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

    population = create_order_population(
        orders
    )

    best_order_solution = None
    best_unit_sequence = None
    best_simulation_result = None

    best_fitness = float("inf")

    for generation in range(GENERATIONS):

        print("\n================================================")
        print(
            f"GENERATION "
            f"{generation + 1}/{GENERATIONS}"
        )
        print("================================================")

        fitnesses = []

        for chromosome_index, order_chromosome in enumerate(
            population,
            start=1
        ):

            print(
                f"\nEvaluating chromosome "
                f"{chromosome_index}"
            )

            unit_sequence = (
                order_chromosome_to_unit_sequence(
                    order_chromosome,
                    order_units
                )
            )

            simulation_result = (
                evaluate_schedule_with_simulator(
                    chromosome=order_chromosome,
                    order_units=order_units,
                    units_lookup=units_lookup
                )
            )

            fitness = calculate_fitness(
                simulation_result,
                orders
            )

            fitnesses.append(fitness)

            print(
                f"Fitness: {fitness:.2f}"
            )

            if fitness < best_fitness:

                print(
                    "NEW BEST SOLUTION FOUND"
                )

                best_fitness = fitness

                best_order_solution = (
                    copy.deepcopy(order_chromosome)
                )

                best_unit_sequence = (
                    copy.deepcopy(unit_sequence)
                )

                best_simulation_result = (
                    simulation_result
                )

        print(
            f"\nGeneration best fitness: "
            f"{best_fitness:.2f}"
        )

        ranked = sorted(
            zip(population, fitnesses),
            key=lambda x: x[1]
        )

        new_population = [
            copy.deepcopy(ranked[i][0])
            for i in range(ELITE_SIZE)
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

    return (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n================================================")
    print("STARTING GA SCHEDULER")
    print("================================================")

    print("\nROOT:")
    print(ROOT)

    print("\nLoading files...")

    production_df = load_production_plan(
        INPUT_DIR
    )

    layout_data = load_json(
        LAYOUT_PATH
    )

    process_data = load_json(
        PROCESS_TIMES_PATH
    )

    transport_data = load_json(
        TRANSPORT_TIMES_PATH
    )

    print("\nFiles loaded successfully.")

    orders, units, order_units = (
        load_orders_and_units_from_file(
            production_df
        )
    )

    print("\nExample order to units:")

    for order_id, unit_ids in list(
        order_units.items()
    )[:5]:

        print(
            f"Order {order_id}: "
            f"{unit_ids}"
        )

    print("\nRunning GA...")

    (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness
    ) = run_ga(
        orders=orders,
        units=units,
        order_units=order_units
    )

    print("\n================================================")
    print("GA FINISHED")
    print("================================================")

    print(
        f"\nBest fitness: "
        f"{best_fitness:.2f}"
    )

    print("\nBest order solution:")
    print(best_order_solution)

    if best_simulation_result is not None:

        print(
            f"\nMakespan: "
            f"{best_simulation_result.get('makespan', 'N/A')}"
        )

    units_lookup = {
        u.unit_id: u
        for u in units
    }

    export_schedule(
        chromosome=best_order_solution,
        order_units=order_units,
        units_lookup=units_lookup,
        filename="best_schedule.csv"
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()