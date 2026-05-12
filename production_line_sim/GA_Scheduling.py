import random
import copy
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

import E_production_line_sim_trimmed


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
            f"No generated input folders were found in {input_root}. "
            "Expected folders like orders_DD-MM_HH-MM_N"
        )

    return max(candidate_dirs, key=lambda path: _parse_input_batch_sort_key(path.name))


INPUT_DIR = find_newest_input_batch_dir(ROOT / "input")

LAYOUT_PATH = (
    ROOT / "data" / "Layouts" / "line_layout_robotcell_6staggered_example.json"
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
# CAPACITY PLANNING SETTINGS
# ============================================================

PLANNING_DAYS = 5
DAILY_CAPACITY_SECONDS = 8 * 60 * 60
CAPACITY_BUFFER_FACTOR = 0.90


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

    raise ValueError(f"Unsupported production plan file type: {path.suffix}")


# ============================================================
# LOAD ORDERS AND UNITS FROM FILE
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
                order_units[order_id].append(internal_unit_id)

                print(f"    Created Unit: {internal_unit_id}")

                unit_counter += 1
                total_units += 1

        order = Order(
            order_id=order_id,
            due_date=due_date,
            priority=priority,
            total_units=total_units
        )

        orders.append(order)

        print(f"  Total units for order {order_id}: {total_units}")

    print("\n================================================")
    print(f"TOTAL ORDERS LOADED: {len(orders)}")
    print(f"TOTAL UNITS LOADED: {len(units)}")
    print("================================================\n")

    return orders, units, order_units


# ============================================================
# TRANSPORT LOOKUP
# ============================================================

def build_transport_lookup(transport_data: dict) -> Dict[tuple, float]:
    raw = transport_data["transport_times_between_consecutive_stations"]

    lookup = {}

    for key, value in raw.items():
        left, right = key.split(" -> ")
        lookup[(left, right)] = float(value)

    print("\n================================================")
    print("TRANSPORT LOOKUP TABLE")
    print("================================================")

    for (left, right), time in lookup.items():
        print(f"{left}  -->  {right}  =  {time} sec")

    print("================================================\n")

    return lookup


# ============================================================
# ESTIMATE ORDER PROCESSING TIME
# ============================================================

def estimate_order_processing_time(
    order: Order,
    order_units: Dict[int, List[str]],
    units_lookup: Dict[str, Unit],
    process_data: dict
):
    station_sequence = process_data["station_sequence"]
    process_times = process_data["process_times"]

    total_time = 0.0

    for unit_id in order_units[order.order_id]:
        unit = units_lookup[unit_id]

        for station in station_sequence:
            total_time += process_times[unit.variant][station]

    return total_time


# ============================================================
# ORDER-BASED INITIAL SEED
# ============================================================

def create_order_seed(
    orders: List[Order],
    order_units: Dict[int, List[str]],
    units_lookup: Dict[str, Unit],
    process_data: dict
):
    order_scores = []

    for order in orders:
        estimated_processing_time = estimate_order_processing_time(
            order=order,
            order_units=order_units,
            units_lookup=units_lookup,
            process_data=process_data
        )

        time_until_due = order.due_date
        priority_weight = 1 + order.priority

        score = time_until_due / (estimated_processing_time * priority_weight)

        order_scores.append({
            "order_id": order.order_id,
            "due_date": order.due_date,
            "priority": order.priority,
            "estimated_processing_time": estimated_processing_time,
            "score": score
        })

    order_scores.sort(key=lambda x: x["score"])

    seed = [row["order_id"] for row in order_scores]

    print("\n================================================")
    print("ORDER SEED USING PRIORITIZED CRITICAL RATIO")
    print("================================================")

    for row in order_scores:
        print(
            f"Order {row['order_id']} | "
            f"Due={row['due_date']:.1f} | "
            f"Priority={row['priority']} | "
            f"ProcTime={row['estimated_processing_time']:.1f} | "
            f"CR Score={row['score']:.3f}"
        )

    print("================================================\n")

    return seed


def create_order_population(
    orders: List[Order],
    order_units: Dict[int, List[str]],
    units_lookup: Dict[str, Unit],
    process_data: dict
):
    seed = create_order_seed(
        orders=orders,
        order_units=order_units,
        units_lookup=units_lookup,
        process_data=process_data
    )

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
# CONVERT ORDER CHROMOSOME TO UNIT SEQUENCE
# ============================================================

def order_chromosome_to_unit_sequence(
    order_chromosome: List[int],
    order_units: Dict[int, List[str]]
):
    unit_sequence = []

    for order_id in order_chromosome:
        unit_sequence.extend(order_units[order_id])

    return unit_sequence


# ============================================================
# CONVERT CHROMOSOME TO UNIT DATAFRAME
# ============================================================

def chromosome_to_unit_dataframe(
    chromosome: List[int],
    order_units: Dict[int, List[str]],
    units_lookup: Dict[str, Unit],
    route_id: int = 0
) -> pd.DataFrame:

    unit_sequence = order_chromosome_to_unit_sequence(
        order_chromosome=chromosome,
        order_units=order_units
    )

    rows = []

    for unit_seq, internal_unit_id in enumerate(unit_sequence, start=1):
        unit = units_lookup[internal_unit_id]

        local_unit_id = int(unit.unit_id.split(".")[1])

        rows.append({
            "unit_seq": unit_seq,
            "order_id": unit.order_id,
            "unit_id": local_unit_id,
            "variant": unit.variant,
            "route_id": route_id
        })

    return pd.DataFrame(rows)


# ============================================================
# EXPORT ALL CHROMOSOMES AS UNIT SCHEDULES
# ============================================================

def export_population_as_unit_schedules(
    population: List[List[int]],
    order_units: Dict[int, List[str]],
    units_lookup: Dict[str, Unit],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for chromosome_number, chromosome in enumerate(population, start=1):
        unit_df = chromosome_to_unit_dataframe(
            chromosome=chromosome,
            order_units=order_units,
            units_lookup=units_lookup,
            route_id=0
        )

        output_path = output_dir / f"Iter_{1}_schedule_{chromosome_number}.csv"

        unit_df.to_csv(output_path, index=False)

        print(f"Saved chromosome {chromosome_number}: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    process_data = load_json(PROCESS_TIMES_PATH)
    transport_data = load_json(TRANSPORT_TIMES_PATH)

    transport_lookup = build_transport_lookup(transport_data)

    df = load_production_plan(INPUT_DIR)

    orders, units, order_units = load_orders_and_units_from_file(df)

    units_lookup = {
        unit.unit_id: unit
        for unit in units
    }

    population = create_order_population(
        orders=orders,
        order_units=order_units,
        units_lookup=units_lookup,
        process_data=process_data
    )

    output_dir = INPUT_DIR / "output_chromosomes"

    export_population_as_unit_schedules(
        population=population,
        order_units=order_units,
        units_lookup=units_lookup,
        output_dir=output_dir
    )

    print("\n================================================")
    print("ALL CHROMOSOMES EXPORTED AS UNIT SCHEDULES")
    print("================================================")


if __name__ == "__main__":
    main()


# ============================================================
# INSERT YOUR SIMULATOR HERE
# ============================================================

#E_production_line_sim_trimmed.main() #Kør simulering
#Export chromosom som .csv - skal være for hvert enkelt
#


# ============================================================
# FITNESS
# ============================================================

def calculate_fitness(simulation_result, orders: List[Order]):
    order_info = {
        o.order_id: {
            "due_date": o.due_date,
            "priority": o.priority
        }
        for o in orders
    }

    total_tardiness = 0.0
    total_earliness = 0.0
    late_orders = 0

    for order_id, completion in simulation_result["order_completion_times"].items():
        due = order_info[order_id]["due_date"]
        priority = order_info[order_id]["priority"]

        tardiness = max(0.0, completion - due)
        earliness = max(0.0, due - completion)

        if tardiness > 0:
            late_orders += 1

        total_tardiness += tardiness * (priority + 1)
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

def tournament_selection(population, fitnesses):
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

    fill = [g for g in p2 if g not in child]

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
    order_units,
    layout_data,
    process_data,
    transport_data,
    transport_lookup
):
    units_lookup = {
        u.unit_id: u for u in units
    }

    population = create_order_population(
        orders=orders,
        order_units=order_units,
        units_lookup=units_lookup,
        process_data=process_data
    )

    best_order_solution = None
    best_unit_sequence = None
    best_simulation_result = None
    best_fitness = float("inf")

    for generation in range(GENERATIONS):
        fitnesses = []

        for order_chromosome in population:
            unit_sequence = order_chromosome_to_unit_sequence(
                order_chromosome,
                order_units
            )

            simulation_result = evaluate_schedule_with_simulator(
                unit_sequence=unit_sequence,
                units_lookup=units_lookup,
                layout_data=layout_data,
                process_data=process_data,
                transport_data=transport_data,
                transport_lookup=transport_lookup
            )

            fitness = calculate_fitness(
                simulation_result,
                orders
            )

            fitnesses.append(fitness)

            if fitness < best_fitness:
                best_fitness = fitness
                best_order_solution = copy.deepcopy(order_chromosome)
                best_unit_sequence = copy.deepcopy(unit_sequence)
                best_simulation_result = simulation_result

        print(
            f"Generation {generation + 1}/{GENERATIONS} | "
            f"Best fitness: {best_fitness:.2f}"
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
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)

            if random.random() < CROSSOVER_RATE:
                child = order_crossover(p1, p2)
            else:
                child = copy.deepcopy(p1)

            if random.random() < MUTATION_RATE:
                child = insert_mutation(child)

            new_population.append(child)

        population = new_population

    return (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness
    )


# ============================================================
# EXPORT
# ============================================================

def export_order_schedule(order_chromosome, filename="best_order_schedule.csv"):
    rows = []

    for sequence, order_id in enumerate(order_chromosome):
        rows.append({
            "sequence": sequence,
            "order_id": order_id
        })

    output_path = ROOT / filename

    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Exported order schedule to: {output_path}")


def export_unit_schedule(
    unit_sequence,
    units_lookup,
    filename="best_unit_schedule.csv"
):
    rows = []

    for sequence, unit_id in enumerate(unit_sequence):
        u = units_lookup[unit_id]

        rows.append({
            "sequence": sequence,
            "unit_id": unit_id,
            "order_id": u.order_id,
            "variant": u.variant,
            "due_date": u.due_date,
            "priority": u.priority
        })

    output_path = ROOT / filename

    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Exported unit schedule to: {output_path}")


def export_daily_schedule(
    daily_schedule,
    orders_lookup,
    filename="daily_order_schedule.csv"
):
    rows = []

    for day, data in daily_schedule.items():
        for sequence, order_id in enumerate(data["orders"]):
            order = orders_lookup[order_id]

            rows.append({
                "day": day,
                "sequence_on_day": sequence,
                "order_id": order_id,
                "due_date": order.due_date,
                "priority": order.priority,
                "total_units": order.total_units,
                "used_capacity_day": data["used_capacity"],
                "remaining_capacity_day": data["remaining_capacity"]
            })

    output_path = ROOT / filename

    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Exported daily schedule to: {output_path}")


def export_overflow_orders(
    overflow_orders,
    orders_lookup,
    filename="overflow_orders.csv"
):
    rows = []

    for order_id in overflow_orders:
        order = orders_lookup[order_id]

        rows.append({
            "order_id": order_id,
            "due_date": order.due_date,
            "priority": order.priority,
            "total_units": order.total_units
        })

    output_path = ROOT / filename

    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Exported overflow orders to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("ROOT:")
    print(ROOT)

    print("\nLoading files...")

    production_df = load_production_plan(INPUT_DIR)
    layout_data = load_json(LAYOUT_PATH)
    process_data = load_json(PROCESS_TIMES_PATH)
    transport_data = load_json(TRANSPORT_TIMES_PATH)

    transport_lookup = build_transport_lookup(transport_data)

    orders, units, order_units = load_orders_and_units_from_file(
        production_df
    )

    print("\nExample order to units:")
    for order_id, unit_ids in list(order_units.items())[:5]:
        print(f"Order {order_id}: {unit_ids}")

    print("\nRunning order-based GA...")

    (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness
    ) = run_ga(
        orders=orders,
        units=units,
        order_units=order_units,
        layout_data=layout_data,
        process_data=process_data,
        transport_data=transport_data,
        transport_lookup=transport_lookup
    )

    orders_lookup = {
        o.order_id: o for o in orders
    }

    units_lookup = {
        u.unit_id: u for u in units
    }

    daily_schedule, overflow_orders = create_daily_order_schedule(
        order_sequence=best_order_solution,
        orders_lookup=orders_lookup,
        order_units=order_units,
        units_lookup=units_lookup,
        process_data=process_data,
        planning_days=PLANNING_DAYS,
        daily_capacity_seconds=DAILY_CAPACITY_SECONDS,
        buffer_factor=CAPACITY_BUFFER_FACTOR
    )

    export_order_schedule(best_order_solution)

    export_daily_schedule(
        daily_schedule=daily_schedule,
        orders_lookup=orders_lookup
    )

    export_unit_schedule(
        best_unit_sequence,
        units_lookup
    )

    export_overflow_orders(
        overflow_orders=overflow_orders,
        orders_lookup=orders_lookup
    )

    print("\nFinished.")
    print(f"Best fitness: {best_fitness:.2f}")

    if best_simulation_result is not None:
        print(f"Makespan: {best_simulation_result.get('makespan', 'N/A')}")

    if overflow_orders:
        print(f"Overflow orders: {overflow_orders}")


if __name__ == "__main__":
    main()