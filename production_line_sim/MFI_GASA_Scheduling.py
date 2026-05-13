import random
import copy
import json
import re
import contextlib
import io
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List

import pandas as pd

import E_production_line_sim_schedule_multi_outputs_final_input_fix as simulator


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
# HYBRID GA-SA SETTINGS
# ============================================================

RANDOM_SEED = None

# GA settings
SWAPS = 3
POPULATION_SIZE = 6
GA_GENERATIONS = 4
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.4

# SA settings
SA_ITERATIONS = 20
INITIAL_TEMPERATURE = 5000.0
FINAL_TEMPERATURE = 1.0
COOLING_RATE = 0.90
MOVES_PER_NEIGHBOR = 1


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
# INITIAL SEED - PURE EDD
# ============================================================

def create_order_seed(orders: List[Order]):
    order_scores = []

    for order in orders:
        order_scores.append({
            "order_id": order.order_id,
            "due_date": order.due_date,
            "priority": order.priority,
            "total_units": order.total_units
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
            f"Units={row['total_units']}"
        )

    print("================================================\n")

    return seed


# ============================================================
# GA POPULATION
# ============================================================

def create_order_population(orders: List[Order]):
    seed = create_order_seed(orders)

    population = [seed]

    print("\n================================================")
    print("CREATING INITIAL GA POPULATION")
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
    print(f"TOTAL GA POPULATION CREATED: {len(population)}")
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
    filename,
    verbose=False
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

    if verbose:
        print(f"Saved schedule: {output_path}")

    return output_path


# ============================================================
# SUMMARY READING
# ============================================================

def find_summary_folder(
    generation,
    chromosome_index
):
    output_root = ROOT / "output" / INPUT_DIR.name

    base_name = (
        f"Iter_{generation}_schedule_"
        f"{chromosome_index}_summary"
    )

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
    filename = (
        f"Iter_{generation}_schedule_"
        f"{chromosome_index}.csv"
    )

    export_schedule(
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

    return {
        "fitness": fitness,
        "total_tardiness": total_tardiness,
        "late_orders": late_orders,
        "total_earliness": total_earliness
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
# SA OPERATORS
# ============================================================

def create_neighbor_solution(chromosome):
    neighbor = copy.deepcopy(chromosome)

    for _ in range(MOVES_PER_NEIGHBOR):
        move_type = random.choice([
            "swap",
            "insert"
        ])

        i = random.randint(0, len(neighbor) - 1)
        j = random.randint(0, len(neighbor) - 1)

        if move_type == "swap":
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        elif move_type == "insert":
            gene = neighbor.pop(i)
            neighbor.insert(j, gene)

    return neighbor


def accept_solution(
    current_fitness,
    candidate_fitness,
    temperature
):
    if candidate_fitness < current_fitness:
        return True

    if temperature <= 0:
        return False

    delta = candidate_fitness - current_fitness

    acceptance_probability = math.exp(
        -delta / temperature
    )

    return random.random() < acceptance_probability


# ============================================================
# GA PHASE
# ============================================================

def run_ga_phase(
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
    best_summary_folder = None

    best_fitness = float("inf")

    for generation in range(GA_GENERATIONS):
        generation_number = generation + 1

        print("\n================================================")
        print(f"GA GENERATION {generation_number}/{GA_GENERATIONS}")
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

            best_marker = " <-- NEW GA BEST" if is_new_global_best else ""

            print(
                f"GA Gen {generation_number:02d} | "
                f"Chrom {chromosome_index:02d} | "
                f"Fitness {fitness:12.2f} | "
                f"Late {fitness_result['late_orders']:2d} | "
                f"Tard {fitness_result['total_tardiness']:10.1f} | "
                f"Early {fitness_result['total_earliness']:10.1f} | "
                f"Makespan {simulation_result['makespan']:10.1f}"
                f"{best_marker}"
            )

        print(
            f"\n>>> GA generation {generation_number} done | "
            f"Generation best chromosome: {generation_best_chromosome} | "
            f"Generation best fitness: {generation_best_fitness:.2f} | "
            f"GA global best fitness: {best_fitness:.2f}"
        )

        print(
            f"Best order sequence in GA generation {generation_number}: "
            f"{generation_best_order_sequence}"
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
        best_fitness,
        best_summary_folder
    )


# ============================================================
# SA PHASE
# ============================================================

def run_sa_phase(
    start_solution,
    start_fitness,
    start_simulation_result,
    start_summary_folder,
    orders,
    units,
    order_units
):
    units_lookup = {
        u.unit_id: u
        for u in units
    }

    current_solution = copy.deepcopy(
        start_solution
    )

    current_fitness = start_fitness
    current_simulation_result = copy.deepcopy(
        start_simulation_result
    )
    current_summary_folder = start_summary_folder

    best_order_solution = copy.deepcopy(
        start_solution
    )

    best_unit_sequence = order_chromosome_to_unit_sequence(
        best_order_solution,
        order_units
    )

    best_simulation_result = copy.deepcopy(
        start_simulation_result
    )

    best_summary_folder = start_summary_folder
    best_fitness = start_fitness

    temperature = INITIAL_TEMPERATURE

    print("\n================================================")
    print("STARTING SA PHASE FROM BEST GA SOLUTION")
    print("================================================")

    print(f"Initial SA fitness from GA: {current_fitness:.2f}")
    print("Initial SA order solution:")
    print(current_solution)

    for iteration in range(1, SA_ITERATIONS + 1):
        print("\n================================================")
        print(
            f"SA ITERATION {iteration}/{SA_ITERATIONS} | "
            f"Temperature={temperature:.4f}"
        )
        print("================================================")

        candidate_solution = create_neighbor_solution(
            current_solution
        )

        # Generation index offset makes GA and SA result folders unique.
        sa_generation_number = GA_GENERATIONS + iteration

        candidate_simulation_result, candidate_summary_folder = evaluate_schedule_with_simulator(
            chromosome=candidate_solution,
            order_units=order_units,
            units_lookup=units_lookup,
            chromosome_index=1,
            generation=sa_generation_number
        )

        candidate_fitness_result = calculate_fitness(
            candidate_simulation_result,
            orders
        )

        candidate_fitness = candidate_fitness_result["fitness"]

        accepted = accept_solution(
            current_fitness=current_fitness,
            candidate_fitness=candidate_fitness,
            temperature=temperature
        )

        if accepted:
            current_solution = copy.deepcopy(
                candidate_solution
            )

            current_fitness = candidate_fitness

            current_simulation_result = copy.deepcopy(
                candidate_simulation_result
            )

            current_summary_folder = candidate_summary_folder

        is_new_global_best = candidate_fitness < best_fitness

        if is_new_global_best:
            best_fitness = candidate_fitness

            best_order_solution = copy.deepcopy(
                candidate_solution
            )

            best_unit_sequence = order_chromosome_to_unit_sequence(
                best_order_solution,
                order_units
            )

            best_simulation_result = copy.deepcopy(
                candidate_simulation_result
            )

            best_summary_folder = candidate_summary_folder

        accepted_text = "ACCEPTED" if accepted else "REJECTED"
        best_marker = " <-- NEW HYBRID BEST" if is_new_global_best else ""

        print(
            f"SA Iter {iteration:03d} | "
            f"Candidate fitness {candidate_fitness:12.2f} | "
            f"Current fitness {current_fitness:12.2f} | "
            f"Best fitness {best_fitness:12.2f} | "
            f"Late {candidate_fitness_result['late_orders']:2d} | "
            f"Tard {candidate_fitness_result['total_tardiness']:10.1f} | "
            f"Early {candidate_fitness_result['total_earliness']:10.1f} | "
            f"Makespan {candidate_simulation_result['makespan']:10.1f} | "
            f"{accepted_text}"
            f"{best_marker}"
        )

        print("Current SA order solution:")
        print(current_solution)

        temperature = max(
            FINAL_TEMPERATURE,
            temperature * COOLING_RATE
        )

        if temperature <= FINAL_TEMPERATURE:
            print("\nFinal SA temperature reached.")
            break

    return (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness,
        best_summary_folder
    )


# ============================================================
# HYBRID GA-SA LOOP
# ============================================================

def run_gasa(
    orders,
    units,
    order_units
):
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print("\n================================================")
    print("STARTING HYBRID GA-SA OPTIMIZATION")
    print("================================================")

    (
        ga_best_order_solution,
        ga_best_unit_sequence,
        ga_best_simulation_result,
        ga_best_fitness,
        ga_best_summary_folder
    ) = run_ga_phase(
        orders=orders,
        units=units,
        order_units=order_units
    )

    print("\n================================================")
    print("GA PHASE FINISHED")
    print("================================================")
    print(f"Best GA fitness: {ga_best_fitness:.2f}")
    print("Best GA order solution:")
    print(ga_best_order_solution)

    (
        hybrid_best_order_solution,
        hybrid_best_unit_sequence,
        hybrid_best_simulation_result,
        hybrid_best_fitness,
        hybrid_best_summary_folder
    ) = run_sa_phase(
        start_solution=ga_best_order_solution,
        start_fitness=ga_best_fitness,
        start_simulation_result=ga_best_simulation_result,
        start_summary_folder=ga_best_summary_folder,
        orders=orders,
        units=units,
        order_units=order_units
    )

    return (
        hybrid_best_order_solution,
        hybrid_best_unit_sequence,
        hybrid_best_simulation_result,
        hybrid_best_fitness,
        hybrid_best_summary_folder
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n================================================")
    print("STARTING GASA HYBRID SCHEDULER")
    print("================================================")

    print("\nROOT:")
    print(ROOT)

    print("\nInput folder:")
    print(INPUT_DIR)

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

    orders, units, order_units = load_orders_and_units_from_file(
        production_df
    )

    print("\nExample order to units:")

    for order_id, unit_ids in list(order_units.items())[:5]:
        print(
            f"Order {order_id}: "
            f"{unit_ids}"
        )

    print("\nRunning GASA hybrid optimization...")

    (
        best_order_solution,
        best_unit_sequence,
        best_simulation_result,
        best_fitness,
        best_summary_folder
    ) = run_gasa(
        orders=orders,
        units=units,
        order_units=order_units
    )

    print("\n================================================")
    print("GASA HYBRID FINISHED")
    print("================================================")

    print(f"\nBest hybrid fitness: {best_fitness:.2f}")

    print("\nBest hybrid order solution:")
    print(best_order_solution)

    if best_unit_sequence is not None:
        print("\nBest hybrid unit sequence:")
        print(best_unit_sequence)

    if best_simulation_result is not None:
        print(
            f"\nBest hybrid makespan: "
            f"{best_simulation_result.get('makespan', 'N/A')}"
        )

    if best_summary_folder is not None:
        print("\nBest hybrid summary folder:")
        print(best_summary_folder)

    units_lookup = {
        u.unit_id: u
        for u in units
    }

    export_schedule(
        chromosome=best_order_solution,
        order_units=order_units,
        units_lookup=units_lookup,
        filename="best_schedule_gasa.csv",
        verbose=True
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()