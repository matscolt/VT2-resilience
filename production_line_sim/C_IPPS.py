"""C_IPPS.py

Integrated Process Planning & Scheduling (IPPS) – driver script.

Goal
----
Create a 5-working-day (Mon–Fri) executable schedule from:
  1) A monthly/4-week production plan (CSV) produced by your production_planning script.
  2) Process route options (with transport times) for each variant/order.

Then evaluate candidate schedules with a deterministic simulation (no disruptions) and
optimize on KPIs (lateness/tardiness & earliness), while allowing you to swap in
multiple scheduling/optimization algorithms to test performance.

This file is intentionally structured as:
  - Data loading & normalization
  - Candidate schedule generation (multiple heuristics)
  - KPI scoring
  - Optimization loop (pluggable)
  - Reporting/export

Dependencies (expected)
-----------------------
- D_algo.py : optional – advanced optimizers / neighbor moves / metaheuristics
- E_production_line_sim.py : expected – deterministic simulation of a schedule
- process_routes.py : expected – route options with transport times

If those modules do not yet provide the referenced functions, the placeholders in
this file show the expected interfaces.

"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Optional imports – keep IPPS runnable while modules are under construction
try:
    import D_algo  # your optimization algorithms
except Exception:
    D_algo = None

try:
    import E_production_line_sim as sim  # your deterministic simulator
except Exception:
    sim = None

try:
    import process_routes  # your route definitions and transport times
except Exception:
    process_routes = None


# ==============================================================================
# Data models
# ==============================================================================

@dataclass(frozen=True)
class Order:
    order_id: str
    due_date: int          # use day index or absolute day number
    priority: int
    variant_qty: Dict[str, int]   # e.g. {"FUSE0": 10, "FUSE2": 5}
    planned_week: int      # from 4-week production_plan


@dataclass(frozen=True)
class RouteChoice:
    """A specific route option for an order (or variant) including transport times."""
    route_id: str
    # The simulator/route module can decide what data it needs.
    route_payload: dict


@dataclass
class ScheduleItem:
    """One scheduled chunk of work (can represent lot-splitting)."""
    order_id: str
    day: int               # 1..5
    sequence: int          # position in the day (for single-machine bottleneck)
    route_id: str
    # Optional: amount if you support lot splitting
    lot_fraction: float = 1.0


@dataclass
class CandidateSchedule:
    """A schedule for the 5-day horizon."""
    items: List[ScheduleItem]
    name: str = "candidate"
    meta: dict = None


@dataclass
class KPIs:
    total_tardiness: float
    num_late: int
    total_earliness: float
    makespan: float

    def score(self, alpha=1.0, beta=1000.0, gamma=-0.01) -> float:
        """Lower is better.

        alpha * sum tardiness + beta * #late + gamma * total earliness
        gamma negative means: more earliness => smaller score (i.e., better).
        Increase beta to punish late orders strongly.
        """
        return alpha * self.total_tardiness + beta * self.num_late + gamma * self.total_earliness


# ==============================================================================
# Configuration
# ==============================================================================

WORKING_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ==============================================================================
# Loading utilities
# ==============================================================================

def load_orders_from_production_plan_csv(path: Path) -> List[Order]:
    """Read production_plan.csv created by your production planning step."""
    orders: List[Order] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = str(row.get("order_id", "")).strip()
            if not oid:
                continue

            due = int(float(row.get("due date") or 0))
            prio = int(float(row.get("priority") or 0))
            planned_week = int(float(row.get("planned_week") or 1))

            vqty: Dict[str, int] = {}
            for i in range(3):
                v = (row.get(f"variant{i}") or "").strip().upper()
                q_raw = row.get(f"quantity{i}")
                if not v or q_raw in (None, "", "0"):
                    continue
                q = int(float(q_raw))
                if q > 0:
                    vqty[v] = vqty.get(v, 0) + q

            orders.append(Order(
                order_id=oid,
                due_date=due,
                priority=prio,
                variant_qty=vqty,
                planned_week=planned_week,
            ))

    return orders


def select_horizon_orders(orders: List[Order], weeks: Iterable[int] = (1,), max_days: int = 5) -> List[Order]:
    """Select the subset of orders we actually schedule now.

    Typical use: only schedule week 1, and within that only 5 working days.
    We still keep due dates as-is; the simulator will map days to time.
    """
    wset = set(int(w) for w in weeks)
    horizon = [o for o in orders if o.planned_week in wset]
    return horizon


# ==============================================================================
# Route choice helpers
# ==============================================================================

def get_route_options_for_order(order: Order) -> List[RouteChoice]:
    """Retrieve route choices for the order.

    This is a thin adapter around your process_routes module.

    Expected in process_routes (suggested):
      - process_routes.get_options(order: Order) -> list[RouteChoice-like]

    For now, fallback to a single default route.
    """
    if process_routes and hasattr(process_routes, "get_options"):
        opts = process_routes.get_options(order)
        # normalize
        out = []
        for o in opts:
            if isinstance(o, RouteChoice):
                out.append(o)
            else:
                out.append(RouteChoice(route_id=str(o.get("route_id")), route_payload=o))
        return out

    return [RouteChoice(route_id="default", route_payload={})]


def choose_route_fastest(order: Order) -> RouteChoice:
    """Pick the route with the smallest estimated processing+transport time.

    Requires process_routes to provide a estimator.
    """
    opts = get_route_options_for_order(order)
    if process_routes and hasattr(process_routes, "estimate_route_time"):
        best = min(opts, key=lambda r: process_routes.estimate_route_time(order, r))
        return best
    return opts[0]


# ==============================================================================
# Priority rules / heuristics
# ==============================================================================

def estimate_order_processing_time(order: Order, route: RouteChoice) -> float:
    """Estimate processing time for dispatching rules (SPT/ATC/etc.).

    If you have route-dependent times, hook in here.
    Otherwise, use a simple proxy: total quantity.
    """
    if process_routes and hasattr(process_routes, "estimate_order_time"):
        return float(process_routes.estimate_order_time(order, route))

    return float(sum(order.variant_qty.values()))


def build_sequence_EDD(orders: List[Order]) -> List[Order]:
    """Earliest due date first."""
    return sorted(orders, key=lambda o: (o.due_date, -o.priority))


def build_sequence_SPT(orders: List[Order]) -> List[Order]:
    """Shortest processing time first (proxy)."""
    def p(o: Order):
        r = choose_route_fastest(o)
        return estimate_order_processing_time(o, r)
    return sorted(orders, key=lambda o: (p(o), o.due_date))


def build_sequence_Slack(orders: List[Order], now_day: int = 1) -> List[Order]:
    """Smallest slack first.

    slack = due - now - p
    """
    def slack(o: Order):
        r = choose_route_fastest(o)
        p = estimate_order_processing_time(o, r)
        return (o.due_date - now_day - p)
    return sorted(orders, key=lambda o: (slack(o), o.due_date))


def build_sequence_ATC(orders: List[Order], now_day: int = 1, k: float = 3.0) -> List[Order]:
    """Apparent Tardiness Cost (ATC) dispatch rule.

    priority_i = (w/p) * exp( - max(0, slack) / (k * p_bar) )
    """
    routes = {o.order_id: choose_route_fastest(o) for o in orders}
    p_times = {o.order_id: max(1e-9, estimate_order_processing_time(o, routes[o.order_id])) for o in orders}
    p_bar = sum(p_times.values()) / max(1, len(p_times))

    def atc_value(o: Order):
        p = p_times[o.order_id]
        slack = max(0.0, (o.due_date - now_day - p))
        w = max(1.0, float(o.priority) if o.priority is not None else 1.0)
        return (w / p) * math.exp(-slack / (k * p_bar))

    # Higher ATC priority first
    return sorted(orders, key=lambda o: (-atc_value(o), o.due_date))


# ==============================================================================
# Schedule construction for 5 days
# ==============================================================================

def assigndays_round_robin(seq: List[Order], days: int = 5) -> Dict[str, int]:
    """Assign a day number to each order by round-robin, just to generate a starting guess."""
    out = {}
    d = 1
    for o in seq:
        out[o.order_id] = d
        d = d + 1 if d < days else 1
    return out


def build_candidate_schedule(
    orders: List[Order],
    sequencing_rule: Callable[[List[Order]], List[Order]],
    days: int = 5,
    route_selector: Callable[[Order], RouteChoice] = choose_route_fastest,
    name: str = "heuristic",
) -> CandidateSchedule:
    """Build a schedule candidate from a sequencing rule.

    This is a simple baseline: one sequence split across 5 days (round robin).
    Replace with capacity-aware day packing once your daily simulator is stable.
    """
    seq = sequencing_rule(orders)
    day_by_order = assigndays_round_robin(seq, days=days)

    items: List[ScheduleItem] = []
    # For each day, preserve the sequence order
    day_sequences = {d: 0 for d in range(1, days + 1)}
    for o in seq:
        d = day_by_order[o.order_id]
        day_sequences[d] += 1
        r = route_selector(o)
        items.append(ScheduleItem(order_id=o.order_id, day=d, sequence=day_sequences[d], route_id=r.route_id))

    return CandidateSchedule(items=items, name=name, meta={"rule": sequencing_rule.__name__})


# ==============================================================================
# Simulation + KPI evaluation
# ==============================================================================

def simulate_candidate(schedule: CandidateSchedule, orders: List[Order], routes: Dict[str, RouteChoice]) -> Dict[str, dict]:
    """Run deterministic simulation. Returns per-order results.

    Expected output per order_id (suggested):
      {
        "completion_day": <float>,
        "completion_time": <float>,
        "start_time": <float>,
      }

    If your E_production_line_sim exposes a different interface, adapt here.
    """
    if sim and hasattr(sim, "simulate_deterministic"):
        return sim.simulate_deterministic(schedule, orders, routes)

    # Fallback stub: completion_day = scheduled day
    res = {}
    for it in schedule.items:
        res[it.order_id] = {"completion_day": float(it.day), "completion_time": float(it.day)}
    return res


def compute_kpis(results: Dict[str, dict], orders: List[Order]) -> KPIs:
    """Compute lateness/tardiness & earliness KPIs from simulation results."""
    due = {o.order_id: o.due_date for o in orders}

    total_tardiness = 0.0
    total_earliness = 0.0
    num_late = 0
    makespan = 0.0

    for oid, r in results.items():
        c = float(r.get("completion_day", r.get("completion_time", 0.0)))
        d = float(due.get(oid, 0))
        makespan = max(makespan, c)

        tard = max(0.0, c - d)
        earl = max(0.0, d - c)
        total_tardiness += tard
        total_earliness += earl
        if tard > 0:
            num_late += 1

    return KPIs(
        total_tardiness=total_tardiness,
        num_late=num_late,
        total_earliness=total_earliness,
        makespan=makespan,
    )


# ==============================================================================
# Optimization loop (pluggable)
# ==============================================================================

def neighbor_swap_two_items(candidate: CandidateSchedule) -> CandidateSchedule:
    """Simple neighbor: swap two random items' sequence within the same day."""
    import random
    new = CandidateSchedule(items=[ScheduleItem(**it.__dict__) for it in candidate.items], name=candidate.name, meta=dict(candidate.meta or {}))

    by_day = {}
    for idx, it in enumerate(new.items):
        by_day.setdefault(it.day, []).append(idx)

    days = [d for d, idxs in by_day.items() if len(idxs) >= 2]
    if not days:
        return new

    d = random.choice(days)
    i1, i2 = random.sample(by_day[d], 2)

    # swap sequence numbers
    new.items[i1].sequence, new.items[i2].sequence = new.items[i2].sequence, new.items[i1].sequence
    return new


def optimize(
    initial: CandidateSchedule,
    orders: List[Order],
    routes: Dict[str, RouteChoice],
    objective: Callable[[KPIs], float],
    max_iter: int = 500,
    patience: int = 100,
) -> Tuple[CandidateSchedule, KPIs]:
    """Simple hill-climbing with random neighbor swaps.

    Replace with D_algo methods when you have them.
    """
    best = initial

    best_res = simulate_candidate(best, orders, routes)
    best_kpi = compute_kpis(best_res, orders)
    best_score = objective(best_kpi)

    no_improve = 0

    for it in range(1, max_iter + 1):
        cand = neighbor_swap_two_items(best)

        res = simulate_candidate(cand, orders, routes)
        kpi = compute_kpis(res, orders)
        score = objective(kpi)

        if score < best_score:
            best, best_kpi, best_score = cand, kpi, score
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best, best_kpi


# ==============================================================================
# Reporting / export
# ==============================================================================

def print_orders_by_weekday(schedule: CandidateSchedule):
    """Print order IDs produced in each scheduled day."""
    by_day = {d: [] for d in range(1, 6)}
    for it in sorted(schedule.items, key=lambda x: (x.day, x.sequence)):
        by_day[it.day].append(it.order_id)

    print("\norder_id produced in each day")
    for d in range(1, 6):
        ids = by_day[d]
        label = WORKING_DAYS[d - 1]
        print(f"day {d} ({label}): {', '.join(ids) if ids else '(none)'}")


def export_schedule_csv(schedule: CandidateSchedule, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "day", "sequence", "route_id", "lot_fraction"])
        for it in sorted(schedule.items, key=lambda x: (x.day, x.sequence)):
            w.writerow([it.order_id, it.day, it.sequence, it.route_id, it.lot_fraction])


# ==============================================================================
# Main entry
# ==============================================================================

def run_ipps(
    input_dir: Path,
    horizon_weeks: Iterable[int] = (1,),
    days: int = 5,
    max_iter: int = 500,
):
    """Run IPPS for the selected planning horizon."""
    input_dir = Path(input_dir)

    production_plan_csv = input_dir / "production_plan.csv"
    if not production_plan_csv.exists():
        raise FileNotFoundError(f"production_plan.csv not found: {production_plan_csv}")

    orders_all = load_orders_from_production_plan_csv(production_plan_csv)
    orders = select_horizon_orders(orders_all, weeks=horizon_weeks, max_days=days)

    # Build route choices map (choose one route per order for now)
    routes = {o.order_id: choose_route_fastest(o) for o in orders}

    # Generate several heuristic candidates to compare
    candidates = [
        build_candidate_schedule(orders, build_sequence_EDD, days=days, name="EDD"),
        build_candidate_schedule(orders, build_sequence_SPT, days=days, name="SPT"),
        build_candidate_schedule(orders, lambda xs: build_sequence_Slack(xs, now_day=1), days=days, name="Slack"),
        build_candidate_schedule(orders, lambda xs: build_sequence_ATC(xs, now_day=1, k=3.0), days=days, name="ATC"),
    ]

    # Objective: punish late orders heavily, reward earliness lightly
    objective = lambda k: k.score(alpha=1.0, beta=2000.0, gamma=-0.01)

    # Evaluate candidates
    scored = []
    for c in candidates:
        res = simulate_candidate(c, orders, routes)
        k = compute_kpis(res, orders)
        scored.append((objective(k), c, k))

    scored.sort(key=lambda t: t[0])
    best0_score, best0, best0_kpi = scored[0]

    print("\n=== Heuristic comparison (lower score is better) ===")
    for score, cand, kpi in scored:
        print(f"{cand.name:>4} | score={score:,.2f} | tard={kpi.total_tardiness:.2f} | late={kpi.num_late} | earl={kpi.total_earliness:.2f} | makespan={kpi.makespan:.2f}")

    # Optimization: refine best heuristic
    if D_algo and hasattr(D_algo, "optimize"):
        # If you implement D_algo.optimize(initial, orders, routes, objective, ...)
        best, best_kpi = D_algo.optimize(best0, orders, routes, objective, max_iter=max_iter)
    else:
        best, best_kpi = optimize(best0, orders, routes, objective, max_iter=max_iter, patience=max(50, max_iter // 5))

    print("\n=== Best schedule after optimization ===")
    print(f"name={best.name} | tard={best_kpi.total_tardiness:.2f} | late={best_kpi.num_late} | earl={best_kpi.total_earliness:.2f} | makespan={best_kpi.makespan:.2f}")

    print_orders_by_weekday(best)

    out_csv = input_dir / "ipps_schedule_5days.csv"
    export_schedule_csv(best, out_csv)
    print(f"\nWrote: {out_csv}")


def main(order_dir = None):
    # Example: point to an 'orders run' folder containing production_plan.csv
    # In your pipeline, call run_ipps(...) from your top-level main.py
    if order_dir == None:
        ordername = r"\\orders_07-05_15-29_1"
        order_dir = r"C:\\Users\\mathi\\OneDrive\\Dokumenter\\1.UNI\\8. semester\\Projekt\\Github\\VT2-resilience\\production_line_sim\\input" + ordername
    
    run_ipps(order_dir, horizon_weeks=(1,), days=5, max_iter=500)


if __name__ == "__main__":
    main()
