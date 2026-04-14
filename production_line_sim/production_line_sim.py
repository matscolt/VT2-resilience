from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    MATPLOTLIB_AVAILABLE = False



@dataclass
class OperationRecord:
    unit_id: str
    variant: str
    station_index: int
    station_name: str
    arrival_time_s: float
    start_time_s: float
    finish_time_s: float
    process_time_s: float
    wait_time_s: float
    queue_length_on_arrival: int


@dataclass
class UnitSummary:
    unit_id: str
    variant: str
    first_arrival_time_s: float
    start_time_s: float
    completion_time_s: float
    flow_time_s: float


@dataclass
class StationSummary:
    station_index: int
    station_name: str
    busy_time_s: float
    first_start_time_s: float | None
    last_finish_time_s: float | None
    max_queue_length: int
    average_queue_length: float
    average_wait_time_s: float
    total_wait_time_s: float
    utilization_overall: float
    utilization_active_window: float


@dataclass
class StationState:
    queue: deque[tuple[int, float, int]]
    busy: bool = False
    current_unit_index: int | None = None
    current_operation: OperationRecord | None = None
    busy_time_s: float = 0.0
    first_start_time_s: float | None = None
    last_finish_time_s: float | None = None
    max_queue_length: int = 0
    total_wait_time_s: float = 0.0
    queue_area: float = 0.0
    last_queue_change_time_s: float = 0.0


EVENT_FINISH = "finish"
EVENT_ARRIVAL = "arrival"
EVENT_PRIORITY = {EVENT_FINISH: 0, EVENT_ARRIVAL: 1}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_order(order_text: str, valid_variants: set[str]) -> list[str]:
    """
    Accepts formats such as:
      3xFUSE2, 2xFUSE1, 4xFUSE0
      3 X FUSE2 and 2 X FUSE1 and 4 X FUSE0
    """
    matches = re.findall(r"(\d+)\s*[xX]\s*([A-Za-z0-9_\-]+)", order_text)
    if not matches:
        raise ValueError(
            "Could not read the order string. Example: 3xFUSE2, 2xFUSE1, 4xFUSE0"
        )

    units: list[str] = []
    for qty_text, variant_text in matches:
        qty = int(qty_text)
        variant = variant_text.upper()
        if variant not in valid_variants:
            raise ValueError(
                f"Unknown variant '{variant_text}'. Valid options: {', '.join(sorted(valid_variants))}"
            )
        units.extend([variant] * qty)
    return units


def build_transport_lookup(
    station_sequence: list[str], transport_data: dict[str, Any]
) -> dict[tuple[str, str], float]:
    raw = transport_data["transport_times_between_consecutive_stations"]
    lookup: dict[tuple[str, str], float] = {}

    for key, value in raw.items():
        if "->" not in key:
            raise ValueError(
                f"Transport key '{key}' must be written as 'Station A -> Station B'."
            )
        from_station, to_station = [part.strip() for part in key.split("->", maxsplit=1)]
        lookup[(from_station, to_station)] = float(value)

    for i in range(len(station_sequence) - 1):
        pair = (station_sequence[i], station_sequence[i + 1])
        if pair not in lookup:
            raise ValueError(f"Missing transport time for {pair[0]} -> {pair[1]}")

    return lookup


def calculate_material_requirements(
    ordered_units: list[str], bom_data: dict[str, Any]
) -> dict[str, int]:
    bom = bom_data["bom_units_per_phone"]
    requirements: defaultdict[str, int] = defaultdict(int)

    for variant in ordered_units:
        if variant not in bom:
            raise ValueError(f"Variant '{variant}' is missing from bom.json")
        for material, qty in bom[variant].items():
            requirements[material] += int(qty)

    return dict(requirements)


def check_materials(
    requirements: dict[str, int], material_stock_data: dict[str, Any]
) -> tuple[bool, dict[str, dict[str, int]]]:
    stock = material_stock_data["materials_in_stock"]
    report: dict[str, dict[str, int]] = {}
    feasible = True

    all_materials = set(stock.keys()) | set(requirements.keys())
    for material in sorted(all_materials):
        needed = int(requirements.get(material, 0))
        available = int(stock.get(material, 0))
        remaining = available - needed
        if remaining < 0:
            feasible = False
        report[material] = {
            "needed": needed,
            "available": available,
            "remaining_after_order": remaining,
        }

    return feasible, report


def _update_queue_area(station_state: StationState, current_time_s: float) -> None:
    delta_t = current_time_s - station_state.last_queue_change_time_s
    if delta_t < 0:
        raise ValueError("Queue statistics received a negative time step.")
    station_state.queue_area += len(station_state.queue) * delta_t
    station_state.last_queue_change_time_s = current_time_s


def run_simulation(
    ordered_units: list[str],
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any],
) -> tuple[list[OperationRecord], list[UnitSummary], list[StationSummary], dict[str, float]]:
    station_sequence = process_time_data["station_sequence"]
    process_times = process_time_data["process_times"]
    transport_lookup = build_transport_lookup(station_sequence, transport_time_data)

    station_states: list[StationState] = [StationState(queue=deque()) for _ in station_sequence]
    operations: list[OperationRecord] = []
    operation_lookup: dict[tuple[int, int], OperationRecord] = {}

    unit_first_arrival: dict[int, float] = {}
    unit_first_start: dict[int, float] = {}
    unit_completion: dict[int, float] = {}

    event_queue: list[tuple[float, int, int, str, int, int]] = []
    event_sequence = 0

    def push_event(time_s: float, event_type: str, station_index: int, unit_index: int) -> None:
        nonlocal event_sequence
        heapq.heappush(
            event_queue,
            (
                time_s,
                EVENT_PRIORITY[event_type],
                event_sequence,
                event_type,
                station_index,
                unit_index,
            ),
        )
        event_sequence += 1

    def try_start_next(station_index: int, current_time_s: float) -> None:
        station_state = station_states[station_index]
        if station_state.busy or not station_state.queue:
            return

        _update_queue_area(station_state, current_time_s)
        unit_index, arrival_time_s, queue_length_ahead_on_arrival = station_state.queue.popleft()

        unit_id = f"U{unit_index + 1:03d}"
        variant = ordered_units[unit_index]
        station_name = station_sequence[station_index]
        process_time_s = float(process_times[variant][station_name])
        wait_time_s = current_time_s - arrival_time_s
        queue_length_on_arrival = queue_length_ahead_on_arrival

        operation = OperationRecord(
            unit_id=unit_id,
            variant=variant,
            station_index=station_index + 1,
            station_name=station_name,
            arrival_time_s=arrival_time_s,
            start_time_s=current_time_s,
            finish_time_s=current_time_s + process_time_s,
            process_time_s=process_time_s,
            wait_time_s=wait_time_s,
            queue_length_on_arrival=queue_length_on_arrival,
        )
        operations.append(operation)
        operation_lookup[(station_index, unit_index)] = operation

        station_state.busy = True
        station_state.current_unit_index = unit_index
        station_state.current_operation = operation
        station_state.busy_time_s += process_time_s
        station_state.total_wait_time_s += wait_time_s
        if station_state.first_start_time_s is None:
            station_state.first_start_time_s = current_time_s

        if unit_index not in unit_first_start:
            unit_first_start[unit_index] = current_time_s

        push_event(
            time_s=current_time_s + process_time_s,
            event_type=EVENT_FINISH,
            station_index=station_index,
            unit_index=unit_index,
        )

    for unit_index in range(len(ordered_units)):
        push_event(0.0, EVENT_ARRIVAL, 0, unit_index)

    while event_queue:
        time_s, _, _, event_type, station_index, unit_index = heapq.heappop(event_queue)
        station_state = station_states[station_index]

        if event_type == EVENT_ARRIVAL:
            _update_queue_area(station_state, time_s)
            queue_length_ahead_on_arrival = len(station_state.queue)
            station_state.queue.append((unit_index, time_s, queue_length_ahead_on_arrival))
            station_state.max_queue_length = max(
                station_state.max_queue_length, len(station_state.queue)
            )
            unit_first_arrival.setdefault(unit_index, time_s)
            try_start_next(station_index, time_s)
            continue

        if event_type == EVENT_FINISH:
            if station_state.current_unit_index != unit_index:
                raise RuntimeError(
                    f"Station {station_index + 1} tried to finish unit {unit_index + 1}, "
                    f"but it is currently processing {station_state.current_unit_index}."
                )

            operation = operation_lookup[(station_index, unit_index)]
            operation.finish_time_s = time_s
            station_state.last_finish_time_s = time_s
            station_state.busy = False
            station_state.current_unit_index = None
            station_state.current_operation = None

            if station_index < len(station_sequence) - 1:
                next_station_index = station_index + 1
                from_station = station_sequence[station_index]
                to_station = station_sequence[next_station_index]
                arrival_time_s = time_s + transport_lookup[(from_station, to_station)]
                push_event(arrival_time_s, EVENT_ARRIVAL, next_station_index, unit_index)
            else:
                unit_completion[unit_index] = time_s

            try_start_next(station_index, time_s)
            continue

        raise ValueError(f"Unknown event type: {event_type}")

    if len(unit_completion) != len(ordered_units):
        raise RuntimeError("Simulation ended before all ordered units were completed.")

    makespan_s = max(unit_completion.values()) if unit_completion else 0.0

    station_summaries: list[StationSummary] = []
    for station_index, station_name in enumerate(station_sequence):
        station_state = station_states[station_index]
        _update_queue_area(station_state, makespan_s)

        first_start_time_s = station_state.first_start_time_s
        last_finish_time_s = station_state.last_finish_time_s
        total_wait_time_s = station_state.total_wait_time_s
        station_operation_count = sum(
            1 for op in operations if op.station_name == station_name
        )
        average_wait_time_s = (
            total_wait_time_s / station_operation_count if station_operation_count > 0 else 0.0
        )
        average_queue_length = station_state.queue_area / makespan_s if makespan_s > 0 else 0.0
        utilization_overall = (
            station_state.busy_time_s / makespan_s if makespan_s > 0 else 0.0
        )

        if first_start_time_s is not None and last_finish_time_s is not None:
            active_window_s = last_finish_time_s - first_start_time_s
            utilization_active_window = (
                station_state.busy_time_s / active_window_s if active_window_s > 0 else 1.0
            )
        else:
            utilization_active_window = 0.0

        station_summaries.append(
            StationSummary(
                station_index=station_index + 1,
                station_name=station_name,
                busy_time_s=station_state.busy_time_s,
                first_start_time_s=first_start_time_s,
                last_finish_time_s=last_finish_time_s,
                max_queue_length=station_state.max_queue_length,
                average_queue_length=average_queue_length,
                average_wait_time_s=average_wait_time_s,
                total_wait_time_s=total_wait_time_s,
                utilization_overall=utilization_overall,
                utilization_active_window=utilization_active_window,
            )
        )

    unit_summaries: list[UnitSummary] = []
    for unit_index, variant in enumerate(ordered_units):
        start_time_s = unit_first_start[unit_index]
        completion_time_s = unit_completion[unit_index]
        first_arrival_time_s = unit_first_arrival.get(unit_index, 0.0)
        unit_summaries.append(
            UnitSummary(
                unit_id=f"U{unit_index + 1:03d}",
                variant=variant,
                first_arrival_time_s=first_arrival_time_s,
                start_time_s=start_time_s,
                completion_time_s=completion_time_s,
                flow_time_s=completion_time_s - first_arrival_time_s,
            )
        )

    summary_lookup = {
        summary.station_name: summary.busy_time_s for summary in station_summaries
    }
    return operations, unit_summaries, station_summaries, summary_lookup


def calculate_kpis(
    ordered_units: list[str],
    operations: list[OperationRecord],
    unit_summaries: list[UnitSummary],
    station_summaries: list[StationSummary],
) -> dict[str, float | int]:
    if not unit_summaries:
        raise ValueError("The order is empty, so no KPIs can be calculated.")

    completion_times = [u.completion_time_s for u in unit_summaries]
    makespan_s = max(completion_times)
    total_units = len(unit_summaries)
    throughput_rate_units_per_s = total_units / makespan_s if makespan_s > 0 else 0.0
    throughput_rate_units_per_h = throughput_rate_units_per_s * 3600.0

    if len(completion_times) > 1:
        completion_times_sorted = sorted(completion_times)
        intervals = [
            completion_times_sorted[i] - completion_times_sorted[i - 1]
            for i in range(1, len(completion_times_sorted))
        ]
        average_cycle_time_s = mean(intervals)
    else:
        average_cycle_time_s = makespan_s

    cycle_time_from_inverse_throughput_s = (
        1.0 / throughput_rate_units_per_s if throughput_rate_units_per_s > 0 else math.inf
    )

    average_flow_time_s = mean(u.flow_time_s for u in unit_summaries)
    total_wait_time_s = sum(op.wait_time_s for op in operations)

    kpis: dict[str, float | int] = {
        "total_units_ordered": total_units,
        "makespan_seconds": round(makespan_s, 4),
        "average_cycle_time_seconds": round(average_cycle_time_s, 4),
        "cycle_time_from_1_over_throughput_seconds": round(
            cycle_time_from_inverse_throughput_s, 4
        ),
        "throughput_units_per_second": round(throughput_rate_units_per_s, 6),
        "throughput_units_per_hour": round(throughput_rate_units_per_h, 4),
        "average_flow_time_seconds": round(average_flow_time_s, 4),
        "total_wait_time_seconds": round(total_wait_time_s, 4),
    }

    for summary in station_summaries:
        safe_key = re.sub(r"[^A-Za-z0-9]+", "_", summary.station_name).strip("_").lower()
        kpis[f"utilization_overall_{safe_key}"] = round(summary.utilization_overall, 6)
        kpis[f"utilization_active_window_{safe_key}"] = round(
            summary.utilization_active_window, 6
        )
        kpis[f"max_queue_length_{safe_key}"] = summary.max_queue_length
        kpis[f"average_queue_length_{safe_key}"] = round(summary.average_queue_length, 6)
        kpis[f"average_wait_time_{safe_key}_seconds"] = round(
            summary.average_wait_time_s, 4
        )

    ordered_mix = Counter(ordered_units)
    for variant, qty in sorted(ordered_mix.items()):
        kpis[f"order_qty_{variant.lower()}"] = qty

    return kpis


def write_kpis_csv(kpis: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kpi_name", "value"])
        for key, value in kpis.items():
            writer.writerow([key, value])


def write_material_report_csv(
    material_report: dict[str, dict[str, int]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "needed", "available", "remaining_after_order"])
        for material, values in material_report.items():
            writer.writerow(
                [
                    material,
                    values["needed"],
                    values["available"],
                    values["remaining_after_order"],
                ]
            )


def write_operations_csv(operations: list[OperationRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unit_id",
                "variant",
                "station_index",
                "station_name",
                "arrival_time_s",
                "start_time_s",
                "finish_time_s",
                "process_time_s",
                "wait_time_s",
                "queue_length_on_arrival",
            ]
        )
        for op in operations:
            writer.writerow(
                [
                    op.unit_id,
                    op.variant,
                    op.station_index,
                    op.station_name,
                    round(op.arrival_time_s, 4),
                    round(op.start_time_s, 4),
                    round(op.finish_time_s, 4),
                    round(op.process_time_s, 4),
                    round(op.wait_time_s, 4),
                    op.queue_length_on_arrival,
                ]
            )


def write_unit_summary_csv(unit_summaries: list[UnitSummary], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unit_id",
                "variant",
                "first_arrival_time_s",
                "start_time_s",
                "completion_time_s",
                "flow_time_s",
            ]
        )
        for summary in unit_summaries:
            writer.writerow(
                [
                    summary.unit_id,
                    summary.variant,
                    round(summary.first_arrival_time_s, 4),
                    round(summary.start_time_s, 4),
                    round(summary.completion_time_s, 4),
                    round(summary.flow_time_s, 4),
                ]
            )


def write_station_summary_csv(
    station_summaries: list[StationSummary], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "station_index",
                "station_name",
                "busy_time_s",
                "first_start_time_s",
                "last_finish_time_s",
                "max_queue_length",
                "average_queue_length",
                "average_wait_time_s",
                "total_wait_time_s",
                "utilization_overall",
                "utilization_active_window",
            ]
        )
        for summary in station_summaries:
            writer.writerow(
                [
                    summary.station_index,
                    summary.station_name,
                    round(summary.busy_time_s, 4),
                    round(summary.first_start_time_s, 4)
                    if summary.first_start_time_s is not None
                    else "",
                    round(summary.last_finish_time_s, 4)
                    if summary.last_finish_time_s is not None
                    else "",
                    summary.max_queue_length,
                    round(summary.average_queue_length, 6),
                    round(summary.average_wait_time_s, 4),
                    round(summary.total_wait_time_s, 4),
                    round(summary.utilization_overall, 6),
                    round(summary.utilization_active_window, 6),
                ]
            )


def create_gantt_chart(operations: list[OperationRecord], output_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    station_names = []
    for op in operations:
        if op.station_name not in station_names:
            station_names.append(op.station_name)
    station_to_y = {name: idx for idx, name in enumerate(station_names)}

    fig, ax = plt.subplots(figsize=(16, 8))
    for op in operations:
        y = station_to_y[op.station_name]
        duration = op.finish_time_s - op.start_time_s
        ax.barh(y, duration, left=op.start_time_s, height=0.6)
        ax.text(
            op.start_time_s + duration / 2,
            y,
            f"{op.unit_id}-{op.variant}",
            ha="center",
            va="center",
            fontsize=7,
        )

    ax.set_yticks(list(station_to_y.values()))
    ax.set_yticklabels(list(station_to_y.keys()))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stations")
    ax.set_title("Production line Gantt chart")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_throughput_chart(unit_summaries: list[UnitSummary], output_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    completion_times = sorted(u.completion_time_s for u in unit_summaries)
    x = [0.0] + completion_times
    y = [0] + list(range(1, len(completion_times) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(x, y, where="post")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Completed units")
    ax.set_title("Cumulative completed phones over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_order_slug(order_text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", order_text.strip()).strip("_")
    if not slug:
        slug = "order"
    return slug[:max_length]


def create_run_output_dir(output_root: Path, order_text: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = make_order_slug(order_text)
    run_dir = output_root / f"{timestamp}__{slug}"
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = output_root / f"{timestamp}__{slug}__{counter}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(
    order_text: str,
    ordered_units: list[str],
    output_path: Path,
    data_dir: Path,
    output_dir: Path,
) -> None:
    payload = {
        "order_text": order_text,
        "expanded_order_sequence": ordered_units,
        "data_directory": str(data_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_error_metadata(message: str, output_path: Path) -> None:
    payload = {
        "status": "failed",
        "message": message,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a 6-station phone production line with FIFO queues and transport times."
    )
    parser.add_argument(
        "--order",
        type=str,
        help='Order string, for example: "3xFUSE2, 2xFUSE1, 4xFUSE0"',
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Folder containing process_times.json, transport_times.json, material_stock.json and bom.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Root folder where a new subfolder will be created for every order run",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_root: Path = args.output_root

    process_time_data = load_json(data_dir / "process_times.json")
    transport_time_data = load_json(data_dir / "transport_times.json")
    material_stock_data = load_json(data_dir / "material_stock.json")
    bom_data = load_json(data_dir / "bom.json")

    valid_variants = set(process_time_data["process_times"].keys())

    if args.order:
        order_text = args.order
    else:
        order_text = input(
            "Enter order (example: 3xFUSE2, 2xFUSE1, 4xFUSE0): "
        ).strip()

    ordered_units = parse_order(order_text, valid_variants)
    run_output_dir = create_run_output_dir(output_root, order_text)
    save_run_metadata(
        order_text, ordered_units, run_output_dir / "run_metadata.json", data_dir, run_output_dir
    )

    try:
        requirements = calculate_material_requirements(ordered_units, bom_data)
        feasible, material_report = check_materials(requirements, material_stock_data)
        write_material_report_csv(material_report, run_output_dir / "material_report.csv")

        if not feasible:
            message = (
                "Insufficient material stock for this order. "
                "See material_report.csv in the run folder for details."
            )
            save_error_metadata(message, run_output_dir / "run_error.json")
            raise RuntimeError(message)

        operations, unit_summaries, station_summaries, _ = run_simulation(
            ordered_units=ordered_units,
            process_time_data=process_time_data,
            transport_time_data=transport_time_data,
        )
        kpis = calculate_kpis(
            ordered_units=ordered_units,
            operations=operations,
            unit_summaries=unit_summaries,
            station_summaries=station_summaries,
        )

        write_kpis_csv(kpis, run_output_dir / "kpi_summary.csv")
        write_operations_csv(operations, run_output_dir / "station_schedule.csv")
        write_unit_summary_csv(unit_summaries, run_output_dir / "unit_summary.csv")
        write_station_summary_csv(station_summaries, run_output_dir / "station_summary.csv")
        create_gantt_chart(operations, run_output_dir / "gantt_chart.png")
        create_throughput_chart(unit_summaries, run_output_dir / "throughput_chart.png")

        if not MATPLOTLIB_AVAILABLE:
            with (run_output_dir / "charts_skipped.txt").open("w", encoding="utf-8") as f:
                f.write("Charts were skipped because matplotlib is not installed.\n")
                f.write("Install it with: pip install matplotlib\n")

    except Exception as exc:
        print(f"Simulation failed: {exc}")
        print(f"Run folder: {run_output_dir.resolve()}")
        raise SystemExit(1) from exc

    print("Simulation finished successfully.")
    print(f"Order: {order_text}")
    print(f"Units completed: {kpis['total_units_ordered']}")
    print(f"Makespan [s]: {kpis['makespan_seconds']}")
    print(f"Average cycle time [s]: {kpis['average_cycle_time_seconds']}")
    print(f"Throughput [units/h]: {kpis['throughput_units_per_hour']}")
    print(f"Run folder: {run_output_dir.resolve()}")


if __name__ == "__main__":
    main()
