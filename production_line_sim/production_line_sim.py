from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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
class TransportRecord:
    unit_id: str
    variant: str
    transport_index: int
    transport_name: str
    from_station: str
    to_station: str
    start_time_s: float
    finish_time_s: float
    transport_time_s: float


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


# -----------------------------
# Data loading / order parsing
# -----------------------------
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_order(order_text: str, valid_variants: set[str]) -> list[str]:
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


# -----------------------------
# Materials
# -----------------------------
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


def determine_producible_units(
    ordered_units: list[str],
    bom_data: dict[str, Any],
    material_stock_data: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, dict[str, int]], dict[str, Any]]:
    bom = bom_data["bom_units_per_phone"]
    initial_stock = {
        material: int(qty)
        for material, qty in material_stock_data["materials_in_stock"].items()
    }
    remaining_stock = dict(initial_stock)

    produced_units: list[str] = []
    unproduced_units: list[str] = []
    first_unproduced_position: int | None = None
    shortage_reason: dict[str, Any] | None = None

    for index, variant in enumerate(ordered_units, start=1):
        if variant not in bom:
            raise ValueError(f"Variant '{variant}' is missing from bom.json")

        unit_bom = {material: int(qty) for material, qty in bom[variant].items()}
        shortages = []
        for material, needed in unit_bom.items():
            available = int(remaining_stock.get(material, 0))
            if available < needed:
                shortages.append(
                    {
                        "material": material,
                        "needed": needed,
                        "available": available,
                        "missing": needed - available,
                    }
                )

        if shortages:
            first_unproduced_position = index
            shortage_reason = {
                "unit_position": index,
                "variant": variant,
                "shortages": shortages,
            }
            unproduced_units = ordered_units[index - 1 :]
            break

        for material, needed in unit_bom.items():
            remaining_stock[material] = int(remaining_stock.get(material, 0)) - needed
        produced_units.append(variant)

    requested_requirements = calculate_material_requirements(ordered_units, bom_data)
    consumed_requirements = calculate_material_requirements(produced_units, bom_data)

    all_materials = sorted(
        set(initial_stock.keys())
        | set(requested_requirements.keys())
        | set(consumed_requirements.keys())
    )
    material_report: dict[str, dict[str, int]] = {}
    for material in all_materials:
        available = int(initial_stock.get(material, 0))
        requested = int(requested_requirements.get(material, 0))
        consumed = int(consumed_requirements.get(material, 0))
        remaining = int(remaining_stock.get(material, available))
        unmet = max(0, requested - available)
        material_report[material] = {
            "requested_for_full_order": requested,
            "available_at_start": available,
            "consumed_for_produced_units": consumed,
            "remaining_after_run": remaining,
            "unmet_for_full_order": unmet,
        }

    production_status: dict[str, Any] = {
        "requested_unit_count": len(ordered_units),
        "produced_unit_count": len(produced_units),
        "unproduced_unit_count": len(unproduced_units),
        "produced_mix": dict(Counter(produced_units)),
        "unproduced_mix": dict(Counter(unproduced_units)),
        "first_unproduced_position": first_unproduced_position,
        "shortage_reason": shortage_reason,
        "status": "complete" if not unproduced_units else "partial_due_to_material_shortage",
    }

    return produced_units, unproduced_units, material_report, production_status


# -----------------------------
# Event-based simulation with FIFO queues
# -----------------------------
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
) -> tuple[list[OperationRecord], list[TransportRecord], list[UnitSummary], list[StationSummary], dict[str, float]]:
    station_sequence = process_time_data["station_sequence"]
    process_times = process_time_data["process_times"]
    transport_lookup = build_transport_lookup(station_sequence, transport_time_data)

    station_states: list[StationState] = [StationState(queue=deque()) for _ in station_sequence]
    operations: list[OperationRecord] = []
    operation_lookup: dict[tuple[int, int], OperationRecord] = {}
    transport_records: list[TransportRecord] = []

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
            queue_length_on_arrival=queue_length_ahead_on_arrival,
        )
        operations.append(operation)
        operation_lookup[(station_index, unit_index)] = operation

        station_state.busy = True
        station_state.current_unit_index = unit_index
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
            station_state.max_queue_length = max(station_state.max_queue_length, len(station_state.queue))
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

            if station_index < len(station_sequence) - 1:
                current_station_name = station_sequence[station_index]
                next_station_name = station_sequence[station_index + 1]
                transport_time_s = float(transport_lookup[(current_station_name, next_station_name)])
                arrival_time_s = time_s + transport_time_s
                transport_records.append(
                    TransportRecord(
                        unit_id=f"U{unit_index + 1:03d}",
                        variant=ordered_units[unit_index],
                        transport_index=station_index + 1,
                        transport_name=f"Transportation {station_index + 1}",
                        from_station=current_station_name,
                        to_station=next_station_name,
                        start_time_s=time_s,
                        finish_time_s=arrival_time_s,
                        transport_time_s=transport_time_s,
                    )
                )
                push_event(
                    time_s=arrival_time_s,
                    event_type=EVENT_ARRIVAL,
                    station_index=station_index + 1,
                    unit_index=unit_index,
                )
            else:
                unit_completion[unit_index] = time_s

            try_start_next(station_index, time_s)
            continue

        raise RuntimeError(f"Unknown event type: {event_type}")

    if ordered_units:
        makespan_s = max(unit_completion.values()) if unit_completion else 0.0
    else:
        makespan_s = 0.0

    for station_state in station_states:
        _update_queue_area(station_state, makespan_s)

    unit_summaries: list[UnitSummary] = []
    for unit_index, variant in enumerate(ordered_units):
        unit_id = f"U{unit_index + 1:03d}"
        first_arrival = unit_first_arrival.get(unit_index, 0.0)
        first_start = unit_first_start.get(unit_index, 0.0)
        completion = unit_completion.get(unit_index, first_start)
        unit_summaries.append(
            UnitSummary(
                unit_id=unit_id,
                variant=variant,
                first_arrival_time_s=first_arrival,
                start_time_s=first_start,
                completion_time_s=completion,
                flow_time_s=completion - first_arrival,
            )
        )

    station_summaries: list[StationSummary] = []
    for idx, station_name in enumerate(station_sequence):
        state = station_states[idx]
        active_window = 0.0
        if state.first_start_time_s is not None and state.last_finish_time_s is not None:
            active_window = state.last_finish_time_s - state.first_start_time_s

        station_ops = [op for op in operations if op.station_index == idx + 1]
        average_wait_time_s = mean([op.wait_time_s for op in station_ops]) if station_ops else 0.0
        average_queue_length = state.queue_area / makespan_s if makespan_s > 0 else 0.0
        utilization_overall = state.busy_time_s / makespan_s if makespan_s > 0 else 0.0
        utilization_active_window = state.busy_time_s / active_window if active_window > 0 else 0.0

        station_summaries.append(
            StationSummary(
                station_index=idx + 1,
                station_name=station_name,
                busy_time_s=state.busy_time_s,
                first_start_time_s=state.first_start_time_s,
                last_finish_time_s=state.last_finish_time_s,
                max_queue_length=state.max_queue_length,
                average_queue_length=average_queue_length,
                average_wait_time_s=average_wait_time_s,
                total_wait_time_s=state.total_wait_time_s,
                utilization_overall=utilization_overall,
                utilization_active_window=utilization_active_window,
            )
        )

    station_available_time = {
        summary.station_name: float(summary.last_finish_time_s or 0.0)
        for summary in station_summaries
    }

    return operations, transport_records, unit_summaries, station_summaries, station_available_time


# -----------------------------
# KPI calculation
# -----------------------------
def calculate_kpis(
    ordered_units: list[str],
    operations: list[OperationRecord],
    unit_summaries: list[UnitSummary],
    station_summaries: list[StationSummary],
) -> dict[str, float | int]:
    if not unit_summaries:
        return {
            "total_units_ordered": 0,
            "makespan_seconds": 0.0,
            "average_cycle_time_seconds": 0.0,
            "cycle_time_from_1_over_throughput_seconds": math.inf,
            "throughput_units_per_second": 0.0,
            "throughput_units_per_hour": 0.0,
            "average_flow_time_seconds": 0.0,
            "total_wait_time_seconds": 0.0,
        }

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
        )
        if math.isfinite(cycle_time_from_inverse_throughput_s)
        else math.inf,
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
        kpis[f"max_queue_{safe_key}"] = summary.max_queue_length
        kpis[f"average_queue_{safe_key}"] = round(summary.average_queue_length, 6)
        kpis[f"average_wait_{safe_key}_seconds"] = round(summary.average_wait_time_s, 4)

    ordered_mix = Counter(ordered_units)
    for variant, qty in sorted(ordered_mix.items()):
        kpis[f"order_qty_{variant.lower()}"] = qty

    return kpis


# -----------------------------
# Output writers
# -----------------------------
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
        writer.writerow(
            [
                "material",
                "requested_for_full_order",
                "available_at_start",
                "consumed_for_produced_units",
                "remaining_after_run",
                "unmet_for_full_order",
            ]
        )
        for material, values in material_report.items():
            writer.writerow(
                [
                    material,
                    values["requested_for_full_order"],
                    values["available_at_start"],
                    values["consumed_for_produced_units"],
                    values["remaining_after_run"],
                    values["unmet_for_full_order"],
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


def write_transport_csv(transport_records: list[TransportRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unit_id",
                "variant",
                "transport_index",
                "transport_name",
                "from_station",
                "to_station",
                "start_time_s",
                "finish_time_s",
                "transport_time_s",
            ]
        )
        for tr in transport_records:
            writer.writerow(
                [
                    tr.unit_id,
                    tr.variant,
                    tr.transport_index,
                    tr.transport_name,
                    tr.from_station,
                    tr.to_station,
                    round(tr.start_time_s, 4),
                    round(tr.finish_time_s, 4),
                    round(tr.transport_time_s, 4),
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


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------
# Charts
# -----------------------------
def _unit_color_map(unit_ids: list[str]) -> dict[str, Any]:
    if plt is None:
        return {}
    cmap = plt.get_cmap("tab20")
    sorted_unit_ids = sorted(unit_ids)
    return {unit_id: cmap(i % cmap.N) for i, unit_id in enumerate(sorted_unit_ids)}


def create_gantt_chart(
    operations: list[OperationRecord],
    transport_records: list[TransportRecord],
    output_path: Path,
) -> bool:
    if plt is None:
        return False

    if not operations:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Production line Gantt chart")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Stations / Transport")
        ax.text(0.5, 0.5, "No units were produced.", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True

    station_names = []
    seen_stations = set()
    for op in sorted(operations, key=lambda x: x.station_index):
        if op.station_name not in seen_stations:
            station_names.append(op.station_name)
            seen_stations.add(op.station_name)

    transport_names = []
    seen_transports = set()
    for tr in sorted(transport_records, key=lambda x: x.transport_index):
        if tr.transport_name not in seen_transports:
            transport_names.append(tr.transport_name)
            seen_transports.add(tr.transport_name)

    row_names: list[str] = []
    for idx, station_name in enumerate(station_names):
        row_names.append(station_name)
        if idx < len(transport_names):
            row_names.append(transport_names[idx])

    row_to_y = {name: idx for idx, name in enumerate(row_names)}
    unit_colors = _unit_color_map([op.unit_id for op in operations])

    fig, ax = plt.subplots(figsize=(18, 9))

    for op in operations:
        y = row_to_y[op.station_name]
        duration = op.finish_time_s - op.start_time_s
        color = unit_colors.get(op.unit_id)
        ax.barh(
            y,
            duration,
            left=op.start_time_s,
            height=0.62,
            color=color,
            edgecolor="black",
            linewidth=0.25,
        )
        if duration > 1.0:
            ax.text(
                op.start_time_s + duration / 2,
                y,
                f"{op.unit_id}-{op.variant}",
                ha="center",
                va="center",
                fontsize=5,
            )

    for tr in transport_records:
        y = row_to_y[tr.transport_name]
        duration = tr.finish_time_s - tr.start_time_s
        color = unit_colors.get(tr.unit_id)
        ax.barh(
            y,
            duration,
            left=tr.start_time_s,
            height=0.42,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.65,
        )
        if duration > 1.0:
            ax.text(
                tr.start_time_s + duration / 2,
                y,
                f"{tr.unit_id}-{tr.variant}",
                ha="center",
                va="center",
                fontsize=4.5,
            )

    ax.set_yticks(list(row_to_y.values()))
    ax.set_yticklabels(list(row_to_y.keys()))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stations / Transport")
    ax.set_title("Production line Gantt chart")
    ax.grid(True, axis="x", alpha=0.3)
    # Intentionally do not invert the y-axis so the chart keeps the original bottom-to-top orientation.
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_throughput_chart(unit_summaries: list[UnitSummary], output_path: Path) -> bool:
    if plt is None:
        return False

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
    return True


# -----------------------------
# Run metadata / output folders
# -----------------------------
def make_order_slug(order_text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", order_text.strip()).strip("_")
    return (slug or "order")[:max_length]


def create_run_output_dir(output_root: Path, order_text: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
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
    save_json(payload, output_path)


# -----------------------------
# Main
# -----------------------------
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

    if args.order:
        order_text = args.order
    else:
        order_text = input(
            "Enter order (example: 3xFUSE2, 2xFUSE1, 4xFUSE0): "
        ).strip()

    starttime = time.perf_counter() #start time of the whole execution, including setup and file writing

    process_time_data = load_json(data_dir / "process_times.json")
    transport_time_data = load_json(data_dir / "transport_times.json")
    material_stock_data = load_json(data_dir / "material_stock.json")
    bom_data = load_json(data_dir / "bom.json")

    valid_variants = set(process_time_data["process_times"].keys())

    order_text = args.order or input(
        "Enter order (example: 3xFUSE2, 2xFUSE1, 4xFUSE0): "
    ).strip()

    ordered_units = parse_order(order_text, valid_variants)
    run_output_dir = create_run_output_dir(output_root, order_text)
    save_run_metadata(
        order_text, ordered_units, run_output_dir / "run_metadata.json", data_dir, run_output_dir
    )

    produced_units, unproduced_units, material_report, production_status = determine_producible_units(
        ordered_units=ordered_units,
        bom_data=bom_data,
        material_stock_data=material_stock_data,
    )

    write_material_report_csv(material_report, run_output_dir / "material_report.csv")
    save_json(production_status, run_output_dir / "production_status.json")

    operations: list[OperationRecord] = []
    transport_records: list[TransportRecord] = []
    unit_summaries: list[UnitSummary] = []
    station_summaries: list[StationSummary] = []

    if produced_units:
        operations, transport_records, unit_summaries, station_summaries, _ = run_simulation(
            ordered_units=produced_units,
            process_time_data=process_time_data,
            transport_time_data=transport_time_data,
        )

    kpis = calculate_kpis(
        ordered_units=produced_units,
        operations=operations,
        unit_summaries=unit_summaries,
        station_summaries=station_summaries,
    )

    write_kpis_csv(kpis, run_output_dir / "kpi_summary.csv")
    write_operations_csv(operations, run_output_dir / "station_schedule.csv")
    write_transport_csv(transport_records, run_output_dir / "transport_schedule.csv")
    write_unit_summary_csv(unit_summaries, run_output_dir / "unit_summary.csv")
    write_station_summary_csv(station_summaries, run_output_dir / "station_summary.csv")

    chart_notes: list[str] = []
    if not create_gantt_chart(operations, transport_records, run_output_dir / "gantt_chart.png"):
        chart_notes.append("gantt_chart.png not created because matplotlib is not installed.")
    if not create_throughput_chart(unit_summaries, run_output_dir / "throughput_chart.png"):
        chart_notes.append("throughput_chart.png not created because matplotlib is not installed.")
    if chart_notes:
        (run_output_dir / "charts_skipped.txt").write_text("\n".join(chart_notes), encoding="utf-8")

    print(f"Run folder: {run_output_dir.resolve()}")
    print(f"Requested units: {len(ordered_units)}")
    print(f"Produced units: {len(produced_units)}")
    if unproduced_units:
        print(f"Unproduced units: {len(unproduced_units)}")
        if production_status.get("shortage_reason"):
            pos = production_status["shortage_reason"]["unit_position"]
            variant = production_status["shortage_reason"]["variant"]
            print(f"Stopped at unit {pos} ({variant}) because of material shortage.")

    endtime = time.perf_counter() #end time of the whole execution, including setup and file writing
    print(f"Total execution time: {endtime - starttime:.6f} seconds")


if __name__ == "__main__":
    main()
    
