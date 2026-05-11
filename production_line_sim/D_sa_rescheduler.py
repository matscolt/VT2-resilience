from __future__ import annotations

import math
import random
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_transport_lookup(raw_transport_lookup: dict[str, Any]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for key, value in dict(raw_transport_lookup or {}).items():
        if "->" not in str(key):
            continue
        from_station, to_station = [part.strip() for part in str(key).split("->", maxsplit=1)]
        lookup[(from_station, to_station)] = _safe_float(value, 0.0)
    return lookup


def _group_station_indices_by_stage(snapshot: dict[str, Any]) -> list[list[int]]:
    raw = list(snapshot.get("stage_instance_indices", []))
    grouped: list[list[int]] = []
    for row in raw:
        grouped.append([max(0, _safe_int(value, 1) - 1) for value in list(row)])
    return grouped


def _build_station_rows_by_index(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    stations = list(snapshot.get("stations", []))
    stations_sorted = sorted(stations, key=lambda row: _safe_int(row.get("station_index", 0)))
    return stations_sorted


def _compute_station_penalties(
    snapshot: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    penalties: dict[int, float] = {}
    outputs: list[dict[str, Any]] = []

    disruption = dict(snapshot.get("triggered_disruption", {}))
    stations = _build_station_rows_by_index(snapshot)
    stage_groups = _group_station_indices_by_stage(snapshot)
    current_time_s = _safe_float(snapshot.get("current_time_s"), 0.0)

    if not disruption:
        return penalties, outputs

    disruption_type = str(disruption.get("triggered_disruption_type", "")).strip().lower()
    disruption_station_name = str(disruption.get("station_name", disruption.get("base_station_name", ""))).strip()

    if disruption_station_name == "":
        return penalties, outputs

    disrupted_station_index = None
    disrupted_stage_index = None
    for idx, station_row in enumerate(stations):
        if str(station_row.get("station_name", "")).strip() == disruption_station_name:
            disrupted_station_index = idx
            disrupted_stage_index = _safe_int(station_row.get("stage_index"), idx)
            break

    if disrupted_station_index is None:
        return penalties, outputs

    breakdown_penalty_s = _safe_float(config.get("sa_breakdown_penalty_s", config.get("default_breakdown_penalty_s", 1200.0)), 1200.0)
    efficiency_penalty_s = _safe_float(config.get("sa_efficiency_penalty_s", config.get("default_efficiency_penalty_s", 300.0)), 300.0)
    penalty_horizon_s = _safe_float(config.get("sa_penalty_horizon_s", config.get("penalty_horizon_s", 900.0)), 900.0)

    stage_candidates = stage_groups[disrupted_stage_index] if 0 <= disrupted_stage_index < len(stage_groups) else [disrupted_station_index]
    if len(stage_candidates) <= 1:
        return penalties, outputs

    if disruption_type == "breakdown":
        penalty_s = max(0.0, breakdown_penalty_s)
    elif disruption_type == "efficiency_loss":
        penalty_s = max(0.0, efficiency_penalty_s)
    else:
        penalty_s = max(0.0, efficiency_penalty_s * 0.5)

    if penalty_s <= 0.0:
        return penalties, outputs

    penalties[disrupted_station_index] = penalty_s
    outputs.append(
        {
            "station_index": int(disrupted_station_index + 1),
            "penalty_s": float(penalty_s),
            "expires_at_s": float(current_time_s + penalty_horizon_s),
            "reason": f"sa_{disruption_type or 'disruption'}",
        }
    )
    return penalties, outputs


def _candidate_units(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    waiting_units = list(snapshot.get("waiting_units", []))
    candidates = [
        dict(unit)
        for unit in waiting_units
        if bool(unit.get("already_released_by_time", False)) or bool(unit.get("is_waiting_for_system_slot", False))
    ]
    if not candidates:
        candidates = [dict(unit) for unit in waiting_units if _safe_float(unit.get("release_time_s"), 0.0) >= 0.0]
    return candidates


def _baseline_order(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        units,
        key=lambda unit: (
            -_safe_int(unit.get("priority"), 1),
            -_safe_int(unit.get("attempt"), 1),
            _safe_float(unit.get("release_time_s"), 0.0),
            _safe_float(unit.get("remaining_total_base_process_time_s"), 0.0),
            _safe_int(unit.get("unit_index"), 0),
        ),
    )


def _choose_best_station_for_stage(
    variant: str,
    current_time_s: float,
    from_station_index: int | None,
    stage_station_indices: list[int],
    station_names: list[str],
    station_base_names: list[str],
    station_available_time_s: list[float],
    process_times: dict[str, Any],
    transport_lookup: dict[tuple[str, str], float],
    station_penalties: dict[int, float],
) -> tuple[int, float]:
    best_station_index = stage_station_indices[0]
    best_finish_time_s = float("inf")

    for station_index in stage_station_indices:
        station_name = station_names[station_index]
        transport_time_s = 0.0
        if from_station_index is not None:
            previous_station_name = station_names[from_station_index]
            transport_time_s = _safe_float(transport_lookup.get((previous_station_name, station_name)), 0.0)
        arrival_time_s = current_time_s + transport_time_s
        base_station_name = station_base_names[station_index]
        process_time_s = _safe_float(process_times.get(variant, {}).get(base_station_name), 0.0)
        start_time_s = max(arrival_time_s, station_available_time_s[station_index])
        finish_time_s = start_time_s + process_time_s + _safe_float(station_penalties.get(station_index), 0.0)

        if finish_time_s < best_finish_time_s - 1e-9 or (
            abs(finish_time_s - best_finish_time_s) <= 1e-9 and station_index < best_station_index
        ):
            best_station_index = station_index
            best_finish_time_s = finish_time_s

    return best_station_index, best_finish_time_s


def _evaluate_sequence(
    sequence: list[dict[str, Any]],
    snapshot: dict[str, Any],
    config: dict[str, Any],
    station_penalties: dict[int, float],
) -> float:
    current_time_s = _safe_float(snapshot.get("current_time_s"), 0.0)
    available_system_slots = max(0, _safe_int(snapshot.get("available_system_slots"), 0))
    max_units_in_system = max(1, _safe_int(snapshot.get("max_units_in_system"), 1))
    return_to_station_1_time_s = _safe_float(snapshot.get("return_to_station_1_time_s"), 31.3)
    dispatch_spacing_s = _safe_float(config.get("dispatch_spacing_s", return_to_station_1_time_s), return_to_station_1_time_s)

    station_names = list(snapshot.get("station_sequence", []))
    station_base_names = list(snapshot.get("station_instance_base_names", []))
    process_times = dict(snapshot.get("process_times", {}))
    transport_lookup = _normalize_transport_lookup(snapshot.get("transport_lookup", {}))
    stage_station_indices = _group_station_indices_by_stage(snapshot)
    projected_station_available_time_s = [_safe_float(value, current_time_s) for value in list(snapshot.get("projected_station_available_time_s", []))]

    if not sequence or not stage_station_indices:
        return 0.0

    carrier_ready_times: list[float] = [float(current_time_s)] * available_system_slots
    occupied_slots = max(0, max_units_in_system - available_system_slots)
    carrier_ready_times.extend(float(current_time_s) + dispatch_spacing_s * (idx + 1) for idx in range(occupied_slots))
    if not carrier_ready_times:
        carrier_ready_times = [float(current_time_s)]
    carrier_ready_times.sort()

    makespan_s = current_time_s
    total_flow_time_s = 0.0
    weighted_completion_penalty = 0.0
    rework_bonus_value = _safe_float(config.get("rework_bonus", 250.0), 250.0)

    for unit in sequence:
        variant = str(unit.get("variant", "")).strip()
        if variant == "":
            continue

        earliest_carrier_time_s = carrier_ready_times.pop(0)
        release_time_s = max(
            float(current_time_s),
            _safe_float(unit.get("release_time_s"), current_time_s),
            earliest_carrier_time_s,
        )

        stage_time_s = release_time_s
        from_station_index = None

        for stage_indices in stage_station_indices:
            chosen_station_index, _ = _choose_best_station_for_stage(
                variant=variant,
                current_time_s=stage_time_s,
                from_station_index=from_station_index,
                stage_station_indices=stage_indices,
                station_names=station_names,
                station_base_names=station_base_names,
                station_available_time_s=projected_station_available_time_s,
                process_times=process_times,
                transport_lookup=transport_lookup,
                station_penalties=station_penalties,
            )
            chosen_station_name = station_names[chosen_station_index]
            transport_time_s = 0.0
            if from_station_index is not None:
                previous_station_name = station_names[from_station_index]
                transport_time_s = _safe_float(transport_lookup.get((previous_station_name, chosen_station_name)), 0.0)
            arrival_time_s = stage_time_s + transport_time_s
            base_station_name = station_base_names[chosen_station_index]
            process_time_s = _safe_float(process_times.get(variant, {}).get(base_station_name), 0.0)
            start_time_s = max(arrival_time_s, projected_station_available_time_s[chosen_station_index])
            true_finish_time_s = start_time_s + process_time_s
            projected_station_available_time_s[chosen_station_index] = true_finish_time_s

            from_station_index = chosen_station_index
            stage_time_s = true_finish_time_s

        completion_time_s = stage_time_s
        carrier_ready_times.append(completion_time_s + return_to_station_1_time_s)
        carrier_ready_times.sort()

        makespan_s = max(makespan_s, completion_time_s)
        total_flow_time_s += max(0.0, completion_time_s - release_time_s)

        priority_value = max(1, _safe_int(unit.get("priority"), 1))
        attempt_value = max(1, _safe_int(unit.get("attempt"), 1))
        weighted_completion_penalty += completion_time_s * priority_value
        if attempt_value > 1:
            weighted_completion_penalty -= rework_bonus_value * float(attempt_value - 1)

    makespan_weight = _safe_float(config.get("makespan_weight", 1.0), 1.0)
    flow_weight = _safe_float(config.get("flow_weight", 0.1), 0.1)
    priority_weight = _safe_float(config.get("priority_weight", 0.0005), 0.0005)

    return (
        makespan_weight * makespan_s
        + flow_weight * total_flow_time_s
        + priority_weight * weighted_completion_penalty
    )


def _sequence_neighbor(sequence: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    if len(sequence) <= 1:
        return list(sequence)

    neighbor = list(sequence)
    move_type = rng.choice(("swap", "insert", "reverse"))

    if move_type == "swap":
        i, j = sorted(rng.sample(range(len(neighbor)), 2))
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    if move_type == "insert":
        i, j = rng.sample(range(len(neighbor)), 2)
        item = neighbor.pop(i)
        neighbor.insert(j, item)
        return neighbor

    i, j = sorted(rng.sample(range(len(neighbor)), 2))
    neighbor[i : j + 1] = list(reversed(neighbor[i : j + 1]))
    return neighbor


def _boost_map_from_sequence(sequence: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    boost_step = _safe_float(config.get("score_boost_step", 1000.0), 1000.0)
    total = len(sequence)
    boost_map: dict[str, float] = {}
    for rank, unit in enumerate(sequence):
        boost_map[str(_safe_int(unit.get("unit_index"), 0))] = float((total - rank) * boost_step)
    return boost_map


def optimize_reschedule(snapshot: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidates = _candidate_units(snapshot)
    if not candidates:
        return {
            "waiting_unit_score_boosts": {},
            "station_penalties": [],
            "summary": {
                "algorithm": "simulated_annealing",
                "optimized_units": 0,
                "improved": False,
            },
        }

    pool_size = max(1, _safe_int(config.get("candidate_pool_size", 12), 12))
    initial_order = _baseline_order(candidates)
    optimized_pool = list(initial_order[:pool_size])
    tail_units = list(initial_order[pool_size:])

    station_penalty_lookup, station_penalty_output = _compute_station_penalties(snapshot, config)

    seed_value = config.get("seed", 0)
    rng = random.Random(str(seed_value) + "|" + str(snapshot.get("current_time_s")) + "|" + str(snapshot.get("triggered_disruption", {})))

    current_sequence = list(optimized_pool)
    current_score = _evaluate_sequence(current_sequence + tail_units, snapshot, config, station_penalty_lookup)
    best_sequence = list(current_sequence)
    best_score = float(current_score)

    initial_temperature = max(1e-9, _safe_float(config.get("sa_initial_temperature", 250.0), 250.0))
    cooling_rate = min(0.9999, max(0.5, _safe_float(config.get("sa_cooling_rate", 0.93), 0.93)))
    iterations = max(1, _safe_int(config.get("sa_iterations", 250), 250))
    minimum_improvement = _safe_float(config.get("minimum_improvement", 1e-6), 1e-6)

    temperature = float(initial_temperature)
    accepted_moves = 0

    for _ in range(iterations):
        neighbor_sequence = _sequence_neighbor(current_sequence, rng)
        neighbor_score = _evaluate_sequence(neighbor_sequence + tail_units, snapshot, config, station_penalty_lookup)

        delta = neighbor_score - current_score
        accept = False
        if delta <= 0.0:
            accept = True
        else:
            acceptance_probability = math.exp(-delta / max(temperature, 1e-9))
            if rng.random() < acceptance_probability:
                accept = True

        if accept:
            current_sequence = neighbor_sequence
            current_score = neighbor_score
            accepted_moves += 1
            if current_score < best_score - minimum_improvement:
                best_score = current_score
                best_sequence = list(current_sequence)

        temperature *= cooling_rate
        if temperature < 1e-6:
            temperature = 1e-6

    best_full_sequence = best_sequence + tail_units
    boost_map = _boost_map_from_sequence(best_full_sequence, config)

    baseline_score = _evaluate_sequence(initial_order, snapshot, config, station_penalty_lookup)
    improved = best_score < baseline_score - minimum_improvement

    return {
        "waiting_unit_score_boosts": boost_map,
        "station_penalties": station_penalty_output,
        "summary": {
            "algorithm": "simulated_annealing",
            "candidate_pool_size": int(pool_size),
            "iterations": int(iterations),
            "accepted_moves": int(accepted_moves),
            "baseline_score": float(round(baseline_score, 6)),
            "best_score": float(round(best_score, 6)),
            "improved": bool(improved),
            "optimized_units": int(len(best_full_sequence)),
        },
    }


def optimize(snapshot: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return optimize_reschedule(snapshot, config)


def reschedule(snapshot: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return optimize_reschedule(snapshot, config)
