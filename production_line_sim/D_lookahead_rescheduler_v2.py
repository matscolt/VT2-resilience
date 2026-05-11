from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any


@dataclass
class CandidateResult:
    sequence: list[int]
    score: float
    completion_times: dict[int, float]


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default) if isinstance(config, dict) else default


def _build_base_sort_key(unit_row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -float(unit_row.get("priority", 1)),
        -float(unit_row.get("attempt", 1)),
        float(unit_row.get("release_time_s", 0.0)),
        float(unit_row.get("remaining_total_base_process_time_s", 0.0)),
        int(unit_row.get("unit_index", 0)),
    )


def _estimate_unit_completion_time(
    unit_row: dict[str, Any],
    station_available_times: list[float],
    stage_instance_indices_zero_based: list[list[int]],
    station_sequence: list[str],
    station_instance_base_names: list[str],
    process_times: dict[str, dict[str, float]],
    transport_lookup: dict[str, float],
    dispatch_time_s: float,
    station_penalty_s_by_index: dict[int, float],
    queue_penalty_factor: float,
    busy_penalty_factor: float,
    station_rows_by_index: dict[int, dict[str, Any]],
) -> tuple[float, list[int]]:
    variant = str(unit_row["variant"])
    previous_station_name: str | None = None
    previous_finish_time_s = float(dispatch_time_s)
    chosen_station_indices: list[int] = []

    for stage_station_indices in stage_instance_indices_zero_based:
        best_station_index: int | None = None
        best_station_finish_time_s: float | None = None
        best_station_score: float | None = None

        for station_index in stage_station_indices:
            station_name = station_sequence[station_index]
            base_station_name = station_instance_base_names[station_index]
            transport_time_s = 0.0
            if previous_station_name is not None:
                transport_time_s = float(transport_lookup.get(f"{previous_station_name} -> {station_name}", 0.0))

            arrival_time_s = previous_finish_time_s + transport_time_s
            start_time_s = max(arrival_time_s, float(station_available_times[station_index]))
            process_time_s = float(process_times[variant][base_station_name])
            finish_time_s = start_time_s + process_time_s

            station_row = station_rows_by_index.get(int(station_index + 1), {})
            queue_length = float(station_row.get("queue_length", 0.0))
            busy_flag = 1.0 if bool(station_row.get("busy", False)) else 0.0
            optimizer_penalty_s = float(station_penalty_s_by_index.get(station_index, 0.0))
            candidate_score = (
                finish_time_s
                + optimizer_penalty_s
                + queue_penalty_factor * queue_length
                + busy_penalty_factor * busy_flag
            )

            if best_station_score is None or candidate_score < best_station_score:
                best_station_score = candidate_score
                best_station_index = station_index
                best_station_finish_time_s = finish_time_s

        if best_station_index is None or best_station_finish_time_s is None:
            return float("inf"), chosen_station_indices

        station_available_times[best_station_index] = best_station_finish_time_s
        chosen_station_indices.append(best_station_index)
        previous_station_name = station_sequence[best_station_index]
        previous_finish_time_s = best_station_finish_time_s

    return previous_finish_time_s, chosen_station_indices


def _evaluate_sequence(
    sequence: list[dict[str, Any]],
    current_time_s: float,
    available_system_slots: int,
    dispatch_spacing_s: float,
    stage_instance_indices_zero_based: list[list[int]],
    station_sequence: list[str],
    station_instance_base_names: list[str],
    process_times: dict[str, dict[str, float]],
    transport_lookup: dict[str, float],
    projected_station_available_time_s: list[float],
    station_penalty_s_by_index: dict[int, float],
    queue_penalty_factor: float,
    busy_penalty_factor: float,
    station_rows_by_index: dict[int, dict[str, Any]],
    priority_weight: float,
    flow_weight: float,
    release_hold_weight: float,
) -> CandidateResult:
    station_available_times = [float(value) for value in projected_station_available_time_s]
    completion_times: dict[int, float] = {}
    score = 0.0

    immediate_slots = max(0, int(available_system_slots))
    queued_dispatch_count = 0

    for seq_position, unit_row in enumerate(sequence):
        if seq_position < immediate_slots:
            dispatch_time_s = max(float(current_time_s), float(unit_row.get("release_time_s", 0.0)))
        else:
            queued_dispatch_count += 1
            dispatch_time_s = max(
                float(current_time_s) + float(queued_dispatch_count) * float(dispatch_spacing_s),
                float(unit_row.get("release_time_s", 0.0)),
            )

        completion_time_s, _chosen_stations = _estimate_unit_completion_time(
            unit_row=unit_row,
            station_available_times=station_available_times,
            stage_instance_indices_zero_based=stage_instance_indices_zero_based,
            station_sequence=station_sequence,
            station_instance_base_names=station_instance_base_names,
            process_times=process_times,
            transport_lookup=transport_lookup,
            dispatch_time_s=dispatch_time_s,
            station_penalty_s_by_index=station_penalty_s_by_index,
            queue_penalty_factor=queue_penalty_factor,
            busy_penalty_factor=busy_penalty_factor,
            station_rows_by_index=station_rows_by_index,
        )
        completion_times[int(unit_row["unit_index"])] = float(completion_time_s)

        flow_time_s = max(0.0, completion_time_s - float(current_time_s))
        release_hold_s = max(0.0, dispatch_time_s - float(current_time_s))
        priority_value = max(1.0, float(unit_row.get("priority", 1.0)))
        score += (
            flow_weight * flow_time_s
            + release_hold_weight * release_hold_s
            - priority_weight * priority_value
        )

    return CandidateResult(
        sequence=[int(unit_row["unit_index"]) for unit_row in sequence],
        score=float(score),
        completion_times=completion_times,
    )


def optimize_reschedule(snapshot: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    current_time_s = float(snapshot.get("current_time_s", 0.0))
    waiting_units = list(snapshot.get("waiting_units", []))
    if not waiting_units:
        return {"summary": {"decision": "no_waiting_units"}}

    stage_instance_indices_zero_based = [
        [max(0, int(station_index) - 1) for station_index in stage_row]
        for stage_row in snapshot.get("stage_instance_indices", [])
    ]
    station_sequence = list(snapshot.get("station_sequence", []))
    station_instance_base_names = list(snapshot.get("station_instance_base_names", []))
    process_times = dict(snapshot.get("process_times", {}))
    transport_lookup = dict(snapshot.get("transport_lookup", {}))
    projected_station_available_time_s = [float(value) for value in snapshot.get("projected_station_available_time_s", [])]
    station_rows = list(snapshot.get("stations", []))
    station_rows_by_index = {int(row.get("station_index", 0)): row for row in station_rows}

    if not stage_instance_indices_zero_based or not station_sequence or not process_times:
        return {"summary": {"decision": "invalid_snapshot"}}

    candidate_pool_size = max(3, int(_config_value(config, "candidate_pool_size", 10)))
    beam_width = max(2, int(_config_value(config, "beam_width", 8)))
    search_depth = max(2, int(_config_value(config, "lookahead_depth", 4)))
    dispatch_spacing_s = float(snapshot.get("return_to_station_1_time_s", _config_value(config, "dispatch_spacing_s", 31.3)))
    priority_weight = float(_config_value(config, "priority_weight", 2000.0))
    flow_weight = float(_config_value(config, "flow_weight", 1.0))
    release_hold_weight = float(_config_value(config, "release_hold_weight", 0.1))
    queue_penalty_factor = float(_config_value(config, "queue_penalty_factor", 10.0))
    busy_penalty_factor = float(_config_value(config, "busy_penalty_factor", 3.0))
    disruption_penalty_scale = float(_config_value(config, "disruption_penalty_scale", 1.0))
    score_boost_step = float(_config_value(config, "score_boost_step", 1000.0))

    disruption_payload = dict(snapshot.get("triggered_disruption", {}))
    disrupted_station_index = int(disruption_payload.get("station_index", 0) or 0)
    disrupted_station_penalty_s = max(
        0.0,
        float(disruption_payload.get("breakdown_added_time_s", 0.0) or 0.0),
        float(disruption_payload.get("effective_process_time_s", 0.0) or 0.0) - float(disruption_payload.get("base_process_time_s", 0.0) or 0.0),
    )
    if disrupted_station_penalty_s <= 0.0:
        triggered_type = str(disruption_payload.get("triggered_disruption_type", ""))
        if triggered_type == "breakdown":
            disrupted_station_penalty_s = float(_config_value(config, "default_breakdown_penalty_s", 600.0))
        elif triggered_type == "efficiency_loss":
            disrupted_station_penalty_s = float(_config_value(config, "default_efficiency_penalty_s", 180.0))

    station_penalty_s_by_index: dict[int, float] = defaultdict(float)
    if disrupted_station_index > 0 and disrupted_station_penalty_s > 0.0:
        station_penalty_s_by_index[max(0, disrupted_station_index - 1)] = disrupted_station_penalty_s * disruption_penalty_scale

    sorted_units = sorted(waiting_units, key=_build_base_sort_key)
    candidate_pool = list(islice(sorted_units, candidate_pool_size))
    if not candidate_pool:
        return {"summary": {"decision": "empty_candidate_pool"}}

    depth = min(search_depth, len(candidate_pool))
    baseline_sequence = candidate_pool[:depth]
    baseline_result = _evaluate_sequence(
        sequence=baseline_sequence,
        current_time_s=current_time_s,
        available_system_slots=int(snapshot.get("available_system_slots", 0)),
        dispatch_spacing_s=dispatch_spacing_s,
        stage_instance_indices_zero_based=stage_instance_indices_zero_based,
        station_sequence=station_sequence,
        station_instance_base_names=station_instance_base_names,
        process_times=process_times,
        transport_lookup=transport_lookup,
        projected_station_available_time_s=projected_station_available_time_s,
        station_penalty_s_by_index=station_penalty_s_by_index,
        queue_penalty_factor=queue_penalty_factor,
        busy_penalty_factor=busy_penalty_factor,
        station_rows_by_index=station_rows_by_index,
        priority_weight=priority_weight,
        flow_weight=flow_weight,
        release_hold_weight=release_hold_weight,
    )

    beam: list[list[dict[str, Any]]] = [[]]
    for _depth_index in range(depth):
        expanded_sequences: list[tuple[float, list[dict[str, Any]]]] = []
        for partial_sequence in beam:
            used_unit_indices = {int(unit_row["unit_index"]) for unit_row in partial_sequence}
            for unit_row in candidate_pool:
                if int(unit_row["unit_index"]) in used_unit_indices:
                    continue
                candidate_sequence = partial_sequence + [unit_row]
                candidate_result = _evaluate_sequence(
                    sequence=candidate_sequence,
                    current_time_s=current_time_s,
                    available_system_slots=int(snapshot.get("available_system_slots", 0)),
                    dispatch_spacing_s=dispatch_spacing_s,
                    stage_instance_indices_zero_based=stage_instance_indices_zero_based,
                    station_sequence=station_sequence,
                    station_instance_base_names=station_instance_base_names,
                    process_times=process_times,
                    transport_lookup=transport_lookup,
                    projected_station_available_time_s=projected_station_available_time_s,
                    station_penalty_s_by_index=station_penalty_s_by_index,
                    queue_penalty_factor=queue_penalty_factor,
                    busy_penalty_factor=busy_penalty_factor,
                    station_rows_by_index=station_rows_by_index,
                    priority_weight=priority_weight,
                    flow_weight=flow_weight,
                    release_hold_weight=release_hold_weight,
                )
                expanded_sequences.append((candidate_result.score, candidate_sequence))

        expanded_sequences.sort(key=lambda row: row[0])
        beam = [sequence for _score, sequence in expanded_sequences[:beam_width]]
        if not beam:
            break

    best_sequence = beam[0] if beam else baseline_sequence
    best_result = _evaluate_sequence(
        sequence=best_sequence,
        current_time_s=current_time_s,
        available_system_slots=int(snapshot.get("available_system_slots", 0)),
        dispatch_spacing_s=dispatch_spacing_s,
        stage_instance_indices_zero_based=stage_instance_indices_zero_based,
        station_sequence=station_sequence,
        station_instance_base_names=station_instance_base_names,
        process_times=process_times,
        transport_lookup=transport_lookup,
        projected_station_available_time_s=projected_station_available_time_s,
        station_penalty_s_by_index=station_penalty_s_by_index,
        queue_penalty_factor=queue_penalty_factor,
        busy_penalty_factor=busy_penalty_factor,
        station_rows_by_index=station_rows_by_index,
        priority_weight=priority_weight,
        flow_weight=flow_weight,
        release_hold_weight=release_hold_weight,
    )

    improvement = float(baseline_result.score - best_result.score)
    if improvement <= float(_config_value(config, "minimum_improvement", 1e-6)):
        best_sequence = baseline_sequence
        best_result = baseline_result

    waiting_unit_score_boosts: dict[str, float] = {}
    for position, unit_index in enumerate(best_result.sequence):
        waiting_unit_score_boosts[str(unit_index)] = float((depth - position) * score_boost_step)

    station_penalties: list[dict[str, Any]] = []
    if disrupted_station_index > 0 and station_penalty_s_by_index:
        penalty_s = float(station_penalty_s_by_index[max(0, disrupted_station_index - 1)])
        expires_at_s = current_time_s + max(60.0, penalty_s)
        station_penalties.append(
            {
                "station_index": int(disrupted_station_index),
                "penalty_s": penalty_s,
                "expires_at_s": float(expires_at_s),
                "reason": f"lookahead_penalty_for_{str(disruption_payload.get('triggered_disruption_type', 'disruption'))}",
            }
        )

    return {
        "waiting_unit_score_boosts": waiting_unit_score_boosts,
        "station_penalties": station_penalties,
        "summary": {
            "decision": "beam_search_lookahead",
            "candidate_pool_size": int(len(candidate_pool)),
            "search_depth": int(depth),
            "beam_width": int(beam_width),
            "baseline_score": float(baseline_result.score),
            "best_score": float(best_result.score),
            "estimated_improvement": improvement,
            "recommended_sequence": best_result.sequence,
        },
    }
