from __future__ import annotations

import numpy as np
import argparse
import csv
import heapq
import json
import math
import re
import shutil
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
    order_id: str
    variant: str
    station_index: int
    station_name: str
    arrival_time_s: float
    start_time_s: float
    finish_time_s: float
    process_time_s: float
    base_process_time_s: float
    wait_time_s: float
    queue_length_on_arrival: int


@dataclass
class TransportRecord:
    unit_id: str
    order_id: str
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
    order_id: str
    variant: str
    first_arrival_time_s: float
    start_time_s: float
    completion_time_s: float
    flow_time_s: float
    active_flow_time_s: float
    time_spent_producing: float
    throughput_efficiency: float
    attempts: int
    route_taken: str = "0"


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
    queue: list[tuple[int, float, int, int, int]]
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
EVENT_RELEASE = "release"
EVENT_CART_RETURN = "cart_return"
EVENT_TIMED_FAILED_INSPECTION = "timed_failed_inspection"
EVENT_PRIORITY = {
    EVENT_FINISH: 0,
    EVENT_TIMED_FAILED_INSPECTION: 1,
    EVENT_CART_RETURN: 1,
    EVENT_ARRIVAL: 2,
    EVENT_RELEASE: 3,
}

MAX_UNITS_IN_SYSTEM = 8
RETURN_TO_STATION_1_TIME_S = 31.3

INPUT_BATCH_NAME_RE = re.compile(r"^orders_(\d{2})-(\d{2})_(\d{2})-(\d{2})_(\d+)$", re.IGNORECASE)
INPUT_ORDER_CSV_RE = re.compile(r"^orders_(\d{2})-(\d{2})_(\d{2})-(\d{2})_(\d+)\.csv$", re.IGNORECASE)
LINE_LAYOUT_FILENAME = "line_layout.json"
DISRUPTION_FILENAME = "disruption.json"
TIMED_DISRUPTION_FILENAME = "disruptions.csv"
TIMED_DISRUPTION_V2_FILENAME = "disruption_v2.json"
DISRUPTIONS_DIRNAME = "disruptions"
CURRENT_SCHEDULE_FILENAME = "current_schedule.csv"
CARRIER_SNAPSHOT_FILENAME = "carrier_snapshot.json"
LINE_STATE_SNAPSHOT_FILENAME = "line_state_snapshot.json"
LINE_LAYOUT_SETTINGS_KEYS = (
    "line_layout_file",
    "line_layout_filename",
    "layout_file",
    "layout_filename",
)
STATION_NAME_NUMBER_RE = re.compile(r"^\s*Station\s+(\d+)(?:\.(\d+))?\s*:\s*(.+?)\s*$", re.IGNORECASE)
TRANSPORT_NAME_NUMBER_RE = re.compile(r"^\s*Transportation\s+(\d+)\s*$", re.IGNORECASE)
ROUTE_PATTERN_RE = re.compile(r"^p:(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)$", re.IGNORECASE)

MATERIAL_STAGE_TO_MATERIAL = {
    1: ["Bottom cover"],
    3: ["Fuse", "PCB"],
    5: ["Top cover"],
}
INSPECTION_STAGE_NUMBER = 6
BROKEN_MATERIAL_EXTRA_TIME_DEFAULT_S = 30.0


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


def _load_optional_transport_time_data(data_dir: Path) -> dict[str, Any] | None:
    transport_path = data_dir / "transport_times.json"
    if transport_path.exists():
        return load_json(transport_path)
    return None


def _transport_time_data_for_layout(
    line_layout_config: dict[str, Any],
    transport_time_data: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        isinstance(transport_time_data, dict)
        and isinstance(transport_time_data.get("transport_times_between_consecutive_stations"), dict)
    ):
        return transport_time_data

    layout_transport_times = line_layout_config.get("transport_times_between_consecutive_stations")
    if isinstance(layout_transport_times, dict):
        return {"transport_times_between_consecutive_stations": layout_transport_times}

    raise FileNotFoundError(
        "Transport times were not found. Add 'transport_times_between_consecutive_stations' "
        "to the selected layout JSON, or restore data/transport_times.json."
    )


def _extract_station_name_parts(station_name: str) -> tuple[int | None, int | None, str]:
    match = STATION_NAME_NUMBER_RE.match(str(station_name).strip())
    if not match:
        return None, None, str(station_name).strip()

    stage_number_s, copy_number_s, label = match.groups()
    stage_number = int(stage_number_s)
    copy_number = int(copy_number_s) if copy_number_s is not None else None
    return stage_number, copy_number, label.strip()


def _make_station_instance_name(base_station_name: str, copy_index: int, total_copies: int) -> str:
    stage_number, _, label = _extract_station_name_parts(base_station_name)
    if total_copies <= 1:
        return base_station_name
    if stage_number is not None:
        return f"Station {stage_number}.{copy_index}: {label}"
    return f"{base_station_name}.{copy_index}"


def _transport_stage_number_from_name(transport_name: str) -> int | None:
    match = TRANSPORT_NAME_NUMBER_RE.match(str(transport_name).strip())
    if not match:
        return None
    return int(match.group(1))


def _make_default_line_layout(process_time_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "layout_name": "default_single_path",
        "station_instances": [
            {
                "station_name": station_name,
                "time_scale_factor": 1.0,
                "branch_transport_from_previous_s": 0.0,
                "branch_transport_to_next_s": 0.0,
            }
            for station_name in process_time_data["station_sequence"]
        ],
    }


def _build_base_station_number_lookup(process_time_data: dict[str, Any]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for base_station_name in process_time_data["station_sequence"]:
        stage_number, _, _ = _extract_station_name_parts(base_station_name)
        if stage_number is None:
            raise ValueError(
                f"Base station name '{base_station_name}' in process_times.json must follow 'Station N: Name'."
            )
        lookup[stage_number] = base_station_name
    return lookup


def _build_effective_line_layout_from_stage_definitions(
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any] | None,
    line_layout_config: dict[str, Any],
) -> dict[str, Any]:
    base_station_sequence = list(process_time_data["station_sequence"])
    resolved_transport_time_data = _transport_time_data_for_layout(line_layout_config, transport_time_data)
    base_transport_lookup = build_transport_lookup(base_station_sequence, resolved_transport_time_data)

    raw_stages = line_layout_config.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(
            f"{LINE_LAYOUT_FILENAME} must contain a non-empty 'stages' list."
        )

    resolved_stages: list[dict[str, Any]] = []
    configured_base_sequence: list[str] = []
    for stage_position, stage_entry in enumerate(raw_stages, start=1):
        if not isinstance(stage_entry, dict):
            raise ValueError(f"Stage entry {stage_position} in {LINE_LAYOUT_FILENAME} must be a JSON object.")

        base_station_name = str(stage_entry.get("base_station_name", "")).strip()
        if base_station_name == "":
            raise ValueError(f"Stage entry {stage_position} in {LINE_LAYOUT_FILENAME} is missing 'base_station_name'.")
        if base_station_name not in base_station_sequence:
            raise ValueError(
                f"Unknown base_station_name '{base_station_name}' in {LINE_LAYOUT_FILENAME}. Expected one of: {', '.join(base_station_sequence)}"
            )

        copies = int(stage_entry.get("copies", 1))
        if copies <= 0:
            raise ValueError(f"Stage '{base_station_name}' in {LINE_LAYOUT_FILENAME} must have copies >= 1.")

        branch_transport_from_previous_s = float(stage_entry.get("branch_transport_from_previous_s", 0.0))
        branch_transport_to_next_s = float(stage_entry.get("branch_transport_to_next_s", 0.0))
        time_scale_factor = float(stage_entry.get("time_scale_factor", 1.0))

        configured_base_sequence.append(base_station_name)
        resolved_stages.append(
            {
                "stage_position": stage_position,
                "base_station_name": base_station_name,
                "copies": copies,
                "branch_transport_from_previous_s": branch_transport_from_previous_s,
                "branch_transport_to_next_s": branch_transport_to_next_s,
                "time_scale_factor": time_scale_factor,
            }
        )

    if configured_base_sequence != base_station_sequence:
        raise ValueError(
            f"{LINE_LAYOUT_FILENAME} must keep the same base station order as process_times.json. Expected: {base_station_sequence}"
        )

    station_instance_names: list[str] = []
    station_instance_base_names: list[str] = []
    station_time_scale_factors: list[float] = []
    stage_instance_indices: list[list[int]] = []
    station_to_stage_index: list[int] = []

    for stage_index, stage_entry in enumerate(resolved_stages):
        instance_indices_for_stage: list[int] = []
        for copy_index in range(stage_entry["copies"]):
            instance_name = _make_station_instance_name(
                stage_entry["base_station_name"],
                copy_index,
                stage_entry["copies"],
            )
            station_instance_names.append(instance_name)
            station_instance_base_names.append(stage_entry["base_station_name"])
            station_time_scale_factors.append(float(stage_entry.get("time_scale_factor", 1.0)))
            station_to_stage_index.append(stage_index)
            instance_indices_for_stage.append(len(station_instance_names) - 1)
        stage_instance_indices.append(instance_indices_for_stage)

    effective_transport_lookup: dict[tuple[str, str], float] = {}
    for stage_index in range(len(resolved_stages) - 1):
        current_stage = resolved_stages[stage_index]
        next_stage = resolved_stages[stage_index + 1]
        base_transport_time_s = float(
            base_transport_lookup[(current_stage["base_station_name"], next_stage["base_station_name"])]
        )

        current_instance_indices = stage_instance_indices[stage_index]
        next_instance_indices = stage_instance_indices[stage_index + 1]

        for from_copy_position, from_instance_index in enumerate(current_instance_indices):
            from_station_name = station_instance_names[from_instance_index]
            extra_outbound_s = (
                float(current_stage["branch_transport_to_next_s"])
                if len(current_instance_indices) > 1 and from_copy_position > 0
                else 0.0
            )

            for to_copy_position, to_instance_index in enumerate(next_instance_indices):
                to_station_name = station_instance_names[to_instance_index]
                extra_inbound_s = (
                    float(next_stage["branch_transport_from_previous_s"])
                    if len(next_instance_indices) > 1 and to_copy_position > 0
                    else 0.0
                )
                effective_transport_lookup[(from_station_name, to_station_name)] = (
                    base_transport_time_s + extra_outbound_s + extra_inbound_s
                )

    return {
        "layout_name": line_layout_config.get("layout_name", "custom_layout"),
        "stages": resolved_stages,
        "station_sequence": station_instance_names,
        "station_instance_base_names": station_instance_base_names,
        "station_time_scale_factors": station_time_scale_factors,
        "stage_instance_indices": stage_instance_indices,
        "station_to_stage_index": station_to_stage_index,
        "transport_lookup": effective_transport_lookup,
        "base_station_sequence": base_station_sequence,
    }


def _normalize_station_instance_entries(
    process_time_data: dict[str, Any],
    line_layout_config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_entries = line_layout_config.get("station_instances")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            "Line layout config must contain a non-empty 'station_instances' list when using the station-instance format."
        )

    base_station_by_stage_number = _build_base_station_number_lookup(process_time_data)
    normalized_entries: list[dict[str, Any]] = []
    seen_instance_names: set[str] = set()

    for entry_index, raw_entry in enumerate(raw_entries, start=1):
        if isinstance(raw_entry, str):
            station_name = raw_entry.strip()
            branch_transport_from_previous_s = 0.0
            branch_transport_to_next_s = 0.0
            time_scale_factor = 1.0
        elif isinstance(raw_entry, dict):
            station_name = str(
                raw_entry.get("station_name", raw_entry.get("name", raw_entry.get("station", "")))
            ).strip()
            branch_transport_from_previous_s = float(raw_entry.get("branch_transport_from_previous_s", 0.0))
            branch_transport_to_next_s = float(raw_entry.get("branch_transport_to_next_s", 0.0))
            time_scale_factor = float(raw_entry.get("time_scale_factor", 1.0))
        else:
            raise ValueError(
                f"station_instances entry {entry_index} must be either a string or a JSON object."
            )

        if station_name == "":
            raise ValueError(f"station_instances entry {entry_index} is missing 'station_name'.")
        if station_name in seen_instance_names:
            raise ValueError(f"station_instances contains duplicate station_name '{station_name}'.")
        seen_instance_names.add(station_name)

        stage_number, copy_number, label = _extract_station_name_parts(station_name)
        if stage_number is None:
            raise ValueError(
                f"station_name '{station_name}' must follow the format 'Station N: Name' or 'Station N.M: Name'."
            )
        if stage_number not in base_station_by_stage_number:
            raise ValueError(
                f"station_name '{station_name}' refers to stage {stage_number}, but that stage does not exist in process_times.json."
            )

        base_station_name = base_station_by_stage_number[stage_number]
        _, _, base_label = _extract_station_name_parts(base_station_name)
        if label.casefold() != base_label.casefold():
            raise ValueError(
                f"station_name '{station_name}' does not match the base station label '{base_label}' for stage {stage_number}."
            )

        normalized_entries.append(
            {
                "station_name": station_name,
                "base_station_name": base_station_name,
                "stage_number": stage_number,
                "copy_number": copy_number,
                "branch_transport_from_previous_s": branch_transport_from_previous_s,
                "branch_transport_to_next_s": branch_transport_to_next_s,
                "time_scale_factor": time_scale_factor,
                "declared_order": entry_index,
            }
        )

    return normalized_entries


def _build_effective_line_layout_from_station_instances(
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any] | None,
    line_layout_config: dict[str, Any],
) -> dict[str, Any]:
    base_station_sequence = list(process_time_data["station_sequence"])
    resolved_transport_time_data = _transport_time_data_for_layout(line_layout_config, transport_time_data)
    base_transport_lookup = build_transport_lookup(base_station_sequence, resolved_transport_time_data)
    base_station_by_stage_number = _build_base_station_number_lookup(process_time_data)
    normalized_entries = _normalize_station_instance_entries(process_time_data, line_layout_config)

    entries_by_stage_number: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in normalized_entries:
        entries_by_stage_number[int(entry["stage_number"])].append(entry)

    expected_stage_numbers = list(base_station_by_stage_number.keys())
    missing_stage_numbers = [stage_number for stage_number in expected_stage_numbers if stage_number not in entries_by_stage_number]
    if missing_stage_numbers:
        raise ValueError(
            f"Line layout config is missing stage(s): {', '.join(str(stage_number) for stage_number in missing_stage_numbers)}"
        )

    unexpected_stage_numbers = [stage_number for stage_number in entries_by_stage_number if stage_number not in base_station_by_stage_number]
    if unexpected_stage_numbers:
        raise ValueError(
            f"Line layout config contains unknown stage number(s): {', '.join(str(stage_number) for stage_number in sorted(unexpected_stage_numbers))}"
        )

    station_instance_names: list[str] = []
    station_instance_base_names: list[str] = []
    station_time_scale_factors: list[float] = []
    stage_instance_indices: list[list[int]] = []
    station_to_stage_index: list[int] = []
    stage_entries_resolved: list[dict[str, Any]] = []
    instance_entry_by_name: dict[str, dict[str, Any]] = {}

    for stage_index, base_station_name in enumerate(base_station_sequence):
        stage_number, _, _ = _extract_station_name_parts(base_station_name)
        if stage_number is None:
            raise ValueError(f"Could not determine stage number for base station '{base_station_name}'.")

        stage_entries = sorted(
            entries_by_stage_number[stage_number],
            key=lambda entry: (
                -1 if entry["copy_number"] is None else int(entry["copy_number"]),
                int(entry["declared_order"]),
            ),
        )

        instance_indices_for_stage: list[int] = []
        for entry in stage_entries:
            station_instance_names.append(str(entry["station_name"]))
            station_instance_base_names.append(str(entry["base_station_name"]))
            station_time_scale_factors.append(float(entry.get("time_scale_factor", 1.0)))
            station_to_stage_index.append(stage_index)
            instance_indices_for_stage.append(len(station_instance_names) - 1)
            instance_entry_by_name[str(entry["station_name"])] = entry

        stage_instance_indices.append(instance_indices_for_stage)
        stage_entries_resolved.append(
            {
                "stage_position": stage_index + 1,
                "base_station_name": base_station_name,
                "copies": len(stage_entries),
                "station_names": [str(entry["station_name"]) for entry in stage_entries],
            }
        )

    effective_transport_lookup: dict[tuple[str, str], float] = {}
    for stage_index in range(len(base_station_sequence) - 1):
        current_base_station_name = base_station_sequence[stage_index]
        next_base_station_name = base_station_sequence[stage_index + 1]
        base_transport_time_s = float(
            base_transport_lookup[(current_base_station_name, next_base_station_name)]
        )

        current_instance_indices = stage_instance_indices[stage_index]
        next_instance_indices = stage_instance_indices[stage_index + 1]

        for from_instance_index in current_instance_indices:
            from_station_name = station_instance_names[from_instance_index]
            from_entry = instance_entry_by_name[from_station_name]
            extra_outbound_s = float(from_entry.get("branch_transport_to_next_s", 0.0))

            for to_instance_index in next_instance_indices:
                to_station_name = station_instance_names[to_instance_index]
                to_entry = instance_entry_by_name[to_station_name]
                extra_inbound_s = float(to_entry.get("branch_transport_from_previous_s", 0.0))
                effective_transport_lookup[(from_station_name, to_station_name)] = (
                    base_transport_time_s + extra_outbound_s + extra_inbound_s
                )

    return {
        "layout_name": line_layout_config.get("layout_name", "custom_layout"),
        "stages": stage_entries_resolved,
        "station_sequence": station_instance_names,
        "station_instance_base_names": station_instance_base_names,
        "station_time_scale_factors": station_time_scale_factors,
        "stage_instance_indices": stage_instance_indices,
        "station_to_stage_index": station_to_stage_index,
        "transport_lookup": effective_transport_lookup,
        "base_station_sequence": base_station_sequence,
        "station_instances": normalized_entries,
    }


def load_line_layout_config(layout_path: Path | None, process_time_data: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if layout_path is not None:
        if not layout_path.exists():
            raise FileNotFoundError(f"Line layout file not found: {layout_path}")
        return load_json(layout_path), layout_path
    return _make_default_line_layout(process_time_data), None


def _carriers_from_layout_config(
    line_layout_config: dict[str, Any] | None,
    fallback_carriers: int = MAX_UNITS_IN_SYSTEM,
) -> int:
    if isinstance(line_layout_config, dict) and "carriers" in line_layout_config:
        try:
            layout_carriers = int(float(line_layout_config.get("carriers")))
            if layout_carriers > 0:
                return layout_carriers
        except (TypeError, ValueError):
            pass
    return int(fallback_carriers)


def build_effective_line_layout(
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any] | None,
    line_layout_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if line_layout_config is None:
        line_layout_config = _make_default_line_layout(process_time_data)

    if isinstance(line_layout_config.get("station_instances"), list):
        return _build_effective_line_layout_from_station_instances(
            process_time_data=process_time_data,
            transport_time_data=transport_time_data,
            line_layout_config=line_layout_config,
        )

    if isinstance(line_layout_config.get("stages"), list):
        return _build_effective_line_layout_from_stage_definitions(
            process_time_data=process_time_data,
            transport_time_data=transport_time_data,
            line_layout_config=line_layout_config,
        )

    raise ValueError(
        "Line layout config must contain either 'station_instances' or 'stages'."
    )


def _resolve_line_layout_filename_from_settings(settings_data: dict[str, Any]) -> str | None:
    for key in LINE_LAYOUT_SETTINGS_KEYS:
        value = settings_data.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value.strip()
    return None


def resolve_line_layout_path(
    selected_layout_name: str | None,
    input_root: Path | None,
    batch_dir: Path | None,
    data_dir: Path,
) -> Path | None:
    if selected_layout_name is None or str(selected_layout_name).strip() == "":
        batch_default = (batch_dir / LINE_LAYOUT_FILENAME) if batch_dir is not None else None
        if batch_default is not None and batch_default.exists():
            return batch_default

        input_default = (input_root / LINE_LAYOUT_FILENAME) if input_root is not None else None
        if input_default is not None and input_default.exists():
            return input_default

        data_default = data_dir / LINE_LAYOUT_FILENAME
        if data_default.exists():
            return data_default
        return None

    candidate_text = str(selected_layout_name).strip()
    candidate_path = Path(candidate_text)
    if candidate_path.is_absolute():
        if candidate_path.exists():
            return candidate_path
        raise FileNotFoundError(f"Selected line layout file was not found: {candidate_path}")

    search_locations: list[Path] = []
    if batch_dir is not None:
        search_locations.extend([batch_dir, batch_dir / "layouts"])
    if input_root is not None:
        search_locations.extend([input_root, input_root / "layouts"])
    search_locations.extend([data_dir, data_dir / "layouts"])

    checked_paths: list[Path] = []
    for location in search_locations:
        resolved = location / candidate_text
        checked_paths.append(resolved)
        if resolved.exists():
            return resolved

    recursive_matches: list[Path] = []
    if input_root is not None and input_root.exists():
        recursive_matches.extend(sorted(path for path in input_root.rglob(candidate_text) if path.is_file()))
    recursive_matches.extend(sorted(path for path in data_dir.rglob(candidate_text) if path.is_file()))

    unique_matches: list[Path] = []
    seen_match_keys: set[str] = set()
    for match in recursive_matches:
        match_key = str(match.resolve())
        if match_key not in seen_match_keys:
            unique_matches.append(match)
            seen_match_keys.add(match_key)

    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        raise FileNotFoundError(
            "Multiple line layout files matched "
            f"'{candidate_text}': {', '.join(str(path) for path in unique_matches)}. "
            "Choose a more specific file name or path."
        )

    checked_text = ", ".join(str(path) for path in checked_paths)
    raise FileNotFoundError(
        f"Could not find line layout file '{candidate_text}'. Checked: {checked_text}"
    )


def _parse_input_batch_sort_key(name: str) -> tuple[int, int, int, int, int]:
    stem = Path(name).stem
    match = INPUT_BATCH_NAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"Input batch name '{name}' must match orders_DD-MM_HH-MM_N or orders_DD-MM_HH-MM_N.csv"
        )

    day_s, month_s, hour_s, minute_s, sequence_s = match.groups()
    day = int(day_s)
    month = int(month_s)
    hour = int(hour_s)
    minute = int(minute_s)
    sequence = int(sequence_s)
    return (month, day, hour, minute, sequence)


def _read_int(cell_value: str, default: int = 0) -> int:
    text_value = str(cell_value).strip()
    if text_value == "" or text_value.casefold() == "nan":
        return default
    numeric_value = float(text_value)
    if math.isnan(numeric_value):
        return default
    return int(numeric_value)


def _read_float(cell_value: str, default: float = 0.0) -> float:
    text_value = str(cell_value).strip()
    if text_value == "" or text_value.casefold() == "nan":
        return default
    numeric_value = float(text_value)
    if math.isnan(numeric_value):
        return default
    return numeric_value


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
            f"No generated input folders were found in {input_root}. Expected folders like orders_DD-MM_HH-MM_N"
        )

    return max(candidate_dirs, key=lambda path: _parse_input_batch_sort_key(path.name))


def find_newest_orders_csv(batch_dir: Path) -> Path:
    candidate_csv_files = [
        path
        for path in batch_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and INPUT_ORDER_CSV_RE.match(path.name)
    ]

    if candidate_csv_files:
        return max(candidate_csv_files, key=lambda path: _parse_input_batch_sort_key(path.name))

    fallback_csv_files = [
        path
        for path in batch_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name.casefold() != TIMED_DISRUPTION_FILENAME.casefold()
        and not path.name.casefold().startswith("disruption_list")
        and not path.name.casefold().startswith("disruptions")
    ]
    if fallback_csv_files:
        return max(fallback_csv_files, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(f"No order CSV file was found in {batch_dir}")


def load_latest_generated_input(
    input_root: Path, valid_variants: set[str]
) -> dict[str, Any]:
    batch_dir = find_newest_input_batch_dir(input_root)
    orders_csv_path = find_newest_orders_csv(batch_dir)
    settings_path = batch_dir / "settings.json"

    if not settings_path.exists():
        raise FileNotFoundError(f"settings.json was not found in {batch_dir}")

    settings_data = load_json(settings_path)
    simulation_time_s = float(settings_data.get("sim_time [s]", settings_data.get("Sim_time [s]", 0.0)))
    carriers = int(float(settings_data.get("carriers", {}).get("number of carriers", MAX_UNITS_IN_SYSTEM)))

    expanded_units: list[str] = []
    unit_release_times: list[float] = []
    unit_priorities: list[int] = []
    unit_order_ids: list[str] = []
    order_rows: list[dict[str, Any]] = []

    with orders_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Order CSV is empty: {orders_csv_path}")

        for row_index, row in enumerate(reader, start=1):
            if not row or not any(str(cell).strip() for cell in row):
                continue

            order_id = str(row[0]).strip() if len(row) > 0 else str(row_index)
            order_time_s = _read_float(row[1], 0.0) if len(row) > 1 else 0.0
            priority = max(1, _read_int(row[2], 1)) if len(row) > 2 else 1

            row_variants: list[dict[str, Any]] = []
            for col_idx in range(3, len(row), 2):
                variant_text = str(row[col_idx]).strip().upper() if col_idx < len(row) else ""
                quantity = _read_int(row[col_idx + 1], 0) if col_idx + 1 < len(row) else 0

                if variant_text == "":
                    continue
                if variant_text not in valid_variants:
                    raise ValueError(
                        f"Unknown variant '{variant_text}' in {orders_csv_path.name}. Valid options: {', '.join(sorted(valid_variants))}"
                    )
                if quantity < 0:
                    raise ValueError(
                        f"Negative quantity for variant '{variant_text}' in row {row_index} of {orders_csv_path.name}"
                    )

                row_variants.append(
                    {
                        "variant": variant_text,
                        "quantity": quantity,
                        "variant_slot_index": (col_idx - 3) // 2,
                    }
                )

            order_rows.append(
                {
                    "order_id": order_id,
                    "order_time_s": order_time_s,
                    "priority": priority,
                    "variants": row_variants,
                    "row_index": row_index,
                }
            )

    order_rows.sort(key=lambda row: (row["order_time_s"], -row["priority"], row["row_index"], str(row["order_id"])))

    for row in order_rows:
        if row["order_time_s"] > simulation_time_s:
            continue

        for variant_entry in row["variants"]:
            variant = variant_entry["variant"]
            quantity = int(variant_entry["quantity"])
            for _ in range(quantity):
                expanded_units.append(variant)
                unit_release_times.append(float(row["order_time_s"]))
                unit_priorities.append(int(row["priority"]))
                unit_order_ids.append(str(row["order_id"]))

    batch_name = batch_dir.name
    order_text = batch_name
    if expanded_units:
        order_mix = Counter(expanded_units)
        mix_text = ", ".join(f"{qty}x{variant}" for variant, qty in sorted(order_mix.items()))
        order_text = f"{batch_name}__{mix_text}"

    return {
        "order_text": order_text,
        "ordered_units": expanded_units,
        "unit_release_times": unit_release_times,
        "unit_priorities": unit_priorities,
        "unit_order_ids": unit_order_ids,
        "simulation_time_s": simulation_time_s,
        "carriers": carriers,
        "settings_data": settings_data,
        "selected_line_layout_name": _resolve_line_layout_filename_from_settings(settings_data),
        "input_root": input_root,
        "batch_dir": batch_dir,
        "orders_csv_path": orders_csv_path,
        "settings_path": settings_path,
    }


def _normalize_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        probability = probability / 100.0
    return min(1.0, probability)


def _seed_to_rng(seed_value: Any) -> np.random.Generator:
    if seed_value is None:
        return np.random.default_rng()
    seed_text = str(seed_value).strip()
    if seed_text.isdigit():
        return np.random.default_rng(int(seed_text))
    return np.random.default_rng(abs(hash(seed_text)) % (2**32))


def _sample_linear_from_range(
    rng: np.random.Generator,
    range_values: Any,
    default_value: float = 0.0,
) -> tuple[float, float]:
    if isinstance(range_values, (list, tuple)) and len(range_values) >= 2:
        low = float(range_values[0])
        high = float(range_values[1])
        draw = float(rng.random())
        return low + (high - low) * draw, draw
    return float(default_value), 0.0


def _settings_disruption_mode(settings_data: dict[str, Any]) -> int:
    raw_settings = settings_data.get("random based disruptions", {})
    if isinstance(raw_settings, dict):
        try:
            return int(raw_settings.get("enabled", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _settings_random_disruptions_enabled(settings_data: dict[str, Any]) -> bool:
    return _settings_disruption_mode(settings_data) == 1


def _settings_timed_disruptions_enabled(settings_data: dict[str, Any]) -> bool:
    return _settings_disruption_mode(settings_data) == 2


def _broken_material_extra_time_s(disruption_config: dict[str, Any] | None) -> float:
    if not isinstance(disruption_config, dict):
        return BROKEN_MATERIAL_EXTRA_TIME_DEFAULT_S
    material_config = disruption_config.get("Material", {})
    if not isinstance(material_config, dict):
        return BROKEN_MATERIAL_EXTRA_TIME_DEFAULT_S
    raw_value = material_config.get("broken material extra time [s]", BROKEN_MATERIAL_EXTRA_TIME_DEFAULT_S)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return BROKEN_MATERIAL_EXTRA_TIME_DEFAULT_S


def resolve_disruption_path(input_root: Path | None, batch_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if batch_dir is not None:
        candidates.append(batch_dir / DISRUPTION_FILENAME)
    if input_root is not None:
        candidates.append(input_root / DISRUPTION_FILENAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_timed_disruption_csv_path(input_root: Path | None, batch_dir: Path | None) -> Path | None:
    exact_candidates: list[Path] = []
    if batch_dir is not None:
        exact_candidates.append(batch_dir / TIMED_DISRUPTION_FILENAME)
    if input_root is not None:
        exact_candidates.append(input_root / TIMED_DISRUPTION_FILENAME)

    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    fuzzy_candidates: list[Path] = []
    for search_dir in [path for path in (batch_dir, input_root) if path is not None]:
        fuzzy_candidates.extend(sorted(path for path in search_dir.glob("disruption_list*.csv") if path.is_file()))
        fuzzy_candidates.extend(sorted(path for path in search_dir.glob("disruptions*.csv") if path.is_file()))

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in fuzzy_candidates:
        resolved_key = str(candidate.resolve())
        if resolved_key not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved_key)

    if not unique_candidates:
        return None

    return max(unique_candidates, key=lambda path: path.stat().st_mtime)


def copy_file_if_exists(source_path: Path | None, target_path: Path) -> bool:
    if source_path is None or not source_path.exists():
        return False
    shutil.copy2(source_path, target_path)
    return True


def build_material_report(
    requested_units: list[str],
    consumed_units: list[str],
    bom_data: dict[str, Any],
    material_stock_data: dict[str, Any],
    extra_material_consumed: dict[str, int] | None = None,
    actual_material_consumed: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    initial_stock = {
        material: int(qty)
        for material, qty in material_stock_data["materials_in_stock"].items()
    }
    requested_requirements = calculate_material_requirements(requested_units, bom_data)
    consumed_requirements = calculate_material_requirements(consumed_units, bom_data)
    extra_material_consumed = {str(k): int(v) for k, v in (extra_material_consumed or {}).items()}
    actual_material_consumed = {str(k): int(v) for k, v in (actual_material_consumed or {}).items()}

    all_materials = sorted(
        set(initial_stock.keys())
        | set(requested_requirements.keys())
        | set(consumed_requirements.keys())
        | set(extra_material_consumed.keys())
        | set(actual_material_consumed.keys())
    )

    material_report: dict[str, dict[str, int]] = {}
    for material in all_materials:
        available = int(initial_stock.get(material, 0))
        requested = int(requested_requirements.get(material, 0))
        if actual_material_consumed:
            consumed = int(actual_material_consumed.get(material, 0))
        else:
            consumed = int(consumed_requirements.get(material, 0)) + int(extra_material_consumed.get(material, 0))
        remaining = max(0, available - consumed)
        unmet = max(0, requested - available)
        material_report[material] = {
            "requested_for_full_order": requested,
            "available_at_start": available,
            "consumed_for_produced_units": consumed,
            "remaining_after_run": remaining,
            "unmet_for_full_order": unmet,
        }
    return material_report


def _materials_relevant_for_stage(variant: str, stage_number: int, bom_data: dict[str, Any]) -> list[str]:
    material_names = MATERIAL_STAGE_TO_MATERIAL.get(int(stage_number))
    if material_names is None:
        return []
    if isinstance(material_names, str):
        material_names = [material_names]
    variant_bom = bom_data.get("bom_units_per_phone", {}).get(variant, {})
    return [
        str(material_name)
        for material_name in material_names
        if int(variant_bom.get(str(material_name), 0)) > 0
    ]


def evaluate_operation_disruptions(
    stage_number: int,
    station_name: str,
    variant: str,
    base_process_time_s: float,
    disruption_config: dict[str, Any] | None,
    rng: np.random.Generator | None,
    bom_data: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage_number": int(stage_number),
        "station_name": station_name,
        "base_process_time_s": float(base_process_time_s),
        "effective_process_time_s": float(base_process_time_s),
        "triggered_disruption_type": None,
        "breakdown_triggered": False,
        "breakdown_added_time_s": 0.0,
        "breakdown_random": None,
        "breakdown_duration_random": None,
        "efficiency_loss_triggered": False,
        "efficiency_drop_percent": 0.0,
        "efficiency_multiplier": 1.0,
        "efficiency_random": None,
        "terminal_failure_type": None,
        "terminal_failure_material": None,
        "material_broken_triggered": False,
        "material_broken_random": None,
        "material_broken_added_time_s": 0.0,
        "material_broken_extra_material_name": None,
        "material_broken_extra_material_qty": 0,
        "material_ran_out_triggered": False,
        "material_ran_out_random": None,
        "inspection_failed_triggered": False,
        "inspection_random": None,
    }

    if disruption_config is None or rng is None:
        return result

    stations_config = disruption_config.get("Stations", {})
    station_config = stations_config.get(str(stage_number), {}) if isinstance(stations_config, dict) else {}

    breakdown_config = station_config.get("breakdown", {}) if isinstance(station_config, dict) else {}
    breakdown_probability = _normalize_probability(breakdown_config.get("Machine breakdown chance [%]", 0.0))
    breakdown_random = float(rng.random())
    result["breakdown_random"] = breakdown_random
    if breakdown_random <= breakdown_probability:
        breakdown_added_time_s, breakdown_duration_random = _sample_linear_from_range(
            rng,
            breakdown_config.get("range"),
            default_value=breakdown_config.get("duration [s]", 0.0),
        )
        result["triggered_disruption_type"] = "breakdown"
        result["breakdown_triggered"] = True
        result["breakdown_added_time_s"] = float(breakdown_added_time_s)
        result["breakdown_duration_random"] = float(breakdown_duration_random)
        result["effective_process_time_s"] = float(base_process_time_s) + float(result["breakdown_added_time_s"])
        return result

    efficiency_config = station_config.get("efficiency loss", {}) if isinstance(station_config, dict) else {}
    efficiency_probability = _normalize_probability(efficiency_config.get("efficiency drop chance [%]", 0.0))
    efficiency_random = float(rng.random())
    result["efficiency_random"] = efficiency_random
    if efficiency_random <= efficiency_probability:
        efficiency_drop_percent = float(efficiency_config.get("efficiency drop [%]", 0.0))
        effective_speed_fraction = max(1e-9, 1.0 - (efficiency_drop_percent / 100.0))
        result["triggered_disruption_type"] = "efficiency_loss"
        result["efficiency_loss_triggered"] = True
        result["efficiency_drop_percent"] = efficiency_drop_percent
        result["efficiency_multiplier"] = 1.0 / effective_speed_fraction
        result["effective_process_time_s"] = float(base_process_time_s) * float(result["efficiency_multiplier"])
        return result

    if bom_data is not None:
        materials_config = disruption_config.get("Material", {})
        relevant_materials = _materials_relevant_for_stage(variant, stage_number, bom_data)
        if relevant_materials:
            selected_material = str(relevant_materials[int(rng.integers(0, len(relevant_materials)))])
            broken_probability = _normalize_probability(materials_config.get("Broken material chance [%]", 0.0))
            broken_random = float(rng.random())
            result["material_broken_random"] = broken_random
            if broken_random <= broken_probability:
                result["triggered_disruption_type"] = "broken_material"
                result["material_broken_triggered"] = True
                result["terminal_failure_material"] = selected_material
                result["material_broken_extra_material_name"] = selected_material
                result["material_broken_extra_material_qty"] = 1
                result["material_broken_added_time_s"] = float(_broken_material_extra_time_s(disruption_config))
                result["effective_process_time_s"] = float(base_process_time_s) + float(result["material_broken_added_time_s"])
                return result

            ran_out_probability = _normalize_probability(materials_config.get("ran out of material chance [%]", 0.0))
            ran_out_random = float(rng.random())
            result["material_ran_out_random"] = ran_out_random
            if ran_out_random <= ran_out_probability:
                result["triggered_disruption_type"] = "ran_out_of_material"
                result["material_ran_out_triggered"] = True
                result["terminal_failure_type"] = "ran_out_of_material"
                result["terminal_failure_material"] = selected_material
                return result

    if int(stage_number) == INSPECTION_STAGE_NUMBER:
        inspection_config = station_config.get("failed inspection", {}) if isinstance(station_config, dict) else {}
        inspection_probability = _normalize_probability(inspection_config.get("wrong assembly chance", 0.0))
        inspection_random = float(rng.random())
        result["inspection_random"] = inspection_random
        if inspection_random <= inspection_probability:
            result["triggered_disruption_type"] = "failed_inspection"
            result["inspection_failed_triggered"] = True
            result["terminal_failure_type"] = "failed_inspection"
            return result


    return result


def _is_nan_like(value: Any) -> bool:
    value_text = str(value).strip()
    return value_text == "" or value_text.casefold() == "nan"


def _normalize_station_disruption_id(station_id_value: Any) -> str | None:
    if _is_nan_like(station_id_value):
        return None
    raw_text = str(station_id_value).strip()
    try:
        numeric_value = float(raw_text)
    except (TypeError, ValueError):
        return raw_text
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.12f}".rstrip("0").rstrip(".")


def _station_disruption_id_from_station_name(station_name: str) -> str:
    stage_number, copy_number, _ = _extract_station_name_parts(station_name)
    if stage_number is None:
        return str(station_name).strip()
    if copy_number is None:
        return str(stage_number)
    return f"{stage_number}.{copy_number}"


def _is_emergency_unit_id(unit_id: Any) -> bool:
    unit_id_text = str(unit_id).strip().upper()
    return bool(re.fullmatch(r"E\d+", unit_id_text))


def _next_emergency_unit_number(existing_unit_ids: list[str] | None) -> int:
    highest_emergency_number = 0
    for unit_id in existing_unit_ids or []:
        unit_id_text = str(unit_id).strip().upper()
        match = re.fullmatch(r"E(\d+)", unit_id_text)
        if match:
            highest_emergency_number = max(highest_emergency_number, int(match.group(1)))
    return highest_emergency_number + 1


def _assign_missing_emergency_order_ids(
    timed_disruption_records: list[dict[str, Any]],
    existing_order_ids: list[str],
    existing_unit_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    numeric_order_ids: list[int] = []
    existing_unit_ids = list(existing_unit_ids or [])

    for index, order_id in enumerate(existing_order_ids):
        order_id_text = str(order_id).strip()
        if not order_id_text.isdigit():
            continue

        # Emergency units are named E001, E002, ... .  Those order IDs were
        # generated from this same function in an earlier segment, so they must
        # not push the next emergency order ID upward on later segments.
        if index < len(existing_unit_ids) and _is_emergency_unit_id(existing_unit_ids[index]):
            continue

        numeric_order_ids.append(int(order_id_text))

    next_numeric_order_id = (max(numeric_order_ids) + 1) if numeric_order_ids else 1
    used_generated_order_ids: set[int] = set()

    for record in sorted(timed_disruption_records, key=lambda item: (float(item.get("start_time_s", 0.0) or 0.0), int(item.get("row_index", 0) or 0))):
        if str(record.get("disruption_type", "")).strip().lower() != "emergency_order":
            continue
        if record.get("order_id") is not None:
            continue

        while next_numeric_order_id in used_generated_order_ids:
            next_numeric_order_id += 1

        record["order_id"] = str(next_numeric_order_id)
        used_generated_order_ids.add(next_numeric_order_id)
        next_numeric_order_id += 1

    return timed_disruption_records


def load_timed_disruption_csv(csv_path: Path, valid_variants: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return records
        for row_index, row in enumerate(reader, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            padded_row = list(row) + [""] * max(0, 14 - len(row))
            disruption_type = str(padded_row[0]).strip()
            if disruption_type == "":
                continue

            emergency_variants: list[tuple[str, int]] = []
            for variant_col_idx, quantity_col_idx in ((8, 9), (10, 11), (12, 13)):
                variant_text = str(padded_row[variant_col_idx]).strip().upper()
                quantity_value = 0 if _is_nan_like(padded_row[quantity_col_idx]) else _read_int(padded_row[quantity_col_idx], 0)
                if variant_text == "" or variant_text.casefold() == "nan" or quantity_value <= 0:
                    continue
                if variant_text not in valid_variants:
                    raise ValueError(
                        f"Unknown variant '{variant_text}' in timed disruption CSV row {row_index}. "
                        f"Valid options: {', '.join(sorted(valid_variants))}"
                    )
                emergency_variants.append((variant_text, int(quantity_value)))

            records.append(
                {
                    "disruption_type": disruption_type,
                    "station_id": _normalize_station_disruption_id(padded_row[1]),
                    "start_time_s": _read_float(padded_row[2], 0.0),
                    "end_time_s": None if _is_nan_like(padded_row[3]) else _read_float(padded_row[3], 0.0),
                    "efficiency_percentage": None if _is_nan_like(padded_row[4]) else _read_float(padded_row[4], 100.0),
                    "order_id": None if _is_nan_like(padded_row[5]) else str(padded_row[5]).strip(),
                    "due_date": None if _is_nan_like(padded_row[6]) else _read_float(padded_row[6], _read_float(padded_row[2], 0.0)),
                    "order_time_s": _read_float(padded_row[2], 0.0),
                    "priority": None if _is_nan_like(padded_row[7]) else max(1, _read_int(padded_row[7], 1)),
                    "emergency_variants": emergency_variants,
                    "row_index": int(row_index),
                }
            )
    return records


def prepare_timed_disruption_data(
    station_sequence: list[str],
    timed_disruption_records: list[dict[str, Any]],
    timed_disruption_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    station_index_by_disruption_id = {
        _station_disruption_id_from_station_name(station_name): station_index
        for station_index, station_name in enumerate(station_sequence)
    }

    breakdown_names_by_stage: dict[int, set[str]] = {}
    if isinstance(timed_disruption_config, dict):
        stations_cfg = timed_disruption_config.get("Stations", {})
        if isinstance(stations_cfg, dict):
            for stage_key, stage_cfg in stations_cfg.items():
                try:
                    stage_number = int(stage_key)
                except (TypeError, ValueError):
                    continue
                machine_breakdowns = stage_cfg.get("machine breakdowns", []) if isinstance(stage_cfg, dict) else []
                breakdown_names_by_stage[stage_number] = {
                    str(entry.get("name", "")).strip().casefold()
                    for entry in machine_breakdowns
                    if isinstance(entry, dict) and str(entry.get("name", "")).strip() != ""
                }

    breakdown_windows_by_station: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    efficiency_windows_by_station: defaultdict[int, list[tuple[float, float, float]]] = defaultdict(list)
    failed_inspection_times_by_station: defaultdict[int, list[float]] = defaultdict(list)
    emergency_orders: list[dict[str, Any]] = []

    for record in timed_disruption_records:
        disruption_type_raw = str(record.get("disruption_type", "")).strip()
        disruption_type = disruption_type_raw.casefold()
        station_id = record.get("station_id")
        station_index: int | None = None
        stage_number: int | None = None

        if disruption_type == "emergency_order":
            order_id = record.get("order_id")
            if order_id is None:
                raise ValueError("emergency_order requires order_id.")
            order_time_s = float(record["order_time_s"]) if record.get("order_time_s") is not None else float(record["start_time_s"])
            priority_value = int(record["priority"]) if record.get("priority") is not None else 1
            emergency_orders.append({"order_id": str(order_id), "order_time_s": float(order_time_s), "priority": int(priority_value), "variants": list(record.get("emergency_variants", []))})
            continue

        if station_id is not None:
            station_index = station_index_by_disruption_id.get(str(station_id))
            if station_index is None:
                # Ignore disruptions for station copies not present in this layout.
                continue
            stage_number, _, _ = _extract_station_name_parts(station_sequence[int(station_index)])

        if station_index is None:
            continue

        start_time_s = float(record["start_time_s"])
        end_time_s = float(record["end_time_s"]) if record.get("end_time_s") is not None else start_time_s

        if disruption_type in {"efficiency_loss", "efficiency loss"}:
            if end_time_s < start_time_s:
                continue
            efficiency_percentage = float(record["efficiency_percentage"]) if record.get("efficiency_percentage") is not None else 100.0
            efficiency_fraction = max(0.0, min(1.0, efficiency_percentage / 100.0))
            efficiency_windows_by_station[station_index].append((start_time_s, end_time_s, efficiency_fraction))
            continue

        if disruption_type in {"failed_inspection", "inspection_failure", "inspection failure"}:
            failed_inspection_times_by_station[station_index].append(float(record["start_time_s"]))
            continue

        is_breakdown_type = disruption_type == "machine_breakdown"
        if not is_breakdown_type and stage_number is not None:
            valid_breakdown_names = breakdown_names_by_stage.get(int(stage_number), set())
            if disruption_type in valid_breakdown_names:
                is_breakdown_type = True

        if is_breakdown_type:
            if end_time_s < start_time_s:
                continue
            breakdown_windows_by_station[station_index].append((start_time_s, end_time_s))
            continue

        raise ValueError(
            f"Unknown timed disruption_type '{record.get('disruption_type')}'. "
            "Expected one of: machine_breakdown, efficiency_loss, failed_inspection, emergency_order, "
            "or a breakdown subtype defined in disruption_v2.json."
        )

    for station_index in breakdown_windows_by_station:
        breakdown_windows_by_station[station_index].sort()
    for station_index in efficiency_windows_by_station:
        efficiency_windows_by_station[station_index].sort()
    for station_index in failed_inspection_times_by_station:
        failed_inspection_times_by_station[station_index].sort()
    emergency_orders.sort(key=lambda item: (float(item["order_time_s"]), -int(item["priority"]), str(item["order_id"])))

    return {
        "records": list(timed_disruption_records),
        "breakdown_windows_by_station": dict(breakdown_windows_by_station),
        "efficiency_windows_by_station": dict(efficiency_windows_by_station),
        "failed_inspection_times_by_station": dict(failed_inspection_times_by_station),
        "emergency_orders": emergency_orders,
    }


def _station_speed_factor_at_time(
    time_s: float,
    breakdown_windows: list[tuple[float, float]],
    efficiency_windows: list[tuple[float, float, float]],
) -> float:
    for start_time_s, end_time_s in breakdown_windows:
        if start_time_s <= time_s < end_time_s:
            return 0.0

    active_efficiency_fractions = [
        float(efficiency_fraction)
        for start_time_s, end_time_s, efficiency_fraction in efficiency_windows
        if start_time_s <= time_s < end_time_s
    ]
    if active_efficiency_fractions:
        return max(0.0, min(active_efficiency_fractions))
    return 1.0


def _next_station_schedule_change_after(
    time_s: float,
    breakdown_windows: list[tuple[float, float]],
    efficiency_windows: list[tuple[float, float, float]],
) -> float | None:
    candidate_times: list[float] = []
    for start_time_s, end_time_s in breakdown_windows:
        if start_time_s > time_s:
            candidate_times.append(float(start_time_s))
        if end_time_s > time_s:
            candidate_times.append(float(end_time_s))
    for start_time_s, end_time_s, _ in efficiency_windows:
        if start_time_s > time_s:
            candidate_times.append(float(start_time_s))
        if end_time_s > time_s:
            candidate_times.append(float(end_time_s))
    if not candidate_times:
        return None
    return min(candidate_times)


def calculate_timed_operation_disruption_result(
    station_index: int,
    station_name: str,
    stage_number: int,
    current_time_s: float,
    base_process_time_s: float,
    timed_disruption_data: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage_number": int(stage_number),
        "station_name": station_name,
        "base_process_time_s": float(base_process_time_s),
        "effective_process_time_s": float(base_process_time_s),
        "triggered_disruption_type": None,
        "breakdown_triggered": False,
        "breakdown_added_time_s": 0.0,
        "breakdown_random": None,
        "breakdown_duration_random": None,
        "efficiency_loss_triggered": False,
        "efficiency_drop_percent": 0.0,
        "efficiency_multiplier": 1.0,
        "efficiency_random": None,
        "terminal_failure_type": None,
        "terminal_failure_material": None,
        "material_broken_triggered": False,
        "material_broken_random": None,
        "material_broken_added_time_s": 0.0,
        "material_broken_extra_material_name": None,
        "material_broken_extra_material_qty": 0,
        "material_ran_out_triggered": False,
        "material_ran_out_random": None,
        "inspection_failed_triggered": False,
        "inspection_random": None,
    }

    if timed_disruption_data is None:
        return result

    breakdown_windows = list(timed_disruption_data.get("breakdown_windows_by_station", {}).get(int(station_index), []))
    efficiency_windows = list(timed_disruption_data.get("efficiency_windows_by_station", {}).get(int(station_index), []))
    if not breakdown_windows and not efficiency_windows:
        return result

    current_cursor_s = float(current_time_s)
    work_remaining_s = float(base_process_time_s)
    breakdown_added_time_s = 0.0
    epsilon = 1e-9

    while work_remaining_s > epsilon:
        speed_factor = _station_speed_factor_at_time(current_cursor_s, breakdown_windows, efficiency_windows)
        next_change_time_s = _next_station_schedule_change_after(current_cursor_s, breakdown_windows, efficiency_windows)

        if speed_factor <= 0.0:
            result["breakdown_triggered"] = True
            if next_change_time_s is None:
                break
            breakdown_added_time_s += max(0.0, next_change_time_s - current_cursor_s)
            current_cursor_s = next_change_time_s
            continue

        if next_change_time_s is None:
            time_needed_s = work_remaining_s / speed_factor
            current_cursor_s += time_needed_s
            work_remaining_s = 0.0
            break

        time_until_change_s = max(0.0, next_change_time_s - current_cursor_s)
        producible_work_s = time_until_change_s * speed_factor
        if producible_work_s + epsilon >= work_remaining_s:
            time_needed_s = work_remaining_s / speed_factor
            current_cursor_s += time_needed_s
            work_remaining_s = 0.0
            break

        work_remaining_s -= producible_work_s
        current_cursor_s = next_change_time_s

    result["effective_process_time_s"] = max(0.0, float(current_cursor_s) - float(current_time_s))
    result["breakdown_added_time_s"] = float(breakdown_added_time_s)
    if breakdown_added_time_s > 0.0:
        result["triggered_disruption_type"] = "breakdown"

    productive_elapsed_time_s = max(0.0, result["effective_process_time_s"] - result["breakdown_added_time_s"])
    if productive_elapsed_time_s > float(base_process_time_s) + 1e-9:
        result["efficiency_loss_triggered"] = True
        result["efficiency_multiplier"] = productive_elapsed_time_s / float(base_process_time_s) if float(base_process_time_s) > 0 else 1.0
        if result["efficiency_multiplier"] > 0:
            result["efficiency_drop_percent"] = max(0.0, 100.0 * (1.0 - (1.0 / result["efficiency_multiplier"])))
        if result["triggered_disruption_type"] is None:
            result["triggered_disruption_type"] = "efficiency_loss"

    return result


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
    produced_unit_positions: list[int] = []
    unproduced_unit_positions: list[int] = []
    skipped_units_due_to_shortage: list[dict[str, Any]] = []
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
            if first_unproduced_position is None:
                first_unproduced_position = index
                shortage_reason = {
                    "unit_position": index,
                    "variant": variant,
                    "shortages": shortages,
                }

            unproduced_units.append(variant)
            unproduced_unit_positions.append(index)
            skipped_units_due_to_shortage.append(
                {
                    "unit_position": index,
                    "variant": variant,
                    "shortages": shortages,
                }
            )
            continue

        for material, needed in unit_bom.items():
            remaining_stock[material] = int(remaining_stock.get(material, 0)) - needed
        produced_units.append(variant)
        produced_unit_positions.append(index)

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
        "produced_unit_positions": produced_unit_positions,
        "unproduced_unit_positions": unproduced_unit_positions,
        "first_unproduced_position": first_unproduced_position,
        "shortage_reason": shortage_reason,
        "skipped_units_due_to_shortage": skipped_units_due_to_shortage,
        "status": "complete" if not unproduced_units else "partial_due_to_material_shortage",
    }

    return produced_units, unproduced_units, material_report, production_status


# -----------------------------
# Event-based simulation with FIFO queues
# -----------------------------
def _update_queue_area(station_state: StationState, current_time_s: float) -> None:
    # Segment resume can legitimately restore queues with pre-boundary arrival
    # times. Queue statistics must never move backwards; if an old/legacy
    # snapshot still contains a time that is slightly ahead of the current
    # local event time, clamp the accounting point instead of crashing.
    if current_time_s < station_state.last_queue_change_time_s:
        current_time_s = station_state.last_queue_change_time_s
    delta_t = current_time_s - station_state.last_queue_change_time_s
    station_state.queue_area += len(station_state.queue) * delta_t
    station_state.last_queue_change_time_s = current_time_s


def _calculate_active_production_line_time_s(
    operations: list[OperationRecord],
    disruption_event_log: list[dict[str, Any]] | None = None,
    transport_records: list[TransportRecord] | None = None,
) -> float:
    if not operations and not transport_records:
        return 0.0

    intervals: list[tuple[float, float]] = []
    for op in operations:
        operation_start_time_s = float(op.start_time_s)
        operation_finish_time_s = float(op.finish_time_s)
        if operation_finish_time_s > operation_start_time_s:
            intervals.append((operation_start_time_s, operation_finish_time_s))

    for tr in transport_records or []:
        transport_start_time_s = float(tr.start_time_s)
        transport_finish_time_s = float(tr.finish_time_s)
        if transport_finish_time_s > transport_start_time_s:
            intervals.append((transport_start_time_s, transport_finish_time_s))

    if not intervals:
        return 0.0

    intervals.sort()
    merged_total = 0.0
    current_start, current_end = intervals[0]
    for start_time_s, finish_time_s in intervals[1:]:
        if start_time_s <= current_end:
            current_end = max(current_end, finish_time_s)
        else:
            merged_total += max(0.0, current_end - current_start)
            current_start, current_end = start_time_s, finish_time_s

    merged_total += max(0.0, current_end - current_start)
    return merged_total


def _normalize_route_value(route_value: Any) -> str:
    route_text = str(route_value).strip()
    if route_text == "" or route_text.casefold() == "nan":
        return "0"
    try:
        numeric_value = float(route_text)
    except (TypeError, ValueError):
        return route_text
    if math.isnan(numeric_value) or numeric_value == 0.0:
        return "0"
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return route_text


def _route_value_is_assigned(route_value: Any) -> bool:
    return _normalize_route_value(route_value) != "0"


def _parse_unit_route_for_layout(
    route_value: Any,
    stage_instance_indices: list[list[int]],
) -> list[int] | None:
    route_text = _normalize_route_value(route_value)
    if route_text == "0":
        return None

    if route_text.casefold().startswith("p:"):
        route_text = route_text.split(":", maxsplit=1)[1]

    parts = [part.strip() for part in route_text.split(".") if part.strip() != ""]
    if len(parts) != len(stage_instance_indices):
        return None

    parsed_indices: list[int] = []
    for stage_position, part in enumerate(parts):
        try:
            copy_choice = int(float(part))
        except (TypeError, ValueError):
            return None

        stage_instances = list(stage_instance_indices[stage_position])
        if copy_choice < 1 or copy_choice > len(stage_instances):
            return None

        parsed_indices.append(int(stage_instances[copy_choice - 1]))

    return parsed_indices


def _route_indices_to_route_id(
    route_indices: list[int | None],
    stage_instance_indices: list[list[int]],
) -> str:
    if len(route_indices) != len(stage_instance_indices):
        return "0"

    route_parts: list[str] = []
    for chosen_station_index, stage_instances in zip(route_indices, stage_instance_indices):
        if chosen_station_index is None:
            return "0"
        try:
            copy_position = list(stage_instances).index(int(chosen_station_index)) + 1
        except ValueError:
            return "0"
        route_parts.append(str(copy_position))

    return "p:" + ".".join(route_parts)


def run_simulation(
    ordered_units: list[str],
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any],
    unit_release_times: list[float] | None = None,
    unit_priorities: list[int] | None = None,
    unit_order_ids: list[str] | None = None,
    unit_ids: list[str] | None = None,
    unit_route_ids: list[str] | None = None,
    max_units_in_system: int = MAX_UNITS_IN_SYSTEM,
    return_to_station_1_time_s: float = RETURN_TO_STATION_1_TIME_S,
    line_layout_config: dict[str, Any] | None = None,
    bom_data: dict[str, Any] | None = None,
    material_stock_data: dict[str, Any] | None = None,
    disruptions_enabled: bool = False,
    disruption_config: dict[str, Any] | None = None,
    disruption_seed: Any | None = None,
    simulation_time_s: float | None = None,
    timed_disruption_data: dict[str, Any] | None = None,
    initial_available_system_slots: int | None = None,
    initial_cart_return_times_s: list[float] | None = None,
    initial_line_state_snapshot: dict[str, Any] | None = None,
) -> tuple[list[OperationRecord], list[TransportRecord], list[UnitSummary], list[StationSummary], dict[str, float], dict[str, Any]]:
    effective_line_layout = build_effective_line_layout(
        process_time_data=process_time_data,
        transport_time_data=transport_time_data,
        line_layout_config=line_layout_config,
    )
    station_sequence = effective_line_layout["station_sequence"]
    station_instance_base_names = effective_line_layout["station_instance_base_names"]
    station_time_scale_factors = list(effective_line_layout.get("station_time_scale_factors", [1.0] * len(station_sequence)))
    if len(station_time_scale_factors) != len(station_sequence):
        station_time_scale_factors = [1.0] * len(station_sequence)
    stage_instance_indices = effective_line_layout["stage_instance_indices"]
    station_to_stage_index = effective_line_layout["station_to_stage_index"]
    transport_lookup = effective_line_layout["transport_lookup"]
    process_times = process_time_data["process_times"]

    ordered_units = list(ordered_units)
    initial_requested_unit_count = len(ordered_units)

    if unit_release_times is None:
        unit_release_times = [0.0] * len(ordered_units)
    else:
        unit_release_times = [float(value) for value in unit_release_times]

    if unit_priorities is None:
        unit_priorities = [1] * len(ordered_units)
    else:
        unit_priorities = [max(1, int(value)) for value in unit_priorities]

    if unit_order_ids is None:
        unit_order_ids = ["manual"] * len(ordered_units)
    else:
        unit_order_ids = [str(value) for value in unit_order_ids]

    if unit_ids is None:
        unit_ids = [f"U{idx + 1:03d}" for idx in range(len(ordered_units))]
    else:
        unit_ids = [
            str(value).strip() if str(value).strip() != "" else f"U{idx + 1:03d}"
            for idx, value in enumerate(unit_ids)
        ]

    if unit_route_ids is None:
        unit_route_ids = ["0"] * len(ordered_units)
    else:
        unit_route_ids = [_normalize_route_value(value) for value in unit_route_ids]

    timed_breakdown_windows_by_station: dict[int, list[tuple[float, float]]] = {}
    timed_efficiency_windows_by_station: dict[int, list[tuple[float, float, float]]] = {}
    timed_failed_inspection_times_by_station: dict[int, list[float]] = {}
    if timed_disruption_data is not None:
        timed_breakdown_windows_by_station = {
            int(key): list(value)
            for key, value in timed_disruption_data.get("breakdown_windows_by_station", {}).items()
        }
        timed_efficiency_windows_by_station = {
            int(key): list(value)
            for key, value in timed_disruption_data.get("efficiency_windows_by_station", {}).items()
        }
        timed_failed_inspection_times_by_station = {
            int(key): list(value)
            for key, value in timed_disruption_data.get("failed_inspection_times_by_station", {}).items()
        }
        next_emergency_unit_number = _next_emergency_unit_number(unit_ids)
        for emergency_order in timed_disruption_data.get("emergency_orders", []):
            emergency_order_id = str(emergency_order["order_id"])
            emergency_order_time_s = float(emergency_order["order_time_s"])
            emergency_priority = max(1_000_000, int(emergency_order["priority"]))
            for variant_value, quantity_value in emergency_order.get("variants", []):
                for _ in range(int(quantity_value)):
                    ordered_units.append(str(variant_value))
                    unit_release_times.append(emergency_order_time_s)
                    unit_priorities.append(emergency_priority)
                    unit_order_ids.append(emergency_order_id)
                    unit_ids.append(f"E{next_emergency_unit_number:03d}")
                    next_emergency_unit_number += 1
                    unit_route_ids.append("0")

    initial_requested_unit_count = len(ordered_units)

    if len(unit_release_times) != len(ordered_units):
        raise ValueError("unit_release_times must have the same length as ordered_units")
    if len(unit_priorities) != len(ordered_units):
        raise ValueError("unit_priorities must have the same length as ordered_units")
    if len(unit_order_ids) != len(ordered_units):
        raise ValueError("unit_order_ids must have the same length as ordered_units")
    if len(unit_ids) != len(ordered_units):
        raise ValueError("unit_ids must have the same length as ordered_units")
    if len(unit_route_ids) != len(ordered_units):
        raise ValueError("unit_route_ids must have the same length as ordered_units")
    if max_units_in_system <= 0:
        raise ValueError("max_units_in_system must be greater than 0")
    if return_to_station_1_time_s < 0:
        raise ValueError("return_to_station_1_time_s cannot be negative")

    rng = _seed_to_rng(disruption_seed) if disruptions_enabled else None

    station_states: list[StationState] = [StationState(queue=[]) for _ in station_sequence]
    operations: list[OperationRecord] = []
    operation_lookup: dict[tuple[int, int], OperationRecord] = {}
    operation_disruption_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    transport_records: list[TransportRecord] = []

    unit_first_arrival: dict[int, float] = {}
    unit_first_start: dict[int, float] = {}
    unit_completion: dict[int, float] = {}
    unit_attempt_exit_time: dict[int, float] = {}

    event_queue: list[tuple[float, int, int, str, int, int]] = []
    event_sequence = 0
    if initial_available_system_slots is None:
        available_system_slots = max_units_in_system
    else:
        available_system_slots = max(0, min(max_units_in_system, int(initial_available_system_slots)))
    initial_cart_return_times_s = [float(value) for value in (initial_cart_return_times_s or [])]
    waiting_for_system_slot: list[tuple[int, float, int, bool, int]] = []
    projected_station_available_time_s: list[float] = [0.0] * len(station_sequence)
    waiting_queue_sequence = 0
    station_queue_sequence = 0

    root_indices: list[int] = list(range(initial_requested_unit_count))
    attempt_numbers: list[int] = [1] * initial_requested_unit_count
    root_to_attempt_indices: defaultdict[int, list[int]] = defaultdict(list)
    root_order_ids: list[str] = list(unit_order_ids)
    root_unit_ids: list[str] = list(unit_ids)

    def _root_unit_id_text(root_index: int) -> str:
        if 0 <= int(root_index) < len(root_unit_ids):
            value = str(root_unit_ids[int(root_index)]).strip()
            if value != "":
                return value
        return f"U{int(root_index) + 1:03d}"

    for unit_index in range(initial_requested_unit_count):
        root_to_attempt_indices[unit_index].append(unit_index)

    unit_route_station_indices: list[list[int] | None] = [
        _parse_unit_route_for_layout(route_value, stage_instance_indices)
        for route_value in unit_route_ids
    ]
    unit_chosen_route_by_root: defaultdict[int, dict[int, int]] = defaultdict(dict)

    unit_index_by_unit_id: dict[str, int] = {
        str(unit_id_value).strip(): int(idx)
        for idx, unit_id_value in enumerate(unit_ids)
        if str(unit_id_value).strip() != ""
    }
    resumed_unit_indices: set[int] = set()
    carried_base_process_time_by_unit_id: defaultdict[str, float] = defaultdict(float)
    released_unit_indices: set[int] = set()

    def _unit_index_from_snapshot_unit_id(unit_id_value: Any) -> int | None:
        unit_id_text = str(unit_id_value).strip()
        if unit_id_text == "":
            return None
        return unit_index_by_unit_id.get(unit_id_text)

    def _set_snapshot_first_times(unit_index: int, entry: dict[str, Any]) -> None:
        try:
            first_arrival = entry.get("first_arrival_time_s")
            if first_arrival is not None:
                unit_first_arrival.setdefault(int(unit_index), float(first_arrival))
        except (TypeError, ValueError):
            pass
        try:
            first_start = entry.get("first_start_time_s")
            if first_start is not None:
                unit_first_start.setdefault(int(unit_index), float(first_start))
        except (TypeError, ValueError):
            pass

    def _station_index_from_snapshot(entry: dict[str, Any]) -> int | None:
        try:
            station_index_value = int(entry.get("station_index"))
        except (TypeError, ValueError):
            return None
        if 0 <= station_index_value < len(station_sequence):
            return station_index_value
        return None

    def _remaining_process_result_for_snapshot(
        station_index: int,
        unit_index: int,
        remaining_base_process_time_s: float,
        current_time_s: float,
    ) -> dict[str, Any]:
        station_name = station_sequence[station_index]
        base_station_name = station_instance_base_names[station_index]
        stage_number, _, _ = _extract_station_name_parts(base_station_name)
        if stage_number is None:
            stage_number = station_index + 1
        if timed_disruption_data is not None:
            return calculate_timed_operation_disruption_result(
                station_index=station_index,
                station_name=station_name,
                stage_number=int(stage_number),
                current_time_s=float(current_time_s),
                base_process_time_s=float(remaining_base_process_time_s),
                timed_disruption_data={
                    "breakdown_windows_by_station": timed_breakdown_windows_by_station,
                    "efficiency_windows_by_station": timed_efficiency_windows_by_station,
                },
            )
        return {
            "stage_number": int(stage_number),
            "station_name": station_name,
            "base_process_time_s": float(remaining_base_process_time_s),
            "effective_process_time_s": float(remaining_base_process_time_s),
            "triggered_disruption_type": None,
            "breakdown_triggered": False,
            "breakdown_added_time_s": 0.0,
            "efficiency_loss_triggered": False,
            "efficiency_drop_percent": 0.0,
            "efficiency_multiplier": 1.0,
            "terminal_failure_type": None,
            "terminal_failure_material": None,
            "material_broken_triggered": False,
            "material_ran_out_triggered": False,
            "inspection_failed_triggered": False,
        }

    replacement_variants_created: list[str] = []
    extra_material_consumed: defaultdict[str, int] = defaultdict(int)
    actual_material_consumed: defaultdict[str, int] = defaultdict(int)
    disruption_event_log: list[dict[str, Any]] = []
    disruption_counts: Counter[str] = Counter()
    root_successful_attempt_index: dict[int, int] = {}
    root_failed_without_replacement: dict[int, dict[str, Any]] = {}

    remaining_station_material_stock: dict[str, int] = {}
    if material_stock_data is not None:
        remaining_station_material_stock = {
            material: int(qty)
            for material, qty in material_stock_data["materials_in_stock"].items()
        }

    stop_requested = False
    stop_time_s: float | None = None
    stop_reason: dict[str, Any] | None = None
    material_stop_cutoff_unit_index: int | None = None
    pending_timed_failed_inspection_by_station: defaultdict[int, int] = defaultdict(int)

    def _unit_blocked_by_material_stop(unit_index: int) -> bool:
        return (
            material_stop_cutoff_unit_index is not None
            and int(unit_index) >= int(material_stop_cutoff_unit_index)
        )

    def push_event(time_s: float, event_type: str, station_index: int, unit_index: int) -> None:
        nonlocal event_sequence
        event_priority = EVENT_PRIORITY[event_type]
        if (
            event_type == EVENT_RELEASE
            and 0 <= int(unit_index) < len(unit_ids)
            and _is_emergency_unit_id(unit_ids[int(unit_index)])
        ):
            event_priority = -1
        heapq.heappush(
            event_queue,
            (
                float(time_s),
                event_priority,
                event_sequence,
                event_type,
                station_index,
                unit_index,
            ),
        )
        event_sequence += 1

    def choose_station_instance_for_stage(
        target_stage_index: int,
        current_time_s: float,
        unit_index: int,
        from_station_index: int | None = None,
    ) -> tuple[int, float, float]:
        variant = ordered_units[unit_index]
        best_candidate: tuple[tuple[float, float, float, float, int], tuple[int, float, float, float]] | None = None

        forced_route = unit_route_station_indices[unit_index] if unit_index < len(unit_route_station_indices) else None
        if forced_route is not None and 0 <= target_stage_index < len(forced_route):
            candidate_station_indices = [int(forced_route[target_stage_index])]
        else:
            candidate_station_indices = list(stage_instance_indices[target_stage_index])

        for candidate_station_index in candidate_station_indices:
            candidate_station_name = station_sequence[candidate_station_index]
            transport_time_s = 0.0
            if from_station_index is not None:
                from_station_name = station_sequence[from_station_index]
                transport_time_s = float(transport_lookup[(from_station_name, candidate_station_name)])

            arrival_time_s = current_time_s + transport_time_s
            base_process_time_s = (
                float(process_times[variant][station_instance_base_names[candidate_station_index]])
                * float(station_time_scale_factors[candidate_station_index])
            )
            estimated_start_time_s = max(arrival_time_s, projected_station_available_time_s[candidate_station_index])

            # Route choice must estimate the same timed-disruption effect that
            # the real operation will see.  Without this, the router can keep
            # sending units to a station that is currently broken/slow because
            # the score only used the base process time.
            if timed_disruption_data is not None:
                base_station_name = station_instance_base_names[candidate_station_index]
                stage_number, _, _ = _extract_station_name_parts(base_station_name)
                if stage_number is None:
                    stage_number = int(target_stage_index + 1)
                route_disruption_result = calculate_timed_operation_disruption_result(
                    station_index=int(candidate_station_index),
                    station_name=candidate_station_name,
                    stage_number=int(stage_number),
                    current_time_s=float(estimated_start_time_s),
                    base_process_time_s=float(base_process_time_s),
                    timed_disruption_data={
                        "breakdown_windows_by_station": timed_breakdown_windows_by_station,
                        "efficiency_windows_by_station": timed_efficiency_windows_by_station,
                    },
                )
                estimated_process_time_s = float(
                    route_disruption_result.get("effective_process_time_s", base_process_time_s)
                )
            else:
                estimated_process_time_s = float(base_process_time_s)

            estimated_finish_time_s = estimated_start_time_s + estimated_process_time_s

            candidate_score = (
                estimated_finish_time_s,
                estimated_start_time_s,
                arrival_time_s,
                projected_station_available_time_s[candidate_station_index],
                candidate_station_index,
            )
            candidate_payload = (
                candidate_station_index,
                transport_time_s,
                arrival_time_s,
                estimated_finish_time_s,
            )

            if best_candidate is None or candidate_score < best_candidate[0]:
                best_candidate = (candidate_score, candidate_payload)

        if best_candidate is None:
            raise RuntimeError(f"No station instances found for stage {target_stage_index + 1}.")

        chosen_station_index, chosen_transport_time_s, chosen_arrival_time_s, projected_finish_time_s = best_candidate[1]
        projected_station_available_time_s[chosen_station_index] = projected_finish_time_s
        if 0 <= int(unit_index) < len(root_indices):
            unit_chosen_route_by_root[int(root_indices[unit_index])][int(target_stage_index)] = int(chosen_station_index)
        return chosen_station_index, chosen_transport_time_s, chosen_arrival_time_s

    def _enqueue_waiting_unit(unit_index: int, requested_release_time_s: float, prioritize_front: bool = False) -> None:
        nonlocal waiting_queue_sequence
        is_emergency_unit = (
            0 <= int(unit_index) < len(unit_ids)
            and _is_emergency_unit_id(unit_ids[int(unit_index)])
        )
        waiting_for_system_slot.append(
            (
                unit_index,
                float(requested_release_time_s),
                max(int(unit_priorities[unit_index]), 1_000_000 if is_emergency_unit else int(unit_priorities[unit_index])),
                bool(prioritize_front) or bool(is_emergency_unit),
                waiting_queue_sequence,
            )
        )
        waiting_queue_sequence += 1

    def _pop_best_waiting_unit(current_time_s: float) -> tuple[int, float] | None:
        best_index: int | None = None
        best_score: tuple[int, int, float, int] | None = None
        for idx, (unit_index, requested_release_time_s, priority_value, prioritize_front, queue_seq) in enumerate(waiting_for_system_slot):
            if _unit_blocked_by_material_stop(unit_index):
                continue
            if requested_release_time_s > current_time_s:
                continue
            score = (-int(bool(prioritize_front)), -int(priority_value), float(requested_release_time_s), int(queue_seq))
            if best_score is None or score < best_score:
                best_score = score
                best_index = idx
        if best_index is None:
            return None
        unit_index, requested_release_time_s, _, _, _ = waiting_for_system_slot.pop(best_index)
        return unit_index, requested_release_time_s

    def _enqueue_station_queue(
        station_state: StationState,
        unit_index: int,
        arrival_time_s: float,
        queue_length_ahead_on_arrival: int,
    ) -> None:
        nonlocal station_queue_sequence
        entry = (
            unit_index,
            float(arrival_time_s),
            int(queue_length_ahead_on_arrival),
            int(unit_priorities[unit_index]),
            station_queue_sequence,
        )
        station_queue_sequence += 1
        # Keep station queues FIFO. Emergency orders receive priority only when
        # a carrier becomes available; they must not jump ahead of units that
        # were already waiting inside station queues.
        station_state.queue.append(entry)

    def release_waiting_units_into_system(current_time_s: float) -> None:
        nonlocal available_system_slots
        while available_system_slots > 0 and waiting_for_system_slot and not stop_requested:
            waiting_item = _pop_best_waiting_unit(current_time_s)
            if waiting_item is None:
                break
            unit_index, _requested_release_time_s = waiting_item
            available_system_slots -= 1
            released_unit_indices.add(int(unit_index))
            first_station_index, _, arrival_time_s = choose_station_instance_for_stage(
                target_stage_index=0,
                current_time_s=current_time_s,
                unit_index=unit_index,
                from_station_index=None,
            )
            push_event(arrival_time_s, EVENT_ARRIVAL, first_station_index, unit_index)

    def _stage_material_requirements(variant: str, stage_number: int) -> dict[str, int]:
        if bom_data is None:
            return {}
        material_names = MATERIAL_STAGE_TO_MATERIAL.get(int(stage_number))
        if material_names is None:
            return {}
        if isinstance(material_names, str):
            material_names = [material_names]
        variant_bom = bom_data.get("bom_units_per_phone", {}).get(variant, {})
        requirements: dict[str, int] = {}
        for material_name in material_names:
            qty = int(variant_bom.get(str(material_name), 0))
            if qty > 0:
                requirements[str(material_name)] = qty
        return requirements

    def try_allocate_replacement_unit(
        failed_unit_index: int,
        current_time_s: float,
        prioritize_next_queue: bool = False,
    ) -> tuple[int | None, list[dict[str, Any]]]:
        if bom_data is None:
            return None, []

        root_index = root_indices[failed_unit_index]
        variant = ordered_units[failed_unit_index]
        variant_bom = {material: int(qty) for material, qty in bom_data["bom_units_per_phone"][variant].items()}
        shortages: list[dict[str, Any]] = []
        for material, needed in variant_bom.items():
            available = int(remaining_station_material_stock.get(material, 0))
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
            return None, shortages

        new_unit_index = len(ordered_units)
        ordered_units.append(variant)
        unit_release_times.append(float(current_time_s))
        unit_priorities.append(int(unit_priorities[failed_unit_index]))
        unit_order_ids.append(str(unit_order_ids[failed_unit_index]))
        unit_route_ids.append(str(unit_route_ids[failed_unit_index]) if failed_unit_index < len(unit_route_ids) else "0")
        unit_route_station_indices.append(unit_route_station_indices[failed_unit_index] if failed_unit_index < len(unit_route_station_indices) else None)
        root_indices.append(root_index)
        attempt_numbers.append(int(attempt_numbers[failed_unit_index]) + 1)
        root_to_attempt_indices[root_index].append(new_unit_index)
        replacement_variants_created.append(variant)
        _enqueue_waiting_unit(new_unit_index, current_time_s, prioritize_front=prioritize_next_queue)
        if prioritize_next_queue:
            release_waiting_units_into_system(current_time_s)
        return new_unit_index, []

    def try_start_next(station_index: int, current_time_s: float) -> None:
        nonlocal stop_requested, stop_time_s, stop_reason, material_stop_cutoff_unit_index
        station_state = station_states[station_index]
        if stop_requested or station_state.busy or not station_state.queue:
            return

        _update_queue_area(station_state, current_time_s)
        unit_index, arrival_time_s, queue_length_ahead_on_arrival, _priority_value, _queue_seq = station_state.queue.pop(0)

        root_index = root_indices[unit_index]
        unit_id = _root_unit_id_text(root_index)
        variant = ordered_units[unit_index]
        station_name = station_sequence[station_index]
        base_station_name = station_instance_base_names[station_index]
        stage_number, _, _ = _extract_station_name_parts(base_station_name)
        if stage_number is None:
            stage_number = int(station_index + 1)

        base_process_time_s = (
            float(process_times[variant][base_station_name])
            * float(station_time_scale_factors[station_index])
        )
        if timed_disruption_data is not None:
            disruption_result = calculate_timed_operation_disruption_result(
                station_index=station_index,
                station_name=station_name,
                stage_number=int(stage_number),
                current_time_s=float(current_time_s),
                base_process_time_s=base_process_time_s,
                timed_disruption_data={
                    "breakdown_windows_by_station": timed_breakdown_windows_by_station,
                    "efficiency_windows_by_station": timed_efficiency_windows_by_station,
                },
            )
        else:
            disruption_result = evaluate_operation_disruptions(
                stage_number=stage_number,
                station_name=station_name,
                variant=variant,
                base_process_time_s=base_process_time_s,
                disruption_config=disruption_config if disruptions_enabled else None,
                rng=rng,
                bom_data=bom_data,
            )

        if pending_timed_failed_inspection_by_station.get(int(station_index), 0) > 0:
            pending_timed_failed_inspection_by_station[int(station_index)] -= 1
            disruption_result["triggered_disruption_type"] = disruption_result.get("triggered_disruption_type") or "failed_inspection"
            disruption_result["inspection_failed_triggered"] = True
            disruption_result["terminal_failure_type"] = "failed_inspection"
            disruption_result["inspection_random"] = "timed"

        stage_material_requirements = _stage_material_requirements(variant, int(stage_number))
        broken_extra_material_name = str(disruption_result.get("material_broken_extra_material_name") or "").strip()
        broken_extra_material_qty = int(disruption_result.get("material_broken_extra_material_qty", 0) or 0)

        total_material_requirements = dict(stage_material_requirements)
        if broken_extra_material_name and broken_extra_material_qty > 0:
            total_material_requirements[broken_extra_material_name] = int(total_material_requirements.get(broken_extra_material_name, 0)) + broken_extra_material_qty

        for material, needed in total_material_requirements.items():
            available = int(remaining_station_material_stock.get(material, 0))
            if available < int(needed):
                if material_stop_cutoff_unit_index is None or int(unit_index) < int(material_stop_cutoff_unit_index):
                    material_stop_cutoff_unit_index = int(unit_index)
                    stop_time_s = float(current_time_s)
                    stop_reason = {
                        "type": "material_stockout_at_station",
                        "time_s": float(current_time_s),
                        "unit_id": unit_id,
                        "order_id": str(unit_order_ids[unit_index]),
                        "variant": variant,
                        "station_name": station_name,
                        "stage_number": int(stage_number),
                        "material": material,
                        "needed": int(needed),
                        "available": int(available),
                    }
                    disruption_counts["material_stockout_stop_events"] += 1
                unit_attempt_exit_time[unit_index] = float(current_time_s)
                root_failed_without_replacement[root_indices[unit_index]] = {
                    "root_unit_id": _root_unit_id_text(root_indices[unit_index]),
                    "order_id": str(root_order_ids[root_indices[unit_index]]),
                    "variant": ordered_units[unit_index],
                    "failed_attempt_unit_id": _root_unit_id_text(root_indices[unit_index]),
                    "failed_attempt_number": int(attempt_numbers[unit_index]),
                    "failure_type": "material_stockout_at_station",
                    "station_name": station_name,
                    "shortages_for_replacement": [
                        {
                            "material": material,
                            "needed": int(needed),
                            "available": int(available),
                            "missing": int(needed) - int(available),
                        }
                    ],
                }
                push_event(current_time_s + return_to_station_1_time_s, EVENT_CART_RETURN, -1, unit_index)
                return

        for material, needed in total_material_requirements.items():
            remaining_station_material_stock[material] = int(remaining_station_material_stock.get(material, 0)) - int(needed)
            actual_material_consumed[material] += int(needed)
            if material == broken_extra_material_name and broken_extra_material_qty > 0:
                extra_material_consumed[material] += broken_extra_material_qty
                disruption_counts["broken_material_extra_items_consumed"] += broken_extra_material_qty

        process_time_s = float(disruption_result["effective_process_time_s"])
        breakdown_added_time_s = float(disruption_result.get("breakdown_added_time_s", 0.0) or 0.0)
        productive_process_time_for_utilization_s = max(0.0, process_time_s - breakdown_added_time_s)
        wait_time_s = current_time_s - arrival_time_s

        operation = OperationRecord(
            unit_id=unit_id,
            order_id=str(unit_order_ids[unit_index]),
            variant=variant,
            station_index=station_index + 1,
            station_name=station_name,
            arrival_time_s=arrival_time_s,
            start_time_s=current_time_s,
            finish_time_s=current_time_s + process_time_s,
            process_time_s=process_time_s,
            base_process_time_s=base_process_time_s,
            wait_time_s=wait_time_s,
            queue_length_on_arrival=queue_length_ahead_on_arrival,
        )
        operations.append(operation)
        operation_lookup[(station_index, unit_index)] = operation
        operation_disruption_lookup[(station_index, unit_index)] = disruption_result

        if disruptions_enabled and disruption_result.get("triggered_disruption_type") is not None:
            disruption_event_log.append(
                {
                    "event": "operation_disruption",
                    "disruption_timestamp_s": float(current_time_s),
                    "unit_id": unit_id,
                    "order_id": str(unit_order_ids[unit_index]),
                    "root_unit_id": _root_unit_id_text(root_index),
                    "attempt": int(attempt_numbers[unit_index]),
                    "priority": int(unit_priorities[unit_index]),
                    "variant": variant,
                    "station_name": station_name,
                    "base_station_name": base_station_name,
                    "stage_number": int(stage_number),
                    **disruption_result,
                }
            )
            if disruption_result["breakdown_triggered"]:
                disruption_counts["breakdown_events"] += 1
            if disruption_result["efficiency_loss_triggered"]:
                disruption_counts["efficiency_loss_events"] += 1
            if disruption_result["material_broken_triggered"]:
                disruption_counts["broken_material_events"] += 1
            if disruption_result["material_ran_out_triggered"]:
                disruption_counts["ran_out_material_events"] += 1
            if disruption_result["inspection_failed_triggered"]:
                disruption_counts["failed_inspection_events"] += 1

        station_state.busy = True
        station_state.current_unit_index = unit_index
        station_state.busy_time_s += productive_process_time_for_utilization_s
        station_state.total_wait_time_s += wait_time_s
        if station_state.first_start_time_s is None:
            station_state.first_start_time_s = current_time_s

        projected_station_available_time_s[station_index] = max(
            projected_station_available_time_s[station_index],
            current_time_s + process_time_s,
        )

        if unit_index not in unit_first_start:
            unit_first_start[unit_index] = current_time_s

        push_event(current_time_s + process_time_s, EVENT_FINISH, station_index, unit_index)

    def _restore_initial_line_state_snapshot() -> int:
        """Restore units that were already on the line when this segment starts.

        These units must not be released again from current_schedule.csv.  They
        either resume processing/queue/transport, or keep occupying a carrier
        until a stored cart-return event releases it.
        """
        if not isinstance(initial_line_state_snapshot, dict):
            return 0

        occupied_carrier_count = 0
        restored_unit_ids: set[str] = set()

        for unit_id_text, carried_value in (initial_line_state_snapshot.get("carried_base_process_time_by_unit_id", {}) or {}).items():
            try:
                carried_base_process_time_by_unit_id[str(unit_id_text)] = float(carried_value)
            except (TypeError, ValueError):
                continue

        # Units already being processed at the segment boundary.
        for entry in initial_line_state_snapshot.get("processing_units", []) or []:
            if not isinstance(entry, dict):
                continue
            unit_index_from_entry = _unit_index_from_snapshot_unit_id(entry.get("unit_id"))
            station_index_from_entry = _station_index_from_snapshot(entry)
            if unit_index_from_entry is None or station_index_from_entry is None:
                continue
            if unit_index_from_entry in root_successful_attempt_index:
                continue

            unit_index_int = int(unit_index_from_entry)
            station_index_int = int(station_index_from_entry)
            station_state = station_states[station_index_int]
            if station_state.busy:
                continue

            resumed_unit_indices.add(unit_index_int)
            released_unit_indices.add(unit_index_int)
            restored_unit_ids.add(str(unit_ids[unit_index_int]).strip())
            occupied_carrier_count += 1
            _set_snapshot_first_times(unit_index_int, entry)

            try:
                remaining_base = float(entry.get("remaining_base_process_time_s", entry.get("remaining_process_time_s", 0.0)) or 0.0)
            except (TypeError, ValueError):
                remaining_base = 0.0
            remaining_base = max(0.0, remaining_base)
            if remaining_base <= 1e-9:
                remaining_base = 1e-9

            station_name = station_sequence[station_index_int]
            base_station_name = station_instance_base_names[station_index_int]
            stage_number, _, _ = _extract_station_name_parts(base_station_name)
            if stage_number is None:
                stage_number = station_index_int + 1

            disruption_result = _remaining_process_result_for_snapshot(
                station_index=station_index_int,
                unit_index=unit_index_int,
                remaining_base_process_time_s=remaining_base,
                current_time_s=0.0,
            )
            process_time_s = float(disruption_result.get("effective_process_time_s", remaining_base))
            breakdown_added_time_s = float(disruption_result.get("breakdown_added_time_s", 0.0) or 0.0)
            productive_process_time_for_utilization_s = max(0.0, process_time_s - breakdown_added_time_s)
            arrival_time_s = float(entry.get("arrival_time_s", entry.get("first_arrival_time_s", 0.0)) or 0.0)
            start_time_s = 0.0

            operation = OperationRecord(
                unit_id=_root_unit_id_text(root_indices[unit_index_int]),
                order_id=str(unit_order_ids[unit_index_int]),
                variant=ordered_units[unit_index_int],
                station_index=station_index_int + 1,
                station_name=station_name,
                arrival_time_s=arrival_time_s,
                start_time_s=start_time_s,
                finish_time_s=start_time_s + process_time_s,
                process_time_s=process_time_s,
                base_process_time_s=remaining_base,
                wait_time_s=max(0.0, start_time_s - arrival_time_s),
                queue_length_on_arrival=int(entry.get("queue_length_on_arrival", 0) or 0),
            )
            operations.append(operation)
            operation_lookup[(station_index_int, unit_index_int)] = operation
            operation_disruption_lookup[(station_index_int, unit_index_int)] = disruption_result

            station_state.busy = True
            station_state.current_unit_index = unit_index_int
            station_state.busy_time_s += productive_process_time_for_utilization_s
            if station_state.first_start_time_s is None:
                station_state.first_start_time_s = start_time_s
            unit_first_start.setdefault(unit_index_int, float(entry.get("first_start_time_s", start_time_s) or start_time_s))
            projected_station_available_time_s[station_index_int] = max(
                projected_station_available_time_s[station_index_int],
                start_time_s + process_time_s,
            )
            push_event(start_time_s + process_time_s, EVENT_FINISH, station_index_int, unit_index_int)

        # Units already waiting at a station queue.
        for entry in initial_line_state_snapshot.get("station_queue_units", []) or []:
            if not isinstance(entry, dict):
                continue
            unit_index_from_entry = _unit_index_from_snapshot_unit_id(entry.get("unit_id"))
            station_index_from_entry = _station_index_from_snapshot(entry)
            if unit_index_from_entry is None or station_index_from_entry is None:
                continue
            unit_index_int = int(unit_index_from_entry)
            station_index_int = int(station_index_from_entry)
            resumed_unit_indices.add(unit_index_int)
            released_unit_indices.add(unit_index_int)
            restored_unit_ids.add(str(unit_ids[unit_index_int]).strip())
            occupied_carrier_count += 1
            _set_snapshot_first_times(unit_index_int, entry)
            station_state = station_states[station_index_int]
            _update_queue_area(station_state, 0.0)
            _enqueue_station_queue(
                station_state=station_state,
                unit_index=unit_index_int,
                arrival_time_s=float(entry.get("arrival_time_s", 0.0) or 0.0),
                queue_length_ahead_on_arrival=int(entry.get("queue_length_on_arrival", len(station_state.queue)) or 0),
            )
            station_state.max_queue_length = max(station_state.max_queue_length, len(station_state.queue))

        # Units currently travelling to their next station.
        for entry in initial_line_state_snapshot.get("arrival_events", []) or []:
            if not isinstance(entry, dict):
                continue
            unit_index_from_entry = _unit_index_from_snapshot_unit_id(entry.get("unit_id"))
            station_index_from_entry = _station_index_from_snapshot(entry)
            if unit_index_from_entry is None or station_index_from_entry is None:
                continue
            unit_index_int = int(unit_index_from_entry)
            station_index_int = int(station_index_from_entry)
            resumed_unit_indices.add(unit_index_int)
            released_unit_indices.add(unit_index_int)
            restored_unit_ids.add(str(unit_ids[unit_index_int]).strip())
            occupied_carrier_count += 1
            _set_snapshot_first_times(unit_index_int, entry)
            push_event(float(entry.get("time_s", 0.0) or 0.0), EVENT_ARRIVAL, station_index_int, unit_index_int)

        # Carriers returning for units that completed before the boundary but whose
        # carrier return event is still pending.
        for entry in initial_line_state_snapshot.get("cart_return_events", []) or []:
            if not isinstance(entry, dict):
                continue
            unit_index_from_entry = _unit_index_from_snapshot_unit_id(entry.get("unit_id"))
            unit_index_for_event = -1 if unit_index_from_entry is None else int(unit_index_from_entry)
            if unit_index_for_event >= 0:
                released_unit_indices.add(unit_index_for_event)
                resumed_unit_indices.add(unit_index_for_event)
                restored_unit_ids.add(str(unit_ids[unit_index_for_event]).strip())
            occupied_carrier_count += 1
            push_event(float(entry.get("time_s", 0.0) or 0.0), EVENT_CART_RETURN, -1, unit_index_for_event)

        for station_index_to_start in range(len(station_states)):
            try_start_next(station_index_to_start, 0.0)

        return max(0, int(occupied_carrier_count))

    restored_occupied_carriers = _restore_initial_line_state_snapshot()
    if restored_occupied_carriers > 0:
        available_system_slots = max(0, int(max_units_in_system) - int(restored_occupied_carriers))

    for station_index, trigger_times in timed_failed_inspection_times_by_station.items():
        for trigger_time_s in trigger_times:
            push_event(float(trigger_time_s), EVENT_TIMED_FAILED_INSPECTION, int(station_index), -1)

    for unit_index in range(initial_requested_unit_count):
        if int(unit_index) in resumed_unit_indices:
            continue
        push_event(float(unit_release_times[unit_index]), EVENT_RELEASE, -1, unit_index)

    for carrier_return_time_s in initial_cart_return_times_s:
        if carrier_return_time_s >= 0.0:
            push_event(float(carrier_return_time_s), EVENT_CART_RETURN, -1, -1)

    popped_future_event: tuple[float, int, int, str, int, int] | None = None

    while event_queue:
        time_s, _, _, event_type, station_index, unit_index = heapq.heappop(event_queue)

        if simulation_time_s is not None and time_s > float(simulation_time_s):
            popped_future_event = (time_s, _, _, event_type, station_index, unit_index)
            stop_requested = True
            stop_time_s = float(simulation_time_s)
            stop_reason = {
                "type": "simulation_time_limit",
                "time_s": float(simulation_time_s),
            }
            break

        if stop_requested:
            break

        station_state = station_states[station_index] if station_index >= 0 else None

        if event_type == EVENT_TIMED_FAILED_INSPECTION:
            if station_state is not None and station_state.busy and station_state.current_unit_index is not None:
                current_processing_unit_index = int(station_state.current_unit_index)
                current_disruption_result = operation_disruption_lookup.get((station_index, current_processing_unit_index))
                if current_disruption_result is not None and current_disruption_result.get("terminal_failure_type") != "failed_inspection":
                    current_disruption_result["triggered_disruption_type"] = current_disruption_result.get("triggered_disruption_type") or "failed_inspection"
                    current_disruption_result["inspection_failed_triggered"] = True
                    current_disruption_result["terminal_failure_type"] = "failed_inspection"
                    current_disruption_result["inspection_random"] = "timed"
                    disruption_event_log.append(
                        {
                            "event": "timed_failed_inspection_trigger",
                            "disruption_timestamp_s": float(time_s),
                            "station_name": station_sequence[station_index],
                            "station_index": int(station_index + 1),
                            "affected_unit_id": _root_unit_id_text(root_indices[current_processing_unit_index]),
                            "order_id": str(unit_order_ids[current_processing_unit_index]),
                            "variant": ordered_units[current_processing_unit_index],
                            "timed_action": "scrap_current_unit",
                        }
                    )
                else:
                    pending_timed_failed_inspection_by_station[int(station_index)] += 1
                    disruption_event_log.append(
                        {
                            "event": "timed_failed_inspection_trigger",
                            "disruption_timestamp_s": float(time_s),
                            "station_name": station_sequence[station_index],
                            "station_index": int(station_index + 1),
                            "timed_action": "scrap_next_unit",
                        }
                    )
            else:
                pending_timed_failed_inspection_by_station[int(station_index)] += 1
                disruption_event_log.append(
                    {
                        "event": "timed_failed_inspection_trigger",
                        "disruption_timestamp_s": float(time_s),
                        "station_name": station_sequence[station_index],
                        "station_index": int(station_index + 1),
                        "timed_action": "scrap_next_unit",
                    }
                )
            disruption_counts["timed_failed_inspection_events"] += 1
            continue

        if event_type == EVENT_RELEASE:
            if _unit_blocked_by_material_stop(unit_index):
                continue
            prioritize_front = bool(attempt_numbers[unit_index] > 1)
            _enqueue_waiting_unit(unit_index, time_s, prioritize_front=prioritize_front)
            release_waiting_units_into_system(time_s)
            continue

        if event_type == EVENT_CART_RETURN:
            if int(unit_index) >= 0:
                released_unit_indices.discard(int(unit_index))
            available_system_slots = min(max_units_in_system, available_system_slots + 1)
            release_waiting_units_into_system(time_s)
            continue

        if event_type == EVENT_ARRIVAL:
            if _unit_blocked_by_material_stop(unit_index):
                unit_attempt_exit_time[unit_index] = float(time_s)
                push_event(time_s + return_to_station_1_time_s, EVENT_CART_RETURN, -1, unit_index)
                continue
            _update_queue_area(station_state, time_s)
            queue_length_ahead_on_arrival = len(station_state.queue)
            _enqueue_station_queue(station_state, unit_index, time_s, queue_length_ahead_on_arrival)
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

            station_state.last_finish_time_s = time_s
            station_state.busy = False
            station_state.current_unit_index = None

            if _unit_blocked_by_material_stop(unit_index):
                unit_attempt_exit_time[unit_index] = float(time_s)
                push_event(time_s + return_to_station_1_time_s, EVENT_CART_RETURN, -1, unit_index)
                try_start_next(station_index, time_s)
                continue

            disruption_result = operation_disruption_lookup.get((station_index, unit_index), {})
            terminal_failure_type = disruption_result.get("terminal_failure_type")

            if terminal_failure_type is not None:
                unit_attempt_exit_time[unit_index] = float(time_s)
                replacement_unit_index: int | None = None
                replacement_shortages: list[dict[str, Any]] = []
                if terminal_failure_type == "failed_inspection":
                    replacement_unit_index, replacement_shortages = try_allocate_replacement_unit(
                        unit_index,
                        time_s,
                        prioritize_next_queue=True,
                    )
                    if replacement_unit_index is None:
                        root_failed_without_replacement[root_indices[unit_index]] = {
                            "root_unit_id": _root_unit_id_text(root_indices[unit_index]),
                            "order_id": str(root_order_ids[root_indices[unit_index]]),
                            "variant": ordered_units[unit_index],
                            "failed_attempt_unit_id": _root_unit_id_text(root_indices[unit_index]),
                            "failed_attempt_number": int(attempt_numbers[unit_index]),
                            "failure_type": terminal_failure_type,
                            "station_name": station_sequence[station_index],
                            "shortages_for_replacement": replacement_shortages,
                        }
                        disruption_counts["unreplaced_failed_units"] += 1
                    else:
                        disruption_counts["replacement_units_created"] += 1
                else:
                    root_failed_without_replacement[root_indices[unit_index]] = {
                        "root_unit_id": _root_unit_id_text(root_indices[unit_index]),
                        "order_id": str(root_order_ids[root_indices[unit_index]]),
                        "variant": ordered_units[unit_index],
                        "failed_attempt_unit_id": _root_unit_id_text(root_indices[unit_index]),
                        "failed_attempt_number": int(attempt_numbers[unit_index]),
                        "failure_type": terminal_failure_type,
                        "station_name": station_sequence[station_index],
                        "shortages_for_replacement": [],
                    }
                    disruption_counts["unreplaced_failed_units"] += 1

                disruption_event_log.append(
                    {
                        "event": "unit_scrapped_and_released_for_retry" if replacement_unit_index is not None else "unit_scrapped_without_retry",
                        "disruption_timestamp_s": float(time_s),
                        "unit_id": _root_unit_id_text(root_indices[unit_index]),
                        "order_id": str(root_order_ids[root_indices[unit_index]]),
                        "root_unit_id": _root_unit_id_text(root_indices[unit_index]),
                        "attempt": int(attempt_numbers[unit_index]),
                        "variant": ordered_units[unit_index],
                        "station_name": station_sequence[station_index],
                        "failure_type": terminal_failure_type,
                        "failure_material": disruption_result.get("terminal_failure_material"),
                        "replacement_unit_id": _root_unit_id_text(root_indices[unit_index]) if replacement_unit_index is not None else None,
                        "replacement_attempt_number": int(attempt_numbers[replacement_unit_index]) if replacement_unit_index is not None else None,
                        "replacement_release_time_s": float(time_s) if replacement_unit_index is not None else None,
                        "replacement_shortages": replacement_shortages,
                    }
                )

                push_event(time_s + return_to_station_1_time_s, EVENT_CART_RETURN, -1, unit_index)
                try_start_next(station_index, time_s)
                continue

            current_stage_index = station_to_stage_index[station_index]
            if current_stage_index < len(stage_instance_indices) - 1:
                current_station_name = station_sequence[station_index]
                next_station_index, transport_time_s, arrival_time_s = choose_station_instance_for_stage(
                    target_stage_index=current_stage_index + 1,
                    current_time_s=time_s,
                    unit_index=unit_index,
                    from_station_index=station_index,
                )
                next_station_name = station_sequence[next_station_index]
                transport_records.append(
                    TransportRecord(
                        unit_id=_root_unit_id_text(root_indices[unit_index]),
                        order_id=str(unit_order_ids[unit_index]),
                        variant=ordered_units[unit_index],
                        transport_index=current_stage_index + 1,
                        transport_name=f"Transportation {current_stage_index + 1}",
                        from_station=current_station_name,
                        to_station=next_station_name,
                        start_time_s=time_s,
                        finish_time_s=arrival_time_s,
                        transport_time_s=transport_time_s,
                    )
                )
                push_event(arrival_time_s, EVENT_ARRIVAL, next_station_index, unit_index)
            else:
                unit_completion[unit_index] = float(time_s)
                unit_attempt_exit_time[unit_index] = float(time_s)
                root_successful_attempt_index[root_indices[unit_index]] = unit_index
                push_event(time_s + return_to_station_1_time_s, EVENT_CART_RETURN, -1, unit_index)

            try_start_next(station_index, time_s)
            continue

        raise RuntimeError(f"Unknown event type: {event_type}")

    carrier_snapshot_time_s = float(stop_time_s) if stop_requested and stop_time_s is not None else None
    carrier_snapshot: dict[str, Any] | None = None
    line_state_snapshot: dict[str, Any] | None = None
    if carrier_snapshot_time_s is not None:
        future_events_for_snapshot: list[tuple[float, int, int, str, int, int]] = []
        if popped_future_event is not None:
            future_events_for_snapshot.append(popped_future_event)
        future_events_for_snapshot.extend(list(event_queue))

        # Build a real line-state snapshot, not only a carrier count.  This keeps
        # units that are already on the line from being released again in the next
        # segment.
        processing_units_snapshot: list[dict[str, Any]] = []
        station_queue_units_snapshot: list[dict[str, Any]] = []
        arrival_events_snapshot: list[dict[str, Any]] = []
        cart_return_events_snapshot: list[dict[str, Any]] = []
        carried_base_snapshot: defaultdict[str, float] = defaultdict(float)

        def _snapshot_unit_common(unit_index_value: int) -> dict[str, Any]:
            root_index_value = int(root_indices[int(unit_index_value)])
            unit_id_value = _root_unit_id_text(root_index_value)
            common = {
                "unit_id": unit_id_value,
                "order_id": str(unit_order_ids[int(unit_index_value)]),
                "variant": ordered_units[int(unit_index_value)],
                "root_index": root_index_value,
                "attempt": int(attempt_numbers[int(unit_index_value)]),
                "first_arrival_time_s": float(unit_first_arrival.get(int(unit_index_value), carrier_snapshot_time_s)),
                "first_start_time_s": (
                    float(unit_first_start[int(unit_index_value)])
                    if int(unit_index_value) in unit_first_start
                    else None
                ),
            }
            return common

        active_indices_for_snapshot: set[int] = set(int(value) for value in released_unit_indices)

        # Anything already fully completed in this segment but still waiting for
        # cart return is represented by its pending cart-return event only.
        future_cart_return_times_s: list[float] = []
        for event_item in future_events_for_snapshot:
            if len(event_item) >= 6 and str(event_item[3]) == EVENT_CART_RETURN and float(event_item[0]) >= carrier_snapshot_time_s:
                future_cart_return_times_s.append(float(event_item[0]))
                if int(event_item[5]) >= 0:
                    unit_index_value = int(event_item[5])
                    if unit_index_value < len(unit_ids):
                        cart_return_events_snapshot.append({
                            **_snapshot_unit_common(unit_index_value),
                            "time_s": float(event_item[0]),
                        })

        # Pending arrivals = units in transport.
        arrival_event_unit_indices: set[int] = set()
        for event_item in future_events_for_snapshot:
            if len(event_item) >= 6 and str(event_item[3]) == EVENT_ARRIVAL and float(event_item[0]) >= carrier_snapshot_time_s:
                unit_index_value = int(event_item[5])
                station_index_value = int(event_item[4])
                if unit_index_value < 0 or unit_index_value >= len(unit_ids):
                    continue
                if station_index_value < 0 or station_index_value >= len(station_sequence):
                    continue
                arrival_event_unit_indices.add(unit_index_value)
                active_indices_for_snapshot.add(unit_index_value)
                arrival_events_snapshot.append({
                    **_snapshot_unit_common(unit_index_value),
                    "station_index": station_index_value,
                    "station_name": station_sequence[station_index_value],
                    "time_s": float(event_item[0]),
                })

        # Busy stations and station queues.
        for station_index_value, station_state in enumerate(station_states):
            if station_state.busy and station_state.current_unit_index is not None:
                unit_index_value = int(station_state.current_unit_index)
                if 0 <= unit_index_value < len(unit_ids):
                    active_indices_for_snapshot.add(unit_index_value)
                    operation = operation_lookup.get((station_index_value, unit_index_value))
                    if operation is not None:
                        remaining_effective = max(0.0, float(operation.finish_time_s) - carrier_snapshot_time_s)
                        elapsed_effective = max(0.0, carrier_snapshot_time_s - float(operation.start_time_s))
                        base_process = float(operation.base_process_time_s)
                        effective_process = max(1e-9, float(operation.process_time_s))
                        completed_base_in_current_op = min(base_process, base_process * (elapsed_effective / effective_process))
                        remaining_base = max(0.0, base_process - completed_base_in_current_op)
                        carried_base_snapshot[_root_unit_id_text(root_indices[unit_index_value])] += completed_base_in_current_op
                        processing_units_snapshot.append({
                            **_snapshot_unit_common(unit_index_value),
                            "station_index": station_index_value,
                            "station_name": station_sequence[station_index_value],
                            "arrival_time_s": float(operation.arrival_time_s),
                            "original_start_time_s": float(operation.start_time_s),
                            "remaining_process_time_s": remaining_effective,
                            "remaining_base_process_time_s": remaining_base,
                            "queue_length_on_arrival": int(operation.queue_length_on_arrival),
                        })

            for queue_position, queue_item in enumerate(list(station_state.queue)):
                try:
                    unit_index_value = int(queue_item[0])
                except (TypeError, ValueError):
                    continue
                if unit_index_value < 0 or unit_index_value >= len(unit_ids):
                    continue
                active_indices_for_snapshot.add(unit_index_value)
                station_queue_units_snapshot.append({
                    **_snapshot_unit_common(unit_index_value),
                    "station_index": int(station_index_value),
                    "station_name": station_sequence[station_index_value],
                    "arrival_time_s": float(queue_item[1]),
                    "queue_length_on_arrival": int(queue_item[2]),
                    "queue_position": int(queue_position),
                })

        # Carry completed base process for all active units from operations already
        # finished before the snapshot. This keeps time_spent_producing accurate
        # when a unit spans multiple segments.
        active_unit_id_texts = {
            _root_unit_id_text(root_indices[idx])
            for idx in active_indices_for_snapshot
            if 0 <= idx < len(root_indices)
        }
        for op in operations:
            if str(op.unit_id) in active_unit_id_texts and float(op.finish_time_s) <= carrier_snapshot_time_s + 1e-9:
                carried_base_snapshot[str(op.unit_id)] += float(op.base_process_time_s)
        for unit_id_text, previous_carried in carried_base_process_time_by_unit_id.items():
            if str(unit_id_text) in active_unit_id_texts:
                carried_base_snapshot[str(unit_id_text)] += float(previous_carried)

        occupied_carriers = len(active_indices_for_snapshot)
        # Add anonymous occupied carriers represented by cart-return events whose
        # units are not present in the current schedule anymore.
        anonymous_cart_returns = [
            entry for entry in cart_return_events_snapshot
            if _unit_index_from_snapshot_unit_id(entry.get("unit_id")) is None
        ]
        occupied_carriers += len(anonymous_cart_returns)

        line_state_snapshot = {
            "snapshot_time_s": float(carrier_snapshot_time_s),
            "max_units_in_system": int(max_units_in_system),
            "processing_units": processing_units_snapshot,
            "station_queue_units": station_queue_units_snapshot,
            "arrival_events": arrival_events_snapshot,
            "cart_return_events": cart_return_events_snapshot,
            "carried_base_process_time_by_unit_id": {str(k): round(float(v), 6) for k, v in carried_base_snapshot.items()},
            "active_unit_ids": sorted(active_unit_id_texts),
            "occupied_carriers": int(max(0, min(int(max_units_in_system), occupied_carriers))),
        }

        # Backward-compatible carrier snapshot: derive it from the line-state
        # snapshot and never invent one-second-apart fallback returns.
        future_cart_return_times_s = sorted(set(round(float(value), 6) for value in future_cart_return_times_s))
        occupied_carriers_for_carrier_file = int(line_state_snapshot["occupied_carriers"])
        carrier_snapshot = {
            "snapshot_time_s": float(carrier_snapshot_time_s),
            "max_units_in_system": int(max_units_in_system),
            "available_system_slots": int(max(0, int(max_units_in_system) - occupied_carriers_for_carrier_file)),
            "occupied_carriers": occupied_carriers_for_carrier_file,
            "future_cart_return_times_s": future_cart_return_times_s,
        }

    if stop_requested and stop_time_s is not None:
        for op in operations:
            if op.finish_time_s > float(stop_time_s) and op.start_time_s < float(stop_time_s):
                op.finish_time_s = float(stop_time_s)
                op.process_time_s = max(0.0, float(stop_time_s) - op.start_time_s)

    if root_successful_attempt_index:
        if stop_requested and stop_time_s is not None:
            makespan_s = float(stop_time_s)
        else:
            makespan_s = max(unit_completion.values()) if unit_completion else 0.0
    else:
        makespan_s = float(stop_time_s or 0.0)

    for station_state in station_states:
        _update_queue_area(station_state, makespan_s)

    def _route_taken_for_root(root_index: int) -> str:
        chosen_by_stage = unit_chosen_route_by_root.get(int(root_index), {})
        forced_route = unit_route_station_indices[int(root_index)] if int(root_index) < len(unit_route_station_indices) else None

        route_indices: list[int | None] = []
        for stage_index, stage_instances in enumerate(stage_instance_indices):
            chosen_station_index = chosen_by_stage.get(stage_index)
            if chosen_station_index is None and forced_route is not None and stage_index < len(forced_route):
                chosen_station_index = forced_route[stage_index]
            if chosen_station_index is None:
                chosen_station_index = int(stage_instances[0]) if stage_instances else None
            route_indices.append(chosen_station_index)

        return _route_indices_to_route_id(route_indices, stage_instance_indices)

    unit_summaries: list[UnitSummary] = []
    completed_good_variants: list[str] = []
    completed_root_positions: list[int] = []
    for root_index in range(initial_requested_unit_count):
        successful_attempt_index = root_successful_attempt_index.get(root_index)
        if successful_attempt_index is None:
            continue

        attempt_indices = root_to_attempt_indices[root_index]
        arrival_candidates = [unit_first_arrival[idx] for idx in attempt_indices if idx in unit_first_arrival]
        start_candidates = [unit_first_start[idx] for idx in attempt_indices if idx in unit_first_start]
        completion_time_s = float(unit_completion[successful_attempt_index])
        first_arrival_time_s = min(arrival_candidates) if arrival_candidates else float(unit_release_times[root_index])
        first_start_time_s = min(start_candidates) if start_candidates else first_arrival_time_s
        active_flow_time_s = sum(
            float(unit_attempt_exit_time[idx]) - float(unit_first_arrival[idx])
            for idx in attempt_indices
            if idx in unit_first_arrival and idx in unit_attempt_exit_time
        )
        variant = ordered_units[successful_attempt_index]
        unit_id_text = _root_unit_id_text(root_index)
        time_spent_producing = float(carried_base_process_time_by_unit_id.get(unit_id_text, 0.0)) + sum(
            float(op.base_process_time_s)
            for op in operations
            if op.unit_id == unit_id_text
        )
        throughput_efficiency = (
            time_spent_producing / active_flow_time_s if active_flow_time_s > 0 else 0.0
        )
        unit_summaries.append(
            UnitSummary(
                unit_id=unit_id_text,
                order_id=str(root_order_ids[root_index]),
                variant=variant,
                first_arrival_time_s=first_arrival_time_s,
                start_time_s=first_start_time_s,
                completion_time_s=completion_time_s,
                flow_time_s=completion_time_s - first_arrival_time_s,
                active_flow_time_s=float(active_flow_time_s),
                time_spent_producing=float(time_spent_producing),
                throughput_efficiency=float(throughput_efficiency),
                attempts=max(int(attempt_numbers[idx]) for idx in attempt_indices),
                route_taken=_route_taken_for_root(root_index),
            )
        )
        completed_good_variants.append(variant)
        completed_root_positions.append(root_index + 1)

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

    simulation_details = {
        "completed_good_variants": completed_good_variants,
        "completed_root_positions": completed_root_positions,
        "completed_good_unit_count": len(completed_good_variants),
        "replacement_units_created": list(replacement_variants_created),
        "replacement_units_created_count": len(replacement_variants_created),
        "extra_material_consumed": dict(extra_material_consumed),
        "actual_material_consumed": dict(actual_material_consumed),
        "remaining_stock_after_run": dict(remaining_station_material_stock),
        "disruption_event_log": disruption_event_log,
        "disruption_counts": dict(disruption_counts),
        "root_failed_without_replacement": list(root_failed_without_replacement.values()),
        "unrecoverable_root_count": len(root_failed_without_replacement),
        "disruptions_enabled": bool(disruptions_enabled or timed_disruption_data is not None),
        "stopped_due_to_sim_time_limit": bool(stop_reason and stop_reason.get("type") == "simulation_time_limit"),
        "stopped_due_to_material_shortage": bool(stop_reason and stop_reason.get("type") == "material_stockout_at_station"),
        "stop_reason": stop_reason,
        "stop_time_s": stop_time_s,
        "unit_order_ids": list(root_order_ids),
        "unit_ids": list(root_unit_ids),
        "route_id_by_unit_id": {
            _root_unit_id_text(root_index): _route_taken_for_root(root_index)
            for root_index in range(initial_requested_unit_count)
        },
        "carrier_snapshot": carrier_snapshot,
        "line_state_snapshot": line_state_snapshot,
    }

    return operations, transport_records, unit_summaries, station_summaries, station_available_time, simulation_details


# -----------------------------
# KPI calculation
# -----------------------------
def calculate_kpis(
    ordered_units: list[str],
    operations: list[OperationRecord],
    unit_summaries: list[UnitSummary],
    station_summaries: list[StationSummary],
    transport_records: list[TransportRecord] | None = None,
) -> dict[str, float | int]:
    def _average_cycle_time_from_completion_times(completion_times: list[float]) -> float:
        if not completion_times:
            return 0.0
        if len(completion_times) == 1:
            return float(completion_times[0])
        completion_times_sorted = sorted(float(value) for value in completion_times)
        intervals = [
            completion_times_sorted[i] - completion_times_sorted[i - 1]
            for i in range(1, len(completion_times_sorted))
        ]
        return float(mean(intervals)) if intervals else 0.0

    def _build_line_active_intervals() -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        for op in operations:
            start_time_s = float(op.start_time_s)
            finish_time_s = float(op.finish_time_s)
            if finish_time_s > start_time_s:
                intervals.append((start_time_s, finish_time_s))
        for tr in transport_records or []:
            start_time_s = float(tr.start_time_s)
            finish_time_s = float(tr.finish_time_s)
            if finish_time_s > start_time_s:
                intervals.append((start_time_s, finish_time_s))
        if not intervals:
            return []
        intervals.sort()
        merged: list[tuple[float, float]] = []
        current_start, current_end = intervals[0]
        for start_time_s, finish_time_s in intervals[1:]:
            if start_time_s <= current_end:
                current_end = max(current_end, finish_time_s)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start_time_s, finish_time_s
        merged.append((current_start, current_end))
        return merged

    def _overlap_with_active_intervals(
        interval_start_s: float,
        interval_finish_s: float,
        active_intervals: list[tuple[float, float]],
    ) -> float:
        if interval_finish_s <= interval_start_s or not active_intervals:
            return 0.0
        overlap_s = 0.0
        for active_start_s, active_finish_s in active_intervals:
            if active_finish_s <= interval_start_s:
                continue
            if active_start_s >= interval_finish_s:
                break
            overlap_s += max(
                0.0,
                min(interval_finish_s, active_finish_s) - max(interval_start_s, active_start_s),
            )
        return overlap_s

    def _average_from_intervals(
        intervals: list[float],
        single_fallback_value: float = 0.0,
    ) -> float:
        if intervals:
            return float(mean(intervals))
        return float(single_fallback_value)

    if not unit_summaries:
        return {
            "total_units_ordered": 0,
            "makespan_seconds": 0.0,
            "average_cycle_time_seconds": 0.0,
            "cycle_time_from_1_over_throughput_rate_seconds": math.inf,
            "throughput_rate_per_second": 0.0,
            "throughput_rate_per_hour": 0.0,
            "average_throughput_time_seconds": 0.0,
            "average_active_throughput_time_seconds": 0.0,
            "average_throughput_efficiency": 0.0,
            "total_wait_time_seconds": 0.0,
        }

    completion_times = [float(u.completion_time_s) for u in unit_summaries]
    makespan_s = max(completion_times) if completion_times else 0.0
    total_units = len(unit_summaries)
    throughput_rate_per_s = total_units / makespan_s if makespan_s > 0 else 0.0
    throughput_rate_per_h = throughput_rate_per_s * 3600.0
    average_cycle_time_s = _average_cycle_time_from_completion_times(completion_times)
    cycle_time_from_inverse_throughput_s = (
        1.0 / throughput_rate_per_s if throughput_rate_per_s > 0 else math.inf
    )

    average_throughput_time_s = mean(float(u.flow_time_s) for u in unit_summaries)
    average_active_throughput_time_s = mean(float(u.active_flow_time_s) for u in unit_summaries)
    average_throughput_efficiency = mean(float(u.throughput_efficiency) for u in unit_summaries)
    total_wait_time_s = sum(float(op.wait_time_s) for op in operations)

    kpis: dict[str, float | int] = {
        "total_units_ordered": total_units,
        "makespan_seconds": round(makespan_s, 4),
        "average_cycle_time_seconds": round(average_cycle_time_s, 4),
        "cycle_time_from_1_over_throughput_rate_seconds": round(
            cycle_time_from_inverse_throughput_s, 4
        ) if math.isfinite(cycle_time_from_inverse_throughput_s) else math.inf,
        "throughput_rate_per_second": round(throughput_rate_per_s, 6),
        "throughput_rate_per_hour": round(throughput_rate_per_h, 4),
        "average_throughput_time_seconds": round(average_throughput_time_s, 4),
        "average_active_throughput_time_seconds": round(average_active_throughput_time_s, 4),
        "average_throughput_efficiency": round(average_throughput_efficiency, 6),
        "total_wait_time_seconds": round(total_wait_time_s, 4),
    }

    line_active_intervals = _build_line_active_intervals()
    completed_units_sorted = sorted(
        unit_summaries,
        key=lambda unit_summary: (float(unit_summary.completion_time_s), str(unit_summary.unit_id)),
    )

    variant_wall_clock_intervals: defaultdict[str, list[float]] = defaultdict(list)
    variant_global_active_intervals: defaultdict[str, list[float]] = defaultdict(list)
    variant_global_cycle_intervals: defaultdict[str, list[float]] = defaultdict(list)
    variant_completion_values: defaultdict[str, list[float]] = defaultdict(list)

    variant_line_intervals_raw: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for op in operations:
        start_time_s = float(op.start_time_s)
        finish_time_s = float(op.finish_time_s)
        if finish_time_s > start_time_s:
            variant_line_intervals_raw[str(op.variant)].append((start_time_s, finish_time_s))
    for tr in transport_records or []:
        start_time_s = float(tr.start_time_s)
        finish_time_s = float(tr.finish_time_s)
        if finish_time_s > start_time_s:
            variant_line_intervals_raw[str(tr.variant)].append((start_time_s, finish_time_s))

    variant_line_intervals: dict[str, list[tuple[float, float]]] = {}
    for variant_name, raw_intervals in variant_line_intervals_raw.items():
        if not raw_intervals:
            variant_line_intervals[variant_name] = []
            continue
        raw_intervals.sort()
        merged_variant_intervals: list[tuple[float, float]] = []
        current_start_s, current_finish_s = raw_intervals[0]
        for start_time_s, finish_time_s in raw_intervals[1:]:
            if start_time_s <= current_finish_s:
                current_finish_s = max(current_finish_s, finish_time_s)
            else:
                merged_variant_intervals.append((current_start_s, current_finish_s))
                current_start_s, current_finish_s = start_time_s, finish_time_s
        merged_variant_intervals.append((current_start_s, current_finish_s))
        variant_line_intervals[variant_name] = merged_variant_intervals

    for unit_summary in completed_units_sorted:
        variant_completion_values[str(unit_summary.variant)].append(float(unit_summary.completion_time_s))

    for variant, completion_values in variant_completion_values.items():
        if len(completion_values) < 2:
            continue

        for previous_completion_s, current_completion_s in zip(completion_values, completion_values[1:]):
            interval_start_s = float(previous_completion_s)
            interval_finish_s = float(current_completion_s)
            interval_duration_s = max(0.0, interval_finish_s - interval_start_s)

            variant_wall_clock_intervals[variant].append(interval_duration_s)

            active_same_variant_s = _overlap_with_active_intervals(
                interval_start_s,
                interval_finish_s,
                variant_line_intervals.get(variant, []),
            )
            active_any_variant_s = _overlap_with_active_intervals(
                interval_start_s,
                interval_finish_s,
                line_active_intervals,
            )
            no_activity_s = max(0.0, interval_duration_s - active_any_variant_s)

            variant_global_active_intervals[variant].append(active_same_variant_s)
            variant_global_cycle_intervals[variant].append(active_same_variant_s + no_activity_s)

    for variant in sorted({str(u.variant) for u in unit_summaries}):
        variant_key = variant.lower()
        single_completion_fallback_s = (
            variant_completion_values[variant][0] if len(variant_completion_values[variant]) == 1 else 0.0
        )
        single_active_completion_fallback_s = (
            _overlap_with_active_intervals(
                0.0,
                variant_completion_values[variant][0],
                variant_line_intervals.get(variant, []),
            )
            if len(variant_completion_values[variant]) == 1
            else 0.0
        )
        single_global_completion_fallback_s = (
            float(variant_completion_values[variant][0])
            if len(variant_completion_values[variant]) == 1
            else 0.0
        )

        average_variant_cycle_time_s = _average_from_intervals(
            variant_wall_clock_intervals[variant],
            single_fallback_value=single_completion_fallback_s,
        )
        active_variant_cycle_time_s = _average_from_intervals(
            variant_global_active_intervals[variant],
            single_fallback_value=single_active_completion_fallback_s,
        )
        global_variant_cycle_time_s = _average_from_intervals(
            variant_global_cycle_intervals[variant],
            single_fallback_value=single_global_completion_fallback_s,
        )

        kpis[f"average_cycle_time_seconds_{variant_key}"] = round(average_variant_cycle_time_s, 4)
        kpis[f"Active_average_cycle_time_seconds_{variant_key}"] = round(active_variant_cycle_time_s, 4)
        kpis[f"global_average_cycle_time_seconds_{variant_key}"] = round(global_variant_cycle_time_s, 4)

    for summary in station_summaries:
        safe_key = re.sub(r"[^A-Za-z0-9]+", "_", summary.station_name).strip("_").lower()
        kpis[f"utilization_overall_{safe_key}"] = round(summary.utilization_overall, 6)
        kpis[f"utilization_active_window_{safe_key}"] = round(
            summary.utilization_active_window, 6
        )
        kpis[f"max_queue_{safe_key}"] = summary.max_queue_length
        kpis[f"average_queue_{safe_key}"] = round(summary.average_queue_length, 6)
        kpis[f"average_wait_{safe_key}_seconds"] = round(summary.average_wait_time_s, 4)

        station_ops = [op for op in operations if op.station_name == summary.station_name]
        station_busy_time_s = sum(float(op.process_time_s) for op in station_ops)
        station_disruption_time_s = sum(
            max(0.0, float(op.process_time_s) - float(op.base_process_time_s))
            for op in station_ops
        )
        availability = 1.0 - (station_disruption_time_s / station_busy_time_s) if station_busy_time_s > 0 else 1.0
        availability = max(0.0, min(1.0, availability))
        kpis[f"availability_{safe_key}"] = round(availability, 6)

    ordered_mix = Counter(ordered_units)
    for variant, qty in sorted(ordered_mix.items()):
        kpis[f"order_qty_{variant.lower()}"] = qty

    return kpis

def _timed_record_is_disruption_penalty_record(record: dict[str, Any]) -> bool:
    disruption_type = str(record.get("disruption_type", "")).strip().casefold()
    if disruption_type in {"", "emergency_order", "failed_inspection", "inspection_failure", "inspection failure"}:
        return False
    return True


def _timed_record_penalty_scale(record: dict[str, Any]) -> float:
    disruption_type = str(record.get("disruption_type", "")).strip().casefold()
    if disruption_type in {"efficiency_loss", "efficiency loss"}:
        efficiency_percentage = record.get("efficiency_percentage")
        if efficiency_percentage is None:
            return 0.0
        try:
            efficiency_fraction = max(0.0, min(1.0, float(efficiency_percentage) / 100.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - efficiency_fraction)
    return 1.0


def _build_timed_disruption_penalty_windows(
    timed_records: list[dict[str, Any]],
    cutoff_time_s: float | None,
) -> list[tuple[float, float, float]]:
    windows: list[tuple[float, float, float]] = []
    cutoff = None if cutoff_time_s is None else float(cutoff_time_s)

    for record in timed_records:
        if not _timed_record_is_disruption_penalty_record(record):
            continue

        try:
            start_s = float(record.get("start_time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        end_raw = record.get("end_time_s")
        if end_raw is None:
            continue
        try:
            end_s = float(end_raw)
        except (TypeError, ValueError):
            continue

        if cutoff is not None:
            if start_s >= cutoff:
                continue
            end_s = min(end_s, cutoff)

        if end_s <= start_s:
            continue

        penalty_scale = _timed_record_penalty_scale(record)
        if penalty_scale <= 0.0:
            continue

        windows.append((start_s, end_s, penalty_scale))

    windows.sort(key=lambda item: (item[0], item[1], item[2]))
    return windows


def _penalty_seconds_before_time(time_s: float, penalty_windows: list[tuple[float, float, float]]) -> float:
    value = float(time_s)
    penalty_s = 0.0
    for start_s, end_s, penalty_scale in penalty_windows:
        if value <= start_s:
            break
        penalty_s += max(0.0, min(value, end_s) - start_s) * float(penalty_scale)
    return penalty_s


def _copy_operation_without_disruption_penalties(
    operation: OperationRecord,
    penalty_windows: list[tuple[float, float, float]],
) -> OperationRecord:
    adjusted_arrival_s = max(0.0, float(operation.arrival_time_s) - _penalty_seconds_before_time(float(operation.arrival_time_s), penalty_windows))
    adjusted_start_s = max(adjusted_arrival_s, float(operation.start_time_s) - _penalty_seconds_before_time(float(operation.start_time_s), penalty_windows))
    base_process_time_s = max(0.0, float(operation.base_process_time_s))
    adjusted_finish_s = adjusted_start_s + base_process_time_s
    adjusted_wait_s = max(0.0, adjusted_start_s - adjusted_arrival_s)

    return OperationRecord(
        unit_id=operation.unit_id,
        order_id=operation.order_id,
        variant=operation.variant,
        station_index=operation.station_index,
        station_name=operation.station_name,
        arrival_time_s=adjusted_arrival_s,
        start_time_s=adjusted_start_s,
        finish_time_s=adjusted_finish_s,
        process_time_s=base_process_time_s,
        base_process_time_s=base_process_time_s,
        wait_time_s=adjusted_wait_s,
        queue_length_on_arrival=operation.queue_length_on_arrival,
    )


def _copy_transport_without_disruption_penalties(
    transport: TransportRecord,
    penalty_windows: list[tuple[float, float, float]],
) -> TransportRecord:
    adjusted_start_s = max(0.0, float(transport.start_time_s) - _penalty_seconds_before_time(float(transport.start_time_s), penalty_windows))
    adjusted_finish_s = max(adjusted_start_s, float(transport.finish_time_s) - _penalty_seconds_before_time(float(transport.finish_time_s), penalty_windows))
    return TransportRecord(
        unit_id=transport.unit_id,
        order_id=transport.order_id,
        variant=transport.variant,
        transport_index=transport.transport_index,
        transport_name=transport.transport_name,
        from_station=transport.from_station,
        to_station=transport.to_station,
        start_time_s=adjusted_start_s,
        finish_time_s=adjusted_finish_s,
        transport_time_s=max(0.0, adjusted_finish_s - adjusted_start_s),
    )


def _copy_unit_summary_without_disruption_penalties(
    summary: UnitSummary,
    penalty_windows: list[tuple[float, float, float]],
    base_production_time_by_unit_id: dict[str, float],
) -> UnitSummary:
    first_arrival_penalty_s = _penalty_seconds_before_time(float(summary.first_arrival_time_s), penalty_windows)
    start_penalty_s = _penalty_seconds_before_time(float(summary.start_time_s), penalty_windows)
    completion_penalty_s = _penalty_seconds_before_time(float(summary.completion_time_s), penalty_windows)

    adjusted_first_arrival_s = max(0.0, float(summary.first_arrival_time_s) - first_arrival_penalty_s)
    adjusted_start_s = max(adjusted_first_arrival_s, float(summary.start_time_s) - start_penalty_s)
    adjusted_completion_s = max(adjusted_start_s, float(summary.completion_time_s) - completion_penalty_s)

    adjusted_flow_s = max(0.0, adjusted_completion_s - adjusted_first_arrival_s)
    adjusted_active_flow_s = max(
        0.0,
        float(summary.active_flow_time_s) - max(0.0, completion_penalty_s - first_arrival_penalty_s),
    )

    base_production_time_s = base_production_time_by_unit_id.get(
        str(summary.unit_id),
        max(0.0, float(summary.time_spent_producing)),
    )
    throughput_efficiency = (
        min(1.0, max(0.0, base_production_time_s / adjusted_flow_s))
        if adjusted_flow_s > 0.0
        else 0.0
    )

    return UnitSummary(
        unit_id=summary.unit_id,
        order_id=summary.order_id,
        variant=summary.variant,
        first_arrival_time_s=adjusted_first_arrival_s,
        start_time_s=adjusted_start_s,
        completion_time_s=adjusted_completion_s,
        flow_time_s=adjusted_flow_s,
        active_flow_time_s=adjusted_active_flow_s,
        time_spent_producing=base_production_time_s,
        throughput_efficiency=throughput_efficiency,
        attempts=summary.attempts,
        route_taken=getattr(summary, "route_taken", "0"),
    )


def calculate_kpis_without_disruption_penalties(
    ordered_units: list[str],
    operations: list[OperationRecord],
    unit_summaries: list[UnitSummary],
    station_summaries: list[StationSummary],
    transport_records: list[TransportRecord] | None = None,
    timed_records: list[dict[str, Any]] | None = None,
    cutoff_time_s: float | None = None,
) -> dict[str, float | int]:
    penalty_windows = _build_timed_disruption_penalty_windows(list(timed_records or []), cutoff_time_s)

    adjusted_operations = [
        _copy_operation_without_disruption_penalties(operation, penalty_windows)
        for operation in operations
    ]
    adjusted_transport_records = [
        _copy_transport_without_disruption_penalties(transport, penalty_windows)
        for transport in (transport_records or [])
    ]

    base_production_time_by_unit_id: defaultdict[str, float] = defaultdict(float)
    for operation in adjusted_operations:
        base_production_time_by_unit_id[str(operation.unit_id)] += float(operation.base_process_time_s)

    adjusted_unit_summaries = [
        _copy_unit_summary_without_disruption_penalties(
            summary=summary,
            penalty_windows=penalty_windows,
            base_production_time_by_unit_id=dict(base_production_time_by_unit_id),
        )
        for summary in unit_summaries
    ]

    return calculate_kpis(
        ordered_units=ordered_units,
        operations=adjusted_operations,
        unit_summaries=adjusted_unit_summaries,
        station_summaries=station_summaries,
        transport_records=adjusted_transport_records,
    )



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
                "order_id",
                "variant",
                "station_index",
                "station_name",
                "arrival_time_s",
                "start_time_s",
                "finish_time_s",
                "process_time_s",
                "base_process_time_s",
                "wait_time_s",
                "queue_length_on_arrival",
            ]
        )
        for op in operations:
            writer.writerow(
                [
                    op.unit_id,
                    op.order_id,
                    op.variant,
                    op.station_index,
                    op.station_name,
                    round(op.arrival_time_s, 4),
                    round(op.start_time_s, 4),
                    round(op.finish_time_s, 4),
                    round(op.process_time_s, 4),
                    round(op.base_process_time_s, 4),
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
                "order_id",
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
                    tr.order_id,
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
    def _unit_summary_sort_key(summary: UnitSummary) -> tuple[float, float]:
        try:
            first_arrival_time_s = float(summary.first_arrival_time_s)
        except (TypeError, ValueError):
            first_arrival_time_s = math.inf
        try:
            start_time_s = float(summary.start_time_s)
        except (TypeError, ValueError):
            start_time_s = math.inf
        return (first_arrival_time_s, start_time_s)

    unit_summaries = sorted(unit_summaries, key=_unit_summary_sort_key)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unit_id",
                "orderID",
                "variant",
                "first_arrival_time_s",
                "start_time_s",
                "completion_time_s",
                "flow_time_s",
                "active_flow_time_s",
                "time_spent_producing",
                "throughput_efficiency",
                "Attempts",
                "route_taken",
            ]
        )
        for summary in unit_summaries:
            writer.writerow(
                [
                    summary.unit_id,
                    summary.order_id,
                    summary.variant,
                    round(summary.first_arrival_time_s, 4),
                    round(summary.start_time_s, 4),
                    round(summary.completion_time_s, 4),
                    round(summary.flow_time_s, 4),
                    float(round(summary.active_flow_time_s, 4)),
                    round(summary.time_spent_producing, 4),
                    round(summary.throughput_efficiency, 6),
                    int(summary.attempts),
                    getattr(summary, "route_taken", "0"),
                ]
            )


def write_station_summary_csv(
    station_summaries: list[StationSummary],
    output_path: Path,
    utilization_active_window_without_disruptions_by_station: dict[tuple[int, str], float] | None = None,
    active_order_utilization_by_station: dict[tuple[int, str], float] | None = None,
) -> None:
    utilization_active_window_without_disruptions_by_station = (
        utilization_active_window_without_disruptions_by_station or {}
    )
    active_order_utilization_by_station = active_order_utilization_by_station or {}
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
                "utilization_active_window_without_disruptions",
                "active_order_utilization",
            ]
        )
        for summary in station_summaries:
            station_key = (int(summary.station_index), str(summary.station_name))
            no_disruptions_value = utilization_active_window_without_disruptions_by_station.get(
                station_key
            )
            active_order_utilization_value = active_order_utilization_by_station.get(
                station_key
            )
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
                    round(float(no_disruptions_value), 6)
                    if no_disruptions_value is not None
                    else "",
                    round(float(active_order_utilization_value), 6)
                    if active_order_utilization_value is not None
                    else "",
                ]
            )


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

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
    extra_payload: dict[str, Any] | None = None,
) -> None:
    payload = {
        "order_text": order_text,
        "expanded_order_sequence": ordered_units,
        "data_directory": str(data_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra_payload:
        payload.update(extra_payload)
    save_json(payload, output_path)

# -----------------------------------------------------------------------------
# Integrated GA/main-settings entry points
# -----------------------------------------------------------------------------
def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    text_value = str(value).strip()
    if text_value == "":
        return None
    return Path(text_value).expanduser()


def _pathlist_path(pathlist: dict[str, Any], key: str) -> Path | None:
    if not isinstance(pathlist, dict):
        return None
    return _coerce_path(pathlist.get(key))


def _read_main_settings(main_settings_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    main_settings_path = Path(main_settings_path).expanduser().resolve()
    data = load_json(main_settings_path)
    return data, dict(data.get("settings", {})), dict(data.get("pathlist", {}))


def _resolve_pathlist_disruption_csv(pathlist: dict[str, Any]) -> Path | None:
    exact = _pathlist_path(pathlist, "input_disruptions_csv")
    if exact is not None and exact.exists() and exact.is_file():
        return exact
    folder = _pathlist_path(pathlist, "input_disruptions")
    if folder is not None and folder.exists() and folder.is_dir():
        candidates: list[Path] = []
        for pattern in ("disruptions_*.csv", "disruption_*.csv", "disruptions*.csv", "disruption_list*.csv"):
            candidates.extend(path for path in folder.glob(pattern) if path.is_file())
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def _resolve_pathlist_disruption_json(pathlist: dict[str, Any]) -> Path | None:
    exact = _pathlist_path(pathlist, "input_disruptions_json")
    if exact is not None and exact.exists() and exact.is_file():
        return exact
    folder = _pathlist_path(pathlist, "input_disruptions")
    if folder is not None and folder.exists() and folder.is_dir():
        for filename in (TIMED_DISRUPTION_V2_FILENAME, DISRUPTION_FILENAME):
            candidate = folder / filename
            if candidate.exists() and candidate.is_file():
                return candidate
        candidates = [path for path in folder.glob("disruption*.json") if path.is_file()]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def _read_current_schedule(schedule_path: Path, valid_variants: set[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with Path(schedule_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Schedule CSV is empty: {schedule_path}")
        normalized = [re.sub(r"[^a-z0-9]", "", str(v).strip().lower()) for v in header]
        required = ["unitseq", "orderid", "unitid", "variant"]
        if normalized[:4] != required:
            raise ValueError("current_schedule.csv must start with unit_seq, order_id, unit_id, variant")
        route_idx = normalized.index("routeid") if "routeid" in normalized else None
        for row_index, row in enumerate(reader, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            padded = list(row) + [""] * max(0, (route_idx + 1 if route_idx is not None else 4) - len(row))
            variant = str(padded[3]).strip().upper()
            if variant not in valid_variants:
                raise ValueError(f"Unknown variant '{variant}' in current_schedule.csv")
            rows.append({
                "unit_seq": _read_int(padded[0], row_index),
                "order_id": str(padded[1]).strip(),
                "unit_id": str(padded[2]).strip(),
                "variant": variant,
                "route_id": _normalize_route_value(padded[route_idx]) if route_idx is not None else "0",
                "row_index": row_index,
            })
    rows.sort(key=lambda r: (int(r["unit_seq"]), int(r["row_index"])))
    return {
        "rows": rows,
        "ordered_units": [r["variant"] for r in rows],
        "unit_order_ids": [str(r["order_id"]) for r in rows],
        "unit_ids": [str(r.get("unit_id", f"U{idx + 1:03d}")) for idx, r in enumerate(rows)],
        "unit_route_ids": [str(r.get("route_id", "0")) for r in rows],
        "has_assigned_route": any(_route_value_is_assigned(r.get("route_id", "0")) for r in rows),
    }



def _emergency_order_ids_already_in_schedule(schedule_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("order_id", "")).strip()
        for row in schedule_rows
        if _is_emergency_unit_id(row.get("unit_id", "")) and str(row.get("order_id", "")).strip() != ""
    }


def _sync_visible_emergency_orders_to_current_schedule(
    schedule_path: Path,
    valid_variants: set[str],
    timed_records: list[dict[str, Any]],
    cutoff_time_s: float,
) -> bool:
    """Add visible emergency orders to current_schedule.csv and put them first.

    When an emergency order has reached its start/order time, the next free
    carriers must be used for that order before normal scheduled units.  To make
    that true for both GA evaluations and the real segment run, visible
    emergency units are inserted at the front of current_schedule.csv and all
    unit_seq values are renumbered.  Existing emergency rows are moved to the
    front as well, so an earlier append-only schedule is corrected without
    duplicating emergency units.
    """
    schedule_path = Path(schedule_path)
    if not schedule_path.exists() or not schedule_path.is_file():
        return False

    with schedule_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return False
        rows = [list(row) for row in reader]

    normalized = [re.sub(r"[^a-z0-9]", "", str(value).strip().lower()) for value in header]
    required = ["unitseq", "orderid", "unitid", "variant"]
    if normalized[:4] != required:
        return False

    route_idx = normalized.index("routeid") if "routeid" in normalized else None
    if route_idx is None:
        header = list(header) + ["route_id"]
        normalized = normalized + ["routeid"]
        route_idx = len(header) - 1
        rows = [list(row) + [""] * max(0, len(header) - len(row)) for row in rows]

    min_len = max(4, route_idx + 1, len(header))
    padded_rows = [
        (list(row) + [""] * max(0, min_len - len(row)))[:min_len]
        for row in rows
    ]

    cutoff = float(cutoff_time_s or 0.0)

    visible_emergency_records: list[dict[str, Any]] = []
    for record in sorted(timed_records, key=lambda r: (float(r.get("start_time_s", 0.0) or 0.0), int(r.get("row_index", 0) or 0))):
        if str(record.get("disruption_type", "")).strip().casefold() != "emergency_order":
            continue

        start_time = float(record.get("start_time_s", 0.0) or 0.0)
        order_time_raw = record.get("order_time_s")
        order_time = float(order_time_raw) if order_time_raw is not None else start_time
        if start_time > cutoff + 1e-9 or order_time > cutoff + 1e-9:
            continue

        order_id = str(record.get("order_id", "")).strip()
        if order_id == "":
            continue

        visible_emergency_records.append(record)

    if not visible_emergency_records:
        return False

    visible_order_rank: dict[str, int] = {}
    for rank, record in enumerate(visible_emergency_records):
        order_id = str(record.get("order_id", "")).strip()
        visible_order_rank.setdefault(order_id, rank)

    visible_order_ids = set(visible_order_rank.keys())

    existing_unit_ids = [str(row[2]).strip() for row in padded_rows if len(row) > 2]
    next_emergency_unit_number = _next_emergency_unit_number(existing_unit_ids)

    existing_emergency_counts: Counter[tuple[str, str]] = Counter()
    for row in padded_rows:
        order_id = str(row[1]).strip() if len(row) > 1 else ""
        unit_id = str(row[2]).strip() if len(row) > 2 else ""
        variant = str(row[3]).strip().upper() if len(row) > 3 else ""
        if order_id and variant and _is_emergency_unit_id(unit_id):
            existing_emergency_counts[(order_id, variant)] += 1

    new_emergency_rows: list[list[str]] = []
    for record in visible_emergency_records:
        order_id = str(record.get("order_id", "")).strip()
        if order_id == "":
            continue

        for variant_value, quantity_value in record.get("emergency_variants", []):
            variant = str(variant_value).strip().upper()
            if variant not in valid_variants:
                continue
            required_quantity = max(0, int(quantity_value))
            existing_quantity = int(existing_emergency_counts.get((order_id, variant), 0))
            missing_quantity = max(0, required_quantity - existing_quantity)
            for _ in range(missing_quantity):
                unit_id = f"E{next_emergency_unit_number:03d}"
                next_emergency_unit_number += 1
                new_row = [""] * len(header)
                new_row[0] = "0"  # renumbered below
                new_row[1] = order_id
                new_row[2] = unit_id
                new_row[3] = variant
                new_row[route_idx] = "0"
                new_emergency_rows.append(new_row)
                existing_emergency_counts[(order_id, variant)] += 1

    existing_visible_emergency_rows: list[list[str]] = []
    normal_rows: list[list[str]] = []
    for row in padded_rows:
        row = (list(row) + [""] * max(0, len(header) - len(row)))[:len(header)]
        order_id = str(row[1]).strip() if len(row) > 1 else ""
        unit_id = str(row[2]).strip() if len(row) > 2 else ""
        if order_id in visible_order_ids and _is_emergency_unit_id(unit_id):
            existing_visible_emergency_rows.append(row)
        else:
            normal_rows.append(row)

    def _emergency_sort_key(row: list[str]) -> tuple[int, int, str]:
        order_id = str(row[1]).strip() if len(row) > 1 else ""
        unit_id = str(row[2]).strip() if len(row) > 2 else ""
        match = re.fullmatch(r"[Ee](\d+)", unit_id)
        unit_number = int(match.group(1)) if match else 10**12
        return (int(visible_order_rank.get(order_id, 10**12)), unit_number, unit_id)

    emergency_rows = sorted(existing_visible_emergency_rows + new_emergency_rows, key=_emergency_sort_key)
    reordered_rows = emergency_rows + normal_rows

    for seq, row in enumerate(reordered_rows, start=1):
        row[0] = str(seq)

    old_unit_order = [
        str(row[2]).strip() if len(row) > 2 else ""
        for row in padded_rows
    ]
    new_unit_order = [
        str(row[2]).strip() if len(row) > 2 else ""
        for row in reordered_rows
    ]

    changed = bool(new_emergency_rows) or old_unit_order != new_unit_order
    if changed:
        with schedule_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in reordered_rows:
                writer.writerow((list(row) + [""] * max(0, len(header) - len(row)))[:len(header)])

    return changed


def _resolve_carrier_snapshot_path(pathlist: dict[str, Any], main_settings_path: Path) -> Path:
    explicit_path = _pathlist_path(pathlist, "on_going_run_carrier_snapshot")
    if explicit_path is not None:
        return explicit_path
    ongoing_run_dir = _pathlist_path(pathlist, "on_going_run") or main_settings_path.parent
    return Path(ongoing_run_dir) / CARRIER_SNAPSHOT_FILENAME


def _load_carrier_snapshot_for_segment(
    snapshot_path: Path | None,
    segment_start_s: float,
    max_units_in_system: int,
) -> tuple[int | None, list[float]]:
    if snapshot_path is None:
        return None, []
    path = Path(snapshot_path)
    if not path.exists() or not path.is_file():
        return None, []

    try:
        snapshot = load_json(path)
    except Exception:
        return None, []

    segment_start_s = float(segment_start_s or 0.0)
    try:
        snapshot_time_s = float(snapshot.get("snapshot_time_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        snapshot_time_s = 0.0

    # Only use a snapshot that was taken before or at this segment boundary.
    if snapshot_time_s > segment_start_s + 1e-9:
        return None, []

    all_return_times_abs: list[float] = []
    for value in snapshot.get("future_cart_return_times_s", []) if isinstance(snapshot.get("future_cart_return_times_s"), list) else []:
        try:
            all_return_times_abs.append(float(value))
        except (TypeError, ValueError):
            continue

    try:
        occupied_carriers = int(snapshot.get("occupied_carriers", 0) or 0)
    except (TypeError, ValueError):
        occupied_carriers = 0
    occupied_carriers = max(0, min(int(max_units_in_system), occupied_carriers))

    # Carriers whose return time already passed before this segment are available again.
    already_returned = sum(1 for value_f in all_return_times_abs if value_f < segment_start_s - 1e-9)
    still_future_returns_abs = [value_f for value_f in all_return_times_abs if value_f >= segment_start_s - 1e-9]

    occupied_carriers = max(0, occupied_carriers - already_returned)
    initial_available_slots = max(0, int(max_units_in_system) - occupied_carriers)
    local_return_times = sorted(max(0.0, value_f - segment_start_s) for value_f in still_future_returns_abs)
    return initial_available_slots, local_return_times


def _write_carrier_snapshot(snapshot_path: Path | None, simulation_details: dict[str, Any]) -> None:
    if snapshot_path is None:
        return
    snapshot = simulation_details.get("carrier_snapshot")
    if not isinstance(snapshot, dict):
        return
    path = Path(snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4)


def _resolve_line_state_snapshot_path(pathlist: dict[str, Any], main_settings_path: Path) -> Path:
    explicit_path = _pathlist_path(pathlist, "on_going_run_line_state_snapshot")
    if explicit_path is not None:
        return explicit_path
    ongoing_run_dir = _pathlist_path(pathlist, "on_going_run") or main_settings_path.parent
    return Path(ongoing_run_dir) / LINE_STATE_SNAPSHOT_FILENAME


def _shift_line_state_snapshot_to_local(snapshot: dict[str, Any], segment_start_s: float) -> dict[str, Any]:
    local_snapshot = json.loads(json.dumps(snapshot))
    segment_start_s = float(segment_start_s or 0.0)

    def shift_key(entry: dict[str, Any], key: str) -> None:
        if key in entry and entry[key] is not None:
            try:
                entry[key] = float(entry[key]) - segment_start_s
            except (TypeError, ValueError):
                pass

    for group_name in ("processing_units", "station_queue_units", "arrival_events", "cart_return_events"):
        for entry in local_snapshot.get(group_name, []) or []:
            if not isinstance(entry, dict):
                continue
            for key in ("time_s", "arrival_time_s", "first_arrival_time_s", "first_start_time_s", "original_start_time_s"):
                shift_key(entry, key)

    try:
        local_snapshot["snapshot_time_s"] = float(local_snapshot.get("snapshot_time_s", segment_start_s)) - segment_start_s
    except (TypeError, ValueError):
        local_snapshot["snapshot_time_s"] = 0.0
    return local_snapshot


def _load_line_state_snapshot_for_segment(snapshot_path: Path | None, segment_start_s: float) -> dict[str, Any] | None:
    if snapshot_path is None:
        return None
    path = Path(snapshot_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        snapshot = load_json(path)
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    try:
        snapshot_time_s = float(snapshot.get("snapshot_time_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if snapshot_time_s > float(segment_start_s or 0.0) + 1e-9:
        return None
    return _shift_line_state_snapshot_to_local(snapshot, float(segment_start_s or 0.0))


def _write_line_state_snapshot(snapshot_path: Path | None, simulation_details: dict[str, Any]) -> None:
    if snapshot_path is None:
        return
    snapshot = simulation_details.get("line_state_snapshot")
    if not isinstance(snapshot, dict):
        return
    path = Path(snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4)


def _load_run_context_from_main_settings(main_settings_path: Path, simulation_time_limit_s: float | None = None, segment_start_s: float = 0.0) -> dict[str, Any]:
    main_settings_path = Path(main_settings_path).expanduser().resolve()
    main_settings, run_settings, pathlist = _read_main_settings(main_settings_path)
    base_dir = Path(__file__).resolve().parent
    data_dir = _pathlist_path(pathlist, "data") or (base_dir / "data")
    process_time_data = load_json(data_dir / "process_times.json")
    valid_variants = set(process_time_data["process_times"].keys())
    schedule_path = _pathlist_path(pathlist, "on_going_run_current_schedule")
    if schedule_path is None:
        schedule_path = (_pathlist_path(pathlist, "on_going_run") or main_settings_path.parent) / CURRENT_SCHEDULE_FILENAME
    if not schedule_path.exists():
        raise FileNotFoundError(f"current_schedule.csv was not found at {schedule_path}")
    segment_start_s = max(0.0, float(segment_start_s or 0.0))
    schedule = _read_current_schedule(schedule_path, valid_variants)

    timed_csv_for_schedule_sync = _resolve_pathlist_disruption_csv(pathlist)
    if timed_csv_for_schedule_sync is not None and timed_csv_for_schedule_sync.exists():
        timed_records_for_schedule_sync = load_timed_disruption_csv(timed_csv_for_schedule_sync, valid_variants)
        timed_records_for_schedule_sync = _assign_missing_emergency_order_ids(
            timed_records_for_schedule_sync,
            list(schedule["unit_order_ids"]),
            list(schedule.get("unit_ids", [])),
        )
        if _sync_visible_emergency_orders_to_current_schedule(
            schedule_path=schedule_path,
            valid_variants=valid_variants,
            timed_records=timed_records_for_schedule_sync,
            cutoff_time_s=segment_start_s,
        ):
            schedule = _read_current_schedule(schedule_path, valid_variants)

    default_settings_path = data_dir / "base_settings.json"
    settings_data = load_json(default_settings_path) if default_settings_path.exists() else {}
    settings_data.update(run_settings)
    scenario_layout = run_settings.get("Scenarios")
    if isinstance(scenario_layout, str) and scenario_layout.strip():
        settings_data["line_layout_file"] = scenario_layout.strip()

    absolute_stop_time_s = float(settings_data.get("sim_time [s]", settings_data.get("Sim_time [s]", 0.0)) or 0.0)
    if simulation_time_limit_s is not None:
        absolute_stop_time_s = float(simulation_time_limit_s)
    sim_time = max(0.0, float(absolute_stop_time_s) - segment_start_s)
    carriers = int(float(settings_data.get("carriers", {}).get("number of carriers", MAX_UNITS_IN_SYSTEM))) if isinstance(settings_data.get("carriers", {}), dict) else MAX_UNITS_IN_SYSTEM
    label = str(main_settings.get("label", main_settings_path.stem))
    return {
        "main_settings_path": main_settings_path,
        "main_settings": main_settings,
        "settings_data": settings_data,
        "pathlist": pathlist,
        "data_dir": data_dir,
        "process_time_data": process_time_data,
        "transport_time_data": _load_optional_transport_time_data(data_dir),
        "material_stock_data": load_json(data_dir / "material_stock.json"),
        "bom_data": load_json(data_dir / "bom.json"),
        "schedule_path": schedule_path,
        "schedule_rows": list(schedule.get("rows", [])),
        "ordered_units": schedule["ordered_units"],
        "unit_order_ids": schedule["unit_order_ids"],
        "unit_ids": schedule.get("unit_ids", [f"U{idx + 1:03d}" for idx in range(len(schedule["ordered_units"]))]),
        "unit_route_ids": schedule["unit_route_ids"],
        "has_assigned_route": bool(schedule["has_assigned_route"]),
        "unit_release_times": [0.0] * len(schedule["ordered_units"]),
        "unit_priorities": [1] * len(schedule["ordered_units"]),
        "simulation_time_s": sim_time,
        "absolute_stop_time_s": float(absolute_stop_time_s),
        "segment_start_s": float(segment_start_s),
        "segment_duration_s": float(sim_time),
        "carriers": carriers,
        "selected_line_layout_name": _resolve_line_layout_filename_from_settings(settings_data),
        "input_root": _pathlist_path(pathlist, "input"),
        "batch_dir": _pathlist_path(pathlist, "input_runs_run") or main_settings_path.parent,
        "output_results": _pathlist_path(pathlist, "output_run_results"),
        "ongoing_unit_summary": _pathlist_path(pathlist, "on_going_run_unit_summary"),
        "ongoing_disruption_history": _pathlist_path(pathlist, "on_going_run_dis_his"),
        "ongoing_carrier_snapshot": _resolve_carrier_snapshot_path(pathlist, main_settings_path),
        "ongoing_line_state_snapshot": _resolve_line_state_snapshot_path(pathlist, main_settings_path),
        "order_text": f"{label}__scheduled_{len(schedule['ordered_units'])}units",
    }



def _read_completed_unit_ids_from_unit_summary(unit_summary_path: Path | None, cutoff_time_s: float) -> set[str]:
    completed: set[str] = set()
    if unit_summary_path is None:
        return completed
    path = Path(unit_summary_path)
    if not path.exists() or not path.is_file():
        return completed

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit_id = str(row.get("unit_id", "")).strip()
            if unit_id == "":
                continue
            try:
                completion_time_s = float(row.get("completion_time_s", "nan"))
            except (TypeError, ValueError):
                continue
            if completion_time_s <= float(cutoff_time_s):
                completed.add(unit_id)
    return completed


def _read_previous_unit_summaries(unit_summary_path: Path | None, cutoff_time_s: float) -> list[UnitSummary]:
    summaries: list[UnitSummary] = []
    if unit_summary_path is None:
        return summaries
    path = Path(unit_summary_path)
    if not path.exists() or not path.is_file():
        return summaries

    def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit_id = str(row.get("unit_id", "")).strip()
            if unit_id == "":
                continue
            completion_time_s = _f(row, "completion_time_s", math.inf)
            if completion_time_s > float(cutoff_time_s):
                continue
            summaries.append(
                UnitSummary(
                    unit_id=unit_id,
                    order_id=str(row.get("orderID", row.get("order_id", ""))).strip(),
                    variant=str(row.get("variant", "")).strip(),
                    first_arrival_time_s=_f(row, "first_arrival_time_s"),
                    start_time_s=_f(row, "start_time_s"),
                    completion_time_s=completion_time_s,
                    flow_time_s=_f(row, "flow_time_s"),
                    active_flow_time_s=_f(row, "active_flow_time_s"),
                    time_spent_producing=_f(row, "time_spent_producing"),
                    throughput_efficiency=_f(row, "throughput_efficiency"),
                    attempts=int(_f(row, "Attempts", 1.0)),
                    route_taken=str(row.get("route_taken", row.get("route_id", "0"))).strip() or "0",
                )
            )
    return summaries




def _read_previous_operations_csv(operations_path: Path | None, cutoff_time_s: float) -> list[OperationRecord]:
    records: list[OperationRecord] = []
    if operations_path is None:
        return records
    path = Path(operations_path)
    if not path.exists() or not path.is_file():
        return records

    def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                finish_time_s = _f(row, "finish_time_s", math.inf)
                if finish_time_s > float(cutoff_time_s) + 1e-9:
                    continue
                records.append(
                    OperationRecord(
                        unit_id=str(row.get("unit_id", "")).strip(),
                        order_id=str(row.get("order_id", "")).strip(),
                        variant=str(row.get("variant", "")).strip(),
                        station_index=int(_f(row, "station_index", 0.0)),
                        station_name=str(row.get("station_name", "")).strip(),
                        arrival_time_s=_f(row, "arrival_time_s"),
                        start_time_s=_f(row, "start_time_s"),
                        finish_time_s=finish_time_s,
                        process_time_s=_f(row, "process_time_s"),
                        base_process_time_s=_f(row, "base_process_time_s"),
                        wait_time_s=_f(row, "wait_time_s"),
                        queue_length_on_arrival=int(_f(row, "queue_length_on_arrival", 0.0)),
                    )
                )
            except Exception:
                continue
    return records


def _read_previous_transport_csv(transport_path: Path | None, cutoff_time_s: float) -> list[TransportRecord]:
    records: list[TransportRecord] = []
    if transport_path is None:
        return records
    path = Path(transport_path)
    if not path.exists() or not path.is_file():
        return records

    def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                finish_time_s = _f(row, "finish_time_s", math.inf)
                if finish_time_s > float(cutoff_time_s) + 1e-9:
                    continue
                records.append(
                    TransportRecord(
                        unit_id=str(row.get("unit_id", "")).strip(),
                        order_id=str(row.get("order_id", "")).strip(),
                        variant=str(row.get("variant", "")).strip(),
                        transport_index=int(_f(row, "transport_index", 0.0)),
                        transport_name=str(row.get("transport_name", "")).strip(),
                        from_station=str(row.get("from_station", "")).strip(),
                        to_station=str(row.get("to_station", "")).strip(),
                        start_time_s=_f(row, "start_time_s"),
                        finish_time_s=finish_time_s,
                        transport_time_s=_f(row, "transport_time_s"),
                    )
                )
            except Exception:
                continue
    return records


def _operation_merge_key(operation: OperationRecord) -> tuple[Any, ...]:
    return (
        str(operation.unit_id),
        str(operation.order_id),
        str(operation.variant),
        int(operation.station_index),
        str(operation.station_name),
        round(float(operation.arrival_time_s), 6),
        round(float(operation.start_time_s), 6),
        round(float(operation.finish_time_s), 6),
    )


def _transport_merge_key(transport: TransportRecord) -> tuple[Any, ...]:
    return (
        str(transport.unit_id),
        str(transport.order_id),
        str(transport.variant),
        int(transport.transport_index),
        str(transport.transport_name),
        str(transport.from_station),
        str(transport.to_station),
        round(float(transport.start_time_s), 6),
        round(float(transport.finish_time_s), 6),
    )


def _combine_operations(previous: list[OperationRecord], current: list[OperationRecord]) -> list[OperationRecord]:
    combined_by_key: dict[tuple[Any, ...], OperationRecord] = {}
    for operation in list(previous) + list(current):
        combined_by_key[_operation_merge_key(operation)] = operation
    return sorted(
        combined_by_key.values(),
        key=lambda op: (float(op.start_time_s), float(op.finish_time_s), int(op.station_index), str(op.unit_id)),
    )


def _combine_transport_records(previous: list[TransportRecord], current: list[TransportRecord]) -> list[TransportRecord]:
    combined_by_key: dict[tuple[Any, ...], TransportRecord] = {}
    for transport in list(previous) + list(current):
        combined_by_key[_transport_merge_key(transport)] = transport
    return sorted(
        combined_by_key.values(),
        key=lambda tr: (float(tr.start_time_s), float(tr.finish_time_s), int(tr.transport_index), str(tr.unit_id)),
    )


def _read_previous_disruption_summary(disruption_summary_path: Path | None, cutoff_time_s: float) -> dict[str, Any]:
    if disruption_summary_path is None:
        return {"disruption_counts": {}, "events": []}
    path = Path(disruption_summary_path)
    if not path.exists() or not path.is_file():
        return {"disruption_counts": {}, "events": []}
    try:
        payload = load_json(path)
    except Exception:
        return {"disruption_counts": {}, "events": []}
    if not isinstance(payload, dict):
        return {"disruption_counts": {}, "events": []}

    cutoff = float(cutoff_time_s)
    previous_events: list[dict[str, Any]] = []
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        event_time = _event_timestamp_for_sort(event)
        if event_time <= cutoff + 1e-9:
            previous_events.append(event)

    previous_counts = payload.get("disruption_counts", {})
    if not isinstance(previous_counts, dict):
        previous_counts = {}
    return {"disruption_counts": dict(previous_counts), "events": previous_events}


def _event_timestamp_for_sort(event: dict[str, Any]) -> float:
    for key in ("disruption_timestamp_s", "time_s", "start_time_s", "end_time_s"):
        if key in event and event.get(key) is not None:
            try:
                return float(event.get(key))
            except (TypeError, ValueError):
                continue
    return 0.0


def _event_merge_key(event: dict[str, Any]) -> str:
    try:
        return json.dumps(event, sort_keys=True, default=str)
    except TypeError:
        return str(sorted(event.items()))


def _combine_disruption_details_with_previous(
    simulation_details: dict[str, Any],
    previous_disruption_summary: dict[str, Any],
) -> dict[str, Any]:
    combined_details = dict(simulation_details)

    combined_events_by_key: dict[str, dict[str, Any]] = {}
    for event in list(previous_disruption_summary.get("events", []) or []) + list(simulation_details.get("disruption_event_log", []) or []):
        if isinstance(event, dict):
            combined_events_by_key[_event_merge_key(event)] = event
    combined_events = sorted(combined_events_by_key.values(), key=lambda event: (_event_timestamp_for_sort(event), _event_merge_key(event)))

    combined_counts = Counter()
    previous_counts = previous_disruption_summary.get("disruption_counts", {})
    if isinstance(previous_counts, dict):
        for key, value in previous_counts.items():
            try:
                combined_counts[str(key)] += int(value)
            except (TypeError, ValueError):
                pass
    current_counts = simulation_details.get("disruption_counts", {})
    if isinstance(current_counts, dict):
        for key, value in current_counts.items():
            try:
                combined_counts[str(key)] += int(value)
            except (TypeError, ValueError):
                pass

    combined_details["disruption_event_log"] = combined_events
    combined_details["disruption_counts"] = dict(combined_counts)
    return combined_details


def _build_station_summaries_from_operations(
    operations: list[OperationRecord],
    station_sequence: list[str],
) -> list[StationSummary]:
    operations_by_station: defaultdict[tuple[int, str], list[OperationRecord]] = defaultdict(list)
    for operation in operations:
        operations_by_station[(int(operation.station_index), str(operation.station_name))].append(operation)

    station_keys: list[tuple[int, str]] = []
    seen_station_keys: set[tuple[int, str]] = set()
    for idx, station_name in enumerate(station_sequence, start=1):
        key = (idx, str(station_name))
        station_keys.append(key)
        seen_station_keys.add(key)
    for key in sorted(operations_by_station.keys()):
        if key not in seen_station_keys:
            station_keys.append(key)

    makespan_s = max((float(op.finish_time_s) for op in operations), default=0.0)
    summaries: list[StationSummary] = []
    for station_index, station_name in station_keys:
        station_ops = sorted(
            operations_by_station.get((station_index, station_name), []),
            key=lambda op: (float(op.start_time_s), float(op.finish_time_s), str(op.unit_id)),
        )
        if station_ops:
            busy_time_s = sum(float(op.process_time_s) for op in station_ops)
            first_start_time_s = min(float(op.start_time_s) for op in station_ops)
            last_finish_time_s = max(float(op.finish_time_s) for op in station_ops)
            max_queue_length = max(int(op.queue_length_on_arrival) for op in station_ops)
            average_wait_time_s = mean(float(op.wait_time_s) for op in station_ops)
            total_wait_time_s = sum(float(op.wait_time_s) for op in station_ops)
            active_window_s = max(0.0, last_finish_time_s - first_start_time_s)
            average_queue_length = mean(float(op.queue_length_on_arrival) for op in station_ops)
            utilization_overall = busy_time_s / makespan_s if makespan_s > 0.0 else 0.0
            utilization_active_window = busy_time_s / active_window_s if active_window_s > 0.0 else 0.0
        else:
            busy_time_s = 0.0
            first_start_time_s = None
            last_finish_time_s = None
            max_queue_length = 0
            average_queue_length = 0.0
            average_wait_time_s = 0.0
            total_wait_time_s = 0.0
            utilization_overall = 0.0
            utilization_active_window = 0.0

        summaries.append(
            StationSummary(
                station_index=station_index,
                station_name=station_name,
                busy_time_s=busy_time_s,
                first_start_time_s=first_start_time_s,
                last_finish_time_s=last_finish_time_s,
                max_queue_length=max_queue_length,
                average_queue_length=average_queue_length,
                average_wait_time_s=average_wait_time_s,
                total_wait_time_s=total_wait_time_s,
                utilization_overall=utilization_overall,
                utilization_active_window=utilization_active_window,
            )
        )
    return summaries


def _read_material_report_consumed(material_report_path: Path | None) -> dict[str, int]:
    consumed_by_material: dict[str, int] = {}
    if material_report_path is None:
        return consumed_by_material
    path = Path(material_report_path)
    if not path.exists() or not path.is_file():
        return consumed_by_material
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            material = str(row.get("material", "")).strip()
            if material == "":
                continue
            try:
                consumed_by_material[material] = int(float(row.get("consumed_for_produced_units", 0) or 0))
            except (TypeError, ValueError):
                consumed_by_material[material] = 0
    return consumed_by_material


def _calculate_cumulative_extra_material_consumed(
    previous_material_report_path: Path | None,
    previous_unit_summaries: list[UnitSummary],
    current_extra_material_consumed: dict[str, int] | None,
    bom_data: dict[str, Any],
) -> dict[str, int]:
    previous_consumed = _read_material_report_consumed(previous_material_report_path)
    previous_completed_variants = [str(summary.variant) for summary in previous_unit_summaries]
    previous_base_requirements = calculate_material_requirements(previous_completed_variants, bom_data)

    cumulative_extra: defaultdict[str, int] = defaultdict(int)
    for material, consumed_qty in previous_consumed.items():
        base_qty = int(previous_base_requirements.get(material, 0))
        extra_qty = int(consumed_qty) - base_qty
        if extra_qty > 0:
            cumulative_extra[material] += extra_qty

    for material, qty in (current_extra_material_consumed or {}).items():
        try:
            qty_int = int(qty)
        except (TypeError, ValueError):
            qty_int = 0
        if qty_int > 0:
            cumulative_extra[str(material)] += qty_int

    return dict(cumulative_extra)

def _filter_context_to_segment_remaining_units(ctx: dict[str, Any]) -> dict[str, Any]:
    segment_start_s = float(ctx.get("segment_start_s", 0.0) or 0.0)
    ongoing_unit_summary = ctx.get("ongoing_unit_summary")
    completed_ids = _read_completed_unit_ids_from_unit_summary(ongoing_unit_summary, segment_start_s)
    previous_summaries = _read_previous_unit_summaries(ongoing_unit_summary, segment_start_s)

    original_ordered_unit_count = len(ctx.get("ordered_units", []))
    unit_ids = list(ctx.get("unit_ids", [f"U{idx + 1:03d}" for idx in range(len(ctx.get("ordered_units", [])))]))
    keep_indices = [
        idx for idx, unit_id in enumerate(unit_ids)
        if str(unit_id) not in completed_ids
    ]

    for key in ("ordered_units", "unit_order_ids", "unit_ids", "unit_route_ids", "unit_release_times", "unit_priorities"):
        values = list(ctx.get(key, []))
        if values:
            ctx[key] = [values[idx] for idx in keep_indices if idx < len(values)]

    schedule_rows = list(ctx.get("schedule_rows", []))
    if schedule_rows:
        ctx["schedule_rows"] = [schedule_rows[idx] for idx in keep_indices if idx < len(schedule_rows)]

    ctx["previous_unit_summaries"] = previous_summaries
    ctx["completed_unit_ids_at_segment_start"] = sorted(completed_ids)
    ctx["segment_original_ordered_unit_count"] = int(original_ordered_unit_count)
    ctx["cumulative_requested_unit_count"] = int(len(previous_summaries) + len(ctx.get("ordered_units", [])))
    ctx["order_text"] = f"{ctx.get('order_text', 'schedule')}__segment_from_{int(segment_start_s)}__remaining_{len(ctx.get('ordered_units', []))}units"
    return ctx


def _shift_timed_records_for_segment(
    timed_records: list[dict[str, Any]],
    segment_start_s: float,
    segment_stop_s: float | None,
    visibility_cutoff_s: float | None = None,
) -> list[dict[str, Any]]:
    """Return only disruptions visible at the current segment start.

    Important:
    - The simulation/GA must not see future disruptions.
    - A timed disruption becomes visible only when its start_time has been reached.
    - The current segment may still simulate an already-visible active disruption
      until the next segment boundary.
    - Emergency orders are hidden until their start_time is reached. Once visible,
      they are only released into this simulation window when their Order_time falls
      inside the current simulated window.
    """
    segment_start_s = float(segment_start_s or 0.0)
    segment_stop_s = None if segment_stop_s is None else float(segment_stop_s)
    visibility_cutoff_s = segment_start_s if visibility_cutoff_s is None else float(visibility_cutoff_s)
    shifted: list[dict[str, Any]] = []
    eps = 1e-9

    for record in timed_records:
        start_abs = float(record.get("start_time_s", 0.0) or 0.0)
        end_raw = record.get("end_time_s")
        end_abs = float(end_raw) if end_raw is not None else start_abs
        dtype = str(record.get("disruption_type", "")).strip().casefold()

        # Do not expose future disruption events to the simulator/GA.
        if start_abs > visibility_cutoff_s + eps:
            continue

        new_record = dict(record)

        if dtype == "emergency_order":
            order_time_raw = record.get("order_time_s")
            order_time_abs = float(order_time_raw) if order_time_raw is not None else start_abs

            # Already released emergency orders should now be represented by the
            # current schedule/snapshot, not appended repeatedly every segment.
            if order_time_abs < segment_start_s - eps:
                continue

            # Do not release the emergency order inside this simulated window
            # before its Order_time is reached.
            if segment_stop_s is not None and order_time_abs > segment_stop_s + eps:
                continue

            new_record["start_time_s"] = max(0.0, start_abs - segment_start_s)
            new_record["order_time_s"] = max(0.0, order_time_abs - segment_start_s)
            shifted.append(new_record)
            continue

        if dtype in {"failed_inspection", "inspection_failure", "inspection failure"}:
            # Failed inspection is a one-shot event. It should only be injected
            # into the segment where the event timestamp is reached.
            if start_abs < segment_start_s - eps:
                continue
            if segment_stop_s is not None and start_abs > segment_stop_s + eps:
                continue
            new_record["start_time_s"] = max(0.0, start_abs - segment_start_s)
            if end_raw is not None:
                new_record["end_time_s"] = max(0.0, end_abs - segment_start_s)
            shifted.append(new_record)
            continue

        # Breakdown/efficiency-loss subtype: include only if it is already known
        # and intersects this segment window.
        if end_abs < segment_start_s - eps:
            continue
        if segment_stop_s is not None and start_abs > segment_stop_s + eps:
            continue

        new_record["start_time_s"] = max(0.0, start_abs - segment_start_s)
        if end_raw is not None:
            new_record["end_time_s"] = max(0.0, end_abs - segment_start_s)

        shifted.append(new_record)

    return shifted



def _offset_simulation_times_to_absolute(
    operations: list[OperationRecord],
    transport_records: list[TransportRecord],
    unit_summaries: list[UnitSummary],
    station_summaries: list[StationSummary],
    simulation_details: dict[str, Any],
    offset_s: float,
) -> None:
    offset_s = float(offset_s or 0.0)
    if offset_s == 0.0:
        return

    for op in operations:
        op.arrival_time_s += offset_s
        op.start_time_s += offset_s
        op.finish_time_s += offset_s

    for tr in transport_records:
        tr.start_time_s += offset_s
        tr.finish_time_s += offset_s

    for summary in unit_summaries:
        summary.first_arrival_time_s += offset_s
        summary.start_time_s += offset_s
        summary.completion_time_s += offset_s
        # flow/active times stay as durations

    for summary in station_summaries:
        if summary.first_start_time_s is not None:
            summary.first_start_time_s += offset_s
        if summary.last_finish_time_s is not None:
            summary.last_finish_time_s += offset_s

    for event in simulation_details.get("disruption_event_log", []) or []:
        for key in ("disruption_timestamp_s", "time_s", "start_time_s", "end_time_s"):
            if key in event and event[key] is not None:
                try:
                    event[key] = float(event[key]) + offset_s
                except (TypeError, ValueError):
                    pass

    carrier_snapshot = simulation_details.get("carrier_snapshot")
    if isinstance(carrier_snapshot, dict):
        if carrier_snapshot.get("snapshot_time_s") is not None:
            try:
                carrier_snapshot["snapshot_time_s"] = float(carrier_snapshot["snapshot_time_s"]) + offset_s
            except (TypeError, ValueError):
                pass
        if isinstance(carrier_snapshot.get("future_cart_return_times_s"), list):
            shifted_return_times: list[float] = []
            for value in carrier_snapshot.get("future_cart_return_times_s", []):
                try:
                    shifted_return_times.append(float(value) + offset_s)
                except (TypeError, ValueError):
                    continue
            carrier_snapshot["future_cart_return_times_s"] = shifted_return_times

    line_state_snapshot = simulation_details.get("line_state_snapshot")
    if isinstance(line_state_snapshot, dict):
        if line_state_snapshot.get("snapshot_time_s") is not None:
            try:
                line_state_snapshot["snapshot_time_s"] = float(line_state_snapshot["snapshot_time_s"]) + offset_s
            except (TypeError, ValueError):
                pass

        def _shift_snapshot_entry(entry: dict[str, Any], key: str) -> None:
            if key in entry and entry[key] is not None:
                try:
                    entry[key] = float(entry[key]) + offset_s
                except (TypeError, ValueError):
                    pass

        for group_name in ("processing_units", "station_queue_units", "arrival_events", "cart_return_events"):
            for entry in line_state_snapshot.get(group_name, []) or []:
                if isinstance(entry, dict):
                    for key in ("time_s", "arrival_time_s", "first_arrival_time_s", "first_start_time_s", "original_start_time_s"):
                        _shift_snapshot_entry(entry, key)

    if simulation_details.get("stop_time_s") is not None:
        try:
            simulation_details["stop_time_s"] = float(simulation_details["stop_time_s"]) + offset_s
        except (TypeError, ValueError):
            pass

    stop_reason = simulation_details.get("stop_reason")
    if isinstance(stop_reason, dict) and stop_reason.get("time_s") is not None:
        try:
            stop_reason["time_s"] = float(stop_reason["time_s"]) + offset_s
        except (TypeError, ValueError):
            pass



def _prepare_disruption_inputs(ctx: dict[str, Any], effective_line_layout: dict[str, Any], valid_variants: set[str]) -> tuple[int, bool, bool, bool, Any, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None, Path | None, Path | None]:
    settings_data = ctx["settings_data"]
    pathlist = ctx.get("pathlist", {})
    disruption_mode = _settings_disruption_mode(settings_data)
    disruptions_enabled = disruption_mode != 0
    chance_based = disruption_mode == 1
    timed = disruption_mode == 2
    seed = settings_data.get("seed")
    disruption_config = None
    timed_records: list[dict[str, Any]] = []
    timed_data = None
    disruption_json_path: Path | None = None
    timed_csv_path: Path | None = None
    if chance_based:
        disruption_json_path = _resolve_pathlist_disruption_json(pathlist) or resolve_disruption_path(ctx.get("input_root"), ctx.get("batch_dir"))
        if disruption_json_path is None or not disruption_json_path.exists():
            raise FileNotFoundError("random based disruptions are enabled, but disruption.json was not found")
        disruption_config = load_json(disruption_json_path)
    elif timed:
        timed_csv_path = _resolve_pathlist_disruption_csv(pathlist) or resolve_timed_disruption_csv_path(ctx.get("input_root"), ctx.get("batch_dir"))
        disruption_json_path = _resolve_pathlist_disruption_json(pathlist)
        if timed_csv_path is None or not timed_csv_path.exists():
            raise FileNotFoundError("time based disruptions are enabled, but the timed disruption CSV was not found")
        if disruption_json_path is None or not disruption_json_path.exists():
            raise FileNotFoundError("time based disruptions are enabled, but disruption_v2.json/disruption.json was not found")
        disruption_config = load_json(disruption_json_path)
        timed_records = load_timed_disruption_csv(timed_csv_path, valid_variants)
        timed_records = _assign_missing_emergency_order_ids(
            timed_records,
            list(ctx["unit_order_ids"]),
            list(ctx.get("unit_ids", [])),
        )
        timed_records_for_sim = _shift_timed_records_for_segment(
            timed_records=timed_records,
            segment_start_s=float(ctx.get("segment_start_s", 0.0) or 0.0),
            segment_stop_s=float(ctx.get("absolute_stop_time_s", ctx.get("simulation_time_s", 0.0)) or 0.0),
            visibility_cutoff_s=float(ctx.get("segment_start_s", 0.0) or 0.0),
        )
        scheduled_emergency_order_ids = _emergency_order_ids_already_in_schedule(list(ctx.get("schedule_rows", [])))
        if scheduled_emergency_order_ids:
            timed_records_for_sim = [
                record for record in timed_records_for_sim
                if not (
                    str(record.get("disruption_type", "")).strip().casefold() == "emergency_order"
                    and str(record.get("order_id", "")).strip() in scheduled_emergency_order_ids
                )
            ]
        timed_data = prepare_timed_disruption_data(
            list(effective_line_layout["station_sequence"]),
            timed_records_for_sim,
            timed_disruption_config=disruption_config,
        )
    return disruption_mode, disruptions_enabled, chance_based, timed, seed, disruption_config, timed_records, timed_data, disruption_json_path, timed_csv_path


def _simulation_result_from_summaries(unit_summaries: list[UnitSummary], details: dict[str, Any]) -> dict[str, Any]:
    order_completion_times: dict[str, float] = {}
    for summary in unit_summaries:
        key = str(summary.order_id)
        order_completion_times[key] = max(float(summary.completion_time_s), order_completion_times.get(key, 0.0))
    makespan = max([float(s.completion_time_s) for s in unit_summaries], default=0.0)
    return {
        "order_completion_times": order_completion_times,
        "makespan": makespan,
        "route_id_by_unit_id": dict(details.get("route_id_by_unit_id", {})),
        "completed_good_unit_count": int(details.get("completed_good_unit_count", len(unit_summaries))),
    }


def simulate_for_ga(main_settings_path: str | Path, current_time_s: float = 0.0) -> dict[str, Any]:
    ctx = _load_run_context_from_main_settings(Path(main_settings_path), None, segment_start_s=float(current_time_s or 0.0))
    ctx = _filter_context_to_segment_remaining_units(ctx)
    line_layout_path = resolve_line_layout_path(ctx["selected_line_layout_name"], ctx.get("input_root"), ctx.get("batch_dir"), ctx["data_dir"])
    line_layout_config, _ = load_line_layout_config(line_layout_path, ctx["process_time_data"])
    ctx["carriers"] = _carriers_from_layout_config(line_layout_config, ctx.get("carriers", MAX_UNITS_IN_SYSTEM))
    initial_line_state_snapshot = _load_line_state_snapshot_for_segment(
        ctx.get("ongoing_line_state_snapshot"),
        float(ctx.get("segment_start_s", 0.0) or 0.0),
    )
    initial_slots, initial_return_times = _load_carrier_snapshot_for_segment(
        ctx.get("ongoing_carrier_snapshot"),
        float(ctx.get("segment_start_s", 0.0) or 0.0),
        int(ctx["carriers"]),
    )
    if isinstance(initial_line_state_snapshot, dict) and int(initial_line_state_snapshot.get("occupied_carriers", 0) or 0) > 0:
        initial_slots, initial_return_times = None, []
    ctx["initial_available_system_slots"] = initial_slots
    ctx["initial_cart_return_times_s"] = initial_return_times
    ctx["initial_line_state_snapshot"] = initial_line_state_snapshot
    effective = build_effective_line_layout(ctx["process_time_data"], ctx["transport_time_data"], line_layout_config)
    valid_variants = set(ctx["process_time_data"]["process_times"].keys())
    mode, enabled, chance_based, timed, seed, dis_cfg, timed_records, timed_data, _, _ = _prepare_disruption_inputs(ctx, effective, valid_variants)
    ops, trs, units, stations, _, details = run_simulation(
        ordered_units=list(ctx["ordered_units"]),
        process_time_data=ctx["process_time_data"],
        transport_time_data=ctx["transport_time_data"],
        unit_release_times=list(ctx["unit_release_times"]),
        unit_priorities=list(ctx["unit_priorities"]),
        unit_order_ids=list(ctx["unit_order_ids"]),
        unit_ids=list(ctx.get("unit_ids", [f"U{idx + 1:03d}" for idx in range(len(ctx["ordered_units"]))])),
        unit_route_ids=list(ctx["unit_route_ids"]),
        max_units_in_system=ctx["carriers"],
        line_layout_config=line_layout_config,
        bom_data=ctx["bom_data"],
        material_stock_data=ctx["material_stock_data"],
        disruptions_enabled=enabled,
        disruption_config=dis_cfg if chance_based else None,
        disruption_seed=seed if chance_based else None,
        simulation_time_s=ctx["simulation_time_s"],
        timed_disruption_data=timed_data,
        initial_available_system_slots=ctx.get("initial_available_system_slots"),
        initial_cart_return_times_s=ctx.get("initial_cart_return_times_s", []),
        initial_line_state_snapshot=ctx.get("initial_line_state_snapshot"),
    )
    return _simulation_result_from_summaries(units, details)



def _estimate_breakdown_duration_from_config(
    disruption_config: dict[str, Any] | None,
    stage_number: int | None,
    disruption_name: str,
) -> float:
    """Return estimated disruption duration as mean + 3 * std.

    In disruption_v2.json, std is stored as a percent of the mean, so:
        estimated = mean + 3 * (mean * std_percent / 100)

    Returns 0.0 for non-breakdown disruptions or if the disruption subtype
    cannot be found in the config.
    """
    if not isinstance(disruption_config, dict) or stage_number is None:
        return 0.0

    stations_cfg = disruption_config.get("Stations", {})
    if not isinstance(stations_cfg, dict):
        return 0.0

    station_cfg = stations_cfg.get(str(int(stage_number)), {})
    if not isinstance(station_cfg, dict):
        return 0.0

    disruption_name_cf = str(disruption_name or "").strip().casefold()

    machine_breakdowns = station_cfg.get("machine breakdowns", [])
    if isinstance(machine_breakdowns, list):
        for entry in machine_breakdowns:
            if not isinstance(entry, dict):
                continue
            entry_name = str(entry.get("name", "")).strip().casefold()
            if disruption_name_cf and entry_name != disruption_name_cf:
                continue
            mean_s = float(entry.get("mean [s]", entry.get("mean", 0.0)) or 0.0)
            std_pct = float(entry.get("std [% of mean]", entry.get("std", 0.0)) or 0.0)
            return mean_s + 3.0 * mean_s * (std_pct / 100.0)

    # Backward-compatible v1 fallback if the config has a single breakdown dict.
    breakdown_cfg = station_cfg.get("breakdown", {})
    if isinstance(breakdown_cfg, dict):
        if isinstance(breakdown_cfg.get("range"), (list, tuple)) and len(breakdown_cfg.get("range", [])) >= 2:
            low = float(breakdown_cfg["range"][0])
            high = float(breakdown_cfg["range"][1])
            mean_s = (low + high) / 2.0
        else:
            mean_s = float(breakdown_cfg.get("duration [s]", breakdown_cfg.get("mean [s]", 0.0)) or 0.0)
        std_pct = float(breakdown_cfg.get("std [% of mean]", breakdown_cfg.get("std", 0.0)) or 0.0)
        return mean_s + 3.0 * mean_s * (std_pct / 100.0)

    return 0.0

def _write_disruption_history_from_timed_csv(
    path: Path | None,
    timed_records: list[dict[str, Any]],
    disruption_config: dict[str, Any] | None,
    cutoff_time_s: float | None = None,
) -> None:
    """Write cumulative disruption history only up to the current known timestamp.

    Future disruptions are not written. If a disruption has started but its
    end_time is still in the future relative to cutoff_time_s, end_time is left
    blank until a later segment reaches that end timestamp.
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cutoff = None if cutoff_time_s is None else float(cutoff_time_s)
    fieldnames = ["disruption_type", "estimated_duration", "station_id", "efficiency_loss", "start_time", "end_time"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(timed_records, key=lambda r: (float(r.get("start_time_s", 0.0)), int(r.get("row_index", 0)))):
            start = float(record.get("start_time_s", 0.0) or 0.0)
            if cutoff is not None and start > cutoff:
                continue

            dtype = str(record.get("disruption_type", "")).strip()
            dtype_cf = dtype.casefold()
            station_id = "" if dtype_cf in {"failed_inspection", "inspection_failure", "inspection failure", "emergency_order"} else ("" if record.get("station_id") is None else str(record.get("station_id")))

            eff_loss = 0.0
            if dtype_cf in {"efficiency_loss", "efficiency loss"} and record.get("efficiency_percentage") is not None:
                eff_loss = max(0.0, 100.0 - float(record.get("efficiency_percentage")))

            estimated = 0.0
            stage = None
            if station_id:
                try:
                    stage = int(float(str(station_id).split(".")[0]))
                except Exception:
                    stage = None
            if dtype_cf not in {"efficiency_loss", "efficiency loss", "failed_inspection", "inspection_failure", "inspection failure", "emergency_order"}:
                estimated = _estimate_breakdown_duration_from_config(disruption_config, stage, dtype)

            end_value: str | float = ""
            end_raw = record.get("end_time_s")
            if end_raw is not None:
                end_abs = float(end_raw)
                if cutoff is None or end_abs <= cutoff:
                    end_value = round(end_abs, 6)

            writer.writerow({
                "disruption_type": dtype,
                "estimated_duration": round(float(estimated), 6),
                "station_id": station_id,
                "efficiency_loss": round(float(eff_loss), 6),
                "start_time": round(start, 6),
                "end_time": end_value,
            })



def _write_outputs_for_integrated_run(ctx: dict[str, Any], operations, transport_records, unit_summaries, station_summaries, simulation_details, material_report, kpis, kpis_without_disruptions, disruptions_enabled, disruption_mode, disruption_seed, disruption_config, timed_records, dis_json_path, timed_csv_path, settings_path, run_output_dir: Path) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    copy_file_if_exists(settings_path, run_output_dir / "settings_used.json")
    if int(disruption_mode) == 1:
        copy_file_if_exists(dis_json_path, run_output_dir / "disruption_used.json")
    elif int(disruption_mode) == 2:
        copy_file_if_exists(dis_json_path, run_output_dir / "disruption_used.json")
        copy_file_if_exists(timed_csv_path, run_output_dir / "disruption_used.csv")
    save_run_metadata(ctx["order_text"], ctx["ordered_units"], run_output_dir / "run_metadata.json", ctx["data_dir"], run_output_dir, extra_payload={"main_settings_json": str(ctx["main_settings_path"]), "input_current_schedule": str(ctx["schedule_path"])})
    write_material_report_csv(material_report, run_output_dir / "material_report.csv")
    completed_good_unit_count = len(unit_summaries)
    requested_unit_count = int(ctx.get("cumulative_requested_unit_count", len(ctx["ordered_units"])))
    save_json({"requested_unit_count": requested_unit_count, "produced_unit_count": completed_good_unit_count, "completed_good_units": completed_good_unit_count, "disruptions_enabled": bool(disruptions_enabled), "disruption_mode": int(disruption_mode), "stop_reason": simulation_details.get("stop_reason")}, run_output_dir / "production_status.json")
    write_kpis_csv(kpis, run_output_dir / "kpi_summary.csv")
    write_kpis_csv(kpis_without_disruptions, run_output_dir / "kpi_summary_without_disruptions.csv")
    write_operations_csv(operations, run_output_dir / "station_schedule.csv")
    write_transport_csv(transport_records, run_output_dir / "transport_schedule.csv")
    write_unit_summary_csv(unit_summaries, run_output_dir / "unit_summary.csv")
    write_station_summary_csv(station_summaries, run_output_dir / "station_summary.csv")
    if disruptions_enabled or simulation_details.get("disruption_event_log"):
        save_json({"disruptions_enabled": bool(disruptions_enabled), "disruption_mode": int(disruption_mode), "seed": str(disruption_seed) if disruption_seed is not None else None, "timed_disruption_records": list(timed_records), "disruption_counts": dict(simulation_details.get("disruption_counts", {})), "events": list(simulation_details.get("disruption_event_log", []))}, run_output_dir / "disruption_summary.json")


def main(main_settings_path: str | Path | None = None, simulation_time_limit_s: float | None = None, segment_start_s: float = 0.0) -> None:
    starttime = time.perf_counter()
    ctx = _load_run_context_from_main_settings(Path(main_settings_path), simulation_time_limit_s, segment_start_s=float(segment_start_s or 0.0))
    ctx = _filter_context_to_segment_remaining_units(ctx)
    line_layout_path = resolve_line_layout_path(ctx["selected_line_layout_name"], ctx.get("input_root"), ctx.get("batch_dir"), ctx["data_dir"])
    line_layout_config, line_layout_path_loaded = load_line_layout_config(line_layout_path, ctx["process_time_data"])
    ctx["carriers"] = _carriers_from_layout_config(line_layout_config, ctx.get("carriers", MAX_UNITS_IN_SYSTEM))
    initial_line_state_snapshot = _load_line_state_snapshot_for_segment(
        ctx.get("ongoing_line_state_snapshot"),
        float(ctx.get("segment_start_s", 0.0) or 0.0),
    )
    initial_slots, initial_return_times = _load_carrier_snapshot_for_segment(
        ctx.get("ongoing_carrier_snapshot"),
        float(ctx.get("segment_start_s", 0.0) or 0.0),
        int(ctx["carriers"]),
    )
    if isinstance(initial_line_state_snapshot, dict) and int(initial_line_state_snapshot.get("occupied_carriers", 0) or 0) > 0:
        initial_slots, initial_return_times = None, []
    ctx["initial_available_system_slots"] = initial_slots
    ctx["initial_cart_return_times_s"] = initial_return_times
    ctx["initial_line_state_snapshot"] = initial_line_state_snapshot
    effective = build_effective_line_layout(ctx["process_time_data"], ctx["transport_time_data"], line_layout_config)
    valid_variants = set(ctx["process_time_data"]["process_times"].keys())
    mode, enabled, chance_based, timed, seed, dis_cfg, timed_records, timed_data, dis_json_path, timed_csv_path = _prepare_disruption_inputs(ctx, effective, valid_variants)
    operations, transport_records, unit_summaries, station_summaries, _, details = run_simulation(
        ordered_units=list(ctx["ordered_units"]),
        process_time_data=ctx["process_time_data"],
        transport_time_data=ctx["transport_time_data"],
        unit_release_times=list(ctx["unit_release_times"]),
        unit_priorities=list(ctx["unit_priorities"]),
        unit_order_ids=list(ctx["unit_order_ids"]),
        unit_ids=list(ctx.get("unit_ids", [f"U{idx + 1:03d}" for idx in range(len(ctx["ordered_units"]))])),
        unit_route_ids=list(ctx["unit_route_ids"]),
        max_units_in_system=ctx["carriers"],
        line_layout_config=line_layout_config,
        bom_data=ctx["bom_data"],
        material_stock_data=ctx["material_stock_data"],
        disruptions_enabled=enabled,
        disruption_config=dis_cfg if chance_based else None,
        disruption_seed=seed if chance_based else None,
        simulation_time_s=ctx["simulation_time_s"],
        timed_disruption_data=timed_data,
        initial_available_system_slots=ctx.get("initial_available_system_slots"),
        initial_cart_return_times_s=ctx.get("initial_cart_return_times_s", []),
        initial_line_state_snapshot=ctx.get("initial_line_state_snapshot"),
    )
    _offset_simulation_times_to_absolute(
        operations=operations,
        transport_records=transport_records,
        unit_summaries=unit_summaries,
        station_summaries=station_summaries,
        simulation_details=details,
        offset_s=float(ctx.get("segment_start_s", 0.0) or 0.0),
    )
    _write_carrier_snapshot(ctx.get("ongoing_carrier_snapshot"), details)
    _write_line_state_snapshot(ctx.get("ongoing_line_state_snapshot"), details)

    previous_unit_summaries = list(ctx.get("previous_unit_summaries", []))
    combined_unit_summaries = previous_unit_summaries + unit_summaries

    output_results = Path(ctx.get("output_results") or (Path(__file__).resolve().parent / "output" / "results"))
    segment_start_s = float(ctx.get("segment_start_s", 0.0) or 0.0)
    previous_operations = _read_previous_operations_csv(output_results / "station_schedule.csv", segment_start_s) if bool(ctx.get("has_assigned_route")) else []
    previous_transport_records = _read_previous_transport_csv(output_results / "transport_schedule.csv", segment_start_s) if bool(ctx.get("has_assigned_route")) else []
    combined_operations = _combine_operations(previous_operations, operations)
    combined_transport_records = _combine_transport_records(previous_transport_records, transport_records)
    station_summaries_for_output = _build_station_summaries_from_operations(
        combined_operations,
        list(effective.get("station_sequence", [])),
    )

    previous_disruption_summary = _read_previous_disruption_summary(output_results / "disruption_summary.json", segment_start_s) if bool(ctx.get("has_assigned_route")) else {"disruption_counts": {}, "events": []}
    details_for_output = _combine_disruption_details_with_previous(details, previous_disruption_summary)

    completed_good_variants = list(details.get("completed_good_variants", []))
    combined_completed_good_variants = [str(summary.variant) for summary in combined_unit_summaries]
    all_requested_units_for_outputs = [str(summary.variant) for summary in previous_unit_summaries] + list(ctx["ordered_units"])
    cumulative_extra_material_consumed = _calculate_cumulative_extra_material_consumed(
        output_results / "material_report.csv" if bool(ctx.get("has_assigned_route")) else None,
        previous_unit_summaries,
        details.get("extra_material_consumed", {}),
        ctx["bom_data"],
    )
    material_report = build_material_report(
        all_requested_units_for_outputs,
        combined_completed_good_variants,
        ctx["bom_data"],
        ctx["material_stock_data"],
        extra_material_consumed=cumulative_extra_material_consumed,
    )
    kpis = calculate_kpis(
        combined_completed_good_variants,
        combined_operations,
        combined_unit_summaries,
        station_summaries_for_output,
        combined_transport_records,
    )

    kpis_without_disruptions = calculate_kpis_without_disruption_penalties(
        ordered_units=combined_completed_good_variants,
        operations=combined_operations,
        unit_summaries=combined_unit_summaries,
        station_summaries=station_summaries_for_output,
        transport_records=combined_transport_records,
        timed_records=timed_records if enabled else [],
        cutoff_time_s=ctx.get("absolute_stop_time_s", ctx["simulation_time_s"]),
    )

    ongoing_unit_summary = ctx.get("ongoing_unit_summary")
    if ongoing_unit_summary is not None:
        write_unit_summary_csv(combined_unit_summaries, ongoing_unit_summary)
    _write_disruption_history_from_timed_csv(
        ctx.get("ongoing_disruption_history"),
        timed_records,
        dis_cfg,
        cutoff_time_s=ctx.get("absolute_stop_time_s", ctx["simulation_time_s"]),
    )
    if bool(ctx.get("has_assigned_route")):
        _write_outputs_for_integrated_run(ctx, combined_operations, combined_transport_records, combined_unit_summaries, station_summaries_for_output, details_for_output, material_report, kpis, kpis_without_disruptions, enabled, mode, seed, dis_cfg, timed_records, dis_json_path, timed_csv_path, ctx["main_settings_path"], output_results)
    print(f"Run folder: {ctx.get('output_results') if ctx.get('has_assigned_route') else '(output skipped: no assigned routes)'}")
    print(f"Simulation time: {ctx['simulation_time_s']} s")
    print(f"Requested units: {len(ctx['ordered_units'])}")
    print(f"Completed good units: {len(completed_good_variants)}")
    print(f"Total execution time: {time.perf_counter() - starttime:.6f} seconds")


if __name__ == "__main__":
    main()
