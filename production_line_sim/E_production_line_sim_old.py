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
MAIN_INPUT_BATCH_NAME_RE = re.compile(r"^main_(\d{2})-(\d{2})_(\d{2})-(\d{2})_(\d+)$", re.IGNORECASE)
RUN_DIR_NAME_RE = re.compile(r"^run_(\d+)$", re.IGNORECASE)
INPUT_ORDER_CSV_RE = re.compile(r"^orders_(\d{2})-(\d{2})_(\d{2})-(\d{2})_(\d+)\.csv$", re.IGNORECASE)
LINE_LAYOUT_FILENAME = "line_layout.json"
DISRUPTION_FILENAME = "disruption.json"
TIMED_DISRUPTION_FILENAME = "disruptions.csv"
TIMED_DISRUPTION_V2_FILENAME = "disruption_v2.json"
DISRUPTIONS_DIRNAME = "disruptions"
ONGOING_DIRNAME = "on_going"
CURRENT_SCHEDULE_FILENAME = "current_schedule.csv"
DISRUPTION_HISTORY_FILENAME = "disruption_his.csv"
LINE_LAYOUT_SETTINGS_KEYS = (
    "line_layout_file",
    "line_layout_filename",
    "layout_file",
    "layout_filename",
)
STATION_NAME_NUMBER_RE = re.compile(r"^\s*Station\s+(\d+)(?:\.(\d+))?\s*:\s*(.+?)\s*$", re.IGNORECASE)
TRANSPORT_NAME_NUMBER_RE = re.compile(r"^\s*Transportation\s+(\d+)\s*$", re.IGNORECASE)

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
    transport_time_data: dict[str, Any],
    line_layout_config: dict[str, Any],
) -> dict[str, Any]:
    base_station_sequence = list(process_time_data["station_sequence"])
    base_transport_lookup = build_transport_lookup(base_station_sequence, transport_time_data)

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

        configured_base_sequence.append(base_station_name)
        resolved_stages.append(
            {
                "stage_position": stage_position,
                "base_station_name": base_station_name,
                "copies": copies,
                "branch_transport_from_previous_s": branch_transport_from_previous_s,
                "branch_transport_to_next_s": branch_transport_to_next_s,
            }
        )

    if configured_base_sequence != base_station_sequence:
        raise ValueError(
            f"{LINE_LAYOUT_FILENAME} must keep the same base station order as process_times.json. Expected: {base_station_sequence}"
        )

    station_instance_names: list[str] = []
    station_instance_base_names: list[str] = []
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
        elif isinstance(raw_entry, dict):
            station_name = str(
                raw_entry.get("station_name", raw_entry.get("name", raw_entry.get("station", "")))
            ).strip()
            branch_transport_from_previous_s = float(raw_entry.get("branch_transport_from_previous_s", 0.0))
            branch_transport_to_next_s = float(raw_entry.get("branch_transport_to_next_s", 0.0))
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
                "declared_order": entry_index,
            }
        )

    return normalized_entries


def _build_effective_line_layout_from_station_instances(
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any],
    line_layout_config: dict[str, Any],
) -> dict[str, Any]:
    base_station_sequence = list(process_time_data["station_sequence"])
    base_transport_lookup = build_transport_lookup(base_station_sequence, transport_time_data)
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


def build_effective_line_layout(
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any],
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


def _parse_main_input_batch_sort_key(name: str) -> tuple[int, int, int, int, int]:
    stem = Path(name).stem
    match = MAIN_INPUT_BATCH_NAME_RE.match(stem)
    if not match:
        raise ValueError(f"Input batch name '{name}' must match main_DD-MM_HH-MM_N")

    day_s, month_s, hour_s, minute_s, sequence_s = match.groups()
    return (int(month_s), int(day_s), int(hour_s), int(minute_s), int(sequence_s))


def _parse_run_dir_sort_key(name: str) -> int:
    match = RUN_DIR_NAME_RE.match(str(name).strip())
    if not match:
        raise ValueError(f"Run folder name '{name}' must match run_N")
    return int(match.group(1))


def resolve_ongoing_run_dir(base_dir: Path) -> Path:
    ongoing_root = base_dir / ONGOING_DIRNAME
    ongoing_root.mkdir(parents=True, exist_ok=True)

    main_dirs = [
        path for path in ongoing_root.iterdir()
        if path.is_dir() and MAIN_INPUT_BATCH_NAME_RE.match(path.name)
    ]
    if main_dirs:
        newest_main_dir = max(main_dirs, key=lambda path: _parse_main_input_batch_sort_key(path.name))
    else:
        newest_main_dir = ongoing_root / f"main_{datetime.now().strftime('%d-%m_%H-%M_0')}"
        newest_main_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [
        path for path in newest_main_dir.iterdir()
        if path.is_dir() and RUN_DIR_NAME_RE.match(path.name)
    ]
    if run_dirs:
        return min(run_dirs, key=lambda path: _parse_run_dir_sort_key(path.name))

    run_dir = newest_main_dir / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_current_schedule_path(base_dir: Path) -> Path | None:
    ongoing_run_dir = resolve_ongoing_run_dir(base_dir)
    schedule_path = ongoing_run_dir / CURRENT_SCHEDULE_FILENAME
    if schedule_path.exists() and schedule_path.is_file():
        return schedule_path
    return None


def _route_value_is_assigned(route_value: Any) -> bool:
    route_text = str(route_value).strip()
    if route_text == "" or route_text.casefold() == "nan":
        return False
    try:
        numeric_value = float(route_text)
    except (TypeError, ValueError):
        return route_text != "0"
    if math.isnan(numeric_value):
        return False
    return numeric_value != 0.0


def _load_current_schedule_csv(schedule_csv_path: Path, valid_variants: set[str]) -> dict[str, Any]:
    scheduled_rows: list[dict[str, Any]] = []

    with schedule_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Schedule CSV is empty: {schedule_csv_path}")

        normalized_header = [re.sub(r"[^a-z0-9]", "", str(value).strip().lower()) for value in header]
        if normalized_header[:4] != ["unitseq", "orderid", "unitid", "variant"]:
            raise ValueError("current_schedule.csv must start with columns: unit_seq, order_id, unit_id, variant")
        route_col_idx = normalized_header.index("routeid") if "routeid" in normalized_header else None

        for row_index, row in enumerate(reader, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            min_row_length = max(4, (route_col_idx + 1) if route_col_idx is not None else 4)
            padded_row = list(row) + [""] * max(0, min_row_length - len(row))
            unit_seq = _read_int(padded_row[0], row_index - 1)
            order_id = str(padded_row[1]).strip() if str(padded_row[1]).strip() != "" else str(row_index - 1)
            unit_id = str(padded_row[2]).strip() if str(padded_row[2]).strip() != "" else f"U{int(unit_seq):03d}"
            variant = str(padded_row[3]).strip().upper()
            route_id = str(padded_row[route_col_idx]).strip() if route_col_idx is not None else "0"

            if variant not in valid_variants:
                raise ValueError(
                    f"Unknown variant '{variant}' in {schedule_csv_path.name}. Valid options: {', '.join(sorted(valid_variants))}"
                )

            scheduled_rows.append(
            {
                "unit_seq": int(unit_seq),
                "order_id": order_id,
                "unit_id": unit_id,
                    "variant": variant,
                    "route_id": route_id,
                    "row_index": int(row_index),
                }
            )

    scheduled_rows.sort(key=lambda row: (int(row["unit_seq"]), int(row["row_index"])))

    return {
        "ordered_units": [str(row["variant"]) for row in scheduled_rows],
        "unit_ids": [str(row.get("unit_id", "")) for row in scheduled_rows],
        "unit_release_times": [0.0] * len(scheduled_rows),
        "unit_priorities": [1] * len(scheduled_rows),
        "unit_order_ids": [str(row["order_id"]) for row in scheduled_rows],
        "unit_route_ids": [str(row.get("route_id", "0")) for row in scheduled_rows],
        "has_assigned_route": any(_route_value_is_assigned(row.get("route_id", "0")) for row in scheduled_rows),
    }

def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    value_text = str(value).strip()
    if value_text == "":
        return None
    return Path(value_text)


def _pathlist_path(pathlist: dict[str, Any], key: str) -> Path | None:
    if not isinstance(pathlist, dict):
        return None
    return _coerce_path(pathlist.get(key))


def _existing_pathlist_file(pathlist: dict[str, Any], key: str) -> Path | None:
    path = _pathlist_path(pathlist, key)
    if path is not None and path.exists() and path.is_file():
        return path
    return None


def _resolve_pathlist_disruption_csv(pathlist: dict[str, Any]) -> Path | None:
    exact_path = _pathlist_path(pathlist, "input_disruptions_csv")
    if exact_path is not None:
        if exact_path.exists() and exact_path.is_file():
            return exact_path
        if exact_path.name.startswith("disruption_"):
            plural_candidate = exact_path.with_name("disruptions_" + exact_path.name[len("disruption_"):])
            if plural_candidate.exists() and plural_candidate.is_file():
                return plural_candidate

    disruption_dir = _pathlist_path(pathlist, "input_disruptions")
    if disruption_dir is not None and disruption_dir.exists() and disruption_dir.is_dir():
        candidates: list[Path] = []
        for pattern in ("disruptions_*.csv", "disruption_*.csv", "disruptions*.csv", "disruption_list*.csv"):
            candidates.extend([path for path in disruption_dir.glob(pattern) if path.is_file()])
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    return None


def _resolve_pathlist_disruption_json(pathlist: dict[str, Any]) -> Path | None:
    exact_path = _pathlist_path(pathlist, "input_disruptions_json")
    if exact_path is not None and exact_path.exists() and exact_path.is_file():
        return exact_path

    disruption_dir = _pathlist_path(pathlist, "input_disruptions")
    if disruption_dir is not None and disruption_dir.exists() and disruption_dir.is_dir():
        for filename in ("disruption_v2.json", "disruption.json"):
            candidate = disruption_dir / filename
            if candidate.exists() and candidate.is_file():
                return candidate

        candidates = [path for path in disruption_dir.glob("disruption*.json") if path.is_file()]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    return None


def load_input_from_main_settings(
    main_settings_path: Path,
    data_dir: Path,
    valid_variants: set[str],
) -> dict[str, Any]:
    main_settings_path = Path(main_settings_path)
    main_settings_data = load_json(main_settings_path)
    pathlist = dict(main_settings_data.get("pathlist", {}))
    run_settings = dict(main_settings_data.get("settings", {}))

    base_settings_path = data_dir / "settings.json"
    settings_data: dict[str, Any] = load_json(base_settings_path) if base_settings_path.exists() else {}
    settings_data.update(run_settings)

    scenario_layout = run_settings.get("Scenarios")
    if isinstance(scenario_layout, str) and scenario_layout.strip() != "":
        settings_data["line_layout_file"] = scenario_layout.strip()

    schedule_path = _pathlist_path(pathlist, "on_going_run_current_schedule")
    if schedule_path is None:
        raise FileNotFoundError("main_settings.json pathlist is missing 'on_going_run_current_schedule'.")
    if not schedule_path.exists():
        raise FileNotFoundError(f"current_schedule.csv was not found at {schedule_path}")

    schedule_payload = _load_current_schedule_csv(schedule_path, valid_variants)
    ordered_units = list(schedule_payload["ordered_units"])
    unit_ids = list(schedule_payload.get("unit_ids", []))

    input_root = _pathlist_path(pathlist, "input") or main_settings_path.parent
    batch_dir = _pathlist_path(pathlist, "input_runs_run") or main_settings_path.parent
    output_root = _pathlist_path(pathlist, "output_run_results")

    simulation_time_s = float(settings_data.get("sim_time [s]", settings_data.get("Sim_time [s]", 0.0)))
    carriers = int(float(settings_data.get("carriers", {}).get("number of carriers", MAX_UNITS_IN_SYSTEM)))

    label_text = str(main_settings_data.get("label", main_settings_path.stem))
    order_text = f"{label_text}__scheduled_{len(ordered_units)}units"

    return {
        "order_text": order_text,
        "ordered_units": ordered_units,
        "unit_ids": unit_ids,
        "unit_release_times": list(schedule_payload["unit_release_times"]),
        "unit_priorities": list(schedule_payload["unit_priorities"]),
        "unit_order_ids": list(schedule_payload["unit_order_ids"]),
        "unit_route_ids": list(schedule_payload.get("unit_route_ids", [])),
        "has_assigned_route": bool(schedule_payload.get("has_assigned_route", False)),
        "simulation_time_s": simulation_time_s,
        "carriers": carriers,
        "settings_data": settings_data,
        "selected_line_layout_name": _resolve_line_layout_filename_from_settings(settings_data),
        "input_root": input_root,
        "batch_dir": batch_dir,
        "orders_csv_path": schedule_path,
        "settings_path": main_settings_path,
        "main_settings_path": main_settings_path,
        "main_settings_data": main_settings_data,
        "pathlist": pathlist,
        "output_root": output_root,
    }

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

    main_dirs = [
        path for path in input_root.iterdir()
        if path.is_dir() and MAIN_INPUT_BATCH_NAME_RE.match(path.name)
    ]
    if main_dirs:
        newest_main_dir = max(main_dirs, key=lambda path: _parse_main_input_batch_sort_key(path.name))
        run_dirs = [
            path for path in newest_main_dir.iterdir()
            if path.is_dir() and RUN_DIR_NAME_RE.match(path.name)
        ]
        if not run_dirs:
            raise FileNotFoundError(f"No run folders like run_1 were found in {newest_main_dir}")
        return min(run_dirs, key=lambda path: _parse_run_dir_sort_key(path.name))

    candidate_dirs = [
        path for path in input_root.iterdir()
        if path.is_dir() and INPUT_BATCH_NAME_RE.match(path.name)
    ]
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No generated input folders were found in {input_root}. Expected folders like main_DD-MM_HH-MM_N or orders_DD-MM_HH-MM_N"
        )
    return max(candidate_dirs, key=lambda path: _parse_input_batch_sort_key(path.name))


def find_newest_orders_csv(batch_dir: Path) -> Path:
    current_schedule_path = resolve_current_schedule_path(Path(__file__).resolve().parent)
    if current_schedule_path is None:
        raise FileNotFoundError(
            f"{CURRENT_SCHEDULE_FILENAME} was not found in {Path(__file__).resolve().parent / ONGOING_DIRNAME}/main_<newest>/run_<lowest>"
        )
    return current_schedule_path


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

    schedule_payload = _load_current_schedule_csv(orders_csv_path, valid_variants)
    ordered_units = list(schedule_payload["ordered_units"])
    unit_ids = list(schedule_payload.get("unit_ids", []))
    order_text = f"{batch_dir.parent.name}__{batch_dir.name}__scheduled_{len(ordered_units)}units"

    return {
        "order_text": order_text,
        "ordered_units": ordered_units,
        "unit_ids": unit_ids,
        "unit_release_times": list(schedule_payload["unit_release_times"]),
        "unit_priorities": list(schedule_payload["unit_priorities"]),
        "unit_order_ids": list(schedule_payload["unit_order_ids"]),
        "unit_route_ids": list(schedule_payload.get("unit_route_ids", [])),
        "has_assigned_route": bool(schedule_payload.get("has_assigned_route", False)),
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
    search_dirs: list[Path] = []

    if batch_dir is not None:
        if batch_dir.parent is not None and MAIN_INPUT_BATCH_NAME_RE.match(batch_dir.parent.name):
            search_dirs.append(batch_dir.parent / DISRUPTIONS_DIRNAME)
        search_dirs.append(batch_dir / DISRUPTIONS_DIRNAME)
        search_dirs.append(batch_dir)

    if input_root is not None:
        main_dirs = [
            path for path in input_root.iterdir()
            if path.is_dir() and MAIN_INPUT_BATCH_NAME_RE.match(path.name)
        ] if input_root.exists() else []
        if main_dirs:
            newest_main_dir = max(main_dirs, key=lambda path: _parse_main_input_batch_sort_key(path.name))
            search_dirs.append(newest_main_dir / DISRUPTIONS_DIRNAME)
        search_dirs.append(input_root / DISRUPTIONS_DIRNAME)
        search_dirs.append(input_root)

    unique_search_dirs: list[Path] = []
    seen_dirs: set[str] = set()
    for search_dir in search_dirs:
        dir_key = str(search_dir.resolve()) if search_dir.exists() else str(search_dir)
        if dir_key not in seen_dirs:
            unique_search_dirs.append(search_dir)
            seen_dirs.add(dir_key)

    for search_dir in unique_search_dirs:
        if not search_dir.exists() or not search_dir.is_dir():
            continue

        exact_candidate = search_dir / TIMED_DISRUPTION_FILENAME
        if exact_candidate.exists() and exact_candidate.is_file():
            return exact_candidate

        preferred_candidates = sorted(
            [path for path in search_dir.glob("disruptions_*.csv") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )
        if preferred_candidates:
            return max(preferred_candidates, key=lambda path: path.stat().st_mtime)

        fallback_candidates = sorted(
            [path for path in search_dir.glob("disruptions*.csv") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )
        if fallback_candidates:
            return max(fallback_candidates, key=lambda path: path.stat().st_mtime)

        disruption_list_candidates = sorted(
            [path for path in search_dir.glob("disruption_list*.csv") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )
        if disruption_list_candidates:
            return max(disruption_list_candidates, key=lambda path: path.stat().st_mtime)

    return None


def resolve_timed_disruption_json_path(input_root: Path | None, batch_dir: Path | None) -> Path | None:
    search_dirs: list[Path] = []

    if batch_dir is not None:
        if batch_dir.parent is not None and MAIN_INPUT_BATCH_NAME_RE.match(batch_dir.parent.name):
            search_dirs.append(batch_dir.parent / DISRUPTIONS_DIRNAME)
        search_dirs.append(batch_dir / DISRUPTIONS_DIRNAME)
        search_dirs.append(batch_dir)

    if input_root is not None:
        main_dirs = [
            path for path in input_root.iterdir()
            if path.is_dir() and MAIN_INPUT_BATCH_NAME_RE.match(path.name)
        ] if input_root.exists() else []
        if main_dirs:
            newest_main_dir = max(main_dirs, key=lambda path: _parse_main_input_batch_sort_key(path.name))
            search_dirs.append(newest_main_dir / DISRUPTIONS_DIRNAME)
        search_dirs.append(input_root / DISRUPTIONS_DIRNAME)
        search_dirs.append(input_root)

    unique_search_dirs: list[Path] = []
    seen_dirs: set[str] = set()
    for search_dir in search_dirs:
        dir_key = str(search_dir.resolve()) if search_dir.exists() else str(search_dir)
        if dir_key not in seen_dirs:
            unique_search_dirs.append(search_dir)
            seen_dirs.add(dir_key)

    for search_dir in unique_search_dirs:
        if not search_dir.exists() or not search_dir.is_dir():
            continue

        exact_candidate = search_dir / TIMED_DISRUPTION_V2_FILENAME
        if exact_candidate.exists() and exact_candidate.is_file():
            return exact_candidate

        fallback_candidates = sorted(
            [path for path in search_dir.glob("disruption*.json") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )
        if fallback_candidates:
            return max(fallback_candidates, key=lambda path: path.stat().st_mtime)

    return None


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


def _assign_missing_emergency_order_ids(
    timed_disruption_records: list[dict[str, Any]],
    existing_order_ids: list[str],
) -> list[dict[str, Any]]:
    numeric_order_ids: list[int] = []
    for order_id in existing_order_ids:
        order_id_text = str(order_id).strip()
        if order_id_text.isdigit():
            numeric_order_ids.append(int(order_id_text))

    next_numeric_order_id = (max(numeric_order_ids) + 1) if numeric_order_ids else 1
    generated_non_numeric_index = 1

    for record in timed_disruption_records:
        if str(record.get("disruption_type", "")).strip().lower() != "emergency_order":
            continue
        if record.get("order_id") is not None:
            continue

        if numeric_order_ids or next_numeric_order_id > 1:
            record["order_id"] = str(next_numeric_order_id)
            next_numeric_order_id += 1
        else:
            record["order_id"] = f"EMERGENCY_{generated_non_numeric_index}"
            generated_non_numeric_index += 1

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
                    "order_time_s": None if _is_nan_like(padded_row[6]) else _read_float(padded_row[6], _read_float(padded_row[2], 0.0)),
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
            emergency_orders.append(
                {
                    "order_id": str(order_id),
                    "order_time_s": float(order_time_s),
                    "priority": int(priority_value),
                    "variants": list(record.get("emergency_variants", [])),
                }
            )
            continue

        if station_id is not None:
            station_index = station_index_by_disruption_id.get(str(station_id))
            if station_index is None:
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
    delta_t = current_time_s - station_state.last_queue_change_time_s
    if delta_t < 0:
        raise ValueError("Queue statistics received a negative time step.")
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


def run_simulation(
    ordered_units: list[str],
    process_time_data: dict[str, Any],
    transport_time_data: dict[str, Any],
    unit_release_times: list[float] | None = None,
    unit_priorities: list[int] | None = None,
    unit_order_ids: list[str] | None = None,
    unit_ids: list[str] | None = None,
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
) -> tuple[list[OperationRecord], list[TransportRecord], list[UnitSummary], list[StationSummary], dict[str, float], dict[str, Any]]:
    effective_line_layout = build_effective_line_layout(
        process_time_data=process_time_data,
        transport_time_data=transport_time_data,
        line_layout_config=line_layout_config,
    )
    station_sequence = effective_line_layout["station_sequence"]
    station_instance_base_names = effective_line_layout["station_instance_base_names"]
    stage_instance_indices = effective_line_layout["stage_instance_indices"]
    station_to_stage_index = effective_line_layout["station_to_stage_index"]
    transport_lookup = effective_line_layout["transport_lookup"]
    process_times = process_time_data["process_times"]

    ordered_units = list(ordered_units)
    initial_requested_unit_count = len(ordered_units)

    # Stable root unit IDs (from schedule)
    root_unit_ids: list[str] = []
    if unit_ids is not None:
        root_unit_ids = [str(u).strip() for u in list(unit_ids)]
    if len(root_unit_ids) < initial_requested_unit_count:
        start_n = len(root_unit_ids)
        root_unit_ids.extend([f"U{i+1:03d}" for i in range(start_n, initial_requested_unit_count)])
    else:
        root_unit_ids = root_unit_ids[:initial_requested_unit_count]

    def _root_unit_id(root_index: int) -> str:
        return str(root_unit_ids[int(root_index)])

    def _attempt_unit_id(unit_index: int) -> str:
        return _root_unit_id(root_indices[int(unit_index)])

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
        for emergency_order in timed_disruption_data.get("emergency_orders", []):
            emergency_order_id = str(emergency_order["order_id"])
            emergency_order_time_s = float(emergency_order["order_time_s"])
            emergency_priority = max(1, int(emergency_order["priority"]))
            for variant_value, quantity_value in emergency_order.get("variants", []):
                for _ in range(int(quantity_value)):
                    ordered_units.append(str(variant_value))
                    unit_release_times.append(emergency_order_time_s)
                    unit_priorities.append(emergency_priority)
                    unit_order_ids.append(emergency_order_id)

    initial_requested_unit_count = len(ordered_units)

    # Stable root unit IDs (from schedule)
    root_unit_ids: list[str] = []
    if unit_ids is not None:
        root_unit_ids = [str(u).strip() for u in list(unit_ids)]
    if len(root_unit_ids) < initial_requested_unit_count:
        start_n = len(root_unit_ids)
        root_unit_ids.extend([f"U{i+1:03d}" for i in range(start_n, initial_requested_unit_count)])
    else:
        root_unit_ids = root_unit_ids[:initial_requested_unit_count]

    def _root_unit_id(root_index: int) -> str:
        return str(root_unit_ids[int(root_index)])

    def _attempt_unit_id(unit_index: int) -> str:
        return _root_unit_id(root_indices[int(unit_index)])

    if len(unit_release_times) != len(ordered_units):
        raise ValueError("unit_release_times must have the same length as ordered_units")
    if len(unit_priorities) != len(ordered_units):
        raise ValueError("unit_priorities must have the same length as ordered_units")
    if len(unit_order_ids) != len(ordered_units):
        raise ValueError("unit_order_ids must have the same length as ordered_units")
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
    available_system_slots = max_units_in_system
    waiting_for_system_slot: list[tuple[int, float, int, bool, int]] = []
    projected_station_available_time_s: list[float] = [0.0] * len(station_sequence)
    waiting_queue_sequence = 0
    station_queue_sequence = 0

    root_indices: list[int] = list(range(initial_requested_unit_count))
    attempt_numbers: list[int] = [1] * initial_requested_unit_count
    root_to_attempt_indices: defaultdict[int, list[int]] = defaultdict(list)
    root_order_ids: list[str] = list(unit_order_ids)
    for unit_index in range(initial_requested_unit_count):
        root_to_attempt_indices[unit_index].append(unit_index)

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
        heapq.heappush(
            event_queue,
            (
                float(time_s),
                EVENT_PRIORITY[event_type],
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
        best_candidate: tuple[tuple[float, float, float, int], tuple[int, float, float, float]] | None = None

        for candidate_station_index in stage_instance_indices[target_stage_index]:
            candidate_station_name = station_sequence[candidate_station_index]
            transport_time_s = 0.0
            if from_station_index is not None:
                from_station_name = station_sequence[from_station_index]
                transport_time_s = float(transport_lookup[(from_station_name, candidate_station_name)])

            arrival_time_s = current_time_s + transport_time_s
            process_time_s = float(process_times[variant][station_instance_base_names[candidate_station_index]])
            estimated_start_time_s = max(arrival_time_s, projected_station_available_time_s[candidate_station_index])
            estimated_finish_time_s = estimated_start_time_s + process_time_s

            candidate_score = (
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
        return chosen_station_index, chosen_transport_time_s, chosen_arrival_time_s

    def _enqueue_waiting_unit(unit_index: int, requested_release_time_s: float, prioritize_front: bool = False) -> None:
        nonlocal waiting_queue_sequence
        waiting_for_system_slot.append(
            (
                unit_index,
                float(requested_release_time_s),
                int(unit_priorities[unit_index]),
                bool(prioritize_front),
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
            score = (-int(priority_value), -int(bool(prioritize_front)), float(requested_release_time_s), int(queue_seq))
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
        station_state.queue.append(entry)

    def release_waiting_units_into_system(current_time_s: float) -> None:
        nonlocal available_system_slots
        while available_system_slots > 0 and waiting_for_system_slot and not stop_requested:
            waiting_item = _pop_best_waiting_unit(current_time_s)
            if waiting_item is None:
                break
            unit_index, _requested_release_time_s = waiting_item
            available_system_slots -= 1
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
        root_indices.append(root_index)
        attempt_numbers.append(int(attempt_numbers[failed_unit_index]) + 1)
        root_to_attempt_indices[root_index].append(new_unit_index)
        replacement_variants_created.append(variant)
        _enqueue_waiting_unit(new_unit_index, current_time_s, prioritize_front=prioritize_next_queue)
        return new_unit_index, []

    def try_start_next(station_index: int, current_time_s: float) -> None:
        nonlocal stop_requested, stop_time_s, stop_reason, material_stop_cutoff_unit_index
        station_state = station_states[station_index]
        if stop_requested or station_state.busy or not station_state.queue:
            return

        _update_queue_area(station_state, current_time_s)
        unit_index, arrival_time_s, queue_length_ahead_on_arrival, _priority_value, _queue_seq = station_state.queue.pop(0)

        root_index = root_indices[unit_index]
        unit_id = _root_unit_id(root_index)
        variant = ordered_units[unit_index]
        station_name = station_sequence[station_index]
        base_station_name = station_instance_base_names[station_index]
        stage_number, _, _ = _extract_station_name_parts(base_station_name)
        if stage_number is None:
            stage_number = int(station_index + 1)

        base_process_time_s = float(process_times[variant][base_station_name])
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
                    "root_unit_id": _attempt_unit_id(unit_index),
                    "order_id": str(root_order_ids[root_indices[unit_index]]),
                    "variant": ordered_units[unit_index],
                    "failed_attempt_unit_id": _attempt_unit_id(unit_index),
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
                    "root_unit_id": _root_unit_id(root_index),
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

    for station_index, trigger_times in timed_failed_inspection_times_by_station.items():
        for trigger_time_s in trigger_times:
            push_event(float(trigger_time_s), EVENT_TIMED_FAILED_INSPECTION, int(station_index), -1)

    for unit_index in range(initial_requested_unit_count):
        push_event(float(unit_release_times[unit_index]), EVENT_RELEASE, -1, unit_index)

    while event_queue:
        time_s, _, _, event_type, station_index, unit_index = heapq.heappop(event_queue)

        if simulation_time_s is not None and time_s > float(simulation_time_s):
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
                            "affected_unit_id": f"U{root_indices[current_processing_unit_index] + 1:03d}",
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
            if available_system_slots > 0:
                available_system_slots -= 1
                first_station_index, _, arrival_time_s = choose_station_instance_for_stage(
                    target_stage_index=0,
                    current_time_s=time_s,
                    unit_index=unit_index,
                    from_station_index=None,
                )
                push_event(arrival_time_s, EVENT_ARRIVAL, first_station_index, unit_index)
            else:
                prioritize_front = bool(attempt_numbers[unit_index] > 1)
                _enqueue_waiting_unit(unit_index, time_s, prioritize_front=prioritize_front)
            continue

        if event_type == EVENT_CART_RETURN:
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
                            "root_unit_id": _attempt_unit_id(unit_index),
                            "order_id": str(root_order_ids[root_indices[unit_index]]),
                            "variant": ordered_units[unit_index],
                            "failed_attempt_unit_id": _attempt_unit_id(unit_index),
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
                        "root_unit_id": _attempt_unit_id(unit_index),
                        "order_id": str(root_order_ids[root_indices[unit_index]]),
                        "variant": ordered_units[unit_index],
                        "failed_attempt_unit_id": _attempt_unit_id(unit_index),
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
                        "unit_id": _attempt_unit_id(unit_index),
                        "order_id": str(root_order_ids[root_indices[unit_index]]),
                        "root_unit_id": _attempt_unit_id(unit_index),
                        "attempt": int(attempt_numbers[unit_index]),
                        "variant": ordered_units[unit_index],
                        "station_name": station_sequence[station_index],
                        "failure_type": terminal_failure_type,
                        "failure_material": disruption_result.get("terminal_failure_material"),
                        "replacement_unit_id": _attempt_unit_id(unit_index) if replacement_unit_index is not None else None,
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
                        unit_id=_attempt_unit_id(unit_index),
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
        unit_id_text = _root_unit_id(root_index)
        time_spent_producing = sum(
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
    }


    # -----------------------------
    # WIP snapshot at segment end (on-line only: station/queue/transport)
    # -----------------------------
    wip_snapshot: list[dict[str, Any]] = []
    snapshot_time_s = float(stop_time_s) if stop_time_s is not None else float(makespan_s)

    def _active_transport(unit_id_text: str):
        for tr in transport_records:
            if str(tr.unit_id) != str(unit_id_text):
                continue
            if float(tr.start_time_s) <= snapshot_time_s < float(tr.finish_time_s):
                return tr
        return None

    seen_root: set[int] = set()

    for st_idx, state in enumerate(station_states):
        if not state.busy or state.current_unit_index is None:
            continue
        uidx = int(state.current_unit_index)
        rix = int(root_indices[uidx])
        if rix in seen_root:
            continue
        seen_root.add(rix)
        uid = _root_unit_id(rix)
        op = operation_lookup.get((st_idx, uidx))
        proc_left = ''
        if op is not None:
            proc_left = str(round(max(0.0, float(op.finish_time_s) - snapshot_time_s), 6))
        wip_snapshot.append({
            'unit_id': uid,
            'orderID': str(root_order_ids[rix]),
            'variant': str(ordered_units[uidx]),
            'current_station_instance': str(station_sequence[st_idx]),
            'next_station_instance': '',
            'transport_left_s': '',
            'process_time_left_s': proc_left,
        })

    for st_idx, state in enumerate(station_states):
        for entry in list(state.queue):
            uidx = int(entry[0])
            rix = int(root_indices[uidx])
            if rix in seen_root:
                continue
            seen_root.add(rix)
            uid = _root_unit_id(rix)
            wip_snapshot.append({
                'unit_id': uid,
                'orderID': str(root_order_ids[rix]),
                'variant': str(ordered_units[uidx]),
                'current_station_instance': str(station_sequence[st_idx]),
                'next_station_instance': '',
                'transport_left_s': '',
                'process_time_left_s': '',
            })

    for rix in range(initial_requested_unit_count):
        if rix in seen_root:
            continue
        uid = _root_unit_id(rix)
        tr = _active_transport(uid)
        if tr is None:
            continue
        seen_root.add(rix)
        t_left = str(round(max(0.0, float(tr.finish_time_s) - snapshot_time_s), 6))
        wip_snapshot.append({
            'unit_id': uid,
            'orderID': str(root_order_ids[rix]),
            'variant': str(tr.variant),
            'current_station_instance': str(tr.from_station),
            'next_station_instance': str(tr.to_station),
            'transport_left_s': t_left,
            'process_time_left_s': '',
        })

    simulation_details['wip_snapshot'] = wip_snapshot


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




def write_unit_summary_snapshot_csv(completed_unit_summaries: list[UnitSummary], wip_snapshot: list[dict[str, Any]], output_path: Path) -> None:
    """Overwrite unit_summary.csv with cumulative completed units + current WIP (on line only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_completed = {}
    if output_path.exists():
        try:
            with output_path.open('r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    uid = str(r.get('unit_id','')).strip()
                    cts = str(r.get('completion_time_s','')).strip()
                    if uid and cts != '':
                        existing_completed[uid] = dict(r)
        except Exception:
            existing_completed = {}

    for s in completed_unit_summaries:
        existing_completed[str(s.unit_id)] = {
            'unit_id': s.unit_id,
            'orderID': s.order_id,
            'variant': s.variant,
            'first_arrival_time_s': round(float(s.first_arrival_time_s), 4),
            'start_time_s': round(float(s.start_time_s), 4),
            'completion_time_s': round(float(s.completion_time_s), 4),
            'flow_time_s': round(float(s.flow_time_s), 4),
            'active_flow_time_s': round(float(s.active_flow_time_s), 4),
            'time_spent_producing': round(float(s.time_spent_producing), 4),
            'throughput_efficiency': round(float(s.throughput_efficiency), 6),
            'Attempts': int(s.attempts),
            'current_station_instance': '',
            'next_station_instance': '',
            'transport_left_s': '',
            'process_time_left_s': '',
        }

    fieldnames = [
        'unit_id','orderID','variant','first_arrival_time_s','start_time_s','completion_time_s',
        'flow_time_s','active_flow_time_s','time_spent_producing','throughput_efficiency','Attempts',
        'current_station_instance','next_station_instance','transport_left_s','process_time_left_s'
    ]

    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_completed.values():
            writer.writerow({k: row.get(k,'') for k in fieldnames})
        for w in wip_snapshot or []:
            row = {
                'unit_id': w.get('unit_id',''),
                'orderID': w.get('orderID',''),
                'variant': w.get('variant',''),
                'first_arrival_time_s': '',
                'start_time_s': '',
                'completion_time_s': '',
                'flow_time_s': '',
                'active_flow_time_s': '',
                'time_spent_producing': '',
                'throughput_efficiency': '',
                'Attempts': '',
                'current_station_instance': w.get('current_station_instance',''),
                'next_station_instance': w.get('next_station_instance',''),
                'transport_left_s': w.get('transport_left_s',''),
                'process_time_left_s': w.get('process_time_left_s',''),
            }
            writer.writerow({k: row.get(k,'') for k in fieldnames})


def write_unit_summary_csv(unit_summaries: list[UnitSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    station_entries: list[tuple[int, int, str]] = []
    seen_station_names: set[str] = set()
    for op in sorted(operations, key=lambda x: x.station_index):
        if op.station_name in seen_station_names:
            continue
        stage_number, copy_number, _ = _extract_station_name_parts(op.station_name)
        station_entries.append((stage_number or op.station_index, copy_number or 0, op.station_name))
        seen_station_names.add(op.station_name)

    transport_entries: dict[int, str] = {}
    for tr in sorted(transport_records, key=lambda x: x.transport_index):
        transport_stage_number = _transport_stage_number_from_name(tr.transport_name)
        if transport_stage_number is None:
            transport_stage_number = tr.transport_index
        transport_entries.setdefault(transport_stage_number, tr.transport_name)

    stations_by_stage: defaultdict[int, list[tuple[int, str]]] = defaultdict(list)
    for stage_number, copy_number, station_name in station_entries:
        stations_by_stage[stage_number].append((copy_number, station_name))

    row_names: list[str] = []
    all_stage_numbers = sorted(stations_by_stage.keys())
    for stage_number in all_stage_numbers:
        for _, station_name in sorted(stations_by_stage[stage_number], key=lambda item: item[0]):
            row_names.append(station_name)
        if stage_number in transport_entries:
            row_names.append(transport_entries[stage_number])

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


def create_gantt_chart_no_transport(
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

    row_names: list[str] = []
    for idx, station_name in enumerate(station_names):
        row_names.append(station_name)

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


def resolve_results_output_dir(output_root: Path, order_text: str) -> Path:
    ongoing_run_dir = resolve_ongoing_run_dir(Path(__file__).resolve().parent)
    ongoing_main_name = ongoing_run_dir.parent.name
    ongoing_run_name = ongoing_run_dir.name

    if output_root.name == "results":
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    if output_root.name == ongoing_run_name and output_root.parent.name == ongoing_main_name:
        results_dir = output_root / "results"
    elif output_root.name == ongoing_main_name:
        results_dir = output_root / ongoing_run_name / "results"
    else:
        results_dir = output_root / ongoing_main_name / ongoing_run_name / "results"

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _find_matching_timed_record_for_event(
    event: dict[str, Any],
    timed_disruption_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    station_name = str(event.get("station_name", "")).strip()
    target_time = float(event.get("disruption_timestamp_s", 0.0))
    reserved_types = {"efficiency_loss", "efficiency loss", "failed_inspection", "inspection_failure", "inspection failure", "emergency_order"}

    matches: list[dict[str, Any]] = []
    for record in timed_disruption_records:
        record_type = str(record.get("disruption_type", "")).strip().casefold()
        if record_type == "emergency_order":
            continue

        if station_name != "":
            record_station_id = record.get("station_id")
            if record_station_id is None or str(record_station_id) != _station_disruption_id_from_station_name(station_name):
                continue

        start_time_s = float(record.get("start_time_s", 0.0))
        end_time_raw = record.get("end_time_s")
        end_time_s = float(end_time_raw) if end_time_raw is not None else start_time_s
        if start_time_s - 1e-9 <= target_time <= end_time_s + 1e-9:
            matches.append(record)

    if not matches:
        return None

    triggered_type = str(event.get("triggered_disruption_type", "")).strip().casefold()
    if triggered_type == "breakdown":
        subtype_matches = [
            record for record in matches
            if str(record.get("disruption_type", "")).strip().casefold() not in reserved_types
        ]
        if subtype_matches:
            return max(subtype_matches, key=lambda record: float(record.get("start_time_s", 0.0)))

    if triggered_type == "efficiency_loss":
        eff_matches = [
            record for record in matches
            if str(record.get("disruption_type", "")).strip().casefold() in {"efficiency_loss", "efficiency loss"}
        ]
        if eff_matches:
            return max(eff_matches, key=lambda record: float(record.get("start_time_s", 0.0)))

    if triggered_type == "failed_inspection":
        insp_matches = [
            record for record in matches
            if str(record.get("disruption_type", "")).strip().casefold() in {"failed_inspection", "inspection_failure", "inspection failure"}
        ]
        if insp_matches:
            return max(insp_matches, key=lambda record: float(record.get("start_time_s", 0.0)))

    return max(matches, key=lambda record: float(record.get("start_time_s", 0.0)))


def _estimate_breakdown_duration_from_config(
    disruption_config: dict[str, Any] | None,
    stage_number: int | None,
    disruption_name: str,
) -> float:
    if not isinstance(disruption_config, dict) or stage_number is None:
        return 0.0

    stations_cfg = disruption_config.get("Stations", {})
    station_cfg = stations_cfg.get(str(int(stage_number)), {}) if isinstance(stations_cfg, dict) else {}
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
            mean_s = float(entry.get("mean [s]", 0.0))
            std_pct = float(entry.get("std [% of mean]", 0.0))
            return mean_s + 3.0 * mean_s * (std_pct / 100.0)

    breakdown_cfg = station_cfg.get("breakdown", {})
    if isinstance(breakdown_cfg, dict):
        if isinstance(breakdown_cfg.get("range"), (list, tuple)) and len(breakdown_cfg.get("range", [])) >= 2:
            low = float(breakdown_cfg["range"][0])
            high = float(breakdown_cfg["range"][1])
            mean_s = (low + high) / 2.0
        else:
            mean_s = float(breakdown_cfg.get("duration [s]", 0.0))
        std_pct = float(breakdown_cfg.get("std [% of mean]", 0.0))
        return mean_s + 3.0 * mean_s * (std_pct / 100.0)

    return 0.0


def _stage_number_from_disruption_station_id(station_id: Any) -> int | None:
    if station_id is None:
        return None
    station_id_text = str(station_id).strip()
    if station_id_text == "" or station_id_text.casefold() == "nan":
        return None
    try:
        return int(float(station_id_text.split(".", maxsplit=1)[0]))
    except (TypeError, ValueError):
        return None


def _is_timed_breakdown_record(
    record: dict[str, Any],
    disruption_config: dict[str, Any] | None,
) -> bool:
    disruption_type = str(record.get("disruption_type", "")).strip().casefold()
    if disruption_type == "machine_breakdown":
        return True
    if disruption_type in {
        "efficiency_loss",
        "efficiency loss",
        "failed_inspection",
        "inspection_failure",
        "inspection failure",
        "emergency_order",
    }:
        return False

    stage_number = _stage_number_from_disruption_station_id(record.get("station_id"))
    if not isinstance(disruption_config, dict) or stage_number is None:
        return False

    stations_cfg = disruption_config.get("Stations", {})
    station_cfg = stations_cfg.get(str(int(stage_number)), {}) if isinstance(stations_cfg, dict) else {}
    machine_breakdowns = station_cfg.get("machine breakdowns", []) if isinstance(station_cfg, dict) else []
    if not isinstance(machine_breakdowns, list):
        return False

    return any(
        isinstance(entry, dict)
        and str(entry.get("name", "")).strip().casefold() == disruption_type
        for entry in machine_breakdowns
    )


def build_disruption_history_rows(
    simulation_details: dict[str, Any],
    disruption_config: dict[str, Any] | None,
    timed_disruption_records: list[dict[str, Any]],
    current_time_s: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if timed_disruption_records:
        current_time_s_for_history = float(current_time_s) if current_time_s is not None else None
        sorted_records = sorted(
            timed_disruption_records,
            key=lambda record: (
                float(record.get("start_time_s", 0.0) or 0.0),
                int(record.get("row_index", 0) or 0),
            ),
        )

        for record in sorted_records:
            disruption_type = str(record.get("disruption_type", "")).strip()
            disruption_type_cf = disruption_type.casefold()
            start_time = float(record.get("start_time_s", 0.0) or 0.0)
            if current_time_s_for_history is not None and start_time > current_time_s_for_history:
                continue
            end_time_raw = record.get("end_time_s")
            end_time_value = float(end_time_raw) if end_time_raw is not None else None
            if current_time_s_for_history is not None and (end_time_value is None or end_time_value > current_time_s_for_history):
                end_time = None
            else:
                end_time = end_time_value

            station_id = ""
            if disruption_type_cf not in {"failed_inspection", "inspection_failure", "inspection failure", "emergency_order"}:
                raw_station_id = record.get("station_id")
                station_id = "" if raw_station_id is None else str(raw_station_id)

            efficiency_loss = 0.0
            if disruption_type_cf in {"efficiency_loss", "efficiency loss"}:
                efficiency_percentage = record.get("efficiency_percentage")
                if efficiency_percentage is not None:
                    efficiency_loss = max(0.0, 100.0 - float(efficiency_percentage))

            estimated_duration = 0.0
            if _is_timed_breakdown_record(record, disruption_config):
                stage_number = _stage_number_from_disruption_station_id(record.get("station_id"))
                estimated_duration = _estimate_breakdown_duration_from_config(
                    disruption_config=disruption_config,
                    stage_number=stage_number,
                    disruption_name=disruption_type,
                )

            rows.append(
                {
                    "disruption_type": disruption_type,
                    "estimated_duration": round(float(estimated_duration), 6),
                    "station_id": station_id,
                    "efficiency_loss": round(float(efficiency_loss), 6),
                    "start_time": round(float(start_time), 6),
                    "end_time": ("" if end_time is None else round(float(end_time), 6)),
                }
            )

        return rows

    for event in list(simulation_details.get("disruption_event_log", [])):
        event_name = str(event.get("event", "")).strip()
        if event_name not in {"operation_disruption", "timed_failed_inspection_trigger"}:
            continue

        start_time = float(event.get("disruption_timestamp_s", 0.0))
        station_name = str(event.get("station_name", "")).strip()
        stage_number_raw = event.get("stage_number")
        stage_number = int(stage_number_raw) if stage_number_raw is not None else None
        station_id = _station_disruption_id_from_station_name(station_name) if station_name else ""

        disruption_type = str(event.get("triggered_disruption_name") or event.get("triggered_disruption_type") or "").strip()
        if not disruption_type:
            disruption_type = "failed_inspection" if event_name == "timed_failed_inspection_trigger" else "disruption"

        estimated_duration = 0.0
        end_time = start_time
        efficiency_loss = 0.0

        if event_name == "timed_failed_inspection_trigger" or bool(event.get("inspection_failed_triggered", False)):
            station_id = ""
            end_time = start_time
        elif bool(event.get("breakdown_triggered", False)):
            end_time = start_time + float(event.get("breakdown_added_time_s", 0.0) or 0.0)
            estimated_duration = _estimate_breakdown_duration_from_config(
                disruption_config=disruption_config,
                stage_number=stage_number,
                disruption_name=disruption_type,
            )
        elif bool(event.get("efficiency_loss_triggered", False)):
            efficiency_loss = float(event.get("efficiency_drop_percent", 0.0) or 0.0)
            end_time = start_time + float(event.get("effective_process_time_s", 0.0) or 0.0)
        elif bool(event.get("material_broken_triggered", False)):
            end_time = start_time + float(event.get("material_broken_added_time_s", 0.0) or 0.0)
        elif bool(event.get("material_ran_out_triggered", False)):
            end_time = start_time

        if disruption_type.casefold() in {"failed_inspection", "inspection_failure", "inspection failure", "emergency_order"}:
            station_id = ""

        rows.append(
            {
                "disruption_type": disruption_type,
                "estimated_duration": round(float(estimated_duration), 6),
                "station_id": station_id,
                "efficiency_loss": round(float(efficiency_loss), 6),
                "start_time": round(float(start_time), 6),
                "end_time": ("" if end_time is None else round(float(end_time), 6)),
            }
        )

    return rows

def write_disruption_history_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["disruption_type", "estimated_duration", "station_id", "efficiency_loss", "start_time", "end_time"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


# -----------------------------
# Main
# -----------------------------

# -----------------------------
# GA evaluation entry point
# -----------------------------

def load_disruption_history_rows(history_csv_path: Path | None) -> list[dict[str, Any]]:
    """Load disruption_his.csv rows (best-effort).

    The project has used slightly different column names over time, so this function
    accepts multiple options:
      - disruption_type / type
      - estimated_duration / duration / duration_s
      - station_id / station
      - efficiency_loss / efficiency_drop
      - start_time / start_time_s / start
      - end_time / end_time_s / end

    Returns a list of normalized dicts with keys:
      disruption_type, station_id, estimated_duration_s, efficiency_loss, start_time_s, end_time_s
    """
    if history_csv_path is None:
        return []
    history_csv_path = Path(history_csv_path)
    if not history_csv_path.exists() or not history_csv_path.is_file():
        return []

    def _norm_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())

    def _to_float(value: Any, default: float | None = None) -> float | None:
        text_value = str(value).strip() if value is not None else ""
        if text_value == "" or text_value.casefold() == "nan":
            return default
        try:
            numeric = float(text_value)
        except (TypeError, ValueError):
            return default
        if math.isnan(numeric):
            return default
        return numeric

    rows: list[dict[str, Any]] = []
    with history_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        # map normalized header -> original header
        if reader.fieldnames is None:
            return []
        header_map = {_norm_key(h): h for h in reader.fieldnames}

        def _get(d: dict[str, Any], *candidates: str) -> Any:
            for c in candidates:
                key = header_map.get(_norm_key(c))
                if key is not None and key in d:
                    return d.get(key)
            return None

        for raw in reader:
            if not raw:
                continue
            disruption_type = str(_get(raw, 'disruption_type', 'type') or '').strip()
            if disruption_type == '':
                continue
            station_id = _get(raw, 'station_id', 'station')
            station_id = None if station_id is None or str(station_id).strip() == '' else str(station_id).strip()

            estimated_duration_s = _to_float(_get(raw, 'estimated_duration', 'duration', 'duration_s'), 0.0) or 0.0
            efficiency_loss = _to_float(_get(raw, 'efficiency_loss', 'efficiency_drop'), 0.0) or 0.0
            start_time_s = _to_float(_get(raw, 'start_time', 'start_time_s', 'start'), 0.0) or 0.0
            end_time_s = _to_float(_get(raw, 'end_time', 'end_time_s', 'end'), None)

            rows.append({
                'disruption_type': disruption_type,
                'station_id': station_id,
                'estimated_duration_s': float(estimated_duration_s),
                'efficiency_loss': float(efficiency_loss),
                'start_time_s': float(start_time_s),
                'end_time_s': None if end_time_s is None else float(end_time_s),
            })

    return rows


def build_timed_disruption_data_from_history(
    history_rows: list[dict[str, Any]],
    station_sequence: list[str],
    current_time_s: float,
) -> dict[str, Any]:
    """Convert disruption history rows into timed disruption data.

    IMPORTANT for GA evaluation:
    - We include ONLY currently-active disruptions to avoid leaking future events.

    A disruption is treated as active when:
      start_time_s <= current_time_s < end_time_s

    If end_time_s is missing and estimated_duration_s > 0, we compute:
      end_time_s = start_time_s + estimated_duration_s
    """
    now = float(current_time_s)
    timed_records: list[dict[str, Any]] = []

    for idx, row in enumerate(history_rows or [], start=1):
        start_time_s = float(row.get('start_time_s', 0.0) or 0.0)
        if start_time_s > now:
            continue

        end_time_s = row.get('end_time_s')
        end_time_s = None if end_time_s is None else float(end_time_s)

        if end_time_s is None:
            est = float(row.get('estimated_duration_s', 0.0) or 0.0)
            end_time_s = (start_time_s + est) if est > 0 else start_time_s

        # not active anymore?
        if not (start_time_s <= now < float(end_time_s) if float(end_time_s) > start_time_s else start_time_s <= now <= float(end_time_s)):
            continue

        disruption_type = str(row.get('disruption_type', '')).strip()
        station_id = row.get('station_id')
        efficiency_loss = float(row.get('efficiency_loss', 0.0) or 0.0)
        efficiency_percentage = None
        if efficiency_loss > 0.0:
            # efficiency_loss interpreted as percentage points lost
            efficiency_percentage = max(0.0, 100.0 - efficiency_loss)

        timed_records.append({
            'disruption_type': disruption_type,
            'station_id': station_id,
            'start_time_s': float(start_time_s),
            'end_time_s': float(end_time_s),
            'efficiency_percentage': efficiency_percentage,
            'row_index': int(idx),
        })

    return prepare_timed_disruption_data(
        station_sequence=station_sequence,
        timed_disruption_records=timed_records,
        timed_disruption_config=None,
    )


def simulate_for_ga(main_settings_path: str | Path, current_time_s: float = 0.0) -> dict[str, Any]:
    """Lightweight simulator entry point used by GA_Scheduling.

    GA_Scheduling expects this function to exist and return a dict with:
      - order_completion_times: dict[str, float]
      - makespan: float

    This function runs the simulator on the *current_schedule.csv* referenced by main_settings.json,
    and applies ONLY currently-active disruption entries from disruption_his.csv (if present).
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'

    process_time_data = load_json(data_dir / 'process_times.json')
    transport_time_data = load_json(data_dir / 'transport_times.json')

    # Optional (only used if your run_simulation consumes them in your setup)
    bom_data = load_json(data_dir / 'bom.json') if (data_dir / 'bom.json').exists() else None
    material_stock_data = load_json(data_dir / 'material_stock.json') if (data_dir / 'material_stock.json').exists() else None

    valid_variants = set(process_time_data.get('process_times', {}).keys())

    payload = load_input_from_main_settings(Path(main_settings_path), data_dir=data_dir, valid_variants=valid_variants)

    # Line layout
    line_layout_path = resolve_line_layout_path(
        payload.get('selected_line_layout_name'),
        payload.get('input_root'),
        payload.get('batch_dir'),
        data_dir,
    )
    line_layout_config = load_json(line_layout_path) if line_layout_path is not None else None

    effective_layout = build_effective_line_layout(
        process_time_data=process_time_data,
        transport_time_data=transport_time_data,
        line_layout_config=line_layout_config,
    )

    # Timed disruptions from history (active only)
    pathlist = payload.get('pathlist', {}) or {}
    history_path = _pathlist_path(pathlist, 'on_going_run_dis_his')
    history_rows = load_disruption_history_rows(history_path)
    timed_disruption_data = build_timed_disruption_data_from_history(
        history_rows=history_rows,
        station_sequence=effective_layout.get('station_sequence', []),
        current_time_s=float(current_time_s),
    )

    operations, transport_records, unit_summaries, station_summaries, _, simulation_details = run_simulation(
        ordered_units=payload.get('ordered_units', []),
        process_time_data=process_time_data,
        transport_time_data=transport_time_data,
        unit_release_times=payload.get('unit_release_times'),
        unit_priorities=payload.get('unit_priorities'),
        unit_order_ids=payload.get('unit_order_ids'),
        unit_ids=payload.get('unit_ids'),
        max_units_in_system=int(payload.get('carriers', MAX_UNITS_IN_SYSTEM) or MAX_UNITS_IN_SYSTEM),
        line_layout_config=line_layout_config,
        bom_data=bom_data,
        material_stock_data=material_stock_data,
        disruptions_enabled=False,
        disruption_config=None,
        disruption_seed=None,
        simulation_time_s=None,
        timed_disruption_data=timed_disruption_data,
    )

    order_completion_times: dict[str, float] = {}
    for summary in unit_summaries:
        order_id = str(summary.order_id)
        order_completion_times[order_id] = max(order_completion_times.get(order_id, 0.0), float(summary.completion_time_s))

    makespan = max((float(summary.completion_time_s) for summary in unit_summaries), default=0.0)

    return {
        'order_completion_times': order_completion_times,
        'makespan': makespan,
        'simulation_details': simulation_details,
    }

def main(main_settings_json: str | Path | None = None,timestamp: int = None) -> None:
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
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent / "input",
        help="Folder containing generated input batch folders such as orders_DD-MM_HH-MM_N",
    )
    parser.add_argument(
        "--line-layout-file",
        type=str,
        help="Optional line layout file name or path. If omitted, settings.json is checked first and then line_layout.json defaults are used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Root folder where a new subfolder will be created for every order run",
    )
    parser.add_argument(
        "--main-settings",
        type=Path,
        help="Path to a run-specific main_settings.json containing the pathlist for this simulation run.",
    )
    args = parser.parse_args([] if main_settings_json is not None else None)
    if main_settings_json is not None:
        args.main_settings = Path(main_settings_json)

    data_dir: Path = args.data_dir
    input_root: Path = args.input_root
    output_root: Path = args.output_root

    starttime = time.perf_counter()

    process_time_data = load_json(data_dir / "process_times.json")
    transport_time_data = load_json(data_dir / "transport_times.json")
    material_stock_data = load_json(data_dir / "material_stock.json")
    bom_data = load_json(data_dir / "bom.json")

    valid_variants = set(process_time_data["process_times"].keys())

    order_text: str
    ordered_units: list[str]
    unit_release_times: list[float]
    unit_priorities: list[int]
    unit_order_ids: list[str]
    simulation_time_s: float | None = None
    carriers = MAX_UNITS_IN_SYSTEM
    run_metadata_extra: dict[str, Any] = {}
    selected_line_layout_name: str | None = args.line_layout_file
    batch_dir_for_layout: Path | None = None
    settings_path: Path | None = None
    settings_data: dict[str, Any] = {}
    main_settings_pathlist: dict[str, Any] = {}
    write_outputs_for_assigned_route = True

    if args.main_settings is not None:
        generated_input = load_input_from_main_settings(Path(args.main_settings), data_dir, valid_variants)
        order_text = generated_input["order_text"]
        ordered_units = generated_input["ordered_units"]
        unit_release_times = generated_input["unit_release_times"]
        unit_priorities = generated_input.get("unit_priorities", [1] * len(ordered_units))
        unit_order_ids = generated_input.get("unit_order_ids", ["1"] * len(ordered_units))
        unit_ids = generated_input.get("unit_ids", [])
        write_outputs_for_assigned_route = bool(generated_input.get("has_assigned_route", False))
        simulation_time_s = generated_input["simulation_time_s"]
        carriers = max(1, int(generated_input.get("carriers", MAX_UNITS_IN_SYSTEM)))
        settings_data = generated_input.get("settings_data", {})
        settings_path = generated_input.get("settings_path")
        batch_dir_for_layout = generated_input["batch_dir"]
        input_root = Path(generated_input["input_root"])
        main_settings_pathlist = dict(generated_input.get("pathlist", {}))
        if generated_input.get("output_root") is not None:
            output_root = Path(generated_input["output_root"])
        if selected_line_layout_name is None:
            selected_line_layout_name = generated_input.get("selected_line_layout_name")
        run_metadata_extra = {
            "main_settings_json": str(Path(args.main_settings).resolve()),
            "input_root": str(Path(generated_input["input_root"]).resolve()),
            "input_batch_directory": str(Path(generated_input["batch_dir"]).resolve()),
            "input_orders_csv": str(Path(generated_input["orders_csv_path"]).resolve()),
            "input_settings_json": str(Path(generated_input["settings_path"]).resolve()),
            "simulation_time_seconds": simulation_time_s,
            "carriers": carriers,
            "return_to_station_1_time_seconds": RETURN_TO_STATION_1_TIME_S,
            "unit_priorities": list(unit_priorities),
            "unit_order_ids": list(unit_order_ids),
        }
    elif args.order:
        order_text = args.order
        ordered_units = parse_order(order_text, valid_variants)
        unit_release_times = [0.0] * len(ordered_units)
        unit_priorities = [1] * len(ordered_units)
        unit_order_ids = ["manual"] * len(ordered_units)
    else:
        generated_input = load_latest_generated_input(input_root, valid_variants)
        order_text = generated_input["order_text"]
        ordered_units = generated_input["ordered_units"]
        unit_release_times = generated_input["unit_release_times"]
        unit_priorities = generated_input.get("unit_priorities", [1] * len(ordered_units))
        unit_order_ids = generated_input.get("unit_order_ids", ["1"] * len(ordered_units))
        unit_ids = generated_input.get("unit_ids", [])
        write_outputs_for_assigned_route = bool(generated_input.get("has_assigned_route", False))
        simulation_time_s = generated_input["simulation_time_s"]
        carriers = max(1, int(generated_input.get("carriers", MAX_UNITS_IN_SYSTEM)))
        settings_data = generated_input.get("settings_data", {})
        settings_path = generated_input.get("settings_path")
        batch_dir_for_layout = generated_input["batch_dir"]
        if selected_line_layout_name is None:
            selected_line_layout_name = generated_input.get("selected_line_layout_name")
        run_metadata_extra = {
            "input_root": str(generated_input["input_root"].resolve()),
            "input_batch_directory": str(generated_input["batch_dir"].resolve()),
            "input_orders_csv": str(generated_input["orders_csv_path"].resolve()),
            "input_settings_json": str(generated_input["settings_path"].resolve()),
            "simulation_time_seconds": simulation_time_s,
            "carriers": carriers,
            "return_to_station_1_time_seconds": RETURN_TO_STATION_1_TIME_S,
            "unit_priorities": list(unit_priorities),
            "unit_order_ids": list(unit_order_ids),
        }

    line_layout_path_resolved = resolve_line_layout_path(
        selected_layout_name=selected_line_layout_name,
        input_root=input_root,
        batch_dir=batch_dir_for_layout,
        data_dir=data_dir,
    )
    line_layout_config, line_layout_path = load_line_layout_config(line_layout_path_resolved, process_time_data)
    effective_line_layout = build_effective_line_layout(
        process_time_data=process_time_data,
        transport_time_data=transport_time_data,
        line_layout_config=line_layout_config,
    )

    if main_settings_pathlist:
        disruption_path = _resolve_pathlist_disruption_json(main_settings_pathlist)
        timed_disruption_csv_path = _resolve_pathlist_disruption_csv(main_settings_pathlist)
        timed_disruption_json_path = _resolve_pathlist_disruption_json(main_settings_pathlist)
    else:
        disruption_path = resolve_disruption_path(input_root=input_root, batch_dir=batch_dir_for_layout)
        timed_disruption_csv_path = resolve_timed_disruption_csv_path(input_root=input_root, batch_dir=batch_dir_for_layout)
        timed_disruption_json_path = resolve_timed_disruption_json_path(input_root=input_root, batch_dir=batch_dir_for_layout)
    disruption_mode = _settings_disruption_mode(settings_data)
    disruptions_enabled = disruption_mode != 0
    chance_based_disruptions_enabled = disruption_mode == 1
    timed_disruptions_enabled = disruption_mode == 2
    disruption_seed = settings_data.get("seed")
    disruption_config: dict[str, Any] | None = None
    timed_disruption_records: list[dict[str, Any]] = []
    timed_disruption_data: dict[str, Any] | None = None

    if chance_based_disruptions_enabled:
        if disruption_path is None or not disruption_path.exists():
            raise FileNotFoundError(
                "random based disruptions are enabled in settings.json, but disruption.json was not found in the input batch."
            )
        disruption_config = load_json(disruption_path)
    elif timed_disruptions_enabled:
        if timed_disruption_csv_path is None or not timed_disruption_csv_path.exists():
            raise FileNotFoundError(
                "time based disruptions are enabled in settings.json, but the timed disruption CSV was not found in the input batch."
            )
        if timed_disruption_json_path is None or not timed_disruption_json_path.exists():
            raise FileNotFoundError(
                "time based disruptions are enabled in settings.json, but disruption_v2.json was not found in the input batch."
            )
        disruption_config = load_json(timed_disruption_json_path)
        timed_disruption_records = load_timed_disruption_csv(
            timed_disruption_csv_path,
            valid_variants=valid_variants,
        )
        timed_disruption_records = _assign_missing_emergency_order_ids(
            timed_disruption_records=timed_disruption_records,
            existing_order_ids=list(unit_order_ids),
        )
        timed_disruption_data = prepare_timed_disruption_data(
            station_sequence=list(effective_line_layout["station_sequence"]),
            timed_disruption_records=timed_disruption_records,
            timed_disruption_config=disruption_config,
        )

    run_metadata_extra["line_layout_file"] = (
        str(line_layout_path.resolve()) if line_layout_path is not None else "default_generated_layout"
    )
    if selected_line_layout_name:
        run_metadata_extra["requested_line_layout_file"] = str(selected_line_layout_name)
    run_metadata_extra["line_layout_name"] = str(effective_line_layout.get("layout_name", "default_single_path"))
    run_metadata_extra["effective_station_sequence"] = list(effective_line_layout["station_sequence"])
    run_metadata_extra["disruptions_enabled"] = bool(disruptions_enabled)
    run_metadata_extra["disruption_mode"] = int(disruption_mode)
    run_metadata_extra["disruption_seed"] = str(disruption_seed) if disruption_seed is not None else None
    if disruption_path is not None:
        run_metadata_extra["input_disruption_json"] = str(disruption_path.resolve())
    if timed_disruption_json_path is not None:
        run_metadata_extra["input_timed_disruption_json"] = str(timed_disruption_json_path.resolve())
    if timed_disruption_csv_path is not None:
        run_metadata_extra["input_timed_disruption_csv"] = str(timed_disruption_csv_path.resolve())

    if write_outputs_for_assigned_route:
        run_output_dir = resolve_results_output_dir(output_root, order_text)

        copy_file_if_exists(settings_path, run_output_dir / "settings_used.json")
        if int(disruption_mode) == 1:
            copy_file_if_exists(disruption_path, run_output_dir / "disruption_used.json")
        elif int(disruption_mode) == 2:
            copy_file_if_exists(timed_disruption_json_path, run_output_dir / "disruption_used.json")
            copy_file_if_exists(timed_disruption_csv_path, run_output_dir / "disruption_used.csv")

        save_run_metadata(
            order_text,
            ordered_units,
            run_output_dir / "run_metadata.json",
            data_dir,
            run_output_dir,
            extra_payload=run_metadata_extra,
        )
    else:
        run_output_dir = output_root

    operations: list[OperationRecord] = []
    transport_records: list[TransportRecord] = []
    unit_summaries: list[UnitSummary] = []
    station_summaries: list[StationSummary] = []

    sim_with_dtimestart = time.perf_counter()
        # If called programmatically, treat timestamp as the segment stop time (seconds)
    if timestamp is not None:
        try:
            simulation_time_s = float(timestamp)
        except (TypeError, ValueError):
            pass

    # Bulletproof: ensure unit_ids is always defined before calling run_simulation
    if 'unit_ids' not in locals():
        unit_ids = []
    # Prefer IDs from the loaded schedule payload (current_schedule.csv)
    if (not unit_ids) and 'generated_input' in locals():
        try:
            unit_ids = generated_input.get('unit_ids', []) or []
        except Exception:
            unit_ids = []
    # Normalize to a plain list
    unit_ids = [] if unit_ids is None else list(unit_ids)

    print(">> Running sim with disruptions")
    operations, transport_records, unit_summaries, station_summaries, _, simulation_details = run_simulation(
        ordered_units=ordered_units,
        process_time_data=process_time_data,
        transport_time_data=transport_time_data,
        unit_release_times=unit_release_times,
        unit_priorities=unit_priorities,
        unit_order_ids=unit_order_ids,
        unit_ids=unit_ids,
        max_units_in_system=carriers,
        line_layout_config=line_layout_config,
        bom_data=bom_data,
        material_stock_data=material_stock_data,
        disruptions_enabled=disruptions_enabled,
        disruption_config=disruption_config if chance_based_disruptions_enabled else None,
        disruption_seed=disruption_seed if chance_based_disruptions_enabled else None,
        simulation_time_s=simulation_time_s,
        timed_disruption_data=timed_disruption_data,
    )
    print(">> Done running sim with disruptions")
    sim_with_dtimeend = time.perf_counter()
    print(f"Simulation with disruptions time: {sim_with_dtimeend - sim_with_dtimestart:.6f} seconds")

    completed_good_variants = list(simulation_details.get("completed_good_variants", []))
    completed_root_positions = list(simulation_details.get("completed_root_positions", []))
    completed_order_ids = [unit_order_ids[pos - 1] for pos in completed_root_positions if 1 <= int(pos) <= len(unit_order_ids)]

    material_report = build_material_report(
        requested_units=ordered_units,
        consumed_units=completed_good_variants,
        bom_data=bom_data,
        material_stock_data=material_stock_data,
        extra_material_consumed=simulation_details.get("extra_material_consumed", {}),
        actual_material_consumed=simulation_details.get("actual_material_consumed", {}),
    )

    completed_root_position_set = set(int(pos) for pos in completed_root_positions)
    unproduced_positions = [idx + 1 for idx in range(len(ordered_units)) if (idx + 1) not in completed_root_position_set]
    unproduced_units = [ordered_units[idx - 1] for idx in unproduced_positions]

    production_status: dict[str, Any] = {
        "requested_unit_count": len(ordered_units),
        "produced_unit_count": len(completed_good_variants),
        "unproduced_unit_count": len(unproduced_units),
        "produced_mix": dict(Counter(completed_good_variants)),
        "unproduced_mix": dict(Counter(unproduced_units)),
        "produced_unit_positions": completed_root_positions,
        "unproduced_unit_positions": unproduced_positions,
        "completed_order_ids": completed_order_ids,
        "carriers": carriers,
        "return_to_station_1_time_seconds": RETURN_TO_STATION_1_TIME_S,
        "disruptions_enabled": bool(disruptions_enabled),
        "disruption_seed": str(disruption_seed) if disruption_seed is not None else None,
        "simulation_time_seconds": simulation_time_s,
        "replacement_units_created": list(simulation_details.get("replacement_units_created", [])),
        "replacement_units_created_count": int(simulation_details.get("replacement_units_created_count", 0)),
        "extra_material_consumed_due_to_disruptions": dict(simulation_details.get("extra_material_consumed", {})),
        "remaining_stock_after_run": dict(simulation_details.get("remaining_stock_after_run", {})),
        "units_lost_due_to_disruptions_without_replacement": int(simulation_details.get("unrecoverable_root_count", 0)),
        "root_failures_without_replacement": list(simulation_details.get("root_failed_without_replacement", [])),
        "disruption_counts": dict(simulation_details.get("disruption_counts", {})),
        "stop_reason": simulation_details.get("stop_reason"),
        "stopped_due_to_sim_time_limit": bool(simulation_details.get("stopped_due_to_sim_time_limit", False)),
        "stopped_due_to_material_shortage": bool(simulation_details.get("stopped_due_to_material_shortage", False)),
        "status": "complete" if len(unproduced_units) == 0 else "partial_or_stopped",
    }

    if write_outputs_for_assigned_route:
        write_material_report_csv(material_report, run_output_dir / "material_report.csv")
        save_json(production_status, run_output_dir / "production_status.json")

    kpis = calculate_kpis(
        ordered_units=completed_good_variants,
        operations=operations,
        unit_summaries=unit_summaries,
        station_summaries=station_summaries,
        transport_records=transport_records,
    )

    completed_units_sorted = sorted(
        unit_summaries,
        key=lambda unit_summary: (
            float(unit_summary.completion_time_s),
            str(unit_summary.unit_id),
        ),
    )
    fuse0_wall_clock_intervals: list[float] = []
    fuse0_active_intervals: list[float] = []
    fuse0_global_intervals: list[float] = []

    line_active_intervals: list[tuple[float, float]] = []
    for op in operations:
        start_time_s = float(op.start_time_s)
        finish_time_s = float(op.finish_time_s)
        if finish_time_s > start_time_s:
            line_active_intervals.append((start_time_s, finish_time_s))
    for tr in transport_records:
        start_time_s = float(tr.start_time_s)
        finish_time_s = float(tr.finish_time_s)
        if finish_time_s > start_time_s:
            line_active_intervals.append((start_time_s, finish_time_s))
    if line_active_intervals:
        line_active_intervals.sort()
        merged_line_active_intervals: list[tuple[float, float]] = []
        current_start_s, current_finish_s = line_active_intervals[0]
        for start_time_s, finish_time_s in line_active_intervals[1:]:
            if start_time_s <= current_finish_s:
                current_finish_s = max(current_finish_s, finish_time_s)
            else:
                merged_line_active_intervals.append((current_start_s, current_finish_s))
                current_start_s, current_finish_s = start_time_s, finish_time_s
        merged_line_active_intervals.append((current_start_s, current_finish_s))
        line_active_intervals = merged_line_active_intervals

    fuse0_variant_intervals: list[tuple[float, float]] = []
    for op in operations:
        if str(op.variant).upper() != "FUSE0":
            continue
        start_time_s = float(op.start_time_s)
        finish_time_s = float(op.finish_time_s)
        if finish_time_s > start_time_s:
            fuse0_variant_intervals.append((start_time_s, finish_time_s))
    for tr in transport_records:
        if str(tr.variant).upper() != "FUSE0":
            continue
        start_time_s = float(tr.start_time_s)
        finish_time_s = float(tr.finish_time_s)
        if finish_time_s > start_time_s:
            fuse0_variant_intervals.append((start_time_s, finish_time_s))
    if fuse0_variant_intervals:
        fuse0_variant_intervals.sort()
        merged_fuse0_variant_intervals: list[tuple[float, float]] = []
        current_start_s, current_finish_s = fuse0_variant_intervals[0]
        for start_time_s, finish_time_s in fuse0_variant_intervals[1:]:
            if start_time_s <= current_finish_s:
                current_finish_s = max(current_finish_s, finish_time_s)
            else:
                merged_fuse0_variant_intervals.append((current_start_s, current_finish_s))
                current_start_s, current_finish_s = start_time_s, finish_time_s
        merged_fuse0_variant_intervals.append((current_start_s, current_finish_s))
        fuse0_variant_intervals = merged_fuse0_variant_intervals

    def _overlap_with_intervals(
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

    for previous_unit_summary, current_unit_summary in zip(
        completed_units_sorted,
        completed_units_sorted[1:],
    ):
        previous_variant = str(previous_unit_summary.variant).upper()
        current_variant = str(current_unit_summary.variant).upper()
        if previous_variant != "FUSE0" or current_variant != "FUSE0":
            continue

        interval_start_s = float(previous_unit_summary.completion_time_s)
        interval_finish_s = float(current_unit_summary.completion_time_s)
        interval_duration_s = max(0.0, interval_finish_s - interval_start_s)

        fuse0_wall_clock_intervals.append(interval_duration_s)
        fuse0_active_intervals.append(
            _overlap_with_intervals(
                interval_start_s,
                interval_finish_s,
                fuse0_variant_intervals,
            )
        )

        active_any_variant_s = _overlap_with_intervals(
            interval_start_s,
            interval_finish_s,
            line_active_intervals,
        )
        no_activity_s = max(0.0, interval_duration_s - active_any_variant_s)
        fuse0_global_intervals.append(
            fuse0_active_intervals[-1] + no_activity_s
        )

    fuse0_completion_times = [
        float(unit_summary.completion_time_s)
        for unit_summary in completed_units_sorted
        if str(unit_summary.variant).upper() == "FUSE0"
    ]
    if len(fuse0_completion_times) == 1:
        fuse0_average_cycle_time = float(fuse0_completion_times[0])
        fuse0_active_cycle_time = _overlap_with_intervals(
            0.0,
            float(fuse0_completion_times[0]),
            fuse0_variant_intervals,
        )
        fuse0_global_cycle_time = float(fuse0_completion_times[0])
    else:
        fuse0_average_cycle_time = (
            float(mean(fuse0_wall_clock_intervals))
            if fuse0_wall_clock_intervals
            else 0.0
        )
        fuse0_active_cycle_time = (
            float(mean(fuse0_active_intervals))
            if fuse0_active_intervals
            else 0.0
        )
        fuse0_global_cycle_time = (
            float(mean(fuse0_global_intervals))
            if fuse0_global_intervals
            else 0.0
        )

    kpis["average_cycle_time_seconds_fuse0"] = round(
        fuse0_average_cycle_time,
        4,
    )
    kpis["Active_average_cycle_time_seconds_fuse0"] = round(
        fuse0_active_cycle_time,
        4,
    )
    kpis["global_average_cycle_time_seconds_fuse0"] = round(
        fuse0_global_cycle_time,
        4,
    )

    kpis["carriers"] = carriers
    kpis["return_to_station_1_time_seconds"] = round(RETURN_TO_STATION_1_TIME_S, 4)
    kpis["disruptions_enabled"] = int(disruptions_enabled)
    kpis["disruption_mode"] = int(disruption_mode)
    if disruption_seed is not None:
        kpis["disruption_seed"] = str(disruption_seed)
    if simulation_time_s is not None:
        kpis["simulation_time_seconds"] = round(simulation_time_s, 4)
    kpis["line_layout_name"] = str(effective_line_layout.get("layout_name", "default_single_path"))
    kpis["effective_station_count"] = len(effective_line_layout["station_sequence"])
    kpis["completed_good_units"] = len(completed_good_variants)
    if simulation_details.get("stop_reason"):
        kpis["stopped_due_to_sim_time_limit"] = int(bool(simulation_details.get("stopped_due_to_sim_time_limit", False)))
        kpis["stopped_due_to_material_shortage"] = int(bool(simulation_details.get("stopped_due_to_material_shortage", False)))
    for key, value in sorted(simulation_details.get("disruption_counts", {}).items()):
        kpis[f"disruption_{key}"] = value

    operations_no_disruptions: list[OperationRecord] = operations
    transport_records_no_disruptions: list[TransportRecord] = transport_records
    unit_summaries_no_disruptions: list[UnitSummary] = unit_summaries
    station_summaries_no_disruptions: list[StationSummary] = station_summaries
    completed_good_variants_no_disruptions: list[str] = completed_good_variants
    simulation_details_no_disruptions: dict[str, Any] = simulation_details

    if disruptions_enabled:
        print(">> Running sim without disruptions")
        sim_no_dtimestart = time.perf_counter()
        (
            operations_no_disruptions,
            transport_records_no_disruptions,
            unit_summaries_no_disruptions,
            station_summaries_no_disruptions,
            _,
            simulation_details_no_disruptions,
        ) = run_simulation(
            ordered_units=ordered_units,
            process_time_data=process_time_data,
            transport_time_data=transport_time_data,
            unit_release_times=unit_release_times,
            unit_priorities=unit_priorities,
            unit_order_ids=unit_order_ids,
            max_units_in_system=carriers,
            line_layout_config=line_layout_config,
            bom_data=bom_data,
            material_stock_data=material_stock_data,
            disruptions_enabled=False,
            disruption_config=None,
            disruption_seed=None,
            simulation_time_s=simulation_time_s,
            timed_disruption_data=None,
        )
        print(">> Done running sim without disruptions")
        sim_no_dtimeend = time.perf_counter()
        print(f"Simulation without disruptions time: {sim_no_dtimeend - sim_no_dtimestart:.6f} seconds")
        completed_good_variants_no_disruptions = list(
            simulation_details_no_disruptions.get("completed_good_variants", [])
        )

    kpis_no_disruptions = calculate_kpis(
        ordered_units=completed_good_variants_no_disruptions,
        operations=operations_no_disruptions,
        unit_summaries=unit_summaries_no_disruptions,
        station_summaries=station_summaries_no_disruptions,
        transport_records=transport_records_no_disruptions,
    )

    completed_units_sorted_no_disruptions = sorted(
        unit_summaries_no_disruptions,
        key=lambda unit_summary: (
            float(unit_summary.completion_time_s),
            str(unit_summary.unit_id),
        ),
    )
    fuse0_wall_clock_intervals_no_disruptions: list[float] = []
    fuse0_active_intervals_no_disruptions: list[float] = []
    fuse0_global_intervals_no_disruptions: list[float] = []

    line_active_intervals_no_disruptions: list[tuple[float, float]] = []
    for op in operations_no_disruptions:
        start_time_s = float(op.start_time_s)
        finish_time_s = float(op.finish_time_s)
        if finish_time_s > start_time_s:
            line_active_intervals_no_disruptions.append((start_time_s, finish_time_s))
    for tr in transport_records_no_disruptions:
        start_time_s = float(tr.start_time_s)
        finish_time_s = float(tr.finish_time_s)
        if finish_time_s > start_time_s:
            line_active_intervals_no_disruptions.append((start_time_s, finish_time_s))
    if line_active_intervals_no_disruptions:
        line_active_intervals_no_disruptions.sort()
        merged_line_active_intervals_no_disruptions: list[tuple[float, float]] = []
        current_start_s, current_finish_s = line_active_intervals_no_disruptions[0]
        for start_time_s, finish_time_s in line_active_intervals_no_disruptions[1:]:
            if start_time_s <= current_finish_s:
                current_finish_s = max(current_finish_s, finish_time_s)
            else:
                merged_line_active_intervals_no_disruptions.append((current_start_s, current_finish_s))
                current_start_s, current_finish_s = start_time_s, finish_time_s
        merged_line_active_intervals_no_disruptions.append((current_start_s, current_finish_s))
        line_active_intervals_no_disruptions = merged_line_active_intervals_no_disruptions

    fuse0_variant_intervals_no_disruptions: list[tuple[float, float]] = []
    for op in operations_no_disruptions:
        if str(op.variant).upper() != "FUSE0":
            continue
        start_time_s = float(op.start_time_s)
        finish_time_s = float(op.finish_time_s)
        if finish_time_s > start_time_s:
            fuse0_variant_intervals_no_disruptions.append((start_time_s, finish_time_s))
    for tr in transport_records_no_disruptions:
        if str(tr.variant).upper() != "FUSE0":
            continue
        start_time_s = float(tr.start_time_s)
        finish_time_s = float(tr.finish_time_s)
        if finish_time_s > start_time_s:
            fuse0_variant_intervals_no_disruptions.append((start_time_s, finish_time_s))
    if fuse0_variant_intervals_no_disruptions:
        fuse0_variant_intervals_no_disruptions.sort()
        merged_fuse0_variant_intervals_no_disruptions: list[tuple[float, float]] = []
        current_start_s, current_finish_s = fuse0_variant_intervals_no_disruptions[0]
        for start_time_s, finish_time_s in fuse0_variant_intervals_no_disruptions[1:]:
            if start_time_s <= current_finish_s:
                current_finish_s = max(current_finish_s, finish_time_s)
            else:
                merged_fuse0_variant_intervals_no_disruptions.append((current_start_s, current_finish_s))
                current_start_s, current_finish_s = start_time_s, finish_time_s
        merged_fuse0_variant_intervals_no_disruptions.append((current_start_s, current_finish_s))
        fuse0_variant_intervals_no_disruptions = merged_fuse0_variant_intervals_no_disruptions

    def _overlap_with_intervals_no_disruptions(
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

    for previous_unit_summary, current_unit_summary in zip(
        completed_units_sorted_no_disruptions,
        completed_units_sorted_no_disruptions[1:],
    ):
        previous_variant = str(previous_unit_summary.variant).upper()
        current_variant = str(current_unit_summary.variant).upper()
        if previous_variant != "FUSE0" or current_variant != "FUSE0":
            continue

        interval_start_s = float(previous_unit_summary.completion_time_s)
        interval_finish_s = float(current_unit_summary.completion_time_s)
        interval_duration_s = max(0.0, interval_finish_s - interval_start_s)

        fuse0_wall_clock_intervals_no_disruptions.append(interval_duration_s)
        fuse0_active_intervals_no_disruptions.append(
            _overlap_with_intervals_no_disruptions(
                interval_start_s,
                interval_finish_s,
                fuse0_variant_intervals_no_disruptions,
            )
        )

        active_any_variant_s = _overlap_with_intervals_no_disruptions(
            interval_start_s,
            interval_finish_s,
            line_active_intervals_no_disruptions,
        )
        no_activity_s = max(0.0, interval_duration_s - active_any_variant_s)
        fuse0_global_intervals_no_disruptions.append(
            fuse0_active_intervals_no_disruptions[-1] + no_activity_s
        )

    fuse0_completion_times_no_disruptions = [
        float(unit_summary.completion_time_s)
        for unit_summary in completed_units_sorted_no_disruptions
        if str(unit_summary.variant).upper() == "FUSE0"
    ]
    if len(fuse0_completion_times_no_disruptions) == 1:
        fuse0_average_cycle_time_no_disruptions = float(fuse0_completion_times_no_disruptions[0])
        fuse0_active_cycle_time_no_disruptions = _overlap_with_intervals_no_disruptions(
            0.0,
            float(fuse0_completion_times_no_disruptions[0]),
            fuse0_variant_intervals_no_disruptions,
        )
        fuse0_global_cycle_time_no_disruptions = float(fuse0_completion_times_no_disruptions[0])
    else:
        fuse0_average_cycle_time_no_disruptions = (
            float(mean(fuse0_wall_clock_intervals_no_disruptions))
            if fuse0_wall_clock_intervals_no_disruptions
            else 0.0
        )
        fuse0_active_cycle_time_no_disruptions = (
            float(mean(fuse0_active_intervals_no_disruptions))
            if fuse0_active_intervals_no_disruptions
            else 0.0
        )
        fuse0_global_cycle_time_no_disruptions = (
            float(mean(fuse0_global_intervals_no_disruptions))
            if fuse0_global_intervals_no_disruptions
            else 0.0
        )

    kpis_no_disruptions["average_cycle_time_seconds_fuse0"] = round(
        fuse0_average_cycle_time_no_disruptions,
        4,
    )
    kpis_no_disruptions["Active_average_cycle_time_seconds_fuse0"] = round(
        fuse0_active_cycle_time_no_disruptions,
        4,
    )
    kpis_no_disruptions["global_average_cycle_time_seconds_fuse0"] = round(
        fuse0_global_cycle_time_no_disruptions,
        4,
    )

    kpis_no_disruptions["carriers"] = carriers
    kpis_no_disruptions["return_to_station_1_time_seconds"] = round(RETURN_TO_STATION_1_TIME_S, 4)
    kpis_no_disruptions["disruptions_enabled"] = 0
    kpis_no_disruptions["disruption_mode"] = 0
    if simulation_time_s is not None:
        kpis_no_disruptions["simulation_time_seconds"] = round(simulation_time_s, 4)
    kpis_no_disruptions["line_layout_name"] = str(effective_line_layout.get("layout_name", "default_single_path"))
    kpis_no_disruptions["effective_station_count"] = len(effective_line_layout["station_sequence"])
    kpis_no_disruptions["completed_good_units"] = len(completed_good_variants_no_disruptions)

    station_utilization_active_window_without_disruptions = {
        (int(summary.station_index), str(summary.station_name)): float(summary.utilization_active_window)
        for summary in station_summaries_no_disruptions
    }

    active_production_line_time_s = _calculate_active_production_line_time_s(
        operations,
        simulation_details.get("disruption_event_log", []),
        transport_records,
    )
    station_active_order_utilization = {}
    for summary in station_summaries:
        station_key = (int(summary.station_index), str(summary.station_name))
        if active_production_line_time_s > 0:
            station_active_order_utilization[station_key] = float(summary.busy_time_s) / float(active_production_line_time_s)
        else:
            station_active_order_utilization[station_key] = 0.0
        safe_key = re.sub(r"[^A-Za-z0-9]+", "_", str(summary.station_name)).strip("_").lower()
        kpis[f"active_order_utilization_{safe_key}"] = round(
            float(station_active_order_utilization[station_key]),
            6,
        )

    active_production_line_time_no_disruptions_s = _calculate_active_production_line_time_s(
        operations_no_disruptions,
        simulation_details_no_disruptions.get("disruption_event_log", []),
        transport_records_no_disruptions,
    )
    station_active_order_utilization_no_disruptions = {}
    for summary in station_summaries_no_disruptions:
        station_key = (int(summary.station_index), str(summary.station_name))
        if active_production_line_time_no_disruptions_s > 0:
            station_active_order_utilization_no_disruptions[station_key] = float(summary.busy_time_s) / float(active_production_line_time_no_disruptions_s)
        else:
            station_active_order_utilization_no_disruptions[station_key] = 0.0
        safe_key = re.sub(r"[^A-Za-z0-9]+", "_", str(summary.station_name)).strip("_").lower()
        kpis_no_disruptions[f"active_order_utilization_{safe_key}"] = round(
            float(station_active_order_utilization_no_disruptions[station_key]),
            6,
        )

    if write_outputs_for_assigned_route:
        write_kpis_csv(kpis, run_output_dir / "kpi_summary.csv")
        write_kpis_csv(kpis_no_disruptions, run_output_dir / "kpi_summary_without_disruptions.csv")
        write_operations_csv(operations, run_output_dir / "station_schedule.csv")
        write_transport_csv(transport_records, run_output_dir / "transport_schedule.csv")
        write_unit_summary_csv(unit_summaries, run_output_dir / "unit_summary.csv")
        write_station_summary_csv(
            station_summaries,
            run_output_dir / "station_summary.csv",
            utilization_active_window_without_disruptions_by_station=station_utilization_active_window_without_disruptions,
            active_order_utilization_by_station=station_active_order_utilization,
        )

        if disruptions_enabled or simulation_details.get("disruption_event_log"):
            save_json(
                {
                    "disruptions_enabled": bool(disruptions_enabled),
                    "disruption_mode": int(disruption_mode),
                    "seed": str(disruption_seed) if disruption_seed is not None else None,
                    "broken_material_extra_time_seconds": _broken_material_extra_time_s(disruption_config),
                    "timed_disruption_records": list(timed_disruption_records),
                    "disruption_counts": dict(simulation_details.get("disruption_counts", {})),
                    "stop_reason": simulation_details.get("stop_reason"),
                    "events": list(simulation_details.get("disruption_event_log", [])),
                },
                run_output_dir / "disruption_summary.json",
            )

            disruption_history_rows = build_disruption_history_rows(
                simulation_details=simulation_details,
                disruption_config=disruption_config,
                timed_disruption_records=timed_disruption_records,
                current_time_s=simulation_time_s,
            )
            if main_settings_pathlist and _pathlist_path(main_settings_pathlist, "on_going_run_dis_his") is not None:
                disruption_history_path = _pathlist_path(main_settings_pathlist, "on_going_run_dis_his")
            else:
                ongoing_run_dir = resolve_ongoing_run_dir(Path(__file__).resolve().parent)
                disruption_history_path = ongoing_run_dir / DISRUPTION_HISTORY_FILENAME
            write_disruption_history_csv(
                disruption_history_path,
                disruption_history_rows,
            )

    if main_settings_pathlist and _pathlist_path(main_settings_pathlist, "on_going_run_unit_summary") is not None:
        ongoing_unit_summary_path = _pathlist_path(main_settings_pathlist, "on_going_run_unit_summary")
    else:
        ongoing_run_dir_for_summary = resolve_ongoing_run_dir(Path(__file__).resolve().parent)
        ongoing_unit_summary_path = ongoing_run_dir_for_summary / "unit_summary.csv"
    write_unit_summary_snapshot_csv(unit_summaries, simulation_details.get("wip_snapshot", []), ongoing_unit_summary_path)

    if not write_outputs_for_assigned_route:
        print("Output files skipped: all current_schedule.csv route values are 0")
    print(f"Run folder: {run_output_dir.resolve()}")
    if simulation_time_s is not None:
        print(f"Simulation time: {simulation_time_s} s")
    print(f"Carriers: {carriers}")
    print(f"Return time to Station 1: {RETURN_TO_STATION_1_TIME_S} s")
    print(f"Line layout: {effective_line_layout.get('layout_name', 'default_single_path')}")
    if line_layout_path is not None:
        print(f"Line layout file: {line_layout_path.resolve()}")
    print(f"Disruptions enabled: {int(disruptions_enabled)}")
    print(f"Disruption mode: {int(disruption_mode)}")
    if chance_based_disruptions_enabled and disruption_path is not None:
        print(f"Disruption file: {disruption_path.resolve()}")
    if timed_disruptions_enabled and timed_disruption_csv_path is not None:
        print(f"Timed disruption file: {timed_disruption_csv_path.resolve()}")
    print(f"Effective station count: {len(effective_line_layout['station_sequence'])}")
    print(f"Requested units: {len(ordered_units)}")
    print(f"Completed good units: {len(completed_good_variants)}")
    if simulation_details.get("stop_reason"):
        print(f"Stop reason: {simulation_details['stop_reason']}")
    if simulation_details.get("unrecoverable_root_count", 0):
        print(f"Units lost due to disruptions without replacement: {simulation_details['unrecoverable_root_count']}")

    endtime = time.perf_counter()
    print(f"Total execution time: {endtime - starttime:.6f} seconds")

if __name__ == "__main__":
    main(timestamp=1244000)
