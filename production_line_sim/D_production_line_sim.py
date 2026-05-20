from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def _insert_helper_if_needed(text: str) -> str:
    if "_next_emergency_unit_number" in text:
        return text

    helper = '''def _is_emergency_unit_id(unit_id: object) -> bool:
    unit_id_text = str(unit_id).strip().upper()
    return bool(re.fullmatch(r"E\\d+", unit_id_text))


def _next_emergency_unit_number(existing_unit_ids: list[str] | None) -> int:
    highest_emergency_number = 0
    for unit_id in existing_unit_ids or []:
        unit_id_text = str(unit_id).strip().upper()
        match = re.fullmatch(r"E(\\d+)", unit_id_text)
        if match:
            highest_emergency_number = max(highest_emergency_number, int(match.group(1)))
    return highest_emergency_number + 1


'''
    anchor = "def _assign_missing_emergency_order_ids("
    if anchor in text:
        return text.replace(anchor, helper + anchor, 1)

    anchor = "def load_timed_disruption_csv("
    if anchor in text:
        return text.replace(anchor, helper + anchor, 1)

    raise RuntimeError("Could not find a safe location to insert emergency unit helper functions.")


def patch_emergency_fastest_first(source_path: Path, output_path: Path | None = None) -> Path:
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Could not find source file: {source_path}")

    text = source_path.read_text(encoding="utf-8", errors="ignore")
    original_text = text

    has_unit_ids = bool(re.search(r"\bunit_ids\s*:", text)) or bool(re.search(r"\bunit_ids\s*=", text))
    has_unit_custom_ids = bool(re.search(r"\bunit_custom_ids\s*:", text)) or bool(re.search(r"\bunit_custom_ids\s*=", text))

    if has_unit_ids or has_unit_custom_ids:
        text = _insert_helper_if_needed(text)

    unit_id_prepend_line = ""
    unit_id_init_line = ""
    unit_id_append_line = ""

    if has_unit_ids:
        unit_id_init_line = "        emergency_unit_ids: list[str] = []\n        next_emergency_unit_number = _next_emergency_unit_number(unit_ids)\n"
        unit_id_append_line = "                    emergency_unit_ids.append(f\"E{next_emergency_unit_number:03d}\")\n                    next_emergency_unit_number += 1\n"
        unit_id_prepend_line = "            unit_ids = emergency_unit_ids + unit_ids\n"
    elif has_unit_custom_ids:
        unit_id_init_line = "        emergency_unit_ids: list[str] = []\n        next_emergency_unit_number = _next_emergency_unit_number(unit_custom_ids)\n"
        unit_id_append_line = "                    emergency_unit_ids.append(f\"E{next_emergency_unit_number:03d}\")\n                    next_emergency_unit_number += 1\n"
        unit_id_prepend_line = "            unit_custom_ids = emergency_unit_ids + unit_custom_ids\n"

    emergency_replacement = (
        "        emergency_units: list[str] = []\n"
        "        emergency_release_times: list[float] = []\n"
        "        emergency_priorities: list[int] = []\n"
        "        emergency_order_ids: list[str] = []\n"
        "        emergency_route_ids: list[str] = []\n"
        f"{unit_id_init_line}"
        "        emergency_priority_base = max([int(value) for value in unit_priorities], default=1) + 1000000\n\n"
        "        for emergency_order in timed_disruption_data.get(\"emergency_orders\", []):\n"
        "            emergency_order_id = str(emergency_order[\"order_id\"])\n"
        "            emergency_order_time_s = float(emergency_order[\"order_time_s\"])\n"
        "            emergency_priority = emergency_priority_base + max(1, int(emergency_order[\"priority\"]))\n"
        "            for variant_value, quantity_value in emergency_order.get(\"variants\", []):\n"
        "                for _ in range(int(quantity_value)):\n"
        "                    emergency_units.append(str(variant_value))\n"
        "                    emergency_release_times.append(emergency_order_time_s)\n"
        "                    emergency_priorities.append(emergency_priority)\n"
        "                    emergency_order_ids.append(emergency_order_id)\n"
        f"{unit_id_append_line}"
        "                    emergency_route_ids.append(\"0\")\n\n"
        "        if emergency_units:\n"
        "            ordered_units = emergency_units + ordered_units\n"
        "            unit_release_times = emergency_release_times + unit_release_times\n"
        "            unit_priorities = emergency_priorities + unit_priorities\n"
        "            unit_order_ids = emergency_order_ids + unit_order_ids\n"
        f"{unit_id_prepend_line}"
        "            unit_route_ids = emergency_route_ids + unit_route_ids"
    )

    pattern = re.compile(
        r'''        for emergency_order in timed_disruption_data\.get\("emergency_orders", \[\]\):\n'''
        r'''(?:            .*\n)+?'''
        r'''(?=\n    initial_requested_unit_count = len\(ordered_units\))''',
        re.MULTILINE,
    )

    text, count = pattern.subn(emergency_replacement, text, count=1)
    if count != 1:
        raise RuntimeError(
            "Could not find the emergency-order append block to patch. "
            "No file was modified."
        )

    if text == original_text:
        raise RuntimeError("Patch made no changes.")

    if output_path is None:
        output_path = source_path.with_name(source_path.stem + "_emergency_fastest_first.py")
    else:
        output_path = Path(output_path).expanduser().resolve()

    output_path.write_text(text, encoding="utf-8")

    backup_path = source_path.with_suffix(source_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)

    return output_path


def main() -> None:
    source_arg = sys.argv[1] if len(sys.argv) > 1 else "D_production_line_sim.py"
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None
    out = patch_emergency_fastest_first(Path(source_arg), Path(output_arg) if output_arg else None)
    print(f"Patched file written to: {out}")


if __name__ == "__main__":
    main()
