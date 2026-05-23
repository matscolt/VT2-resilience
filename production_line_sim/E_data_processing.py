from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ============================================================
# Hardcoded configuration
# ============================================================
# This script is assumed to sit BESIDE the RESULTS folder, not inside it.
# Example:
#   project_root/
#   ├─ RESULTS/
#   └─ E_data_processing.py

OUTPUT_DIR = Path(__file__).resolve().parent / "RESULTS" / "output"
ON_GOING_DIR = Path(__file__).resolve().parent / "RESULTS" / "on_going"
DAY_SECONDS = 8 * 60 * 60   # 8 hours per day
WEEK_DAYS = 5               # 5 days per week


# ============================================================
# Helpers
# ============================================================

def norm(name: str) -> str:
    """Normalize a column name for robust matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def build_col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {norm(c): c for c in df.columns}


def find_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> Optional[str]:
    """Find a column in a dataframe using possible aliases."""
    col_map = build_col_map(df)
    for alias in aliases:
        key = norm(alias)
        if key in col_map:
            return col_map[key]
    if required:
        raise KeyError(
            f"Could not find any of these columns: {list(aliases)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def seconds_to_week_day(seconds_value: float, day_seconds: float, week_days: int) -> tuple[Optional[int], Optional[int]]:
    """Convert simulation seconds to (week, day), both 1-based."""
    if pd.isna(seconds_value):
        return None, None

    day_index = int(float(seconds_value) // day_seconds)
    week = day_index // week_days + 1
    day = day_index + 1
    return week, day


def _parse_main_folder_name(name: str, assumed_year: int) -> Optional[Tuple[datetime, int]]:
    """
    Parse folder names like: main_22-05_22-02_0
    -> datetime(assumed_year, 5, 22, 22, 2), trailing index 0
    """
    m = re.fullmatch(r"main_(\d{2})-(\d{2})_(\d{2})-(\d{2})_(\d+)", name)
    if not m:
        return None

    day, month, hour, minute, idx = map(int, m.groups())
    try:
        ts = datetime(assumed_year, month, day, hour, minute)
    except ValueError:
        return None
    return ts, idx


def list_main_folders(output_dir: Path) -> List[Path]:
    """List candidate main_* folders, newest first when parseable."""
    assumed_year = datetime.now().year
    parsed: List[Tuple[datetime, int, Path]] = []
    unparsed: List[Path] = []

    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        info = _parse_main_folder_name(p.name, assumed_year)
        if info is None:
            if p.name.startswith("main"):
                unparsed.append(p)
            continue
        ts, idx = info
        parsed.append((ts, idx, p))

    parsed.sort(key=lambda x: (x[0], x[1]), reverse=True)
    unparsed.sort(key=lambda x: x.name, reverse=True)
    return [p for _, _, p in parsed] + unparsed


def select_main_folder(output_dir: Path) -> Path:
    """Interactive main-folder picker."""
    candidates = list_main_folders(output_dir)
    if not candidates:
        raise FileNotFoundError(f"No valid main folders found in {output_dir}")

    print(f"Available main folders in {output_dir} (newest first):")
    for i, path in enumerate(candidates, start=1):
        print(f"{i}) {path.name}")

    while True:
        choice = input("\nChoose main folder number (Enter = 1): ").strip()
        if choice == "":
            return candidates[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print("Invalid selection. Please choose a valid number from the list.")


def list_run_folders(main_output_dir: Path) -> List[Path]:
    runs = [p for p in main_output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name)
    return runs


def find_production_plan_file(main_name: str, run_name: str) -> Path:
    """
    Find the single production_plan*.csv for a run in:
      RESULTS/on_going/<main_name>/<run_name>/
    """
    run_dir = ON_GOING_DIR / main_name / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"on_going run folder not found: {run_dir}")

    plans = sorted(run_dir.glob("production_plan*.csv"))
    if not plans:
        raise FileNotFoundError(f"No production_plan*.csv found in {run_dir}")
    if len(plans) > 1:
        print(f"WARNING: Multiple production plans found in {run_dir}. Using: {plans[0].name}")
    return plans[0]


def extract_plan_metadata(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Extract metadata columns from the production plan."""
    order_col = find_col(plan_df, ["order_id", "orderID"])
    due_col = find_col(plan_df, ["due date", "due_date", "due"])
    priority_col = find_col(plan_df, ["priority"])
    planned_week_col = find_col(plan_df, ["planned_week", "planned week"])
    planned_day_col = find_col(plan_df, ["planned_day", "planned day"])

    meta = plan_df[[order_col, due_col, priority_col, planned_week_col, planned_day_col]].copy()
    meta = meta.rename(columns={
        order_col: "order_id",
        due_col: "due date",
        priority_col: "priority",
        planned_week_col: "planned_week",
        planned_day_col: "planned_day",
    })
    meta["order_id"] = safe_numeric(meta["order_id"]).astype("Int64")
    meta = meta.dropna(subset=["order_id"]).copy()
    meta["order_id"] = meta["order_id"].astype(int)
    for c in ["due date", "priority", "planned_week", "planned_day"]:
        meta[c] = safe_numeric(meta[c])
    meta = meta.drop_duplicates(subset=["order_id"])
    return meta


def extract_plan_variants(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Extract variant0..2 and quantity0..2 from the production plan."""
    order_col = find_col(plan_df, ["order_id", "orderID"])

    col_aliases = {
        "variant0": ["variant0", "variant_0"],
        "quantity0": ["quantity0", "quantity_0"],
        "variant1": ["variant1", "variant_1"],
        "quantity1": ["quantity1", "quantity_1"],
        "variant2": ["variant2", "variant_2"],
        "quantity2": ["quantity2", "quantity_2"],
    }

    resolved = {"order_id": order_col}
    for out_col, aliases in col_aliases.items():
        resolved[out_col] = find_col(plan_df, aliases, required=False)

    keep_cols = [order_col] + [c for c in resolved.values() if c is not None and c != order_col]
    variants = plan_df[keep_cols].copy()
    rename_map = {order_col: "order_id"}
    for out_col, source_col in resolved.items():
        if out_col != "order_id" and source_col is not None:
            rename_map[source_col] = out_col
    variants = variants.rename(columns=rename_map)

    variants["order_id"] = safe_numeric(variants["order_id"]).astype("Int64")
    variants = variants.dropna(subset=["order_id"]).copy()
    variants["order_id"] = variants["order_id"].astype(int)

    for vcol in ["variant0", "variant1", "variant2"]:
        if vcol not in variants.columns:
            variants[vcol] = ""
        variants[vcol] = variants[vcol].fillna("").astype(str)

    for qcol in ["quantity0", "quantity1", "quantity2"]:
        if qcol not in variants.columns:
            variants[qcol] = 0
        variants[qcol] = safe_numeric(variants[qcol]).fillna(0).astype(int)

    variants = variants[[
        "order_id", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2"
    ]].drop_duplicates(subset=["order_id"])
    return variants


def build_actual_variant_summary(unit_df: pd.DataFrame, order_col: str, variant_col: str) -> pd.DataFrame:
    """Build actual variant counts from unit_summary.csv."""
    counts = (
        unit_df.groupby([order_col, variant_col], dropna=False)
        .size()
        .reset_index(name="qty")
        .rename(columns={order_col: "order_id", variant_col: "variant"})
    )

    rows = []
    for order_id, grp in counts.groupby("order_id"):
        grp = grp.sort_values("variant", kind="stable").reset_index(drop=True)
        row = {"order_id": int(order_id)}
        for i in range(3):
            if i < len(grp):
                row[f"variant{i}"] = str(grp.loc[i, "variant"])
                row[f"quantity{i}"] = int(grp.loc[i, "qty"])
            else:
                row[f"variant{i}"] = ""
                row[f"quantity{i}"] = 0
        if len(grp) > 3:
            extras = ", ".join(grp.loc[3:, "variant"].astype(str).tolist())
            print(f"WARNING: order {order_id} has more than 3 variants in unit_summary. Extra variants ignored: {extras}")
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["order_id", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2"])
    return pd.DataFrame(rows)


def validate_summary(summary: pd.DataFrame) -> None:
    if summary["order_id"].duplicated().any():
        dupes = summary.loc[summary["order_id"].duplicated(), "order_id"].tolist()
        raise ValueError(f"Duplicate order_id values found in output summary: {dupes[:10]}")

    for col in ["quantity0", "quantity1", "quantity2"]:
        if (safe_numeric(summary[col]).fillna(0) < 0).any():
            raise ValueError(f"Negative quantities found in {col}.")

    mask = summary["finish_time"].notna() & summary["start_time"].notna()
    bad = summary.loc[mask & (summary["finish_time"] < summary["start_time"])]
    if not bad.empty:
        bad_orders = bad["order_id"].tolist()[:10]
        raise ValueError(
            f"Found orders where finish_time < start_time. Example order_id values: {bad_orders}"
        )


def build_order_summary(results_dir: Path, main_name: str, run_name: str) -> Path:
    """
    Create order_summary.csv using:
      - output/<main>/run_x/results/unit_summary.csv
      - on_going/<main>/run_x/production_plan*.csv
    """
    unit_path = results_dir / "unit_summary.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing {unit_path}")

    plan_path = find_production_plan_file(main_name, run_name)

    unit_df = pd.read_csv(unit_path)
    plan_df = pd.read_csv(plan_path)

    # --- unit_summary columns ---
    unit_order_col = find_col(unit_df, ["order_id", "orderID"])
    unit_variant_col = find_col(unit_df, ["variant"])
    unit_start_col = find_col(unit_df, ["first_arrival_time_s", "first_arrival_time", "first_arrival"])
    unit_finish_col = find_col(unit_df, ["completion_time_s", "completion_time", "completion"])

    unit_df[unit_order_col] = safe_numeric(unit_df[unit_order_col]).astype("Int64")
    unit_df[unit_start_col] = safe_numeric(unit_df[unit_start_col])
    unit_df[unit_finish_col] = safe_numeric(unit_df[unit_finish_col])
    unit_df = unit_df.dropna(subset=[unit_order_col]).copy()
    unit_df[unit_order_col] = unit_df[unit_order_col].astype(int)
    unit_df[unit_variant_col] = unit_df[unit_variant_col].astype(str)

    # --- Actual timing from unit_summary ---
    timing = (
        unit_df.groupby(unit_order_col, dropna=False)
        .agg(
            start_time=(unit_start_col, "min"),
            finish_time=(unit_finish_col, "max"),
        )
        .reset_index()
        .rename(columns={unit_order_col: "order_id"})
    )
    timing["through_put_time"] = timing["finish_time"] - timing["start_time"]

    # --- Actual variant counts from unit_summary ---
    actual_variants = build_actual_variant_summary(unit_df, unit_order_col, unit_variant_col)

    # --- Planned metadata and planned variants from production plan ---
    plan_meta = extract_plan_metadata(plan_df)
    plan_variants = extract_plan_variants(plan_df)

    # --- Merge with plan as the base so planned orders with no completed units still appear ---
    summary = plan_meta.merge(timing, on="order_id", how="left")
    summary = summary.merge(plan_variants, on="order_id", how="left", suffixes=("", "_plan"))
    summary = summary.merge(actual_variants, on="order_id", how="left", suffixes=("_plan", ""))

    # Prefer actual variant/quantity values when present; otherwise fall back to plan values
    for i in range(3):
        v_actual = f"variant{i}"
        q_actual = f"quantity{i}"
        v_plan = f"variant{i}_plan"
        q_plan = f"quantity{i}_plan"

        if v_actual not in summary.columns:
            summary[v_actual] = ""
        if q_actual not in summary.columns:
            summary[q_actual] = pd.NA
        if v_plan not in summary.columns:
            summary[v_plan] = ""
        if q_plan not in summary.columns:
            summary[q_plan] = pd.NA

        summary[v_actual] = summary[v_actual].fillna("")
        summary[v_plan] = summary[v_plan].fillna("")
        summary[q_actual] = safe_numeric(summary[q_actual])
        summary[q_plan] = safe_numeric(summary[q_plan])

        use_plan_variant = summary[v_actual].eq("") & summary[v_plan].ne("")
        summary.loc[use_plan_variant, v_actual] = summary.loc[use_plan_variant, v_plan]

        use_plan_qty = summary[q_actual].isna() & summary[q_plan].notna()
        summary.loc[use_plan_qty, q_actual] = summary.loc[use_plan_qty, q_plan]

        summary[q_actual] = summary[q_actual].fillna(0).astype(int)

    # Lateness from finish_time - due date (both in seconds)
    summary["lateness"] = pd.NA
    due_numeric = safe_numeric(summary["due date"])
    can_compute_lateness = summary["finish_time"].notna() & due_numeric.notna()
    summary.loc[can_compute_lateness, "lateness"] = (
        summary.loc[can_compute_lateness, "finish_time"] - due_numeric.loc[can_compute_lateness]
    )

    finished_week_day = summary["finish_time"].apply(lambda x: seconds_to_week_day(x, DAY_SECONDS, WEEK_DAYS))
    summary["finished_week"] = finished_week_day.apply(lambda x: x[0])
    summary["finished_day"] = finished_week_day.apply(lambda x: x[1])

    final_cols = [
        "order_id",
        "due date",
        "start_time",
        "finish_time",
        "through_put_time",
        "lateness",
        "priority",
        "variant0",
        "quantity0",
        "variant1",
        "quantity1",
        "variant2",
        "quantity2",
        "planned_week",
        "finished_week",
        "planned_day",
        "finished_day",
    ]

    for c in final_cols:
        if c not in summary.columns:
            summary[c] = pd.NA

    summary = summary[final_cols].sort_values("order_id").reset_index(drop=True)
    validate_summary(summary)

    out_path = results_dir / "order_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Output folder does not exist: {OUTPUT_DIR}")
    if not ON_GOING_DIR.exists():
        raise FileNotFoundError(f"on_going folder does not exist: {ON_GOING_DIR}")

    chosen_main = select_main_folder(OUTPUT_DIR)
    main_name = chosen_main.name
    runs = list_run_folders(chosen_main)

    if not runs:
        raise FileNotFoundError(f"No run_* folders found in {chosen_main}")

    print(f"\nProcessing main folder: {main_name}")
    print(f"Found {len(runs)} run folder(s).")

    created = []
    for run_dir in runs:
        results_dir = run_dir / "results"
        if not results_dir.exists():
            print(f"WARNING: No results folder in {run_dir}. Skipping.")
            continue
        try:
            out_path = build_order_summary(results_dir, main_name=main_name, run_name=run_dir.name)
            created.append(out_path)
            print(f"Created: {out_path}")
        except Exception as exc:
            print(f"ERROR in {run_dir}: {exc}")

    print(f"\nDone. Created {len(created)} order_summary.csv file(s) in main folder: {main_name}")


if __name__ == "__main__":
    main()
