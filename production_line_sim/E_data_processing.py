from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ============================================================
# Hardcoded configuration
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "RESULTS" / "output"
DAY_SECONDS = 8 * 60 * 60   # 8 hours per day
WEEK_DAYS = 5               # 5 days per week
DUE_DATE_SCALE = 1.0        # due date already in seconds


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
    """
    Convert simulation seconds to (week, day), both 1-based.

    With the hardcoded defaults:
    - 1 day = 8 hours = 28,800 seconds
    - 1 week = 5 days
    """
    if pd.isna(seconds_value):
        return None, None

    day_index = int(float(seconds_value) // day_seconds)
    week = day_index // week_days + 1
    day = day_index % week_days + 1
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
    """
    Show a numbered list like:

    Available main folders in ./output (newest first):
    1) main_...
    2) main_...

    Choose main folder number (Enter = 1):
    """
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


def find_metadata_file(results_dir: Path) -> Optional[Path]:
    """
    Optional helper: look for a results-side order/plan-like CSV that may contain
    due date / priority / planned week / planned day.

    This script intentionally does NOT use the input/ tree.
    """
    patterns = [
        "*order*.csv",
        "*orders*.csv",
        "*plan*.csv",
        "*production*.csv",
    ]
    unit_name = "unit_summary.csv"
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend([p for p in results_dir.glob(pattern) if p.name != unit_name])

    # stable de-duplication
    seen = set()
    unique = []
    for p in sorted(matches):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[0] if unique else None


def extract_optional_order_metadata(results_dir: Path) -> pd.DataFrame:
    """
    Try to extract order-level metadata from a results-side CSV if one exists.
    If not found, return an empty dataframe with the right columns.
    """
    meta_file = find_metadata_file(results_dir)
    desired_cols = ["order_id", "due date", "priority", "planned_week", "planned_day"]

    if meta_file is None:
        return pd.DataFrame(columns=desired_cols)

    try:
        df = pd.read_csv(meta_file)
    except Exception as exc:
        print(f"WARNING: Could not read metadata file {meta_file}: {exc}")
        return pd.DataFrame(columns=desired_cols)

    try:
        order_col = find_col(df, ["order_id", "orderID"])
    except KeyError:
        print(f"WARNING: Metadata file {meta_file.name} does not contain order_id. Ignoring it.")
        return pd.DataFrame(columns=desired_cols)

    due_col = find_col(df, ["due date", "due_date", "due"], required=False)
    priority_col = find_col(df, ["priority"], required=False)
    planned_week_col = find_col(df, ["planned_week", "planned week"], required=False)
    planned_day_col = find_col(df, ["planned_day", "planned day"], required=False)

    keep = {order_col: "order_id"}
    if due_col:
        keep[due_col] = "due date"
    if priority_col:
        keep[priority_col] = "priority"
    if planned_week_col:
        keep[planned_week_col] = "planned_week"
    if planned_day_col:
        keep[planned_day_col] = "planned_day"

    meta = df[list(keep.keys())].rename(columns=keep).copy()
    meta["order_id"] = safe_numeric(meta["order_id"]).astype("Int64")
    meta = meta.dropna(subset=["order_id"]).copy()
    meta["order_id"] = meta["order_id"].astype(int)

    for col in ["due date", "priority", "planned_week", "planned_day"]:
        if col in meta.columns:
            meta[col] = safe_numeric(meta[col])
        else:
            meta[col] = pd.NA

    meta = meta[["order_id", "due date", "priority", "planned_week", "planned_day"]]
    meta = meta.drop_duplicates(subset=["order_id"])
    return meta


def validate_summary(summary: pd.DataFrame) -> None:
    """Light sanity checks before writing the output CSV."""
    if summary["order_id"].duplicated().any():
        dupes = summary.loc[summary["order_id"].duplicated(), "order_id"].tolist()
        raise ValueError(f"Duplicate order_id values found in output summary: {dupes[:10]}")

    for col in ["quantity0", "quantity1", "quantity2"]:
        if (pd.to_numeric(summary[col], errors="coerce").fillna(0) < 0).any():
            raise ValueError(f"Negative quantities found in {col}.")

    mask = summary["finish_time"].notna() & summary["start_time"].notna()
    bad = summary.loc[mask & (summary["finish_time"] < summary["start_time"])]
    if not bad.empty:
        bad_orders = bad["order_id"].tolist()[:10]
        raise ValueError(
            f"Found orders where finish_time < start_time. Example order_id values: {bad_orders}"
        )


def build_order_summary(results_dir: Path) -> Path:
    """
    Create order_summary.csv using results-side data only.

    Required source:
      - unit_summary.csv

    Optional metadata source (if present in results_dir):
      - any CSV with order_id plus due date / priority / planned week / planned day

    If optional metadata is not found, the columns are still created but left blank.
    """
    unit_path = results_dir / "unit_summary.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing {unit_path}")

    unit_df = pd.read_csv(unit_path)

    # --- Required unit_summary columns ---
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

    # --- Aggregate timing per order ---
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

    # --- Count variants per order ---
    counts = (
        unit_df.groupby([unit_order_col, unit_variant_col], dropna=False)
        .size()
        .reset_index(name="qty")
        .rename(columns={unit_order_col: "order_id", unit_variant_col: "variant"})
    )

    variant_rows = []
    for order_id, grp in counts.groupby("order_id"):
        grp = grp.sort_values("variant", kind="stable").reset_index(drop=True)
        row = {"order_id": int(order_id)}

        for i in range(3):
            if i < len(grp):
                row[f"variant{i}"] = grp.loc[i, "variant"]
                row[f"quantity{i}"] = int(grp.loc[i, "qty"])
            else:
                row[f"variant{i}"] = ""
                row[f"quantity{i}"] = 0

        if len(grp) > 3:
            extras = ", ".join(grp.loc[3:, "variant"].astype(str).tolist())
            print(f"WARNING: order {order_id} has more than 3 variants. Extra variants ignored: {extras}")

        variant_rows.append(row)

    variant_df = pd.DataFrame(variant_rows)

    # --- Optional metadata from results-side files only ---
    order_meta = extract_optional_order_metadata(results_dir)
    if not order_meta.empty and "due date" in order_meta.columns:
        order_meta["due date"] = safe_numeric(order_meta["due date"]) * DUE_DATE_SCALE

    # --- Merge everything ---
    summary = timing.merge(variant_df, on="order_id", how="outer")
    summary = summary.merge(order_meta, on="order_id", how="left")

    # Ensure columns exist even when metadata is missing
    for i in range(3):
        vcol = f"variant{i}"
        qcol = f"quantity{i}"
        if vcol not in summary.columns:
            summary[vcol] = ""
        if qcol not in summary.columns:
            summary[qcol] = 0
        summary[vcol] = summary[vcol].fillna("")
        summary[qcol] = safe_numeric(summary[qcol]).fillna(0).astype(int)

    for col in ["due date", "priority", "planned_week", "planned_day"]:
        if col not in summary.columns:
            summary[col] = pd.NA

    # Derived columns
    summary["lateness"] = pd.NA
    due_numeric = safe_numeric(summary["due date"])
    can_compute_lateness = summary["finish_time"].notna() & due_numeric.notna()
    summary.loc[can_compute_lateness, "lateness"] = (
        summary.loc[can_compute_lateness, "finish_time"] - due_numeric.loc[can_compute_lateness]
    )

    finished_week_day = summary["finish_time"].apply(
        lambda x: seconds_to_week_day(x, DAY_SECONDS, WEEK_DAYS)
    )
    summary["finished_week"] = finished_week_day.apply(lambda x: x[0])
    summary["finished_day"] = finished_week_day.apply(lambda x: x[1])

    # Final column order requested by user
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

    chosen_main = select_main_folder(OUTPUT_DIR)
    runs = list_run_folders(chosen_main)

    if not runs:
        raise FileNotFoundError(f"No run_* folders found in {chosen_main}")

    print(f"\nProcessing main folder: {chosen_main.name}")
    print(f"Found {len(runs)} run folder(s).")

    created = []
    for run_dir in runs:
        results_dir = run_dir / "results"
        if not results_dir.exists():
            print(f"WARNING: No results folder in {run_dir}. Skipping.")
            continue
        try:
            out_path = build_order_summary(results_dir)
            created.append(out_path)
            print(f"Created: {out_path}")
        except Exception as exc:
            print(f"ERROR in {run_dir}: {exc}")

    print(f"\nDone. Created {len(created)} order_summary.csv file(s) in main folder: {chosen_main.name}")

    if created:
        missing_meta = []
        for p in created:
            try:
                df = pd.read_csv(p)
                if (
                    df["due date"].isna().all()
                    and df["priority"].isna().all()
                    and df["planned_week"].isna().all()
                    and df["planned_day"].isna().all()
                ):
                    missing_meta.append(p)
            except Exception:
                pass
        if missing_meta:
            print(
                "\nNOTE: Some metadata columns (due date / priority / planned week / planned day) were left blank "
                "because no results-side metadata file containing those columns was found."
            )


if __name__ == "__main__":
    main()
