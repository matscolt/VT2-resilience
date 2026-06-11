from __future__ import annotations

import math
import csv
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
POST_PROCESSING_DIR = Path(__file__).resolve().parent / "RESULTS" / "post_processing"

DAY_SECONDS = 8 * 60 * 60   # 8 hours per day
WEEK_DAYS = 5               # 5 days per week

# Order fitness settings
ALPHA = 0.05
BETA = 0.001
GAMMA = 1.5
DELTA = 1000
TIME_SCALE = 60 * 60  # seconds -> hours

# Output KPI filenames per layout
LAYOUT_1_FILENAME = "KPIs_layout_2_2_5_2_2_2.csv"
LAYOUT_2_FILENAME = "KPIs_layout_2_3_5_2_3_2.csv"


# ============================================================
# Helpers
# ============================================================

def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def build_col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {norm(c): c for c in df.columns}


def find_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> Optional[str]:
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
    if pd.isna(seconds_value):
        return None, None
    day_index = int(float(seconds_value) // day_seconds)
    week = day_index // week_days + 1
    day = day_index % week_days + 1
    return week, day


def _parse_main_folder_name(name: str, assumed_year: int) -> Optional[Tuple[datetime, int]]:
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


def run_number(path: Path) -> int:
    m = re.match(r"run_(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def list_run_folders(main_output_dir: Path) -> List[Path]:
    runs = [p for p in main_output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=run_number)
    return runs


def find_production_plan_file(main_name: str, run_name: str) -> Path:
    run_dir = ON_GOING_DIR / main_name / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"on_going run folder not found: {run_dir}")

    plans = sorted(run_dir.glob("production_plan*.csv"))
    if not plans:
        raise FileNotFoundError(f"No production_plan*.csv found in {run_dir}")
    if len(plans) > 1:
        print(f"WARNING: Multiple production plans found in {run_dir}. Using: {plans[0].name}")
    return plans[0]


def read_kpi_summary(kpi_path: Path) -> Dict[str, object]:
    if not kpi_path.exists():
        raise FileNotFoundError(f"Missing KPI file: {kpi_path}")

    df = pd.read_csv(kpi_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"KPI file does not look like a key/value csv: {kpi_path}")

    keys = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1].tolist()

    record: Dict[str, object] = {}
    for k, v in zip(keys, values):
        try:
            numeric_v = float(v)
            if numeric_v.is_integer():
                record[k] = int(numeric_v)
            else:
                record[k] = numeric_v
        except Exception:
            record[k] = v
    return record


def write_layout_kpi_csv(main_name: str, runs: List[Path], output_filename: str) -> Optional[Path]:
    """
    Build one layout KPI aggregation CSV.

    Output shape:
      - rows = KPI names
      - columns = run names (run_1, run_2, ...)
      - extra column = average across numeric run values
    """
    if not runs:
        print(f"WARNING: No runs available for {output_filename}. Skipping.")
        return None

    kpi_records = {}
    run_names: List[str] = []
    for run_dir in runs:
        kpi_path = run_dir / "results" / "kpi_summary.csv"
        record = read_kpi_summary(kpi_path)
        kpi_records[run_dir.name] = record
        run_names.append(run_dir.name)

    df = pd.DataFrame(kpi_records)
    df.index.name = "kpi"

    numeric_part = df[run_names].apply(pd.to_numeric, errors="coerce")
    df["average"] = numeric_part.mean(axis=1, skipna=True)

    main_post_dir = POST_PROCESSING_DIR / main_name
    main_post_dir.mkdir(parents=True, exist_ok=True)
    out_path = main_post_dir / output_filename
    df.to_csv(out_path)
    return out_path


def create_layout_kpi_csvs(main_name: str, runs: List[Path]) -> List[Path]:
    if not runs:
        return []

    ordered_runs = sorted(runs, key=run_number)
    layout_1_runs = ordered_runs[:5]
    layout_2_runs = ordered_runs[5:10]

    created: List[Path] = []
    out_1 = write_layout_kpi_csv(main_name, layout_1_runs, LAYOUT_1_FILENAME)
    if out_1 is not None:
        created.append(out_1)
    out_2 = write_layout_kpi_csv(main_name, layout_2_runs, LAYOUT_2_FILENAME)
    if out_2 is not None:
        created.append(out_2)
    return created


def extract_plan_metadata(plan_df: pd.DataFrame) -> pd.DataFrame:
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

    variants = variants[["order_id", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2"]].drop_duplicates(subset=["order_id"])
    return variants


def build_actual_variant_summary(unit_df: pd.DataFrame, order_col: str, variant_col: str) -> pd.DataFrame:
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
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["order_id", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2"])
    return pd.DataFrame(rows)


def compute_order_fitness_row(due_date: object, priority: object, finish_time: object) -> Optional[float]:
    if pd.isna(due_date) or pd.isna(priority) or pd.isna(finish_time):
        return None

    due = float(due_date)
    priority_int = max(1, int(float(priority)))
    completion = float(finish_time)

    w = float(priority_int) ** float(GAMMA)
    lateness_s = completion - due
    tardiness_s = max(0.0, lateness_s)
    earliness_s = max(0.0, -lateness_s)

    T_hours = tardiness_s / float(TIME_SCALE)
    E_hours = earliness_s / float(TIME_SCALE)

    exp_term = math.exp(float(ALPHA) * T_hours) - 1.0
    weighted_exp_tardiness = float(DELTA) * w * exp_term
    weighted_earliness_reward = float(BETA) * w * E_hours

    return weighted_exp_tardiness - weighted_earliness_reward


def update_total_fitness_kpi(order_summary_path: Path) -> Optional[Path]:
    order_summary_path = Path(order_summary_path)
    if not order_summary_path.exists():
        return None

    order_df = pd.read_csv(order_summary_path)
    fitness_col = find_col(order_df, ["fitness"], required=False)
    total_fitness = 0.0
    if fitness_col is not None:
        total_fitness = float(safe_numeric(order_df[fitness_col]).fillna(0).sum())

    kpi_path = order_summary_path.with_name("kpi_summary.csv")
    rows: List[List[str]] = []
    header = None

    if kpi_path.exists():
        with kpi_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(pd.read_csv(f, header=None, dtype=str, keep_default_na=False).itertuples(index=False, name=None))
            rows = [list(row) for row in rows]

    if rows:
        first_row_norm = [norm(cell) for cell in rows[0][:2]]
        if len(first_row_norm) >= 2 and first_row_norm[0] in {"kpi", "metric", "name"} and first_row_norm[1] in {"value", "val", "result"}:
            header = rows.pop(0)
        updated = False
        for row in rows:
            if row and norm(row[0]) == "total_fitness":
                if len(row) < 2:
                    row.append(str(total_fitness))
                else:
                    row[1] = str(total_fitness)
                updated = True
                break
        if not updated:
            rows.append(["total fitness", str(total_fitness)])
    else:
        header = ["kpi", "value"]
        rows = [["total fitness", str(total_fitness)]]

    with kpi_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)

    return kpi_path


def validate_summary(summary: pd.DataFrame) -> None:
    if summary["order_id"].duplicated().any():
        dupes = summary.loc[summary["order_id"].duplicated(), "order_id"].tolist()
        raise ValueError(f"Duplicate order_id values found in output summary: {dupes[:10]}")


def build_order_summary(results_dir: Path, main_name: str, run_name: str) -> Path:
    unit_path = results_dir / "unit_summary.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing {unit_path}")

    plan_path = find_production_plan_file(main_name, run_name)

    unit_df = pd.read_csv(unit_path)
    plan_df = pd.read_csv(plan_path)

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

    timing = (
        unit_df.groupby(unit_order_col, dropna=False)
        .agg(start_time=(unit_start_col, "min"), finish_time=(unit_finish_col, "max"))
        .reset_index()
        .rename(columns={unit_order_col: "order_id"})
    )
    timing["through_put_time"] = timing["finish_time"] - timing["start_time"]

    actual_variants = build_actual_variant_summary(unit_df, unit_order_col, unit_variant_col)
    plan_meta = extract_plan_metadata(plan_df)
    plan_variants = extract_plan_variants(plan_df)

    summary = plan_meta.merge(timing, on="order_id", how="left")
    summary = summary.merge(plan_variants, on="order_id", how="left", suffixes=("", "_plan"))
    summary = summary.merge(actual_variants, on="order_id", how="left", suffixes=("_plan", ""))

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

    summary["lateness"] = pd.NA
    due_numeric = safe_numeric(summary["due date"])
    can_compute_lateness = summary["finish_time"].notna() & due_numeric.notna()
    summary.loc[can_compute_lateness, "lateness"] = summary.loc[can_compute_lateness, "finish_time"] - due_numeric.loc[can_compute_lateness]

    finished_week_day = summary["finish_time"].apply(lambda x: seconds_to_week_day(x, DAY_SECONDS, WEEK_DAYS))
    summary["finished_week"] = finished_week_day.apply(lambda x: x[0])
    summary["finished_day"] = finished_week_day.apply(lambda x: x[1])

    summary["fitness"] = summary.apply(lambda row: compute_order_fitness_row(row["due date"], row["priority"], row["finish_time"]), axis=1)

    final_cols = [
        "order_id", "due date", "start_time", "finish_time", "through_put_time", "lateness", "fitness",
        "priority", "variant0", "quantity0", "variant1", "quantity1", "variant2", "quantity2",
        "planned_week", "finished_week", "planned_day", "finished_day",
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

    #chosen_main = select_main_folder(OUTPUT_DIR)
    candidates = list_main_folders(OUTPUT_DIR)
    for chosen_main in candidates:
        main_name = chosen_main.name
        runs = list_run_folders(chosen_main)
        if not runs:
            raise FileNotFoundError(f"No run_* folders found in {chosen_main}")

        print(f"\nProcessing main folder: {main_name}")
        print(f"Found {len(runs)} run folder(s).")

        created_order_summaries: List[Path] = []
        for run_dir in runs:
            results_dir = run_dir / "results"
            if not results_dir.exists():
                print(f"WARNING: No results folder in {run_dir}. Skipping.")
                continue
            try:
                out_path = build_order_summary(results_dir, main_name=main_name, run_name=run_dir.name)
                update_total_fitness_kpi(out_path)
                created_order_summaries.append(out_path)
                print(f"Created order summary: {out_path}")
            except Exception as exc:
                print(f"ERROR building order summary in {run_dir}: {exc}")

        created_kpi_csvs = create_layout_kpi_csvs(main_name, runs)
        for path in created_kpi_csvs:
            print(f"Created KPI summary: {path}")

        print(f"\nDone. Created {len(created_order_summaries)} order_summary.csv file(s) in main folder: {main_name}")
        print(f"Created {len(created_kpi_csvs)} layout KPI CSV file(s) in post_processing/{main_name}")


if __name__ == "__main__":
    main()
