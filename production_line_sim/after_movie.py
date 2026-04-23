#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""after_movie.py

Renderer for an "after movie" of a production line simulation.

Folder layout (as you requested):

production_line_sim/
  after_movie.py
  data/Layouts/
    1_LAYOUT.png
    carrier.png
    carrier_loading.png
  output/
    yyyymmdd_hhmmss.../
      station_schedule.csv
      transport_schedule.csv
      movie/
        after_movie.mp4
        frames/
          1.png
          2.png
          ...

What this script does:
- Reads CSVs from ONE chosen output run folder: output/<run_folder>/
- Loads the static background + carrier icons from data/Layouts/
- Renders frames and saves them to output/<run_folder>/movie/frames/1.png, 2.png, ...
- Also writes an MP4 to output/<run_folder>/movie/after_movie.mp4 (requires ffmpeg via imageio)

Animation logic:
- Queue: units with arrival_time_s <= t < start_time_s are shown in queue slots (bottom->top).
- Processing: units with start_time_s <= t < finish_time_s are shown in the production subbox.
- Disruption split (if process_time_s and base_process_time_s exist):
    disruption_time = max(process_time_s - base_process_time_s, 0)
  During disruption: show a countdown text near the station and NO loading bar.
  After disruption: show a white loading bar above the unit, filling green until finish_time_s.
- Transport: units with start_time_s <= t < finish_time_s in transport_schedule.csv move linearly
  from from_station.center_pos to to_station.center_pos.

Run:
  python after_movie.py render --run yyyymmdd_hhmmss

If --run is omitted, the script picks the newest folder inside ./output.

Optional helper:
  python after_movie.py pick-coords
    Click on the layout image to print pixel coordinates.

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Paths (match your tree)
# ----------------------------
ROOTDIR = Path(__file__).resolve().parent
LAYOUTDIR = ROOTDIR / "data" / "Layouts"
OUTPUTDIR = ROOTDIR / "output"

BACKGROUND_PNG = LAYOUTDIR / "1_LAYOUT.png"
CARRIER_PNG = LAYOUTDIR / "carrier.png"
PRODUCTION_ICON_PNG = LAYOUTDIR / "carrier_loading.png"

STATION_SCHEDULE_CSV = "station_schedule.csv"
TRANSPORT_SCHEDULE_CSV = "transport_schedule.csv"


# ----------------------------
# Video / rendering settings
# ----------------------------
FPS = 30
SIM_SECONDS_PER_FRAME = 0.5

CARRIER_SIZE_PX = 28
PRODUCTION_ICON_SIZE_PX = 40

FONT_SIZE = 12
TEXT_COLOR = (0, 0, 0, 255)

DRAW_TIME_LABEL = True
TIME_LABEL_POS = (20, 20)
TIME_LABEL_COLOR = (255, 255, 255, 255)

DISRUPTION_TEXT_COLOR = (255, 80, 80, 255)
DISRUPTION_PREFIX = "DISR"

BAR_W = 50
BAR_H = 8
BAR_GAP = 6
BAR_BG_COLOR = (255, 255, 255, 255)
BAR_BORDER_COLOR = (255, 255, 255, 255)
BAR_FILL_COLOR = (0, 200, 0, 255)


# ----------------------------
# Station geometry
# NOTE: These are FIRST-PASS coordinates (centers) extracted from your provided layout PNG.
# You can tweak them if needed.
# queue_slots order: bottom -> top (lowest on screen first)
# ----------------------------

@dataclass(frozen=True)
class StationGeom:
    station_name_in_csv: str
    queue_slots: List[Tuple[int, int]]
    process_pos: Tuple[int, int]
    center_pos: Tuple[int, int]
    label_pos: Optional[Tuple[int, int]] = None


STATION_GEOMETRY_BY_INDEX: Dict[int, StationGeom] = {
    1: StationGeom(
        station_name_in_csv="Station 1: Bottom cover",
        queue_slots=[(449, 851), (520, 851), (591, 851),
                    (449, 768), (520, 768), (591, 768), (662, 768)],
        process_pos=(662, 851),
        center_pos=(555, 809),
        label_pos=(375, 920),  # near "BOTTOM COVER" text
    ),
    2: StationGeom(
        station_name_in_csv="Station 2: Drill station",
        queue_slots=[(823, 851), (894, 851), (981, 851),
                    (823, 768), (894, 768), (981, 768), (1068, 768)],
        process_pos=(1068, 851),
        center_pos=(945, 809),
        label_pos=(820, 920),  # near "DRILLING STATION" text
    ),
    3: StationGeom(
        station_name_in_csv="Station 3: Top cover",
        queue_slots=[(681, 651), (752, 651), (823, 651),
                    (681, 568), (752, 568), (823, 568), (894, 568)],
        process_pos=(894, 651),
        center_pos=(787, 609),
        label_pos=(700, 430),  # near "TOP COVER" text
    ),
    4: StationGeom(
        station_name_in_csv="Station 4: Inspection",
        queue_slots=[(1174, 652), (1245, 652), (1316, 652),
                    (1174, 559), (1245, 559), (1316, 559), (1387, 559)],
        process_pos=(1387, 652),
        center_pos=(1280, 600),
        label_pos=(1180, 430),  # near "INSPECTION" text
    ),
    5: StationGeom(
        station_name_in_csv="Station 5: Robot cell",
        queue_slots=[(1713, 834),
                    (1713, 751), (1784, 751),
                    (1713, 668), (1784, 668),
                    (1713, 567), (1784, 567)],
        process_pos=(1784, 834),
        center_pos=(1748, 691),
        label_pos=(1600, 430),  # near "ROBOT CELL" text
    ),
    6: StationGeom(
        station_name_in_csv="Station 6: Packaging",
        queue_slots=[(181, 460),
                    (181, 377), (252, 377),
                    (181, 294), (252, 294),
                    (181, 211), (252, 211)],
        process_pos=(252, 460),
        center_pos=(216, 335),
        label_pos=(130, 120),  # near "PACKAGING" text
    ),
}


# ----------------------------
# Folder helpers (THIS is the logic you asked for)
# ----------------------------

def pick_newest_output_run(output_dir: Path) -> Path:
    """Return newest directory inside output_dir (by modification time)."""
    runs = [p for p in output_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run folders found in: {output_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def build_movie_paths(order_dir: Path) -> Tuple[Path, Path, Path]:
    """Create movie/ and movie/frames/ inside order_dir; return (movie_dir, frames_dir, mp4_path)."""
    movie_dir = order_dir / "movie"
    frames_dir = movie_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = movie_dir / "after_movie.mp4"
    return movie_dir, frames_dir, mp4_path


# ----------------------------
# Drawing helpers
# ----------------------------

def load_font_calibri_prefer(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load Calibri (place calibri.ttf next to the script if needed), else fallback."""
    candidates = [
        str(ROOTDIR / "calibri.ttf"),
        str(ROOTDIR / "Calibri.ttf"),
        os.path.join(os.getcwd(), "calibri.ttf"),
        os.path.join(os.getcwd(), "Calibri.ttf"),
        "/usr/share/fonts/truetype/msttcorefonts/Calibri.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def paste_icon_with_centered_text(
    frame_rgba: Image.Image,
    icon_rgba_resized: Image.Image,
    center_xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int, int] = TEXT_COLOR,
):
    """Paste an RGBA icon centered at center_xy and draw centered text."""
    cx, cy = center_xy
    w, h = icon_rgba_resized.size
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))

    frame_rgba.alpha_composite(icon_rgba_resized, dest=(x0, y0))

    draw = ImageDraw.Draw(frame_rgba)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(round(cx - tw / 2))
    ty = int(round(cy - th / 2))
    draw.text((tx, ty), text, font=font, fill=text_color)


def draw_loading_bar(frame_rgba: Image.Image, center_xy: Tuple[float, float], progress_0_1: float):
    """Draw a white bar above the production icon; fill green by progress."""
    progress = float(max(0.0, min(1.0, progress_0_1)))
    cx, cy = center_xy

    x0 = int(round(cx - BAR_W / 2))
    y0 = int(round(cy - PRODUCTION_ICON_SIZE_PX / 2 - BAR_GAP - BAR_H))
    x1 = x0 + BAR_W
    y1 = y0 + BAR_H

    draw = ImageDraw.Draw(frame_rgba)
    draw.rectangle([x0, y0, x1, y1], fill=BAR_BG_COLOR, outline=BAR_BORDER_COLOR)

    fill_w = int(round(BAR_W * progress))
    if fill_w > 0:
        draw.rectangle([x0, y0, x0 + fill_w, y1], fill=BAR_FILL_COLOR)


def pick_coords_interactive(image_path: Path):
    """Click on the layout to print pixel coordinates."""
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)

    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.set_title("Click points; close window when done. Coordinates print in console.")
    ax.axis("off")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print(f"Clicked: ({x}, {y})")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)


# ----------------------------
# Schedule + simulation helpers
# ----------------------------

@dataclass
class Schedules:
    station_df: pd.DataFrame
    transport_df: pd.DataFrame
    station_name_to_index: Dict[str, int]
    station_index_to_geom: Dict[int, StationGeom]
    station_name_to_geom: Dict[str, StationGeom]
    t_end: float


def load_schedules(order_dir: Path) -> Schedules:
    station_path = order_dir / STATION_SCHEDULE_CSV
    transport_path = order_dir / TRANSPORT_SCHEDULE_CSV

    station_df = pd.read_csv(station_path)
    transport_df = pd.read_csv(transport_path)

    needed_station = {"unit_id", "station_index", "station_name", "arrival_time_s", "start_time_s", "finish_time_s"}
    needed_transport = {"unit_id", "from_station", "to_station", "start_time_s", "finish_time_s", "transport_time_s"}

    ms = needed_station - set(station_df.columns)
    mt = needed_transport - set(transport_df.columns)
    if ms:
        raise ValueError(f"{station_path.name} missing columns: {sorted(ms)}")
    if mt:
        raise ValueError(f"{transport_path.name} missing columns: {sorted(mt)}")

    station_name_to_index = (
        station_df[["station_name", "station_index"]]
        .drop_duplicates()
        .set_index("station_name")["station_index"]
        .to_dict()
    )

    station_index_to_geom = STATION_GEOMETRY_BY_INDEX
    station_name_to_geom = {g.station_name_in_csv: g for g in station_index_to_geom.values()}

    t_end = float(np.nanmax([station_df["finish_time_s"].max(), transport_df["finish_time_s"].max()]))

    station_df = station_df.sort_values(["station_index", "arrival_time_s", "unit_id"]).reset_index(drop=True)
    transport_df = transport_df.sort_values(["start_time_s", "unit_id"]).reset_index(drop=True)

    return Schedules(
        station_df=station_df,
        transport_df=transport_df,
        station_name_to_index=station_name_to_index,
        station_index_to_geom=station_index_to_geom,
        station_name_to_geom=station_name_to_geom,
        t_end=t_end,
    )


def get_waiting(station_rows: pd.DataFrame, t: float) -> pd.DataFrame:
    return station_rows[(station_rows["arrival_time_s"] <= t) & (station_rows["start_time_s"] > t)]


def get_processing(station_rows: pd.DataFrame, t: float) -> pd.DataFrame:
    return station_rows[(station_rows["start_time_s"] <= t) & (station_rows["finish_time_s"] > t)]


def get_transports(transport_df: pd.DataFrame, t: float) -> pd.DataFrame:
    return transport_df[(transport_df["start_time_s"] <= t) & (transport_df["finish_time_s"] > t)]


def disruption_and_processing_times(row: pd.Series) -> Tuple[float, float, float]:
    """Compute disruption time and the normal-processing segment for loading bar."""
    ts = float(row["start_time_s"])
    tf = float(row["finish_time_s"])

    total = max(tf - ts, 0.0)

    if "process_time_s" in row.index and "base_process_time_s" in row.index:
        try:
            pt = float(row["process_time_s"])
            bt = float(row["base_process_time_s"])
            disruption = max(pt - bt, 0.0)
        except Exception:
            disruption = 0.0
    else:
        disruption = 0.0

    disruption = min(disruption, total)
    proc_start = ts + disruption
    proc_dur = max(tf - proc_start, 0.0)

    return disruption, proc_start, proc_dur


# ----------------------------
# Rendering
# ----------------------------

def render_after_movie(order_dir: Path, fps: int = FPS, sim_seconds_per_frame: float = SIM_SECONDS_PER_FRAME):
    order_dir = order_dir.resolve()
    if not order_dir.exists():
        raise FileNotFoundError(f"Order folder not found: {order_dir}")

    # Create output folders: output/<run>/movie and output/<run>/movie/frames
    _, frames_dir, mp4_path = build_movie_paths(order_dir)

    schedules = load_schedules(order_dir)

    # Load images
    background = Image.open(BACKGROUND_PNG).convert("RGBA")

    carrier_icon = Image.open(CARRIER_PNG).convert("RGBA").resize(
        (CARRIER_SIZE_PX, CARRIER_SIZE_PX), resample=Image.Resampling.LANCZOS
    )

    prod_icon_path = PRODUCTION_ICON_PNG if PRODUCTION_ICON_PNG.exists() else CARRIER_PNG
    production_icon = Image.open(prod_icon_path).convert("RGBA").resize(
        (PRODUCTION_ICON_SIZE_PX, PRODUCTION_ICON_SIZE_PX), resample=Image.Resampling.LANCZOS
    )

    font = load_font_calibri_prefer(FONT_SIZE)

    # Group for speed
    station_groups = {idx: df for idx, df in schedules.station_df.groupby("station_index")}

    # MP4 writer (still save frames even if MP4 fails)
    writer = None
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(mp4_path), fps=fps)
    except Exception as e:
        print("WARNING: Could not create MP4 writer. Frames will still be saved.")
        print("Reason:", e)

    t = 0.0
    frame_idx = 0

    while t <= schedules.t_end + 1e-9:
        frame = background.copy()
        draw = ImageDraw.Draw(frame)

        # --- Stations: queue + processing + disruption countdown + loading bar ---
        for station_index, geom in schedules.station_index_to_geom.items():
            station_rows = station_groups.get(station_index)
            if station_rows is None or station_rows.empty:
                continue

            # Queue
            waiting = get_waiting(station_rows, t).sort_values(["arrival_time_s", "unit_id"])
            for i, (_, row) in enumerate(waiting.iterrows()):
                if i >= len(geom.queue_slots):
                    break
                paste_icon_with_centered_text(
                    frame_rgba=frame,
                    icon_rgba_resized=carrier_icon,
                    center_xy=geom.queue_slots[i],
                    text=str(row["unit_id"]),
                    font=font,
                )

            # Processing
            processing = get_processing(station_rows, t)
            for k, (_, row) in enumerate(processing.iterrows()):
                unit_id = str(row["unit_id"])
                disruption, proc_start, proc_dur = disruption_and_processing_times(row)
                ts = float(row["start_time_s"])
                disruption_end = ts + disruption

                # position for the unit (in case of parallel processing draw slight offset)
                offset = k * (PRODUCTION_ICON_SIZE_PX * 0.35)
                center = (geom.process_pos[0] + offset, geom.process_pos[1])

                # During disruption: show countdown; no loading bar
                if disruption > 0 and ts <= t < disruption_end:
                    remaining = max(disruption_end - t, 0.0)
                    label_pos = geom.label_pos or (int(center[0] + 25), int(center[1] - 35))
                    draw.text(
                        label_pos,
                        f"{DISRUPTION_PREFIX} {remaining:0.1f}s",
                        fill=DISRUPTION_TEXT_COLOR,
                        font=font,
                    )
                    paste_icon_with_centered_text(
                        frame_rgba=frame,
                        icon_rgba_resized=production_icon,
                        center_xy=center,
                        text=unit_id,
                        font=font,
                    )
                else:
                    # Normal processing: loading bar fills from proc_start -> finish
                    if t < proc_start:
                        progress = 0.0
                    elif proc_dur <= 1e-9:
                        progress = 1.0
                    else:
                        progress = (t - proc_start) / proc_dur

                    draw_loading_bar(frame, center_xy=center, progress_0_1=progress)
                    paste_icon_with_centered_text(
                        frame_rgba=frame,
                        icon_rgba_resized=production_icon,
                        center_xy=center,
                        text=unit_id,
                        font=font,
                    )

        # --- Transports ---
        moving = get_transports(schedules.transport_df, t)
        if not moving.empty:
            for _, r in moving.iterrows():
                from_name = str(r["from_station"])
                to_name = str(r["to_station"])
                unit_id = str(r["unit_id"])

                from_geom = schedules.station_name_to_geom.get(from_name)
                to_geom = schedules.station_name_to_geom.get(to_name)

                # fallback using station schedule mapping
                if from_geom is None and from_name in schedules.station_name_to_index:
                    from_geom = schedules.station_index_to_geom.get(schedules.station_name_to_index[from_name])
                if to_geom is None and to_name in schedules.station_name_to_index:
                    to_geom = schedules.station_index_to_geom.get(schedules.station_name_to_index[to_name])

                if from_geom is None or to_geom is None:
                    continue

                x0, y0 = from_geom.center_pos
                x1, y1 = to_geom.center_pos

                ts = float(r["start_time_s"])
                tf = float(r["finish_time_s"])
                denom = max(tf - ts, 1e-9)
                alpha = max(0.0, min(1.0, (t - ts) / denom))

                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)

                paste_icon_with_centered_text(
                    frame_rgba=frame,
                    icon_rgba_resized=carrier_icon,
                    center_xy=(x, y),
                    text=unit_id,
                    font=font,
                )

        # --- Time label ---
        if DRAW_TIME_LABEL:
            draw.text(TIME_LABEL_POS, f"t = {t:7.2f} s", fill=TIME_LABEL_COLOR, font=font)

        # --- Save frame to output/<run>/movie/frames/<n>.png ---
        frame_path = frames_dir / f"{frame_idx + 1}.png"
        frame.save(frame_path)

        # --- Append to MP4 ---
        if writer is not None:
            writer.append_data(np.asarray(frame))

        frame_idx += 1
        t += sim_seconds_per_frame

    if writer is not None:
        writer.close()

    print(f"Rendered {frame_idx} frames")
    print(f"Frames saved in: {frames_dir}")
    print(f"MP4 saved to:    {mp4_path}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="After-movie renderer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pick = sub.add_parser("pick-coords", help="Click on the layout image to print pixel coordinates")
    p_pick.add_argument("--image", default=str(BACKGROUND_PNG), help="Path to layout PNG")

    p_render = sub.add_parser("render", help="Render MP4 + PNG frames")
    p_render.add_argument("--run", default=None, help="Run folder name inside ./output (e.g. 20260420_131845)")
    p_render.add_argument("--fps", type=int, default=FPS)
    p_render.add_argument("--sim-seconds-per-frame", type=float, default=SIM_SECONDS_PER_FRAME)

    args = parser.parse_args()

    if args.cmd == "pick-coords":
        pick_coords_interactive(Path(args.image))
        return

    if args.cmd == "render":
        if args.run is None:
            order_dir = pick_newest_output_run(OUTPUTDIR)
            print(f"--run not provided, using newest folder: {order_dir.name}")
        else:
            order_dir = OUTPUTDIR / args.run

        render_after_movie(order_dir, fps=args.fps, sim_seconds_per_frame=args.sim_seconds_per_frame)


if __name__ == "__main__":
    main()
