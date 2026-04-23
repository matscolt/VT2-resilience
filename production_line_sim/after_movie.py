# this script should take info from transport schedule csv
# then it should use a premade layout png based on the selected layout from the order folder
# it then needs to generate new pngs where the carriers are presented with the unit located on it
# it needs to have a loading bar above a unit when producing
# it needs to have a disruption bar above the station telling which disruption
# when transporting it needs to interpolize between the two station based on the time difference
# which should give a smooth transportion between the 
# -------------------------------------
# THIS IS COPILOTS VERSION

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
After-movie renderer for a 6-station production line simulation.

Features:
- Background layout PNG (your 1_LAYOUT.png)
- Carrier icon PNG (your carrier.png)
- Unit id text centered in carrier (Calibri 12 if available)
- Queue visualization with 7 slots per station:
    - Lowest (bottom-most) available slot used on arrival
    - Units move down when a slot opens (computed from arrival/start intervals)
- Processing visualization (unit shown at process box)
- Transport visualization (unit moves between station centers linearly)
- MP4 export (imageio/ffmpeg); fallback to PNG sequence if unavailable.

Usage:
  python after_movie.py pick-coords
      Opens an interactive window to click coordinates on the layout image.
      Useful for building STATION_GEOMETRY.

  python after_movie.py render --output production_line_after_movie.mp4
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Configuration (edit this)
# ----------------------------
ROOTDIR = Path(__file__).parent

#find the correct order
def find_output_folder(output_path=None, target_timestamp=None):
    if output_path is None:
        output_path = ROOTDIR / "output"

    folders = []

    for name in os.listdir(output_path):
        full_path = output_path / name   

        if not full_path.is_dir():
            continue

        try:
            timestamp_str = name.split("__")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            folders.append((timestamp, full_path))

        except Exception:
            continue

    if not folders:
        raise FileNotFoundError(f"No valid output folders found in {output_path}")

    folders.sort(key=lambda x: x[0])

    if target_timestamp:
        target_dt = datetime.strptime(target_timestamp, "%Y%m%d_%H%M%S")
        closest = min(folders, key=lambda x: abs(x[0] - target_dt))
        return closest[1]

    return folders[-1][1]

picturefolder = "data/Layouts"
BACKGROUND_PNG = os.path.join(picturefolder, "1_LAYOUT.png")
CARRIER_PNG = os.path.join(picturefolder, "carrier.png")

DEFAULT_OUTPUT_MP4 = "production_line_after_movie.mp4"

PRODUCTION_ICON_PNG = "image.png"

STATION_SCHEDULE_CSV = "station_schedule.csv"
TRANSPORT_SCHEDULE_CSV = "transport_schedule.csv"

DEFAULT_OUTPUT_MP4 = "production_line_after_movie.mp4"

# Video settings
FPS = 30

# How many simulation seconds each video frame advances.
# Example:
#   1.0  => each frame = 1 sim-second (video plays at 30x sim speed)
#   0.2  => each frame = 0.2 sim-seconds (slower / smoother)
SIM_SECONDS_PER_FRAME = 0.5

# Icon sizes
CARRIER_SIZE_PX = 28
PRODUCTION_ICON_SIZE_PX = 40  # production icon can be larger if you want

# Text style
FONT_SIZE = 12
TEXT_COLOR = (0, 0, 0, 255)

# Time label
DRAW_TIME_LABEL = True
TIME_LABEL_POS = (20, 20)
TIME_LABEL_COLOR = (255, 255, 255, 255)

# Disruption label style
DISRUPTION_TEXT_COLOR = (255, 80, 80, 255)  # reddish
DISRUPTION_PREFIX = "DISR"

# Loading bar appearance (above production icon)
BAR_W = 50
BAR_H = 8
BAR_GAP = 6
BAR_BG_COLOR = (255, 255, 255, 255)     # white background
BAR_BORDER_COLOR = (255, 255, 255, 255) # white border
BAR_FILL_COLOR = (0, 200, 0, 255)       # green fill

@dataclass(frozen=True)
class StationGeom:
    station_name_in_csv: str
    queue_slots: List[Tuple[int, int]]   # length 7, bottom->top
    process_pos: Tuple[int, int]
    center_pos: Tuple[int, int]


STATION_GEOMETRY_BY_INDEX: Dict[int, StationGeom] = {
    # IMPORTANT:
    # Replace the placeholder coordinates below with your real pixel coordinates.
    #
    # The station_name_in_csv must match EXACTLY the station_name/from_station/to_station fields in your CSV.
    #
    # Example names from your sample:
    # "Station 1: Bottom cover"
    # "Station 2: Drill station"
    #
    1: StationGeom(
        station_name_in_csv="Station 1: Bottom cover",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    2: StationGeom(
        station_name_in_csv="Station 2: Drill station",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    3: StationGeom(
        station_name_in_csv="Station 3: Top cover",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    4: StationGeom(
        station_name_in_csv="Station 4: Inspection",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    4: StationGeom(
        station_name_in_csv="Station 4: Inspection",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    5: StationGeom(
        station_name_in_csv="Station 5: Robot cell",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    5: StationGeom(
        station_name_in_csv="Station 5: Robot cell",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    6: StationGeom(
        station_name_in_csv="Station 6: Packaging",
        queue_slots=[(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],  # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
}


#
# ----------------------------
# Helpers
# ----------------------------

def load_font_calibri_prefer(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try to load Calibri. If not found, fallback to a common font.
    Best: place calibri.ttf next to this script.
    """
    candidates = [
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
    """Paste an RGBA icon centered at center_xy and draw centered text on top."""
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


def draw_loading_bar(
    frame_rgba: Image.Image,
    center_xy: Tuple[float, float],
    progress_0_1: float,
):
    """Draw a white bar above the given center point; fill green according to progress."""
    progress = float(max(0.0, min(1.0, progress_0_1)))
    cx, cy = center_xy

    # Position bar above icon
    x0 = int(round(cx - BAR_W / 2))
    y0 = int(round(cy - PRODUCTION_ICON_SIZE_PX / 2 - BAR_GAP - BAR_H))
    x1 = x0 + BAR_W
    y1 = y0 + BAR_H

    draw = ImageDraw.Draw(frame_rgba)

    # Background
    draw.rectangle([x0, y0, x1, y1], fill=BAR_BG_COLOR, outline=BAR_BORDER_COLOR)

    # Fill
    fill_w = int(round(BAR_W * progress))
    if fill_w > 0:
        draw.rectangle([x0, y0, x0 + fill_w, y1], fill=BAR_FILL_COLOR)


def pick_coords_interactive(image_path: str):
    """Interactive coordinate picker using matplotlib. Click points, read coords from console."""
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
# Core rendering logic
# ----------------------------

@dataclass
class Schedules:
    station_df: pd.DataFrame
    transport_df: pd.DataFrame
    station_name_to_index: Dict[str, int]
    station_index_to_geom: Dict[int, StationGeom]
    station_name_to_geom: Dict[str, StationGeom]
    t_end: float


def load_schedules() -> Schedules:
    station_df = pd.read_csv(STATION_SCHEDULE_CSV)
    transport_df = pd.read_csv(TRANSPORT_SCHEDULE_CSV)

    needed_station = {"unit_id", "station_index", "station_name", "arrival_time_s", "start_time_s", "finish_time_s"}
    needed_transport = {"unit_id", "from_station", "to_station", "start_time_s", "finish_time_s", "transport_time_s"}

    missing_station = needed_station - set(station_df.columns)
    missing_transport = needed_transport - set(transport_df.columns)
    if missing_station:
        raise ValueError(f"station_schedule.csv missing columns: {sorted(missing_station)}")
    if missing_transport:
        raise ValueError(f"transport_schedule.csv missing columns: {sorted(missing_transport)}")

    # Optional disruption columns (recommended)
    # process_time_s and base_process_time_s are used if present
    has_process_time = "process_time_s" in station_df.columns
    has_base_time = "base_process_time_s" in station_df.columns

    if not (has_process_time and has_base_time):
        print("WARNING: station_schedule.csv does not include both process_time_s and base_process_time_s.")
        print("         Disruption time will be treated as 0 and loading bar will span full (finish-start).")

    # station name -> index mapping
    station_name_to_index = (
        station_df[["station_name", "station_index"]]
        .drop_duplicates()
        .set_index("station_name")["station_index"]
        .to_dict()
    )

    index_to_geom = STATION_GEOMETRY_BY_INDEX.copy()
    station_name_to_geom = {g.station_name_in_csv: g for g in index_to_geom.values()}

    t_end = float(np.nanmax([station_df["finish_time_s"].max(), transport_df["finish_time_s"].max()]))

    station_df = station_df.sort_values(["station_index", "arrival_time_s", "unit_id"]).reset_index(drop=True)
    transport_df = transport_df.sort_values(["start_time_s", "unit_id"]).reset_index(drop=True)

    return Schedules(
        station_df=station_df,
        transport_df=transport_df,
        station_name_to_index=station_name_to_index,
        station_index_to_geom=index_to_geom,
        station_name_to_geom=station_name_to_geom,
        t_end=t_end
    )


def get_waiting(station_rows: pd.DataFrame, t: float) -> pd.DataFrame:
    return station_rows[(station_rows["arrival_time_s"] <= t) & (station_rows["start_time_s"] > t)]


def get_processing(station_rows: pd.DataFrame, t: float) -> pd.DataFrame:
    return station_rows[(station_rows["start_time_s"] <= t) & (station_rows["finish_time_s"] > t)]


def get_transports(transport_df: pd.DataFrame, t: float) -> pd.DataFrame:
    return transport_df[(transport_df["start_time_s"] <= t) & (transport_df["finish_time_s"] > t)]


def disruption_and_processing_times(row: pd.Series) -> Tuple[float, float, float]:
    """
    Returns:
      disruption_time, processing_start_time, processing_duration
    Logic:
      disruption_time = max(process_time_s - base_process_time_s, 0)
      disruption interval starts at start_time_s
      normal processing begins at start_time_s + disruption_time
      normal processing duration = finish_time_s - processing_start_time
    """
    ts = float(row["start_time_s"])
    tf = float(row["finish_time_s"])

    if ("process_time_s" in row) and ("base_process_time_s" in row):
        try:
            pt = float(row["process_time_s"])
            bt = float(row["base_process_time_s"])
            disruption = max(pt - bt, 0.0)
        except Exception:
            disruption = 0.0
    else:
        disruption = 0.0

    # Clamp disruption so it can't exceed total available interval
    total = max(tf - ts, 0.0)
    disruption = min(disruption, total)

    proc_start = ts + disruption
    proc_dur = max(tf - proc_start, 0.0)

    return disruption, proc_start, proc_dur


def render_after_movie(output_path: str, fps: int = FPS, sim_seconds_per_frame: float = SIM_SECONDS_PER_FRAME):
    schedules = load_schedules()

    background = Image.open(BACKGROUND_PNG).convert("RGBA")

    carrier_icon = Image.open(CARRIER_PNG).convert("RGBA").resize(
        (CARRIER_SIZE_PX, CARRIER_SIZE_PX), resample=Image.Resampling.LANCZOS
    )

    # Production icon: use image.png if present, else carrier.png
    prod_path = PRODUCTION_ICON_PNG if os.path.exists(PRODUCTION_ICON_PNG) else CARRIER_PNG
    production_icon = Image.open(prod_path).convert("RGBA").resize(
        (PRODUCTION_ICON_SIZE_PX, PRODUCTION_ICON_SIZE_PX), resample=Image.Resampling.LANCZOS
    )

    font = load_font_calibri_prefer(FONT_SIZE)

    station_groups = {idx: df for idx, df in schedules.station_df.groupby("station_index")}

    # Writer
    writer = None
    use_png_fallback = False
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(output_path, fps=fps)
    except Exception as e:
        print("WARNING: Could not create MP4 writer (ffmpeg missing?). Falling back to PNG frames in ./frames/")
        print(f"Reason: {e}")
        use_png_fallback = True
        os.makedirs("frames", exist_ok=True)

    # Geometry warnings
    for idx, geom in schedules.station_index_to_geom.items():
        if geom.process_pos == (0, 0) or geom.center_pos == (0, 0) or any(p == (0, 0) for p in geom.queue_slots):
            print(f"WARNING: Station geometry for station_index={idx} still has (0,0) placeholders.")

    t = 0.0
    frame_idx = 0
    t_end = schedules.t_end

    while t <= t_end + 1e-9:
        frame = background.copy()
        draw = ImageDraw.Draw(frame)

        # ---- Stations: queues + processing + disruption countdown + loading bar ----
        for station_index, geom in schedules.station_index_to_geom.items():
            station_rows = station_groups.get(station_index)
            if station_rows is None or station_rows.empty:
                continue

            # Queue: units arrived but not started
            waiting = get_waiting(station_rows, t).sort_values(["arrival_time_s", "unit_id"])
            for i, (_, row) in enumerate(waiting.iterrows()):
                if i >= len(geom.queue_slots):
                    break
                paste_icon_with_centered_text(
                    frame_rgba=frame,
                    icon_rgba_resized=carrier_icon,
                    center_xy=geom.queue_slots[i],
                    text=str(row["unit_id"]),
                    font=font
                )

            # Processing: units started but not finished
            processing = get_processing(station_rows, t)

            # Draw disruption countdown next to station label if disrupted
            # and draw production icon + loading bar only after disruption ends.
            for k, (_, row) in enumerate(processing.iterrows()):
                unit_id = str(row["unit_id"])
                ts = float(row["start_time_s"])
                tf = float(row["finish_time_s"])

                disruption, proc_start, proc_dur = disruption_and_processing_times(row)
                disruption_end = ts + disruption

                # Label position: explicit or auto near process_pos
                label_pos = geom.label_pos
                if label_pos is None:
                    # Auto: slightly above/right of process box
                    label_pos = (geom.process_pos[0] + 25, geom.process_pos[1] - 35)

                if disruption > 0 and ts <= t < disruption_end:
                    remaining = max(disruption_end - t, 0.0)
                    draw.text(
                        label_pos,
                        f"{DISRUPTION_PREFIX} {remaining:0.1f}s",
                        fill=DISRUPTION_TEXT_COLOR,
                        font=font
                    )
                    # During disruption, we still show the unit sitting in production subbox
                    # (your request: unit png within production subbox)
                    offset = (k * (PRODUCTION_ICON_SIZE_PX * 0.35))
                    paste_icon_with_centered_text(
                        frame_rgba=frame,
                        icon_rgba_resized=production_icon,
                        center_xy=(geom.process_pos[0] + offset, geom.process_pos[1]),
                        text=unit_id,
                        font=font
                    )
                else:
                    # Normal processing: show loading bar filling green
                    # If there was no disruption, proc_start==ts
                    # If proc_dur==0 (edge cases), bar is full.
                    if t < proc_start:
                        progress = 0.0
                    elif proc_dur <= 1e-9:
                        progress = 1.0
                    else:
                        progress = (t - proc_start) / proc_dur

                    offset = (k * (PRODUCTION_ICON_SIZE_PX * 0.35))
                    center = (geom.process_pos[0] + offset, geom.process_pos[1])

                    # Loading bar above the unit (starts only after disruption ended)
                    draw_loading_bar(frame, center_xy=center, progress_0_1=progress)

                    # Production unit icon
                    paste_icon_with_centered_text(
                        frame_rgba=frame,
                        icon_rgba_resized=production_icon,
                        center_xy=center,
                        text=unit_id,
                        font=font
                    )

        # ---- Transports: moving carriers between station centers ----
        moving = get_transports(schedules.transport_df, t)
        if not moving.empty:
            for _, r in moving.iterrows():
                from_name = str(r["from_station"])
                to_name = str(r["to_station"])
                unit_id = str(r["unit_id"])

                from_geom = schedules.station_name_to_geom.get(from_name)
                to_geom = schedules.station_name_to_geom.get(to_name)

                # If names mismatch, attempt mapping via station_name_to_index
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
                    font=font
                )

        # ---- Time label ----
        if DRAW_TIME_LABEL:
            draw.text(TIME_LABEL_POS, f"t = {t:7.2f} s", fill=TIME_LABEL_COLOR, font=font)

        # ---- Write frame ----
        frame_np = np.asarray(frame)
        if use_png_fallback:
            frame.save(os.path.join("frames", f"frame_{frame_idx:06d}.png"))
        else:
            writer.append_data(frame_np)

        frame_idx += 1
        t += sim_seconds_per_frame

    if writer is not None:
        writer.close()

    if use_png_fallback:
        print("Saved PNG frames to ./frames/")
        print("Convert to MP4 with ffmpeg example:")
        print("  ffmpeg -r 30 -i frames/frame_%06d.png -pix_fmt yuv420p output.mp4")

    print(f"Done. Rendered {frame_idx} frames. Output: {output_path}")



# ----------------------------
# CLI
# ----------------------------

def main():
    specific_folder = "20260420_131845"#enter the wanted foldername(only the first timestamp) in the output folder
    
    specific_folder_choice = 0 #yes = 1, no = 0
    
    if specific_folder_choice == 1:
        try:
            folder = find_output_folder(target_timestamp=specific_folder)
            print(f">> Using selected folder: {folder}")
        except Exception as e:
            print(f">> Warning: Could not find requested folder ({e})")
            print(">> Falling back to newest folder instead.")
            folder = find_output_folder()
    else:
        folder = find_output_folder()
    foldername = str(folder).split("\\")[-1].split("__")[0]
    print(f"using folder: {foldername}")
    
    parser = argparse.ArgumentParser(description="Production line after-movie renderer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pick = sub.add_parser("pick-coords", help="Click on the layout to get pixel coordinates")
    p_pick.add_argument("--image", default=BACKGROUND_PNG, help="Path to layout PNG")

    p_render = sub.add_parser("render", help="Render after-movie MP4")
    p_render.add_argument("--output", default=DEFAULT_OUTPUT_MP4, help="Output MP4 filename")
    p_render.add_argument("--fps", type=int, default=FPS, help="Video FPS")
    p_render.add_argument("--sim-seconds-per-frame", type=float, default=SIM_SECONDS_PER_FRAME,
                          help="Simulation seconds advanced per rendered frame")

    args = parser.parse_args()

    if args.cmd == "pick-coords":
        pick_coords_interactive(args.image)
    elif args.cmd == "render":
        render_after_movie(args.output, fps=args.fps, sim_seconds_per_frame=args.sim_seconds_per_frame)



if __name__ == "__main__":
    main()

