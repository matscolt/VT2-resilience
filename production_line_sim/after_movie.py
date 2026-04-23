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

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Configuration (edit this)
# ----------------------------

BACKGROUND_PNG = "1_LAYOUT.png"
CARRIER_PNG = "carrier.png"

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

# Carrier icon size in pixels (tweak to fit your boxes)
CARRIER_SIZE_PX = 28

# Text style: Calibri 12 if available, else fallback
FONT_SIZE = 12

# Text color in the white circle
TEXT_COLOR = (0, 0, 0, 255)

# Optional: draw time label
DRAW_TIME_LABEL = True
TIME_LABEL_POS = (20, 20)
TIME_LABEL_COLOR = (255, 255, 255, 255)

# Optional: show station processing as a faint green glow behind the carrier
DRAW_PROCESS_GLOW = False
PROCESS_GLOW_COLOR = (0, 255, 0, 120)
PROCESS_GLOW_RADIUS = 18


# ----------------------------
# Station geometry (YOU MUST FILL THIS)
# ----------------------------
# Provide pixel coordinates (x, y) in the layout PNG for:
# - queue_slots: list of 7 positions, ORDERED bottom->top (lowest on screen first)
# - process_pos: position of the production subbox center
# - center_pos: station center for transport movement endpoints
#
# Tip: run `python after_movie.py pick-coords` and click the points you need.

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
        queue_slots=[(0, 0)] * 7,      # <-- fill
        process_pos=(0, 0),            # <-- fill
        center_pos=(0, 0),             # <-- fill
    ),
    2: StationGeom(
        station_name_in_csv="Station 2: Drill station",
        queue_slots=[(0, 0)] * 7,
        process_pos=(0, 0),
        center_pos=(0, 0),
    ),
    3: StationGeom(
        station_name_in_csv="Station 3: Top cover",
        queue_slots=[(0, 0)] * 7,
        process_pos=(0, 0),
        center_pos=(0, 0),
    ),
    4: StationGeom(
        station_name_in_csv="Station 4: Inspection",
        queue_slots=[(0, 0)] * 7,
        process_pos=(0, 0),
        center_pos=(0, 0),
    ),
    5: StationGeom(
        station_name_in_csv="Station 5: Robot cell",
        queue_slots=[(0, 0)] * 7,
        process_pos=(0, 0),
        center_pos=(0, 0),
    ),
    6: StationGeom(
        station_name_in_csv="Station 6: Packaging",
        queue_slots=[(0, 0)] * 7,
        process_pos=(0, 0),
        center_pos=(0, 0),
    ),
}


# ----------------------------
# Helpers
# ----------------------------

def load_font_calibri_prefer(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try to load Calibri. If not found, fallback to common fonts.
    For best results: place 'calibri.ttf' in the script folder.
    """
    candidates = [
        os.path.join(os.getcwd(), "calibri.ttf"),
        os.path.join(os.getcwd(), "Calibri.ttf"),
        # Common Linux paths (Calibri usually not installed)
        "/usr/share/fonts/truetype/msttcorefonts/Calibri.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/calibri.ttf",
        "/usr/share/fonts/truetype/microsoft/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    # Final fallback
    return ImageFont.load_default()


def paste_carrier_with_text(
    frame_rgba: Image.Image,
    carrier_rgba_resized: Image.Image,
    center_xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int, int] = TEXT_COLOR,
):
    """
    Paste carrier icon centered at center_xy and draw centered text on top.
    """
    cx, cy = center_xy
    w, h = carrier_rgba_resized.size
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))

    # Paste with alpha
    frame_rgba.alpha_composite(carrier_rgba_resized, dest=(x0, y0))

    # Draw centered text
    draw = ImageDraw.Draw(frame_rgba)

    # Compute text box in pixels
    # Using textbbox for better centering across fonts
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = int(round(cx - tw / 2))
    ty = int(round(cy - th / 2))

    draw.text((tx, ty), text, font=font, fill=text_color)


def pick_coords_interactive(image_path: str):
    """
    Interactive coordinate picker using matplotlib.
    Click points on the image; printed coordinates can be copied into STATION_GEOMETRY.
    """
    import matplotlib.pyplot as plt  # local import by design

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

    # Basic sanity: ensure expected columns exist
    needed_station_cols = {
        "unit_id", "station_index", "station_name",
        "arrival_time_s", "start_time_s", "finish_time_s"
    }
    needed_transport_cols = {
        "unit_id", "from_station", "to_station",
        "start_time_s", "finish_time_s", "transport_time_s"
    }
    missing_station = needed_station_cols - set(station_df.columns)
    missing_transport = needed_transport_cols - set(transport_df.columns)
    if missing_station:
        raise ValueError(f"station_schedule.csv missing columns: {sorted(missing_station)}")
    if missing_transport:
        raise ValueError(f"transport_schedule.csv missing columns: {sorted(missing_transport)}")

    # Build station name->index mapping from the station schedule
    name_to_index = (
        station_df[["station_name", "station_index"]]
        .drop_duplicates()
        .set_index("station_name")["station_index"]
        .to_dict()
    )

    # Build station name->geom mapping from your configured geometry
    index_to_geom = STATION_GEOMETRY_BY_INDEX.copy()
    name_to_geom = {g.station_name_in_csv: g for g in index_to_geom.values()}

    # End time
    t_end = float(np.nanmax([
        station_df["finish_time_s"].max(),
        transport_df["finish_time_s"].max()
    ]))

    # Sort for deterministic queue ordering
    station_df = station_df.sort_values(["station_index", "arrival_time_s", "unit_id"]).reset_index(drop=True)
    transport_df = transport_df.sort_values(["start_time_s", "unit_id"]).reset_index(drop=True)

    return Schedules(
        station_df=station_df,
        transport_df=transport_df,
        station_name_to_index=name_to_index,
        station_index_to_geom=index_to_geom,
        station_name_to_geom=name_to_geom,
        t_end=t_end,
    )


def get_station_waiting_units_at_time(station_rows: pd.DataFrame, t: float) -> pd.DataFrame:
    """
    Units waiting in queue are those that have arrived but not started processing yet:
      arrival_time_s <= t < start_time_s
    """
    return station_rows[(station_rows["arrival_time_s"] <= t) & (station_rows["start_time_s"] > t)]


def get_station_processing_units_at_time(station_rows: pd.DataFrame, t: float) -> pd.DataFrame:
    """
    Units in processing:
      start_time_s <= t < finish_time_s
    """
    return station_rows[(station_rows["start_time_s"] <= t) & (station_rows["finish_time_s"] > t)]


def get_transports_at_time(transport_df: pd.DataFrame, t: float) -> pd.DataFrame:
    """
    Active transports:
      start_time_s <= t < finish_time_s
    """
    return transport_df[(transport_df["start_time_s"] <= t) & (transport_df["finish_time_s"] > t)]


def render_after_movie(
    output_path: str,
    fps: int = FPS,
    sim_seconds_per_frame: float = SIM_SECONDS_PER_FRAME,
):
    schedules = load_schedules()

    # Load images
    background = Image.open(BACKGROUND_PNG).convert("RGBA")
    carrier_base = Image.open(CARRIER_PNG).convert("RGBA")
    carrier_resized = carrier_base.resize((CARRIER_SIZE_PX, CARRIER_SIZE_PX), resample=Image.Resampling.LANCZOS)

    font = load_font_calibri_prefer(FONT_SIZE)

    # Group station schedule by station index for faster filtering
    station_groups = {idx: df for idx, df in schedules.station_df.groupby("station_index")}

    # Attempt MP4 writer
    writer = None
    use_png_fallback = False

    try:
        import imageio.v2 as imageio  # imageio v2 API (most stable)
        writer = imageio.get_writer(output_path, fps=fps)
    except Exception as e:
        print("WARNING: Could not create MP4 writer (ffmpeg missing?). Falling back to PNG frames.")
        print(f"Reason: {e}")
        use_png_fallback = True
        os.makedirs("frames", exist_ok=True)

    # Main loop
    t = 0.0
    frame_idx = 0
    t_end = schedules.t_end

    # Pre-check geometry configuration (to help you catch missing coords early)
    for idx, geom in schedules.station_index_to_geom.items():
        if (geom.process_pos == (0, 0)) or (geom.center_pos == (0, 0)) or any(p == (0, 0) for p in geom.queue_slots):
            print(f"WARNING: Station geometry for station_index={idx} contains (0,0) placeholders. "
                  f"Update STATION_GEOMETRY_BY_INDEX before final rendering.")

    while t <= t_end + 1e-9:
        frame = background.copy()
        draw = ImageDraw.Draw(frame)

        # ---- Draw station queues and processing carriers ----
        for station_index, geom in schedules.station_index_to_geom.items():
            station_rows = station_groups.get(station_index)
            if station_rows is None or station_rows.empty:
                continue

            # Processing units (often 0 or 1)
            processing = get_station_processing_units_at_time(station_rows, t)

            # If multiple are processing (parallel machines), draw all (slightly offset)
            for k, (_, row) in enumerate(processing.iterrows()):
                px, py = geom.process_pos
                if DRAW_PROCESS_GLOW:
                    draw.ellipse(
                        [px - PROCESS_GLOW_RADIUS, py - PROCESS_GLOW_RADIUS,
                         px + PROCESS_GLOW_RADIUS, py + PROCESS_GLOW_RADIUS],
                        fill=PROCESS_GLOW_COLOR
                    )
                offset = (k * (CARRIER_SIZE_PX * 0.35))
                paste_carrier_with_text(
                    frame_rgba=frame,
                    carrier_rgba_resized=carrier_resized,
                    center_xy=(px + offset, py),
                    text=str(row["unit_id"]),
                    font=font
                )

            # Queue units
            waiting = get_station_waiting_units_at_time(station_rows, t)

            # Queue ordering: earliest arrival is closest to processing (bottom slot).
            # Because you want "lowest available spot on arrival", queue fills bottom-first.
            waiting = waiting.sort_values(["arrival_time_s", "unit_id"], ascending=[True, True])

            # Map waiting units to slots bottom->top
            max_slots = len(geom.queue_slots)
            for i, (_, row) in enumerate(waiting.iterrows()):
                if i >= max_slots:
                    break  # queue overflow not shown
                slot_xy = geom.queue_slots[i]  # i=0 is the lowest/bottom-most slot
                paste_carrier_with_text(
                    frame_rgba=frame,
                    carrier_rgba_resized=carrier_resized,
                    center_xy=slot_xy,
                    text=str(row["unit_id"]),
                    font=font
                )

        # ---- Draw transports (moving carriers between station centers) ----
        moving = get_transports_at_time(schedules.transport_df, t)
        if not moving.empty:
            for _, r in moving.iterrows():
                from_name = str(r["from_station"])
                to_name = str(r["to_station"])
                unit_id = str(r["unit_id"])

                from_geom = schedules.station_name_to_geom.get(from_name)
                to_geom = schedules.station_name_to_geom.get(to_name)

                # If names don't match config, try mapping via station_name_to_index (from station_schedule)
                if from_geom is None and from_name in schedules.station_name_to_index:
                    from_geom = schedules.station_index_to_geom.get(schedules.station_name_to_index[from_name])
                if to_geom is None and to_name in schedules.station_name_to_index:
                    to_geom = schedules.station_index_to_geom.get(schedules.station_name_to_index[to_name])

                if from_geom is None or to_geom is None:
                    # Can't animate this transport without endpoints
                    continue

                x0, y0 = from_geom.center_pos
                x1, y1 = to_geom.center_pos

                ts = float(r["start_time_s"])
                tf = float(r["finish_time_s"])
                denom = max(tf - ts, 1e-9)
                alpha = float((t - ts) / denom)
                alpha = max(0.0, min(1.0, alpha))

                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)

                paste_carrier_with_text(
                    frame_rgba=frame,
                    carrier_rgba_resized=carrier_resized,
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

        # Advance
        frame_idx += 1
        t += sim_seconds_per_frame

    if writer is not None:
        writer.close()

    if use_png_fallback:
        print("Saved PNG frames to ./frames/")
        print("To convert frames to MP4 with ffmpeg (example):")
        print("  ffmpeg -r 30 -i frames/frame_%06d.png -pix_fmt yuv420p output.mp4")

    print(f"Done. Rendered {frame_idx} frames. Output: {output_path}")


# ----------------------------
# CLI
# ----------------------------

def main():
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

