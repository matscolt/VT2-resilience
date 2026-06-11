from __future__ import annotations
"""
G_after_movie_v2.py

Instance-aware aftermovie renderer.

Key changes from the old approach:
- Stations are addressed as instances: Station 1 -> 1.0, Station 1.1 -> 1.1, etc.
- Transport routes are composed from three parts:
    leaving_paths[source_instance] + common_paths[source_type->target_type] + connecting_paths[target_instance]
- Queued units are NOT drawn individually. Only a queue counter is displayed at queue_label_pos.
- Processing and transporting units are drawn with the carrier PNG, without unit IDs by default.
- Disruption label/timer uses dis_label_pos from the JSON.

Expected files by default:
- aftermovie_config_v2.json
- station_schedule.csv
- transport_schedule.csv
- layout/background PNG and carrier PNG as referenced by config
"""

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Point = Tuple[float, float]
ROOTDIR = Path(__file__).resolve().parent
LAYOUTDIR = ROOTDIR / "data" / "Layouts"

def station_id_from_name(station_name: str) -> str:
    """Return instance id like '1.0', '1.1', '3.4' from names like 'Station 1: ...'."""
    m = re.search(r"Station\s+(\d+)(?:\.(\d+))?", str(station_name))
    if not m:
        raise ValueError(f"Could not parse station id from station_name={station_name!r}")
    station_type = int(m.group(1))
    instance = int(m.group(2) or 0)
    return f"{station_type}.{instance}"


def station_type_from_id(station_id: str) -> int:
    return int(str(station_id).split(".")[0])


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def resolve_path(base_dir: Path, value: str | None) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    candidates = [ROOTDIR / p, base_dir / p, ROOTDIR / "data" / "Layouts" / p, ROOTDIR / "Scenarios" / p]
    for c in candidates:
        if c.exists():
            return c
    return ROOTDIR / p


def choose_layout(config: dict, layout_name: Optional[str]) -> Tuple[str, dict]:
    layouts = config.get("layouts", {})
    if layout_name:
        if layout_name in layouts:
            return layout_name, layouts[layout_name]
        # allow passing source json file name from layout_selection
        selected = config.get("layout_selection", {}).get(layout_name)
        if selected in layouts:
            return selected, layouts[selected]
        raise KeyError(f"Unknown layout {layout_name!r}. Available: {list(layouts)}")
    if not layouts:
        raise ValueError("Config contains no layouts")
    key = next(iter(layouts))
    return key, layouts[key]


def polyline_length(points: List[Point]) -> float:
    return sum(math.hypot(points[i][0]-points[i-1][0], points[i][1]-points[i-1][1]) for i in range(1, len(points)))


def interpolate_polyline(points: List[Point], fraction: float) -> Point:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    fraction = max(0.0, min(1.0, fraction))
    total = polyline_length(points)
    if total <= 0:
        return points[-1]
    target = fraction * total
    travelled = 0.0
    for i in range(1, len(points)):
        a, b = points[i-1], points[i]
        seg = math.hypot(b[0]-a[0], b[1]-a[1])
        if seg <= 0:
            continue
        if travelled + seg >= target:
            local = (target - travelled) / seg
            return (a[0] + local * (b[0]-a[0]), a[1] + local * (b[1]-a[1]))
        travelled += seg
    return points[-1]


def clean_join(paths: Iterable[List[Point]]) -> List[Point]:
    out: List[Point] = []
    for pts in paths:
        for p in pts:
            p = (float(p[0]), float(p[1]))
            if not out or p != out[-1]:
                out.append(p)
    return out


def points_from_segment(segment: dict) -> List[Point]:
    typ = segment.get("type", "polyline")
    if typ != "polyline":
        # Arc support can be added later. For now use start/mid/end as a sampled polyline.
        if all(k in segment for k in ("start", "mid", "end")):
            return [tuple(segment["start"]), tuple(segment["mid"]), tuple(segment["end"])]
        raise ValueError(f"Unsupported segment type: {typ}")
    return [tuple(p) for p in segment.get("points", [])]


def compose_route(layout: dict, from_station: str, to_station: str) -> List[Point]:
    route_parts = layout.get("route_parts", {})
    leaving = route_parts.get("leaving_paths", {})
    common = route_parts.get("common_paths", {})
    connecting = route_parts.get("connecting_paths", {})

    from_id = station_id_from_name(from_station)
    to_id = station_id_from_name(to_station)
    key = f"{station_type_from_id(from_id)}->{station_type_from_id(to_id)}"

    try:
        return clean_join([
            points_from_segment(leaving[from_id]),
            points_from_segment(common[key]),
            points_from_segment(connecting[to_id]),
        ])
    except KeyError as e:
        raise KeyError(
            f"Missing route part {e!s} for movement {from_station!r} -> {to_station!r}. "
            f"Needed leaving_paths[{from_id!r}], common_paths[{key!r}], connecting_paths[{to_id!r}]."
        )


def paste_center(base: Image.Image, sprite: Image.Image, center: Point):
    x = int(round(center[0] - sprite.width / 2))
    y = int(round(center[1] - sprite.height / 2))
    base.alpha_composite(sprite, (x, y))


def draw_text_center(draw: ImageDraw.ImageDraw, pos: Tuple[int, int], text: str, font, fill, anchor="mm"):
    draw.text(tuple(pos), text, font=font, fill=fill, anchor=anchor)


def maybe_make_mp4(frames_dir: Path, output_mp4: Path, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found; PNG frames were written but MP4 was not created.")
        return
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_mp4)
    ]
    subprocess.run(cmd, check=True)



# ----------------------------
# Project / result-folder helpers
# ----------------------------
RESULTS_OUTPUTDIR = ROOTDIR / "RESULTS" / "output"
CONFIG_JSON = LAYOUTDIR / "aftermovie_config_v2.json"
if not CONFIG_JSON.exists():
    CONFIG_JSON = ROOTDIR / "aftermovie_config_v2.json"


def list_folders(parent: Path, prefix: str | None = None) -> List[Path]:
    """Return subfolders newest first. Optionally filter by folder-name prefix."""
    if not parent.exists():
        return []
    folders = [p for p in parent.iterdir() if p.is_dir() and (prefix is None or p.name.startswith(prefix))]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return folders


def prompt_for_main_folder(results_output_dir: Path = RESULTS_OUTPUTDIR) -> Path:
    """Pick a main_* result folder below RESULTS/output, newest first."""
    mains = list_folders(results_output_dir, prefix="main_")
    if not mains:
        # Fallback: show all folders if the naming convention changes.
        mains = list_folders(results_output_dir)
    if not mains:
        raise FileNotFoundError(f"No main folders found in: {results_output_dir}")

    print(f"\nAvailable main folders in {results_output_dir} (newest first):")
    for i, p in enumerate(mains, start=1):
        print(f"  {i:2d}) {p.name}")

    default = 1
    while True:
        s_in = input(f"Choose main folder number (Enter = {default}): ").strip()
        if s_in == "":
            return mains[default - 1]
        if s_in.isdigit() and 1 <= int(s_in) <= len(mains):
            return mains[int(s_in) - 1]
        candidate = results_output_dir / s_in
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")


def prompt_for_run_folder(main_dir: Path) -> Path:
    """Pick a run_* folder below the selected main_* folder, newest first."""
    runs = list_folders(main_dir, prefix="run_")
    if not runs:
        runs = list_folders(main_dir)
    if not runs:
        raise FileNotFoundError(f"No run folders found in: {main_dir}")

    print(f"\nAvailable run folders in {main_dir.name} (newest first):")
    for i, p in enumerate(runs, start=1):
        print(f"  {i:2d}) {p.name}")

    default = 1
    while True:
        s_in = input(f"Choose run folder number (Enter = {default}): ").strip()
        if s_in == "":
            return runs[default - 1]
        if s_in.isdigit() and 1 <= int(s_in) <= len(runs):
            return runs[int(s_in) - 1]
        candidate = main_dir / s_in
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        s_in = input(f"{prompt} (Enter = {default}): ").strip()
        if s_in == "":
            return float(default)
        try:
            return float(s_in)
        except ValueError:
            print("Please enter a number.")


def prompt_int(prompt: str, default: int) -> int:
    while True:
        s_in = input(f"{prompt} (Enter = {default}): ").strip()
        if s_in == "":
            return int(default)
        try:
            return int(s_in)
        except ValueError:
            print("Please enter an integer.")


def data_dir_from_run_folder(run_dir: Path) -> Path:
    """The current project structure stores schedules in run_X/results/."""
    results_dir = run_dir / "results"
    return results_dir if results_dir.exists() else run_dir


def clear_frames_folder(frames_dir: Path) -> None:
    if frames_dir.exists():
        def _onerror(func, path, excinfo):
            try:
                os.chmod(path, 0o666)
                func(path)
            except Exception:
                print(f">> Could not delete locked file/folder: {path}")
        shutil.rmtree(frames_dir, onerror=_onerror)
    frames_dir.mkdir(parents=True, exist_ok=True)


def pick_coords_interactive(image_path: Path) -> None:
    """Click the layout image to print pixel coordinates. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required for pick-coords but could not be imported.")
        print("Reason:", e)
        return

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click points to print x, y coordinates. Close the window when done.")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print(f"[{x}, {y}]")
        ax.plot([x], [y], marker="x")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def render_after_movie(
    run_dir: Path,
    *,
    config_path: Path = CONFIG_JSON,
    layout_name: Optional[str] = None,
    fps_override: Optional[int] = None,
    sim_seconds_per_frame_override: Optional[float] = None,
    max_time: Optional[float] = None,
) -> Tuple[Path, int]:
    """Render one selected run folder and return (movie_dir, number_of_frames)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    data_dir = data_dir_from_run_folder(run_dir)
    station_schedule = data_dir / "station_schedule.csv"
    transport_schedule = data_dir / "transport_schedule.csv"
    if not station_schedule.exists():
        raise FileNotFoundError(f"station_schedule.csv not found: {station_schedule}")
    if not transport_schedule.exists():
        raise FileNotFoundError(f"transport_schedule.csv not found: {transport_schedule}")

    config = json.load(open(config_path, encoding="utf-8"))
    defaults = config.get("defaults", {})
    layout_key, layout = choose_layout(config, layout_name)

    fps = int(fps_override if fps_override is not None else defaults.get("fps", 30))
    sim_seconds_per_frame = float(
        sim_seconds_per_frame_override
        if sim_seconds_per_frame_override is not None
        else defaults.get("sim_seconds_per_frame", 0.5)
    )
    save_every_nth = int(defaults.get("save_every_nth_png", 100))
    carrier_size = int(defaults.get("carrier_size_px", 25))
    write_mp4 = bool(defaults.get("write_mp4", True))
    draw_time_label = bool(defaults.get("draw_time_label", True))
    draw_unit_ids = bool(defaults.get("draw_unit_ids", False))
    queue_fmt = defaults.get("queue_label_format", "Q: {count}")

    assets_base_dir = resolve_path(ROOTDIR, config.get("assets_base_dir", ".")) or ROOTDIR
    bg_path = resolve_path(assets_base_dir, layout.get("background_png"))
    carrier_path = resolve_path(assets_base_dir, layout.get("carrier_png"))
    if not bg_path or not bg_path.exists():
        raise FileNotFoundError(f"Background PNG not found: {bg_path}")
    if not carrier_path or not carrier_path.exists():
        raise FileNotFoundError(f"Carrier PNG not found: {carrier_path}")

    station_df = pd.read_csv(station_schedule)
    transport_df = pd.read_csv(transport_schedule)

    station_df["station_id"] = station_df["station_name"].apply(station_id_from_name)
    transport_df["from_station_id"] = transport_df["from_station"].apply(station_id_from_name)
    transport_df["to_station_id"] = transport_df["to_station"].apply(station_id_from_name)

    stations = layout.get("stations", {})
    for name in station_df["station_name"].dropna().unique():
        if name not in stations:
            raise KeyError(f"Station {name!r} from station_schedule.csv is missing from config layout.stations")
    for name in sorted(set(transport_df["from_station"]).union(set(transport_df["to_station"]))):
        if name not in stations:
            raise KeyError(f"Station {name!r} from transport_schedule.csv is missing from config layout.stations")

    background = Image.open(bg_path).convert("RGBA")
    carrier = Image.open(carrier_path).convert("RGBA").resize((carrier_size, carrier_size), Image.LANCZOS)

    movie_dir = data_dir / "movie"
    frames_dir = movie_dir / "frames"
    saved_frames_dir = movie_dir / "saved_every_nth"
    movie_dir.mkdir(parents=True, exist_ok=True)
    clear_frames_folder(frames_dir)
    clear_frames_folder(saved_frames_dir)

    t0 = 0.0
    t_end = max(float(station_df["finish_time_s"].max()), float(transport_df["finish_time_s"].max()))
    if max_time is not None:
        t_end = min(t_end, max_time)

    label_font = load_font(int(defaults.get("queue_font_size", 24)), bold=True)
    dis_font = load_font(int(defaults.get("disruption_font_size", 24)), bold=True)
    time_font = load_font(int(defaults.get("time_label_font_size", 30)), bold=True)
    unit_font = load_font(10, bold=True)

    print("\nRendering after movie")
    print(f"  main/run folder : {run_dir}")
    print(f"  data folder     : {data_dir}")
    print(f"  config          : {config_path}")
    print(f"  layout          : {layout_key}")
    print(f"  fps             : {fps}")
    print(f"  sim sec/frame   : {sim_seconds_per_frame}")
    print(f"  save every nth  : {save_every_nth}")
    print(f"  movie folder    : {movie_dir}")

    total_frames_est = int(math.floor((t_end - t0) / sim_seconds_per_frame)) + 1
    frame_idx = 0
    t = t0
    while t <= t_end + 1e-9:
        frame = background.copy()
        draw = ImageDraw.Draw(frame)

        # Queue counters only; no individual queued units.
        for station_name, st in stations.items():
            q = station_df[
                (station_df["station_name"] == station_name)
                & (station_df["arrival_time_s"] <= t)
                & (station_df["start_time_s"] > t)
            ]
            count = len(q)
            qpos = st.get("queue_label_pos")
            if qpos is not None:
                draw_text_center(
                    draw,
                    tuple(qpos),
                    queue_fmt.format(count=count),
                    label_font,
                    tuple(defaults.get("queue_label_color", [0, 0, 0, 255])),
                )

        # Processing units are visible at processing_pos.
        processing = station_df[(station_df["start_time_s"] <= t) & (station_df["finish_time_s"] > t)]
        occupied_counts: Dict[str, int] = {}
        for _, row in processing.iterrows():
            st = stations[row["station_name"]]
            x, y = st["processing_pos"]
            k = occupied_counts.get(row["station_name"], 0)
            occupied_counts[row["station_name"]] = k + 1
            dx = (k % 3 - 1) * (carrier_size * 0.35)
            dy = (k // 3) * (carrier_size * 0.35)
            paste_center(frame, carrier, (x + dx, y + dy))
            if draw_unit_ids:
                draw_text_center(draw, (int(x + dx), int(y + dy)), str(row["unit_id"]), unit_font, (0, 0, 0, 255))

            # Disruption timer: show extra processing time beyond base_process_time_s.
            if "base_process_time_s" in row and pd.notna(row.get("base_process_time_s")):
                base_done = float(row["start_time_s"]) + float(row["base_process_time_s"])
                if t >= base_done and float(row["finish_time_s"]) > base_done:
                    elapsed = t - base_done
                    dpos = st.get("dis_label_pos")
                    if dpos is not None:
                        draw_text_center(
                            draw,
                            tuple(dpos),
                            f"DISR {elapsed:0.1f}s",
                            dis_font,
                            tuple(defaults.get("disruption_label_color", [255, 0, 0, 255])),
                        )

        # Transporting units are visible along the composed instance-aware route.
        moving = transport_df[(transport_df["start_time_s"] <= t) & (transport_df["finish_time_s"] > t)]
        for _, row in moving.iterrows():
            duration = float(row["finish_time_s"]) - float(row["start_time_s"])
            frac = 1.0 if duration <= 0 else (t - float(row["start_time_s"])) / duration
            route = compose_route(layout, row["from_station"], row["to_station"])
            pos = interpolate_polyline(route, frac)
            paste_center(frame, carrier, pos)
            if draw_unit_ids:
                draw_text_center(draw, (int(pos[0]), int(pos[1])), str(row["unit_id"]), unit_font, (0, 0, 0, 255))

        if draw_time_label:
            pos = tuple(defaults.get("time_label_pos", [20, 20]))
            draw.text(pos, f"t = {t:0.1f}s", font=time_font, fill=tuple(defaults.get("time_label_color", [0, 0, 0, 255])))

        frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        frame.save(frame_path)
        if save_every_nth > 0 and frame_idx % save_every_nth == 0:
            frame.save(saved_frames_dir / frame_path.name)

        if frame_idx % 50 == 0 or frame_idx == total_frames_est - 1:
            pct = 100.0 * frame_idx / max(total_frames_est - 1, 1)
            print(f"  frame {frame_idx:6d}/{total_frames_est - 1:6d} ({pct:5.1f}%)", end="\r")

        frame_idx += 1
        t += sim_seconds_per_frame

    print()
    if write_mp4:
        maybe_make_mp4(frames_dir, movie_dir / "after_movie.mp4", fps)
    print(f"Done. Wrote {frame_idx} frames to {frames_dir}")
    return movie_dir, frame_idx


def main():

    config_path = CONFIG_JSON
    if not config_path.exists():
        print(f"WARNING: Default config was not found: {config_path}")
        entered = input("Paste path to aftermovie_config_v2.json: ").strip().strip('"')
        config_path = Path(entered)

    config = json.load(open(config_path, encoding="utf-8"))
    defaults = config.get("defaults", {})
    default_fps = int(defaults.get("fps", 30))
    default_spf = float(defaults.get("sim_seconds_per_frame", 0.5))

    main_dir = prompt_for_main_folder(RESULTS_OUTPUTDIR)
    run_dir = prompt_for_run_folder(main_dir)
    fps = prompt_int("FPS", default_fps)
    spf = prompt_float("Simulation seconds per frame", default_spf)
    render_after_movie(run_dir, config_path=config_path, fps_override=fps, sim_seconds_per_frame_override=spf)
if __name__ == "__main__":
    main()
