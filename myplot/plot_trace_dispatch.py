"""
Animates order-driven drone interception traces produced by the new environment.

Usage:
    python plot_trace_dispatch.py --trace ../trace/weight01/train/run_20251019-103000.json

If --trace is omitted, the script will pick the most recent JSON file under the
matching scenario directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

TYPE_COLOR = {
    "ATTACK": "#ff595e",
    "RECON": "#1982c4",
    "EW": "#6a4c93",
}

STATUS_ALPHA = {
    "ACTIVE": 1.0,
    "JAMMED": 0.6,
    "INFILTRATED": 0.9,
    "DESTROYED": 0.3,
    "MISSING": 0.0,
}


@dataclass
class DroneTrace:
    drone_id: int
    drone_type: str
    x: np.ndarray
    y: np.ndarray
    hp: np.ndarray
    status: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate order-driven drone traces.")
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Path to exported trace JSON. If omitted, the latest file in trace/<scenario>/<train|render> is used.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="weight01",
        help="Scenario subdirectory to search when --trace is not provided.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Search the render trace directory instead of train when --trace is omitted.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Animation frame interval in milliseconds.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Number of historical points to keep when plotting each drone trail.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional diagnostics about the loaded trace.",
    )
    return parser.parse_args()


def find_latest_trace(scenario: str, render: bool) -> Optional[Path]:
    base_dir = Path(__file__).resolve().parent.parent / "trace" / scenario
    subdir = "render" if render else "train"
    directory = base_dir / subdir
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("run_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def load_trace(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_drones(frames: List[Dict]) -> List[DroneTrace]:
    drone_map: Dict[int, Dict[str, List]] = {}
    for frame_index, frame in enumerate(frames):
        seen_this_frame = set()
        for entry in frame.get("drones", []):
            drone_id = int(entry["id"])
            container = drone_map.setdefault(
                drone_id,
                {
                    "type": entry.get("type", "ATTACK"),
                    "x": [],
                    "y": [],
                    "hp": [],
                    "status": [],
                },
            )
            # pad previous frames if this drone appears late
            missing = frame_index - len(container["x"])
            if missing > 0:
                container["x"].extend([np.nan] * missing)
                container["y"].extend([np.nan] * missing)
                container["hp"].extend([0.0] * missing)
                container["status"].extend(["MISSING"] * missing)

            container["x"].append(float(entry.get("x", np.nan)))
            container["y"].append(float(entry.get("y", np.nan)))
            container["hp"].append(float(entry.get("hp", 0.0)))
            container["status"].append(entry.get("status", "ACTIVE"))
            seen_this_frame.add(drone_id)

        for drone_id, container in drone_map.items():
            if drone_id not in seen_this_frame and len(container["x"]) <= frame_index:
                container["x"].append(np.nan)
                container["y"].append(np.nan)
                container["hp"].append(0.0)
                container["status"].append("MISSING")

    total_frames = len(frames)
    drones: List[DroneTrace] = []
    for drone_id, container in sorted(drone_map.items()):
        # ensure sequences match total frame count
        while len(container["x"]) < total_frames:
            container["x"].append(np.nan)
            container["y"].append(np.nan)
            container["hp"].append(0.0)
            container["status"].append("MISSING")
        drones.append(
            DroneTrace(
                drone_id=drone_id,
                drone_type=container["type"],
                x=np.asarray(container["x"], dtype=float),
                y=np.asarray(container["y"], dtype=float),
                hp=np.asarray(container["hp"], dtype=float),
                status=list(container["status"]),
            )
        )
    return drones


def extract_orders(frames: List[Dict]) -> List[List[Dict]]:
    return [frame.get("orders", []) for frame in frames]


def compute_plot_limits(drones: List[DroneTrace]) -> Dict[str, float]:
    """Derive axis limits that include both drones and their trails robustly."""

    if not drones:
        return {"xlim": (-1.0, 1.0), "ylim": (-1.0, 1.0)}

    xs = []
    ys = []
    for drone in drones:
        if drone.x.size:
            xs.append(drone.x)
        if drone.y.size:
            ys.append(drone.y)

    if not xs:
        xs = [np.array([0.0])]
    if not ys:
        ys = [np.array([0.0])]

    xs_concat = np.concatenate(xs)
    ys_concat = np.concatenate(ys)

    min_x = float(np.nanmin(xs_concat))
    max_x = float(np.nanmax(xs_concat))
    min_y = float(np.nanmin(ys_concat))
    max_y = float(np.nanmax(ys_concat))

    range_x = max(1.0, max_x - min_x)
    range_y = max(1.0, max_y - min_y)

    padding_x = max(1000.0, 0.1 * range_x)
    padding_y = max(1000.0, 0.1 * range_y)

    if not np.isfinite(min_x):
        min_x, max_x = -1.0, 1.0
    if not np.isfinite(min_y):
        min_y, max_y = -1.0, 1.0

    return {
        "xlim": (min_x - padding_x, max_x + padding_x),
        "ylim": (min_y - padding_y, max_y + padding_y),
    }


def setup_figure(limits: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Order-Driven Drone Interception")
    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.4, alpha=0.3)
    return fig


def main():
    args = parse_args()

    if args.trace:
        trace_path = Path(args.trace)
    else:
        trace_path = find_latest_trace(args.scenario, args.render)
        if trace_path is None:
            print("No trace file found. Please provide --trace.", file=sys.stderr)
            sys.exit(1)

    if not trace_path.exists():
        print(f"Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    data = load_trace(trace_path)
    if args.debug:
        meta = data.get("meta", {})
        if meta:
            print(f"[DEBUG] meta: {meta}")
    frames = data.get("frames", [])
    if not frames:
        print("Trace file contains no frames.", file=sys.stderr)
        sys.exit(1)

    drones = extract_drones(frames)
    if not drones:
        print("Trace frames contain no drone data.", file=sys.stderr)
        sys.exit(1)

    orders = extract_orders(frames)
    num_frames = len(frames)
    if args.debug:
        print(f"[DEBUG] trace file: {trace_path}")
        print(f"[DEBUG] frames = {num_frames}, drones = {len(drones)}, order frames = {len(orders)}")
        for drone in drones:
            print(
                f"  id={drone.drone_id:<3} type={drone.drone_type:<7} "
                f"points={drone.x.size:<4} statuses={len(drone.status):<4}"
            )
        print(f"[DEBUG] computed animation frames: {num_frames}")

    if num_frames <= 1:
        raise RuntimeError(
            "Trace does not contain enough steps to animate (<=1 frame). "
            "Run the scenario for more steps or verify the trace file."
        )
    limits = compute_plot_limits(drones)

    fig = setup_figure(limits)
    ax = fig.axes[0]

    scatters = []
    trails = []
    for drone in drones:
        color = TYPE_COLOR.get(drone.drone_type, "#999999")
        scatter = ax.scatter([], [], s=30, color=color, alpha=0.9, label=f"Drone {drone.drone_id} ({drone.drone_type})")
        trail, = ax.plot([], [], color=color, linewidth=1.2, alpha=0.4)
        scatters.append((scatter, drone))
        trails.append((trail, drone))

    order_scatter = ax.scatter([], [], s=15, color="#f1c40f", alpha=0.5, marker="x", label="Active Orders")
    legend = ax.legend(loc="upper right", framealpha=0.85)

    def update(frame: int):
        for (scatter, drone), (trail, drone_for_trail) in zip(scatters, trails):
            if frame < drone.x.size:
                x = drone.x[frame]
                y = drone.y[frame]
                scatter.set_offsets([x, y])
                status = drone.status[frame] if frame < len(drone.status) else "ACTIVE"
                alpha = STATUS_ALPHA.get(status, 0.9)
                scatter.set_alpha(alpha)
                tail_start = max(0, frame - args.tail)
                trail.set_data(drone_for_trail.x[tail_start:frame + 1], drone_for_trail.y[tail_start:frame + 1])
            else:
                scatter.set_offsets([np.nan, np.nan])
                trail.set_data([], [])

        if frame < len(orders):
            order_entries = orders[frame]
            if order_entries:
                coords = np.column_stack(
                    [
                        [entry.get("x", np.nan) for entry in order_entries],
                        [entry.get("y", np.nan) for entry in order_entries],
                    ]
                )
            else:
                coords = np.empty((0, 2))
            order_scatter.set_offsets(coords)
        else:
            order_scatter.set_offsets(np.empty((0, 2)))

        legend.set_title(f"Step {frame + 1}/{num_frames}")
        return [scatter for scatter, _ in scatters] + [trail for trail, _ in trails] + [order_scatter, legend]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=args.interval,
        blit=False,
        repeat=False,
    )

    if args.debug:
        print("[DEBUG] launching animation window...")

    plt.show(block=True)
    return anim


if __name__ == "__main__":
    animation_handle = main()
    globals()["_TRACE_ANIMATION_HANDLE"] = animation_handle
