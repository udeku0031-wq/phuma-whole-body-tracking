"""Render a robot-like capsule video directly from a WBT motion NPZ.

This script intentionally does not import Isaac Sim. It uses the saved link
world positions in a converted WBT motion file and draws a lightweight G1-like
capsule model, which is useful for presentation assets when headless RTX
recording fails to show the articulation mesh.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wbt_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


CHAINS = (
    ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link", "head_link"),
    (
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
    ),
    (
        "pelvis",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ),
    (
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ),
    (
        "torso_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ),
)


COLORS = {
    "torso": "#e8edf2",
    "pelvis": "#c9d3dc",
    "left": "#73a6ff",
    "right": "#ffb057",
    "joint": "#1e3f66",
    "floor": "#d5dde5",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a presentation-friendly capsule robot motion video.")
    parser.add_argument("--motion_file", required=True, help="Path to a local WBT motion .npz file.")
    parser.add_argument("--output", required=True, help="Output .mp4 path.")
    parser.add_argument("--start_frame", type=int, default=0, help="First motion frame, inclusive.")
    parser.add_argument("--end_frame_exclusive", type=int, default=None, help="Last motion frame, exclusive.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of source frames to render.")
    parser.add_argument("--render_fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Source frame stride.")
    parser.add_argument("--width", type=int, default=1280, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Output height in pixels.")
    parser.add_argument("--dpi", type=int, default=100, help="Matplotlib DPI.")
    parser.add_argument("--azim", type=float, default=-62.0, help="3D camera azimuth.")
    parser.add_argument("--elev", type=float, default=16.0, help="3D camera elevation.")
    parser.add_argument("--view_radius", type=float, default=1.35, help="Meters shown around the pelvis.")
    return parser.parse_args()


def _names(array: np.ndarray) -> list[str]:
    out = []
    for value in array.tolist():
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def _load_motion(path: str) -> tuple[np.ndarray, list[str], float]:
    with np.load(path, allow_pickle=True) as data:
        missing = [name for name in ("body_pos_w", "body_names", "fps") if name not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_names = _names(data["body_names"])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (T, B, 3), got {body_pos.shape}.")
    if len(body_names) != body_pos.shape[1]:
        raise ValueError(f"body_names length {len(body_names)} does not match body count {body_pos.shape[1]}.")
    return body_pos, body_names, fps


def _segments(body_names: list[str]) -> list[tuple[int, int, str, float]]:
    index = {name: i for i, name in enumerate(body_names)}
    result: list[tuple[int, int, str, float]] = []
    seen: set[tuple[int, int]] = set()
    for chain in CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            if a not in index or b not in index:
                continue
            edge = (index[a], index[b])
            if edge in seen:
                continue
            seen.add(edge)
            if a == "pelvis" and "waist" in b:
                group, radius = "pelvis", 0.07
            elif "torso" in a or "torso" in b or "waist" in a or "waist" in b or "head" in b:
                group, radius = "torso", 0.06
            elif a.startswith("left") or b.startswith("left"):
                group, radius = "left", 0.035
            elif a.startswith("right") or b.startswith("right"):
                group, radius = "right", 0.035
            else:
                group, radius = "torso", 0.04
            result.append((edge[0], edge[1], group, radius))
    return result


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(direction, u)
    return u, v


def _cylinder_faces(p0: np.ndarray, p1: np.ndarray, radius: float, sides: int = 10) -> list[list[np.ndarray]]:
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return []
    u, v = _orthonormal_basis(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    c0 = np.asarray([p0 + radius * (np.cos(a) * u + np.sin(a) * v) for a in angles])
    c1 = np.asarray([p1 + radius * (np.cos(a) * u + np.sin(a) * v) for a in angles])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([c0[i], c0[j], c1[j], c1[i]])
    return faces


def _draw_floor(ax, center: np.ndarray, radius: float) -> None:
    low_x, high_x = center[0] - radius, center[0] + radius
    low_y, high_y = center[1] - radius, center[1] + radius
    ticks_x = np.arange(np.floor(low_x / 0.25) * 0.25, high_x + 0.25, 0.25)
    ticks_y = np.arange(np.floor(low_y / 0.25) * 0.25, high_y + 0.25, 0.25)
    for x in ticks_x:
        ax.plot([x, x], [low_y, high_y], [0.0, 0.0], color=COLORS["floor"], linewidth=0.7, alpha=0.7)
    for y in ticks_y:
        ax.plot([low_x, high_x], [y, y], [0.0, 0.0], color=COLORS["floor"], linewidth=0.7, alpha=0.7)


def _render(args: argparse.Namespace) -> None:
    body_pos, body_names, source_fps = _load_motion(args.motion_file)
    start = int(args.start_frame)
    end = body_pos.shape[0] if args.end_frame_exclusive is None else int(args.end_frame_exclusive)
    if args.max_steps is not None:
        end = min(end, start + int(args.max_steps))
    if not 0 <= start < end <= body_pos.shape[0]:
        raise ValueError(f"Invalid frame range [{start}, {end}) for motion length {body_pos.shape[0]}.")
    frames = np.arange(start, end, int(args.stride), dtype=np.int64)
    if frames.size == 0:
        raise ValueError("No frames selected.")
    if "pelvis" not in body_names:
        raise ValueError("Motion file does not contain a pelvis body.")

    pelvis_index = body_names.index("pelvis")
    head_index = body_names.index("head_link") if "head_link" in body_names else None
    segments = _segments(body_names)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7fbff")

    with imageio.get_writer(output, fps=args.render_fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for count, frame in enumerate(frames, start=1):
            ax.clear()
            points = body_pos[frame]
            pelvis = points[pelvis_index]
            _draw_floor(ax, pelvis, args.view_radius)

            for left, right, group, radius in segments:
                faces = _cylinder_faces(points[left], points[right], radius)
                if not faces:
                    continue
                poly = Poly3DCollection(
                    faces,
                    facecolors=COLORS[group],
                    edgecolors="#2f4050",
                    linewidths=0.15,
                    alpha=0.96,
                )
                ax.add_collection3d(poly)

            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=14, color=COLORS["joint"], depthshade=True)
            ax.scatter([pelvis[0]], [pelvis[1]], [pelvis[2]], s=220, color=COLORS["pelvis"], edgecolors="#2f4050")
            torso = points[body_names.index("torso_link")] if "torso_link" in body_names else pelvis
            ax.scatter([torso[0]], [torso[1]], [torso[2]], s=260, color=COLORS["torso"], edgecolors="#2f4050")
            if head_index is not None:
                head = points[head_index]
                ax.scatter([head[0]], [head[1]], [head[2]], s=170, color="#f2f6f9", edgecolors="#2f4050")

            ax.set_xlim(pelvis[0] - args.view_radius, pelvis[0] + args.view_radius)
            ax.set_ylim(pelvis[1] - args.view_radius, pelvis[1] + args.view_radius)
            ax.set_zlim(0.0, 1.55)
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_box_aspect((1.0, 1.0, 0.7))
            ax.set_axis_off()
            ax.set_title(
                f"G1 replay | frame {frame}/{body_pos.shape[0] - 1} | source {source_fps:g} fps",
                fontsize=12,
                color="#163b5c",
                pad=10,
            )
            fig.canvas.draw()
            width_px, height_px = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height_px, width_px, 4)
            writer.append_data(rgba[..., :3].copy())
            if count % 60 == 0:
                print(f"[INFO]: Rendered {count}/{frames.size} frames", flush=True)
    plt.close(fig)
    print(f"[INFO]: Wrote capsule robot video: {output}", flush=True)


def main() -> None:
    args = _parse_args()
    _render(args)


if __name__ == "__main__":
    main()
