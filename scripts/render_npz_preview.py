"""Render a lightweight offline preview video for a WBT motion NPZ.

This script does not import Isaac Sim. It visualizes the stored body positions
and foot sole heights directly from the converted trajectory, which is useful
for manual quality-gate review on machines where Isaac replay is too heavy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wbt_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
DEFAULT_SOLE_OFFSETS = {
    "left_ankle_roll_link": np.asarray([0.04, 0.0, -0.037], dtype=np.float64),
    "right_ankle_roll_link": np.asarray([0.04, 0.0, -0.037], dtype=np.float64),
}

SKELETON_CHAINS = (
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
    ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Isaac-free WBT NPZ preview video.")
    parser.add_argument("--motion_file", required=True, help="Path to a local WBT motion .npz file.")
    parser.add_argument("--output", default=None, help="Output .mp4 or .gif path. Defaults to /tmp/wbt_npz_previews.")
    parser.add_argument("--start_frame", type=int, default=0, help="First motion frame to render, inclusive.")
    parser.add_argument(
        "--end_frame_exclusive",
        type=int,
        default=None,
        help="Frame at which rendering stops, exclusive. Defaults to the motion length.",
    )
    parser.add_argument(
        "--quality_config",
        default="configs/quality/g1_segment_quality.yaml",
        help="Quality config used to read foot body names, sole offsets, and ground z.",
    )
    parser.add_argument("--render_fps", type=float, default=25.0, help="Output video FPS.")
    parser.add_argument("--stride", type=int, default=None, help="Frame stride. Defaults to source_fps/render_fps.")
    parser.add_argument("--dpi", type=int, default=120, help="Matplotlib render DPI.")
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=7.2, help="Figure height in inches.")
    return parser.parse_args()


def _names(array: np.ndarray) -> list[str]:
    output = []
    for value in array.tolist():
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))
    return output


def _load_ground_config(path: str) -> tuple[float, tuple[str, ...], dict[str, np.ndarray]]:
    ground_z = 0.0
    foot_names = DEFAULT_FOOT_NAMES
    sole_offsets = dict(DEFAULT_SOLE_OFFSETS)
    config_path = Path(path)
    if not config_path.exists():
        return ground_z, foot_names, sole_offsets

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    ground = config.get("ground", {})
    ground_z = float(ground.get("z_m", ground_z))
    foot_names = tuple(str(name) for name in ground.get("foot_body_names", foot_names))
    configured_offsets = ground.get("sole_local_offsets_m", {})
    for name in foot_names:
        if name in configured_offsets:
            offset = np.asarray(configured_offsets[name], dtype=np.float64)
            if offset.shape == (3,) and np.all(np.isfinite(offset)):
                sole_offsets[name] = offset
    return ground_z, foot_names, sole_offsets


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-12, None)
    w = quat[..., :1]
    xyz = quat[..., 1:]
    uv = np.cross(xyz, vec)
    uuv = np.cross(xyz, uv)
    return vec + 2.0 * (w * uv + uuv)


def _skeleton_edges(body_names: list[str]) -> list[tuple[int, int]]:
    indexes = {name: index for index, name in enumerate(body_names)}
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for chain in SKELETON_CHAINS:
        for left, right in zip(chain[:-1], chain[1:]):
            if left not in indexes or right not in indexes:
                continue
            edge = (indexes[left], indexes[right])
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return edges


def _axis_limits(values: np.ndarray, pad: float, minimum_span: float = 0.5) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -minimum_span / 2.0, minimum_span / 2.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    if span < minimum_span:
        center = 0.5 * (low + high)
        low = center - minimum_span / 2.0
        high = center + minimum_span / 2.0
    return low - pad, high + pad


def _auto_output_path(motion_file: str, start_frame: int, end_frame_exclusive: int) -> Path:
    output_dir = Path("/tmp/wbt_npz_previews")
    stem = Path(motion_file).stem
    return output_dir / f"{stem}_f{start_frame}_{end_frame_exclusive}.mp4"


def _load_motion(path: str) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as data:
        required = ("fps", "body_pos_w", "body_quat_w", "body_names")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        body_names = _names(data["body_names"])
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (T, B, 3), got {body_pos.shape}.")
    if body_quat.shape[:2] != body_pos.shape[:2] or body_quat.shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape (T, B, 4), got {body_quat.shape}.")
    if len(body_names) != body_pos.shape[1]:
        raise ValueError(f"body_names length {len(body_names)} does not match body count {body_pos.shape[1]}.")
    return fps, body_pos, body_quat, body_names


def _sole_positions(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: list[str],
    foot_names: tuple[str, ...],
    sole_offsets: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    indexes = [body_names.index(name) for name in foot_names if name in body_names]
    matched_names = [body_names[index] for index in indexes]
    if not indexes:
        return [], np.empty((body_pos.shape[0], 0, 3), dtype=np.float64)
    offsets = np.asarray([sole_offsets.get(name, np.zeros(3, dtype=np.float64)) for name in matched_names])
    rotated_offsets = _quat_rotate_wxyz(body_quat[:, indexes, :], offsets[None, :, :])
    return matched_names, body_pos[:, indexes, :] + rotated_offsets


def _draw_preview(
    *,
    motion_file: str,
    output: Path,
    fps: float,
    render_fps: float,
    stride: int,
    body_pos: np.ndarray,
    sole_pos: np.ndarray,
    body_names: list[str],
    foot_names: list[str],
    start_frame: int,
    end_frame_exclusive: int,
    ground_z: float,
    dpi: int,
    width: float,
    height: float,
) -> None:
    frames = np.arange(start_frame, end_frame_exclusive, stride, dtype=np.int64)
    if frames.size == 0:
        raise ValueError("No frames selected for rendering.")

    edges = _skeleton_edges(body_names)
    segment_body = body_pos[start_frame:end_frame_exclusive]
    segment_sole = sole_pos[start_frame:end_frame_exclusive] if sole_pos.size else sole_pos
    combined_x = segment_body[..., 0]
    combined_y = segment_body[..., 1]
    combined_z = segment_body[..., 2]
    if segment_sole.size:
        combined_x = np.concatenate([combined_x.reshape(-1), segment_sole[..., 0].reshape(-1)])
        combined_y = np.concatenate([combined_y.reshape(-1), segment_sole[..., 1].reshape(-1)])
        combined_z = np.concatenate([combined_z.reshape(-1), segment_sole[..., 2].reshape(-1)])

    xlim = _axis_limits(np.asarray(combined_x), pad=0.15)
    ylim = _axis_limits(np.asarray(combined_y), pad=0.15)
    zlim = _axis_limits(np.concatenate([np.asarray(combined_z).reshape(-1), np.asarray([ground_z])]), pad=0.08)
    if zlim[0] > ground_z - 0.12:
        zlim = (ground_z - 0.12, zlim[1])

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=(2.2, 1.0), hspace=0.35, wspace=0.25)
    ax_side = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_foot = fig.add_subplot(gs[1, :])

    ax_side.set_title("side view: x / z")
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.set_xlim(xlim)
    ax_side.set_ylim(zlim)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.axhline(ground_z, color="tab:red", linewidth=1.2, alpha=0.8, label="ground")
    ax_side.grid(True, alpha=0.25)

    ax_top.set_title("top view: x / y")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_xlim(xlim)
    ax_top.set_ylim(ylim)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, alpha=0.25)

    frame_axis = np.arange(start_frame, end_frame_exclusive)
    if segment_sole.size:
        sole_height = segment_sole[..., 2] - ground_z
        foot_ylim = _axis_limits(np.concatenate([sole_height.reshape(-1), np.asarray([0.0])]), pad=0.02, minimum_span=0.12)
        for index, name in enumerate(foot_names):
            ax_foot.plot(frame_axis, sole_height[:, index], linewidth=1.5, label=name)
        min_height = float(np.min(sole_height))
    else:
        foot_ylim = (-0.08, 0.12)
        min_height = float("nan")
    ax_foot.axhline(0.0, color="tab:red", linewidth=1.2, alpha=0.8)
    cursor_line = ax_foot.axvline(start_frame, color="black", linewidth=1.0, alpha=0.7)
    ax_foot.set_xlim(start_frame, end_frame_exclusive - 1)
    ax_foot.set_ylim(foot_ylim)
    ax_foot.set_xlabel("frame")
    ax_foot.set_ylabel("sole height over ground (m)")
    ax_foot.grid(True, alpha=0.25)
    if foot_names:
        ax_foot.legend(loc="upper right", fontsize=8)

    side_lines = [ax_side.plot([], [], color="0.25", linewidth=1.8)[0] for _ in edges]
    top_lines = [ax_top.plot([], [], color="0.25", linewidth=1.8)[0] for _ in edges]
    side_points = ax_side.scatter([], [], s=14, c="tab:blue", alpha=0.75)
    top_points = ax_top.scatter([], [], s=14, c="tab:blue", alpha=0.75)
    side_feet = [
        ax_side.scatter([], [], s=55, marker="x", linewidths=2.0, label=name)
        for name in foot_names
    ]
    top_feet = [
        ax_top.scatter([], [], s=55, marker="x", linewidths=2.0, label=name)
        for name in foot_names
    ]
    title = fig.suptitle("", fontsize=10)

    def update(frame: int) -> None:
        points = body_pos[frame]
        for artist, (left, right) in zip(side_lines, edges):
            artist.set_data([points[left, 0], points[right, 0]], [points[left, 2], points[right, 2]])
        for artist, (left, right) in zip(top_lines, edges):
            artist.set_data([points[left, 0], points[right, 0]], [points[left, 1], points[right, 1]])
        side_points.set_offsets(np.column_stack([points[:, 0], points[:, 2]]))
        top_points.set_offsets(np.column_stack([points[:, 0], points[:, 1]]))

        if sole_pos.size:
            soles = sole_pos[frame]
            for index, artist in enumerate(side_feet):
                color = "tab:red" if soles[index, 2] < ground_z else "tab:green"
                artist.set_offsets([[soles[index, 0], soles[index, 2]]])
                artist.set_color(color)
            for index, artist in enumerate(top_feet):
                color = "tab:red" if soles[index, 2] < ground_z else "tab:green"
                artist.set_offsets([[soles[index, 0], soles[index, 1]]])
                artist.set_color(color)
        cursor_line.set_xdata([frame, frame])
        title.set_text(
            f"{motion_file} | frame {frame}/{end_frame_exclusive - 1} | "
            f"source_fps={fps:g}, render_fps={render_fps:g}, min_sole_height={min_height:.4f} m"
        )

    extension = output.suffix.lower()
    if extension not in {".mp4", ".gif"}:
        raise ValueError(f"Unsupported output extension '{extension}'. Use .mp4 or .gif.")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer_kwargs = {"fps": render_fps}
    if extension == ".mp4":
        writer_kwargs.update({"codec": "libx264", "quality": 8, "macro_block_size": 16})

    with imageio.get_writer(output, **writer_kwargs) as writer:
        for count, frame in enumerate(frames, start=1):
            update(int(frame))
            fig.canvas.draw()
            width_px, height_px = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height_px, width_px, 4)
            writer.append_data(rgba[..., :3].copy())
            if count % 50 == 0:
                print(f"[INFO]: Rendered {count}/{frames.size} frames", flush=True)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    fps, body_pos, body_quat, body_names = _load_motion(args.motion_file)
    motion_length = body_pos.shape[0]
    start_frame = int(args.start_frame)
    end_frame_exclusive = int(args.end_frame_exclusive) if args.end_frame_exclusive is not None else motion_length
    if not 0 <= start_frame < end_frame_exclusive <= motion_length:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame_exclusive}) for motion length {motion_length}."
        )
    if args.render_fps <= 0:
        raise ValueError("--render_fps must be positive.")
    stride = int(args.stride) if args.stride is not None else max(1, round(fps / args.render_fps))
    if stride <= 0:
        raise ValueError("--stride must be positive.")

    ground_z, configured_foot_names, sole_offsets = _load_ground_config(args.quality_config)
    matched_foot_names, sole_pos = _sole_positions(body_pos, body_quat, body_names, configured_foot_names, sole_offsets)
    output = Path(args.output) if args.output else _auto_output_path(args.motion_file, start_frame, end_frame_exclusive)

    if sole_pos.size:
        segment_heights = sole_pos[start_frame:end_frame_exclusive, :, 2] - ground_z
        print("[INFO]: Matched foot bodies:", ", ".join(matched_foot_names), flush=True)
        print(f"[INFO]: Minimum sole height over ground: {float(np.min(segment_heights)):.6f} m", flush=True)
    else:
        print("[WARN]: No configured foot body names were found; foot-height panel will be empty.", flush=True)
    print(f"[INFO]: Rendering {len(range(start_frame, end_frame_exclusive, stride))} frames to {output}", flush=True)

    _draw_preview(
        motion_file=args.motion_file,
        output=output,
        fps=fps,
        render_fps=args.render_fps,
        stride=stride,
        body_pos=body_pos,
        sole_pos=sole_pos,
        body_names=body_names,
        foot_names=matched_foot_names,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        ground_z=ground_z,
        dpi=args.dpi,
        width=args.width,
        height=args.height,
    )
    print(f"[INFO]: Wrote preview video: {output}", flush=True)


if __name__ == "__main__":
    main()
