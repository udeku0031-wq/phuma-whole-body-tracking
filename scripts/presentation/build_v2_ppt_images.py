#!/usr/bin/env python3
"""Build presentation images and inject them into a PPTX copy."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "outputs" / "presentation_v2_assets"
SLIDE_MEDIA_DIR = "ppt/slides/media"

NAVY = (7, 55, 99)
BLUE = (30, 128, 206)
SKY = (112, 190, 230)
ORANGE = (244, 146, 40)
GREEN = (70, 160, 105)
RED = (212, 76, 76)
PURPLE = (134, 94, 190)
MUTED = (86, 112, 139)
GRID = (211, 226, 239)
BG = (247, 252, 255)
PANEL = (255, 255, 255)

LINKS = [
    (0, 3), (3, 6), (6, 9),
    (0, 1), (1, 4), (4, 7), (7, 10), (10, 14), (14, 18),
    (0, 2), (2, 5), (5, 8), (8, 11), (11, 15), (15, 19),
    (9, 12), (12, 16), (16, 20), (20, 22), (22, 24), (24, 26), (26, 28),
    (9, 13), (13, 17), (17, 21), (21, 23), (23, 25), (25, 27), (27, 29),
]

CLUSTER_COLORS = [
    (38, 115, 190),
    (238, 143, 45),
    (74, 158, 100),
    (202, 76, 76),
    (137, 96, 190),
    (106, 153, 178),
    (190, 129, 60),
    (79, 142, 170),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def rounded_panel(draw: ImageDraw.ImageDraw, xy, radius=18, fill=PANEL, outline=(186, 218, 238), width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def save_image(img: Image.Image, name: str) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / name
    img.save(path)
    return path


def load_motion(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["body_pos_w"], dtype=float)


def project(points: np.ndarray) -> np.ndarray:
    return points[..., [0, 2]]


def pose_transform(frames_xy: np.ndarray, rect, keep_world_x: bool = False):
    if keep_world_x:
        xy = frames_xy.copy()
        xy[..., 0] -= np.nanmean(xy[:, 0, 0])
    else:
        root = frames_xy[:, 0:1, :].copy()
        xy = frames_xy - root
    x0, y0, x1, y1 = rect
    lo = np.nanmin(xy.reshape(-1, 2), axis=0)
    hi = np.nanmax(xy.reshape(-1, 2), axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1]) * 0.78
    center_data = (lo + hi) * 0.5
    center_px = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.53])

    def tr(frame_xy: np.ndarray) -> np.ndarray:
        if keep_world_x:
            local = frame_xy.copy()
            local[:, 0] -= np.nanmean(frames_xy[:, 0, 0])
        else:
            local = frame_xy - frame_xy[0]
        out = (local - center_data) * scale
        out[:, 1] *= -1
        out += center_px
        return out

    return tr


def draw_skeleton(draw: ImageDraw.ImageDraw, pts: np.ndarray, color, width=4, joint_radius=3):
    pts_int = [(float(x), float(y)) for x, y in pts]
    for a, b in LINKS:
        if a < len(pts_int) and b < len(pts_int):
            draw.line([pts_int[a], pts_int[b]], fill=color, width=width, joint="curve")
    for x, y in pts_int:
        draw.ellipse((x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius), fill=color)


def motion_thumbnail(path: Path, title: str, size=(640, 420), color=BLUE, accent=ORANGE) -> Image.Image:
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (10, 10, size[0] - 10, size[1] - 10), radius=24)
    draw.text((34, 26), title, font=font(30, True), fill=NAVY)
    draw.line((34, 70, size[0] - 34, 70), fill=(205, 226, 240), width=2)

    pos = load_motion(path)
    idx = np.linspace(0, len(pos) - 1, 7).astype(int)
    frames = project(pos[idx])
    tr = pose_transform(frames, (52, 86, size[0] - 52, size[1] - 52))
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for i, frame in enumerate(frames):
        a = 60 + i * 25
        c = color if i == len(frames) - 1 else (*color, a)
        if i == len(frames) - 1:
            c = (*accent, 245)
            w = 6
        else:
            w = 4
        draw_skeleton(odraw, tr(frame), c, width=w, joint_radius=3)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


def motion_strip(paths: list[Path], title: str, subtitle: str, size=(1500, 430)) -> Image.Image:
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (12, 12, size[0] - 12, size[1] - 12), radius=30)
    draw.text((44, 32), title, font=font(44, True), fill=NAVY)
    draw.text((46, 90), subtitle, font=font(24), fill=MUTED)
    draw.line((44, 128, size[0] - 44, 128), fill=(204, 226, 240), width=2)

    lane_w = (size[0] - 110) / len(paths)
    colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
    for i, path in enumerate(paths):
        x0 = int(55 + i * lane_w)
        x1 = int(55 + (i + 1) * lane_w - 22)
        y0, y1 = 150, size[1] - 44
        pos = load_motion(path)
        idx = np.linspace(0, len(pos) - 1, 5).astype(int)
        frames = project(pos[idx])
        tr = pose_transform(frames, (x0, y0, x1, y1))
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for j, frame in enumerate(frames):
            alpha = 64 + j * 34
            draw_skeleton(odraw, tr(frame), (*colors[i % len(colors)], alpha), width=4, joint_radius=3)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
    return img


def pipeline_panel(path: Path, mode: str, title: str, size=(920, 260)) -> Image.Image:
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (8, 8, size[0] - 8, size[1] - 8), radius=18)
    draw.text((28, 22), title, font=font(30, True), fill=NAVY)
    pos = load_motion(path)
    idx = np.linspace(0, len(pos) - 1, 4).astype(int)
    frames = project(pos[idx])
    tr = pose_transform(frames, (40, 62, size[0] - 40, size[1] - 22), keep_world_x=(mode == "trajectory"))
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    if mode == "human":
        color = (90, 112, 132)
        for i, frame in enumerate(frames):
            draw_skeleton(odraw, tr(frame), (*color, 70 + i * 30), width=4, joint_radius=3)
    elif mode == "reference":
        for i, frame in enumerate(frames):
            draw_skeleton(odraw, tr(frame), (*BLUE, 80 + i * 35), width=4, joint_radius=3)
    else:
        for i, frame in enumerate(frames):
            ref = tr(frame)
            noisy = ref.copy()
            phase = (i + 1) * 0.9
            noisy[:, 0] += np.sin(np.arange(noisy.shape[0]) * 0.8 + phase) * 5
            noisy[:, 1] += np.cos(np.arange(noisy.shape[0]) * 0.7 + phase) * 4
            draw_skeleton(odraw, ref, (*BLUE, 80), width=4, joint_radius=3)
            draw_skeleton(odraw, noisy, (*ORANGE, 155), width=4, joint_radius=3)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def draw_chart_axes(draw, rect, xlim, ylim, xlabel="", ylabel="", title=""):
    x0, y0, x1, y1 = rect
    draw.line((x0, y1, x1, y1), fill=(140, 165, 188), width=2)
    draw.line((x0, y0, x0, y1), fill=(140, 165, 188), width=2)
    for i in range(5):
        y = y0 + i * (y1 - y0) / 4
        draw.line((x0, y, x1, y), fill=GRID, width=1)
    if title:
        draw.text((x0, y0 - 54), title, font=font(30, True), fill=NAVY)
    if xlabel:
        draw.text((x1 - text_width(draw, xlabel, font(18)), y1 + 18), xlabel, font=font(18), fill=MUTED)
    if ylabel:
        draw.text((x0, y0 - 24), ylabel, font=font(18), fill=MUTED)

    def tr(x, y):
        xx = x0 + (x - xlim[0]) / max(1e-9, xlim[1] - xlim[0]) * (x1 - x0)
        yy = y1 - (y - ylim[0]) / max(1e-9, ylim[1] - ylim[0]) * (y1 - y0)
        return xx, yy

    return tr


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def chart_concentration(size=(1000, 560)) -> Image.Image:
    rows = read_dicts(ROOT / "outputs/stage1_learning_gap_v1_diagnostics/sampling_summary.csv")
    names = ["M4", "M5", "old M7", "M7-Raw"]
    row_by = {r["method"]: r for r in rows}
    values = [
        float(row_by["M4"]["motion_top1pct_mass"]) * 100,
        float(row_by["M5"]["motion_top1pct_mass"]) * 100,
        float(row_by["M7"]["motion_top1pct_mass"]) * 100,
        float(row_by["M7Raw"]["motion_top1pct_mass"]) * 100,
    ]
    seg_values = [
        float(row_by["M4"]["segment_global_top1pct_mass"]) * 100,
        float(row_by["M5"]["segment_global_top1pct_mass"]) * 100,
        float(row_by["M7"]["segment_global_top1pct_mass"]) * 100,
        float(row_by["M7Raw"]["segment_global_top1pct_mass"]) * 100,
    ]
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (10, 10, size[0] - 10, size[1] - 10), radius=22)
    rect = (100, 112, size[0] - 56, size[1] - 96)
    ymax = max(values + seg_values) * 1.22
    tr = draw_chart_axes(draw, rect, (-0.6, 3.6), (0, ymax), ylabel="Top-1% mass (%)", title="Learning Gap raises sampling concentration")
    colors = [BLUE, ORANGE, RED, GREEN]
    bar_w = 36
    for i, (name, mv, sv, c) in enumerate(zip(names, values, seg_values, colors)):
        x, y = tr(i - 0.13, mv)
        xb, yb = tr(i - 0.13, 0)
        draw.rounded_rectangle((x - bar_w, y, x, yb), radius=6, fill=c)
        x2, y2 = tr(i + 0.25, sv)
        x2b, y2b = tr(i + 0.25, 0)
        draw.rounded_rectangle((x2 - bar_w, y2, x2, y2b), radius=6, fill=tuple(int(v * 0.75) for v in c))
        draw.text((tr(i, 0)[0] - text_width(draw, name, font(20, True)) / 2, rect[3] + 24), name, font=font(20, True), fill=NAVY)
        label = f"{mv:.1f}"
        draw.text((x - bar_w, y - 28), label, font=font(18, True), fill=c)
    draw.rectangle((646, 54, 672, 72), fill=BLUE)
    draw.text((682, 46), "Motion", font=font(20), fill=MUTED)
    draw.rectangle((780, 54, 806, 72), fill=(22, 96, 154))
    draw.text((816, 46), "Segment", font=font(20), fill=MUTED)
    return img


def summarize_per_motion(path: Path) -> dict[str, float]:
    rows = read_dicts(path)
    success = np.array([float(r["success"]) for r in rows], dtype=float)
    comp = np.array([float(r["completion_ratio"]) for r in rows], dtype=float)
    cats = defaultdict(list)
    for r in rows:
        cats[r.get("category", "")].append(float(r["success"]))
    macro = np.mean([np.mean(v) for v in cats.values()]) if cats else float(np.mean(success))
    return {
        "micro_success_rate": float(np.mean(success)),
        "macro_success_rate": float(macro),
        "mean_completion_ratio": float(np.mean(comp)),
        "num_failures": float(np.sum(success < 0.5)),
    }


def checkpoint_series(label: str, base: Path) -> list[tuple[int, float]]:
    cmp_path = base / "checkpoint_comparison.csv"
    if cmp_path.exists():
        out = []
        for r in read_dicts(cmp_path):
            out.append((int(r["iteration"]), float(r["macro_success_rate"])))
        return sorted(out)
    out = []
    for d in sorted(base.glob("model_*")):
        m = re.search(r"model_(\d+)", d.name)
        p = d / "per_motion.csv"
        if m and p.exists():
            out.append((int(m.group(1)), summarize_per_motion(p)["macro_success_rate"]))
    return sorted(out)


def chart_training_dynamics(size=(1080, 600)) -> Image.Image:
    series = {
        "M0": checkpoint_series("M0", ROOT / "evaluations/formal_v1/M0_seed42/validation_full"),
        "M4": checkpoint_series("M4", ROOT / "evaluations/formal_v1/M4_seed42/validation_full"),
        "old M7": checkpoint_series("old M7", ROOT / "evaluations/formal_v1/M7_seed42/validation_full"),
        "M7-Raw": checkpoint_series("M7-Raw", ROOT / "evaluations/formal_v1/M7Raw_seed42/validation_full"),
    }
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (10, 10, size[0] - 10, size[1] - 10), radius=22)
    rect = (100, 142, size[0] - 58, size[1] - 96)
    all_y = [y for s in series.values() for _, y in s]
    tr = draw_chart_axes(draw, rect, (9000, 34500), (min(all_y) - 0.015, max(all_y) + 0.012), xlabel="checkpoint iteration", ylabel="Macro success")
    draw.text((100, 46), "Full-validation checkpoint trajectory", font=font(30, True), fill=NAVY)
    colors = {"M0": BLUE, "M4": GREEN, "old M7": RED, "M7-Raw": ORANGE}
    for name, pts in series.items():
        pix = [tr(x, y) for x, y in pts]
        if len(pix) > 1:
            draw.line(pix, fill=colors[name], width=4)
        for x, y in pix:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[name])
    x = 100
    for name in series:
        draw.line((x, 102, x + 40, 102), fill=colors[name], width=5)
        draw.text((x + 50, 90), name, font=font(20, True), fill=NAVY)
        x += 190
    return img


def chart_raw_vs_gap(size=(980, 660)) -> Image.Image:
    snap = np.load(ROOT / "outputs/joint_gap_stage3_lambda_replay/snapshots/m7raw_ckpt_33999_joint_snapshot.npz", allow_pickle=True)
    err = np.asarray(snap["segment_joint_error"], dtype=float)
    raw = np.nanmean(err, axis=1)
    diff_bin = np.asarray(snap["difficulty_bin"], dtype=int)
    mask = np.asarray(snap["eligible_mask"], dtype=bool) & np.asarray(snap["observed_mask"], dtype=bool) & np.isfinite(raw)
    gap = np.zeros_like(raw)
    for b in sorted(set(diff_bin[mask].tolist())):
        bm = mask & (diff_bin == b)
        vals = raw[bm]
        mu = np.nanmean(vals)
        sig = max(float(np.nanstd(vals)), 1e-6)
        gap[bm] = np.maximum(0.0, (raw[bm] - mu) / sig)
    raw_m = raw[mask]
    gap_m = gap[mask]
    rng = np.random.default_rng(42)
    idx = np.arange(len(raw_m))
    if len(idx) > 4500:
        idx = rng.choice(idx, 4500, replace=False)
    x = raw_m[idx]
    y = gap_m[idx]
    corr = float(np.corrcoef(raw_m, gap_m)[0, 1])

    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (10, 10, size[0] - 10, size[1] - 10), radius=22)
    rect = (96, 118, size[0] - 58, size[1] - 92)
    xlim = (float(np.quantile(raw_m, 0.01)), float(np.quantile(raw_m, 0.995)))
    ylim = (0, float(np.quantile(gap_m, 0.995)) * 1.08)
    tr = draw_chart_axes(draw, rect, xlim, ylim, xlabel="mean joint error", ylabel="calibrated gap", title="Raw error vs difficulty-calibrated gap")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for xx, yy in zip(x, y):
        if xlim[0] <= xx <= xlim[1] and ylim[0] <= yy <= ylim[1]:
            px, py = tr(xx, yy)
            od.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(*BLUE, 55))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((size[0] - 312, 42, size[0] - 58, 94), radius=14, fill=(235, 247, 253), outline=(185, 219, 239), width=1)
    draw.text((size[0] - 292, 53), f"segments={int(mask.sum())}  r={corr:.2f}", font=font(22, True), fill=NAVY)
    return img


def cluster_pca(size=(1000, 420)) -> tuple[Image.Image, list[Path]]:
    data = np.load(ROOT / "outputs/module4_clusters_random6000_seed42_v1/motion_cluster_metadata.npz", allow_pickle=True)
    x = np.asarray(data["standardized_feature_matrix"], dtype=float)
    cid = np.asarray(data["cluster_id"], dtype=int)
    keys = np.asarray(data["motion_keys"]).astype(str)
    x0 = x - np.nanmean(x, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(np.nan_to_num(x0), full_matrices=False)
    pcs = x0 @ vh[:2].T
    rng = np.random.default_rng(7)
    sample = np.arange(len(pcs))
    if len(sample) > 3200:
        sample = rng.choice(sample, 3200, replace=False)

    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (8, 8, size[0] - 8, size[1] - 8), radius=20)
    rect = (70, 118, size[0] - 42, size[1] - 54)
    xlim = (float(np.quantile(pcs[:, 0], 0.01)), float(np.quantile(pcs[:, 0], 0.99)))
    ylim = (float(np.quantile(pcs[:, 1], 0.01)), float(np.quantile(pcs[:, 1], 0.99)))
    tr = draw_chart_axes(draw, rect, xlim, ylim, xlabel="PC1", ylabel="PC2")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in sample:
        xx, yy = pcs[i]
        if xlim[0] <= xx <= xlim[1] and ylim[0] <= yy <= ylim[1]:
            px, py = tr(xx, yy)
            od.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(*CLUSTER_COLORS[cid[i] % 8], 92))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((70, 30), "K=8 motion diversity clusters", font=font(28, True), fill=NAVY)
    xleg = 78
    for k in range(8):
        draw.ellipse((xleg, 76, xleg + 14, 90), fill=CLUSTER_COLORS[k])
        draw.text((xleg + 20, 67), f"C{k + 1}", font=font(18), fill=MUTED)
        xleg += 86

    reps = []
    for k in range(8):
        m = cid == k
        center = np.mean(x[m], axis=0)
        dist = np.linalg.norm(x[m] - center, axis=1)
        candidates = np.where(m)[0][np.argsort(dist)]
        chosen = None
        for c in candidates[:40]:
            p = ROOT / keys[c]
            if p.exists():
                chosen = p
                break
        if chosen is None:
            chosen = ROOT / keys[np.where(m)[0][0]]
        reps.append(chosen)
    return img, reps


def final_test_chart(size=(1150, 620)) -> Image.Image:
    rows = read_dicts(ROOT / "outputs/final_test/final_test_summary.csv")
    order = ["GlobalRaw", "M4", "D-only", "M7-Raw"]
    by = {r["method"]: r for r in rows}
    values = [float(by[k]["macro_success_rate"]) for k in order]
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (10, 10, size[0] - 10, size[1] - 10), radius=24)
    rect = (96, 118, size[0] - 56, size[1] - 96)
    tr = draw_chart_axes(draw, rect, (-0.5, len(order) - 0.5), (min(values) - 0.012, max(values) + 0.012), ylabel="Macro success", title="Final test: ablation overview")
    colors = [MUTED, GREEN, ORANGE, BLUE]
    for i, (name, v, c) in enumerate(zip(order, values, colors)):
        x, y = tr(i, v)
        xb, yb = tr(i, min(values) - 0.012)
        draw.rounded_rectangle((x - 45, y, x + 45, yb), radius=8, fill=c)
        draw.text((x - text_width(draw, name, font(20, True)) / 2, rect[3] + 22), name, font=font(20, True), fill=NAVY)
        draw.text((x - 42, y - 30), f"{v:.3f}", font=font(18, True), fill=c)
    return img


def ending_visual(paths: list[Path], size=(960, 1120)) -> Image.Image:
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    rounded_panel(draw, (12, 12, size[0] - 12, size[1] - 12), radius=34)
    draw.text((54, 46), "Quality-Gated Diversity Sampling", font=font(42, True), fill=NAVY)
    draw.text((56, 104), "M7-Raw experimental pipeline", font=font(26), fill=MUTED)
    y = 172
    blocks = [("Quality Gate", BLUE), ("Cluster Diversity", GREEN), ("Motion Raw Error", ORANGE), ("Segment Raw Error", PURPLE)]
    for label, c in blocks:
        draw.rounded_rectangle((72, y, size[0] - 72, y + 74), radius=20, fill=(255, 255, 255), outline=(188, 218, 238), width=2)
        draw.rectangle((72, y, 92, y + 74), fill=c)
        draw.text((122, y + 18), label, font=font(28, True), fill=NAVY)
        y += 92
    lane_y = 560
    colors = [BLUE, ORANGE, GREEN]
    for i, path in enumerate(paths[:3]):
        pos = load_motion(path)
        idx = np.linspace(0, len(pos) - 1, 5).astype(int)
        frames = project(pos[idx])
        x0 = 70 + i * 285
        x1 = x0 + 235
        tr = pose_transform(frames, (x0, lane_y, x1, size[1] - 88))
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for j, frame in enumerate(frames):
            draw_skeleton(od, tr(frame), (*colors[i], 65 + j * 38), width=5, joint_radius=3)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
    return img


def add_picture_xml(shape_id: int, rel_id: str, name: str, x: int, y: int, cx: int, cy: int) -> str:
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{name}"/>'
        f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def ensure_png_content_type(xml: str) -> str:
    if 'Extension="png"' in xml:
        return xml
    return xml.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')


def add_relationship(rels_xml: str, rel_id: str, target: str) -> str:
    if f'Id="{rel_id}"' in rels_xml:
        return rels_xml
    rel = (
        f'<Relationship Id="{rel_id}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="{target}"/>'
    )
    return rels_xml.replace("</Relationships>", rel + "</Relationships>")


def default_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>'
    )


def inject_images(input_pptx: Path, output_pptx: Path, placements: list[dict]) -> None:
    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_pptx, "r") as zin:
        contents = {name: zin.read(name) for name in zin.namelist()}

    content_types = contents["[Content_Types].xml"].decode("utf-8")
    contents["[Content_Types].xml"] = ensure_png_content_type(content_types).encode("utf-8")

    by_slide = defaultdict(list)
    for item in placements:
        by_slide[item["slide"]].append(item)

    shape_id = 9000
    for slide, items in by_slide.items():
        slide_name = f"ppt/slides/slide{slide}.xml"
        rels_name = f"ppt/slides/_rels/slide{slide}.xml.rels"
        slide_xml = contents[slide_name].decode("utf-8")
        rels_xml = contents.get(rels_name, default_rels_xml().encode("utf-8")).decode("utf-8")
        pic_xml = []
        for item in items:
            shape_id += 1
            rel_id = f"rIdGen{shape_id}"
            media_name = item["media_name"]
            rels_xml = add_relationship(rels_xml, rel_id, f"media/{media_name}")
            pic_xml.append(add_picture_xml(shape_id, rel_id, media_name, *item["rect"]))
        slide_xml = slide_xml.replace("</p:spTree>", "".join(pic_xml) + "</p:spTree>")
        contents[slide_name] = slide_xml.encode("utf-8")
        contents[rels_name] = rels_xml.encode("utf-8")

    with zipfile.ZipFile(output_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in contents.items():
            zout.writestr(name, data)
        for item in placements:
            media_name = item["media_name"]
            zout.write(item["path"], f"{SLIDE_MEDIA_DIR}/{media_name}")


def make_assets() -> tuple[list[dict], dict[str, Path]]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    motion_paths = {
        "locomotion": ROOT / "PHUMA_wbt_motions/g1_all/idea400/Walking_forward_during_standing_clip1_chunk_0000.npz",
        "turning": ROOT / "PHUMA_wbt_motions/g1_all/idea400/Turn_the_neck_during_walking_clip1_chunk_0000.npz",
        "squat": ROOT / "PHUMA_wbt_motions/g1_all/humman/The_weight_of_Bodyweight_Squat_Squat_0_clip1_chunk_0000.npz",
        "upper": ROOT / "PHUMA_wbt_motions/g1_all/fitness/subset_0025/Squat_To_Upper_Body_Circle_chunk_0001.npz",
        "balance": ROOT / "PHUMA_wbt_motions/g1_all/humman/Single_Leg_Balance_Single_legs_Balance_0_clip1_chunk_0000.npz",
        "single_leg": ROOT / "PHUMA_wbt_motions/g1_all/fitness/subset_0025/Single_Leg_Toe_Touch_R_clip_2_chunk_0000.npz",
        "complex": ROOT / "PHUMA_wbt_motions/g1_all/animation/Ways_to_Jump_+_Sit_+_Fall_Expressive_clip1_clip1_chunk_0000.npz",
    }
    fallback = next((ROOT / "PHUMA_wbt_motions/g1_all").rglob("*.npz"))
    for key, path in list(motion_paths.items()):
        if not path.exists():
            motion_paths[key] = fallback

    assets = {}
    strip_paths = [motion_paths[k] for k in ["locomotion", "squat", "balance", "turning", "complex"]]
    assets["cover"] = save_image(motion_strip(strip_paths, "PHUMA-WBT Motion Library", "Real G1 motion trajectories rendered from local NPZ files"), "slide01_cover_motion_library.png")

    labels = [
        ("locomotion", "Locomotion", BLUE),
        ("turning", "Turning", ORANGE),
        ("squat", "Squat", GREEN),
        ("upper", "Upper Body", PURPLE),
        ("balance", "Balance", RED),
        ("single_leg", "Single Leg", (56, 150, 168)),
        ("complex", "Complex Whole-Body", NAVY),
    ]
    for key, title, c in labels:
        assets[f"slide03_{key}"] = save_image(motion_thumbnail(motion_paths[key], title, color=c, accent=ORANGE), f"slide03_{key}.png")

    pipe_path = motion_paths["locomotion"]
    assets["pipe_human"] = save_image(pipeline_panel(pipe_path, "human", "PHUMA human motion"), "slide05_pipeline_human.png")
    assets["pipe_ref"] = save_image(pipeline_panel(pipe_path, "reference", "G1 retargeted reference"), "slide05_pipeline_reference.png")
    assets["pipe_track"] = save_image(pipeline_panel(pipe_path, "tracking", "Policy tracking overlay"), "slide05_pipeline_tracking.png")

    assets["raw_vs_gap"] = save_image(chart_raw_vs_gap(), "slide09_raw_vs_gap.png")
    assets["concentration"] = save_image(chart_concentration(), "slide10_concentration.png")
    assets["training"] = save_image(chart_training_dynamics(), "slide10_training_dynamics.png")
    pca_img, cluster_reps = cluster_pca()
    assets["cluster_pca"] = save_image(pca_img, "slide12_cluster_pca.png")
    for i, path in enumerate(cluster_reps, start=1):
        assets[f"cluster_{i}"] = save_image(motion_thumbnail(path, f"Cluster {i}", size=(420, 260), color=CLUSTER_COLORS[i - 1], accent=ORANGE), f"slide12_cluster_{i}.png")
    assets["ending"] = save_image(ending_visual(strip_paths), "slide15_ending_visual.png")
    assets["final_test"] = save_image(final_test_chart(), "final_test_ablation_overview.png")

    placements = [
        {"slide": 1, "path": assets["cover"], "media_name": "generated_slide01_cover.png", "rect": (4533900, 3514725, 3124200, 660400)},
        {"slide": 3, "path": assets["slide03_locomotion"], "media_name": "generated_slide03_locomotion.png", "rect": (533400, 1638300, 1714500, 1266825)},
        {"slide": 3, "path": assets["slide03_turning"], "media_name": "generated_slide03_turning.png", "rect": (2343150, 1638300, 1724025, 1266825)},
        {"slide": 3, "path": assets["slide03_squat"], "media_name": "generated_slide03_squat.png", "rect": (4162425, 1638300, 1714500, 1266825)},
        {"slide": 3, "path": assets["slide03_upper"], "media_name": "generated_slide03_upper.png", "rect": (533400, 3000375, 1714500, 1266825)},
        {"slide": 3, "path": assets["slide03_balance"], "media_name": "generated_slide03_balance.png", "rect": (2343150, 3000375, 1724025, 1266825)},
        {"slide": 3, "path": assets["slide03_single_leg"], "media_name": "generated_slide03_single_leg.png", "rect": (4162425, 3000375, 1714500, 1266825)},
        {"slide": 3, "path": assets["slide03_complex"], "media_name": "generated_slide03_complex.png", "rect": (533400, 4362450, 5343525, 1266825)},
        {"slide": 5, "path": assets["pipe_human"], "media_name": "generated_slide05_human.png", "rect": (533400, 3590925, 3286125, 723900)},
        {"slide": 5, "path": assets["pipe_ref"], "media_name": "generated_slide05_reference.png", "rect": (4238625, 3590925, 3295650, 723900)},
        {"slide": 5, "path": assets["pipe_track"], "media_name": "generated_slide05_tracking.png", "rect": (7953375, 3590925, 3705225, 723900)},
        {"slide": 9, "path": assets["raw_vs_gap"], "media_name": "generated_slide09_raw_vs_gap.png", "rect": (7280000, 1350000, 4260000, 3450000)},
        {"slide": 10, "path": assets["concentration"], "media_name": "generated_slide10_concentration.png", "rect": (720000, 2470000, 5020000, 2460000)},
        {"slide": 10, "path": assets["training"], "media_name": "generated_slide10_training.png", "rect": (6420000, 2030000, 4950000, 2750000)},
        {"slide": 12, "path": assets["cluster_pca"], "media_name": "generated_slide12_cluster_pca.png", "rect": (7267575, 1162050, 4391025, 1666875)},
        {"slide": 12, "path": assets["cluster_1"], "media_name": "generated_slide12_cluster_1.png", "rect": (7267575, 3200400, 1038225, 657225)},
        {"slide": 12, "path": assets["cluster_2"], "media_name": "generated_slide12_cluster_2.png", "rect": (8382000, 3200400, 1047750, 657225)},
        {"slide": 12, "path": assets["cluster_3"], "media_name": "generated_slide12_cluster_3.png", "rect": (9505950, 3200400, 1038225, 657225)},
        {"slide": 12, "path": assets["cluster_4"], "media_name": "generated_slide12_cluster_4.png", "rect": (10620375, 3200400, 1038225, 657225)},
        {"slide": 12, "path": assets["cluster_5"], "media_name": "generated_slide12_cluster_5.png", "rect": (7267575, 3933825, 1038225, 657225)},
        {"slide": 12, "path": assets["cluster_6"], "media_name": "generated_slide12_cluster_6.png", "rect": (8382000, 3933825, 1047750, 657225)},
        {"slide": 12, "path": assets["cluster_7"], "media_name": "generated_slide12_cluster_7.png", "rect": (9505950, 3933825, 1038225, 657225)},
        {"slide": 12, "path": assets["cluster_8"], "media_name": "generated_slide12_cluster_8.png", "rect": (10620375, 3933825, 1038225, 657225)},
        {"slide": 15, "path": assets["ending"], "media_name": "generated_slide15_ending.png", "rect": (6820000, 710000, 4320000, 5200000)},
    ]
    return placements, assets


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_v2_ppt_images.py INPUT.pptx OUTPUT.pptx", file=sys.stderr)
        return 2
    input_pptx = Path(sys.argv[1])
    output_pptx = Path(sys.argv[2])
    if not input_pptx.exists():
        print(f"missing input pptx: {input_pptx}", file=sys.stderr)
        return 1
    placements, assets = make_assets()
    inject_images(input_pptx, output_pptx, placements)
    with zipfile.ZipFile(output_pptx) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"bad pptx zip member: {bad}")
    manifest = {
        "input_pptx": str(input_pptx),
        "output_pptx": str(output_pptx),
        "asset_dir": str(ASSET_DIR),
        "num_inserted_images": len(placements),
        "assets": {k: str(v) for k, v in assets.items()},
    }
    (ASSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
