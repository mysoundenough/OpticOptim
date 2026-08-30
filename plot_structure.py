#!/usr/bin/env python3
"""
plot_structure.py — 光学结构专用绘制模块
===========================================
专门用于绘制优化出的镜头结构（离轴反射系统 / EUV 物镜）。

提供两组绘制：
1. plot_full_views(lens, outdir)：整体结构
   - view_3d.png      3D 透视
   - view_x.png       x 视角（沿 x 轴看 → y-z 平面）
   - view_y.png       y 视角（沿 y 轴看 → x-z 平面）
   - view_z.png       z 视角（沿 z 轴看 → x-y 平面俯视）
   - view_all.png     四合一合成图

2. plot_per_mirror(lens, outdir)：每面镜子单独绘制（突出该镜子的工作区域/感光区）
   - M1_3d.png / M1_x.png / M1_y.png / M1_z.png ... M6 同理（6 镜 × 4 视角 = 24 张）
   每张图突出显示该镜子（其余镜子淡化），并标注物面视场。

特性：
- 光线按"反射镜子"分段着色（灰=入射段，M1蓝 M2橙 M3绿 M4红 M5紫 M6棕）
- 每面镜子的工作区域（footprint 环带/区域）高亮
- 物面视场（环形 + 内接矩形 + 视场点）在 z 视角叠加
- 高清（dpi=250），放大可读

用法：
    from plot_structure import plot_full_views, plot_per_mirror
    lens = LensSystem.from_json('output/xxx.json')
    plot_full_views(lens, 'output/figures_structure/full')
    plot_per_mirror(lens, 'output/figures_structure/per_mirror')

    # 或 CLI：
    python plot_structure.py --input output/xxx.json --outdir output/figures_structure
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lens_schema import LensSystem, SurfaceType
from raytrace import (compute_surface_z, generate_offaxis_rays,
                      generate_rect_offaxis_rays, intersect_surface,
                      surface_normal, reflect, aspheric_sag, aspheric_coeffs)
from merit_function import merit_detail

# ===========================================================================
# 配色（每面镜子一种颜色）
# ===========================================================================
MIRROR_COLORS = {
    0: "#888888",          # 入射段（物面→M1）
    1: "#1f77b4",          # M1 蓝
    2: "#ff7f0e",          # M2 橙
    3: "#2ca02c",          # M3 绿
    4: "#d62728",          # M4 红
    5: "#9467bd",          # M5 紫
    6: "#8c564b",          # M6 棕
}
DPI = 250


def get_ray_generator(lens):
    """按系统配置选择光线生成器。"""
    if lens.metadata.optical_constraints.rectangular_fov:
        return generate_rect_offaxis_rays
    return generate_offaxis_rays


def trace_segments(lens):
    """
    追迹全部光线，返回分段路径。

    返回:
        segments: list[(p_start(3,), p_end(3,), reflect_mirror_id, ray_idx)]
                  reflect_mirror_id: 0=入射段(物面→M1), i=M_i 反射后段
        pos/dirs: 光线起点/方向（供物面视场绘制）
    """
    gen = get_ray_generator(lens)
    pos, dirs, wl, fov = gen(lens)
    surf_z = compute_surface_z(lens.surfaces)
    segments = []
    for k in range(len(pos)):
        p = pos[k:k + 1].copy()
        dd = dirs[k:k + 1].copy()
        p0 = p[0].copy()
        for i in range(1, len(lens.surfaces)):
            s = lens.surfaces[i]
            t = intersect_surface(p, dd, surf_z[i], s)
            if not np.isfinite(t[0]):
                break
            p2 = p + t * dd
            rid = i - 1 if i > 1 else 0
            segments.append((p0.copy(), p2[0].copy(), rid, k))
            if s.type == SurfaceType.IMAGE:
                break
            asp = s.aspheric
            A4, A6, A8, A10, A12, A14 = aspheric_coeffs(s)
            norm = surface_normal(p2[:, 0] - s.decenter_x, p2[:, 1] - s.decenter_y,
                                  p2[:, 2] - surf_z[i], s.curvature,
                                  asp.k if asp else 0.0, A4, A6, A8, A10, A12, A14)
            if s.is_reflective:
                dd = reflect(dd, norm)
            p = p2
            p0 = p2[0].copy()
    return segments, pos


def footprint_regions(lens, pos=None):
    """每面镜子的工作区域（r 环带），用于高亮绘制。"""
    gen = get_ray_generator(lens)
    if pos is None:
        pos, dirs, wl, fov = gen(lens)
    else:
        _, dirs, wl, fov = gen(lens)
    surf_z = compute_surface_z(lens.surfaces)
    foot = {}
    p_all = pos.copy()
    d_all = dirs.copy()
    for i in range(1, len(lens.surfaces)):
        s = lens.surfaces[i]
        t = intersect_surface(p_all, d_all, surf_z[i], s)
        h = np.isfinite(t)
        if h.any():
            rl = np.sqrt((p_all[h, 0] + t[h] * d_all[h, 0] - s.decenter_x) ** 2 +
                         (p_all[h, 1] + t[h] * d_all[h, 1] - s.decenter_y) ** 2)
            foot[i] = (float(rl.min()), float(rl.max()))
        if not h.any():
            break
        p_all[h] = p_all[h] + t[h, np.newaxis] * d_all[h]
        if s.type == SurfaceType.IMAGE:
            break
        asp = s.aspheric
        A4, A6, A8, A10, A12, A14 = aspheric_coeffs(s)
        norm = surface_normal(p_all[h, 0] - s.decenter_x, p_all[h, 1] - s.decenter_y,
                              p_all[h, 2] - surf_z[i], s.curvature,
                              asp.k if asp else 0.0, A4, A6, A8, A10, A12, A14)
        if s.is_reflective:
            d_all[h] = reflect(d_all[h], norm)
    return foot


def _title_str(lens):
    d = merit_detail(lens)
    res = d["trace_result"]
    cons = lens.metadata.optical_constraints
    return ("rays=%d/%d depth=%.3f | obj h=%.0f img=%.0f mag=%.2f na=%.3f" % (
        int(res.active.sum()), len(res.active), res.obscuration_depth,
        cons.object_height_mm, cons.image_z_fixed, cons.magnification_target,
        cons.na_target))


# ===========================================================================
# 绘制单个视角（3D / x / y / z）
# ===========================================================================

def _draw_mirror_3d(ax, lens, foot, surf_z, highlight=None, faint_others=True,
                    only_mirror=None):
    th = np.linspace(0, 2 * np.pi, 72)
    for i in range(1, 7):
        if only_mirror is not None and i != only_mirror:
            continue  # 只画指定镜子
        if i not in foot:
            continue
        s = lens.surfaces[i]
        z_v = surf_z[i]
        rmin, rmax = foot[i]
        asp = s.aspheric
        A4, A6, A8, A10, A12, A14 = aspheric_coeffs(s)
        rr = np.linspace(rmin, rmax, 14)
        R, TH = np.meshgrid(rr, th)
        Zz = z_v + aspheric_sag(R, s.curvature, asp.k if asp else 0.0,
                                A4, A6, A8, A10, A12, A14)
        Xg = s.decenter_x + R * np.cos(TH)   # 全局坐标
        Yg = s.decenter_y + R * np.sin(TH)
        if highlight is not None and i != highlight:
            if not faint_others:
                continue
            ax.plot_surface(Xg, Yg, Zz, color=MIRROR_COLORS[i],
                            alpha=0.08, edgecolor="none", rstride=1, cstride=4)
        else:
            ax.plot_surface(Xg, Yg, Zz, color=MIRROR_COLORS[i],
                            alpha=0.5, edgecolor="none", rstride=1, cstride=4,
                            label="M%d" % i if highlight is None else None)
    ax.plot([36.8], [0], "go", ms=10, label="fov center")
    ax.plot([0], [0], [surf_z[-1]], "o", color="orange", ms=10, label="image")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=22, azim=-58)


def _draw_mirror_2d(ax, lens, foot, surf_z, plane="yz", highlight=None, faint_others=True,
                    only_mirror=None):
    """plane: yz(沿x看) / xz(沿y看) / xy(沿z看)"""
    for i in range(1, 7):
        if only_mirror is not None and i != only_mirror:
            continue  # 只画指定镜子
        s = lens.surfaces[i]
        z_v = surf_z[i]
        rr = np.linspace(-max(s.semi_aperture, 10.0), max(s.semi_aperture, 10.0), 401)
        asp = s.aspheric
        A4, A6, A8, A10, A12, A14 = aspheric_coeffs(s)
        sag = aspheric_sag(np.abs(rr), s.curvature, asp.k if asp else 0.0,
                           A4, A6, A8, A10, A12, A14)
        alpha_m = 0.85 if (highlight is None or i == highlight) else 0.25
        lw_m = 1.2 if (highlight is None or i == highlight) else 0.8
        if plane in ("yz", "xz"):
            # 局部坐标 rr → 全局坐标（yz: y+decenter_y, xz: x+decenter_x）
            rr_global = rr + (s.decenter_y if plane == "yz" else s.decenter_x)
            ax.plot(z_v + sag, rr_global, color=MIRROR_COLORS[i], lw=lw_m, alpha=alpha_m)
            if i in foot:
                rmin, rmax = foot[i]
                m = (np.abs(rr) >= rmin - 2) & (np.abs(rr) <= rmax + 2)
                ax.plot(z_v + sag[m], rr_global[m], color=MIRROR_COLORS[i], lw=4.0 * lw_m, alpha=alpha_m)
        else:  # xy 俯视：画环带内外圆
            if i not in foot:
                continue
            rmin, rmax = foot[i]
            th2 = np.linspace(0, 2 * np.pi, 90)
            ax.plot(rmin * np.cos(th2), rmin * np.sin(th2), "--", color=MIRROR_COLORS[i],
                    lw=1.2, alpha=alpha_m)
            ax.plot(rmax * np.cos(th2), rmax * np.sin(th2), "-", color=MIRROR_COLORS[i],
                    lw=2.0, alpha=alpha_m)
            ax.text(rmax + 15, 10, "M%d z=%d" % (i, z_v), fontsize=8,
                    color=MIRROR_COLORS[i] if alpha_m > 0.5 else "#aaaaaa")


def _plot_segments(ax, segments, plane, rid_filter=None):
    for (a, b, rid, k) in segments:
        if rid_filter is not None and rid not in rid_filter:
            continue  # 只画入射/出射段
        c = MIRROR_COLORS.get(rid, "#888888")
        if plane == "yz":
            ax.plot([a[2], b[2]], [a[1], b[1]], "-", color=c, lw=0.8, alpha=0.7)
        elif plane == "xz":
            ax.plot([a[2], b[2]], [a[0], b[0]], "-", color=c, lw=0.8, alpha=0.7)
        elif plane == "xy":
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=c, lw=0.8, alpha=0.7)
        else:  # 3d
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "-", color=c, lw=0.9, alpha=0.7)


def _draw_object_fov_xy(ax, lens):
    """z 视角叠加物面视场（环形 + 内接矩形 + 视场点）。"""
    cons = lens.metadata.optical_constraints
    if not cons.rectangular_fov:
        return
    th = np.linspace(0, 2 * np.pi, 120)
    cx, cy = cons.fov_center_x, cons.fov_center_y
    wx = cons.rect_half_wx_mm or cons.rect_half_width_mm or 50.0
    wy = cons.rect_half_wy_mm or cons.rect_half_width_mm or 50.0
    R_out = np.hypot(cx + wx, cy + wy)  # 外圆 = 矩形角距离
    R_in = min(abs(cx - wx), abs(cy - wy)) if (abs(cx - wx) > 0 and abs(cy - wy) > 0) else 0.0
    ax.plot(R_out * np.cos(th), R_out * np.sin(th), "g-", lw=2.0, label="annulus")
    if R_in > 1:
        ax.plot(R_in * np.cos(th), R_in * np.sin(th), "g--", lw=1.5, label="inner")
    rect = np.array([[cx - wx, cy - wy], [cx + wx, cy - wy],
                     [cx + wx, cy + wy], [cx - wx, cy + wy], [cx - wx, cy - wy]])
    ax.plot(rect[:, 0], rect[:, 1], "b-", lw=2.5, label="rect FOV")
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot(cx + sx * wx, cy + sy * wy, "ro", ms=7)
    ax.plot(cx, cy, "ro", ms=7)
    ax.text(cx, cy + 6, "fov center (%.0f, %.0f)" % (cx, cy), fontsize=9, color="r", ha="center")


def plot_single_view(lens, segments, foot, surf_z, view, save_path, highlight=None,
                    only_mirror=None, hit_highlight=False, anchors=None):
    """
    绘制单个视角。

    view: '3d' / 'x'(y-z) / 'y'(x-z) / 'z'(x-y)
    highlight: 突出显示的镜子 id（None=全部，其他淡化）
    only_mirror: 只画该镜子及其入射/出射光线（每镜单独图）
    hit_highlight: 加亮光线打到该镜子的位置
    """
    # 光线过滤：only_mirror 时只画入射段(rid=i-1)和出射段(rid=i)
    rid_filter = None
    if only_mirror is not None:
        rid_filter = (only_mirror - 1, only_mirror) if only_mirror > 1 else (0, only_mirror)
    # 命中点亮：only_mirror 的入射段终点（光线打到镜面的位置）
    hit_pts = []
    if hit_highlight and only_mirror is not None:
        inc_rid = only_mirror - 1 if only_mirror > 1 else 0
        for (a, b, rid, k) in segments:
            if rid == inc_rid:
                hit_pts.append(b)

    if view == "3d":
        fig = plt.figure(figsize=(13, 10), dpi=DPI)
        ax = fig.add_subplot(111, projection="3d")
        _draw_mirror_3d(ax, lens, foot, surf_z, highlight, only_mirror=only_mirror)
        _plot_segments(ax, segments, "3d", rid_filter)
        if hit_pts:
            hp = np.array(hit_pts)
            ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], c="yellow", s=60, marker="o",
                       edgecolors="k", zorder=10, label="hit points")
        if anchors is not None:
            an = np.array(anchors)
            ax.scatter(an[:, 0], an[:, 1], an[:, 2], c="yellow", s=110, marker="*",
                       edgecolors="k", zorder=15, label="anchor hits")
            for k, (ax0, ay0, az0) in enumerate(an):
                ax.text(ax0, ay0, az0, f" M{k+1}", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("3D | " + _title_str(lens), fontsize=12)
    else:
        if view == "x":
            fig, ax = plt.subplots(figsize=(16, 7), dpi=DPI)
            plane = "yz"
            title = "x view (y-z)"
            xl, yl = "z (mm)", "y (mm)"
        elif view == "y":
            fig, ax = plt.subplots(figsize=(16, 7), dpi=DPI)
            plane = "xz"
            title = "y view (x-z)"
            xl, yl = "z (mm)", "x (mm)"
        else:  # z
            fig, ax = plt.subplots(figsize=(11, 11), dpi=DPI)
            plane = "xy"
            title = "z view (x-y)"
            xl, yl = "x (mm)", "y (mm)"
        _draw_mirror_2d(ax, lens, foot, surf_z, plane, highlight, only_mirror=only_mirror)
        if plane == "xy" and only_mirror is None:
            _draw_object_fov_xy(ax, lens)
        _plot_segments(ax, segments, plane, rid_filter)
        if hit_pts:
            hp = np.array(hit_pts)
            if plane == "yz":
                ax.plot(hp[:, 2], hp[:, 1], "o", color="yellow", ms=8, mec="k", zorder=10)
            elif plane == "xz":
                ax.plot(hp[:, 2], hp[:, 0], "o", color="yellow", ms=8, mec="k", zorder=10)
            else:
                ax.plot(hp[:, 0], hp[:, 1], "o", color="yellow", ms=8, mec="k", zorder=10)
        if anchors is not None:
            an = np.array(anchors)
            if plane == "yz":
                ax.plot(an[:, 2], an[:, 1], "y*", ms=18, mec="k", zorder=15, label="anchor hits")
                for k, (ax0, ay0, az0) in enumerate(an):
                    ax.text(az0, ay0, f" M{k+1}", fontsize=9)
            elif plane == "xz":
                ax.plot(an[:, 2], an[:, 0], "y*", ms=18, mec="k", zorder=15, label="anchor hits")
            else:
                ax.plot(an[:, 0], an[:, 1], "y*", ms=18, mec="k", zorder=15, label="anchor hits")
        ax.axvline(surf_z[-1], color="orange", lw=2.5) if plane != "xy" else None
        ax.set_title(title + " | " + _title_str(lens), fontsize=12)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        lim = 1200
        if plane == "yz":
            ax.set_xlim(-100, surf_z[-1] + 100)
            ax.set_ylim(-lim, lim)
        elif plane == "xz":
            ax.set_xlim(-100, surf_z[-1] + 100)
            ax.set_ylim(-lim, lim)
        else:
            ax.set_xlim(-1400, 1400)
            ax.set_ylim(-1400, 1400)
        if plane == "xy":
            ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_quick(lens, save_path, title=None):
    """快速单张 y-z 布局快照（实时可视化用，快）。"""
    segments, _ = trace_segments(lens)
    foot = footprint_regions(lens)
    surf_z = compute_surface_z(lens.surfaces)
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    _draw_mirror_2d(ax, lens, foot, surf_z, "yz")
    _plot_segments(ax, segments, "yz")
    ax.axvline(surf_z[-1], color="orange", lw=2)
    ax.set_title(title or ("Quick layout | " + _title_str(lens)), fontsize=11)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlim(-100, surf_z[-1] + 100)
    ax.set_ylim(-1200, 1200)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===========================================================================
# 整体结构绘制
# ===========================================================================

def plot_full_views(lens, outdir="output/figures_structure/full"):
    """整体结构：3D + x/y/z 单图 + 四合一合成图。"""
    os.makedirs(outdir, exist_ok=True)
    segments, _ = trace_segments(lens)
    foot = footprint_regions(lens)
    surf_z = compute_surface_z(lens.surfaces)

    for view in ("3d", "x", "y", "z"):
        p = os.path.join(outdir, "view_%s.png" % view)
        plot_single_view(lens, segments, foot, surf_z, view, p)
        print("  saved:", p)

    # 四合一
    fig = plt.figure(figsize=(20, 12), dpi=DPI)
    ax = fig.add_subplot(221, projection="3d")
    _draw_mirror_3d(ax, lens, foot, surf_z)
    _plot_segments(ax, segments, "3d")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("3D", fontsize=12)
    for k, (view, plane, title, pos) in enumerate([("x", "yz", "x view (y-z)", 222),
                                                   ("y", "xz", "y view (x-z)", 223),
                                                   ("z", "xy", "z view (x-y)", 224)]):
        ax = fig.add_subplot(pos)
        _draw_mirror_2d(ax, lens, foot, surf_z, plane)
        if plane == "xy":
            _draw_object_fov_xy(ax, lens)
        _plot_segments(ax, segments, plane)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if plane in ("yz", "xz"):
            ax.set_xlim(-100, surf_z[-1] + 100)
            ax.set_ylim(-1200, 1200)
        else:
            ax.set_xlim(-1400, 1400)
            ax.set_ylim(-1400, 1400)
    fig.suptitle("Full structure | " + _title_str(lens), fontsize=14)
    plt.tight_layout()
    p = os.path.join(outdir, "view_all.png")
    plt.savefig(p)
    plt.close()
    print("  saved:", p)


# ===========================================================================
# 每面镜子单独绘制（突出该镜子的工作区域/感光区）
# ===========================================================================

def plot_per_mirror(lens, outdir="output/figures_structure/per_mirror"):
    """每面镜子 4 视角单独绘制（6 镜 × 4 = 24 张）。"""
    os.makedirs(outdir, exist_ok=True)
    segments, _ = trace_segments(lens)
    foot = footprint_regions(lens)
    surf_z = compute_surface_z(lens.surfaces)

    for i in range(1, 7):
        for view in ("3d", "x", "y", "z"):
            p = os.path.join(outdir, "M%d_%s.png" % (i, view))
            # 只画该镜子 + 其入射/出射光线 + 命中点加亮
            plot_single_view(lens, segments, foot, surf_z, view, p,
                             only_mirror=i, hit_highlight=True)
            print("  saved:", p)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="光学结构专用绘制模块")
    parser.add_argument("--input", required=True, help="镜头 JSON")
    parser.add_argument("--outdir", default="output/figures_structure", help="输出目录")
    parser.add_argument("--full-only", action="store_true", help="只画整体")
    parser.add_argument("--per-mirror-only", action="store_true", help="只画每镜")
    args = parser.parse_args()

    lens = LensSystem.from_json(args.input)
    if not args.per_mirror_only:
        print("[plot] 整体结构视图...")
        plot_full_views(lens, os.path.join(args.outdir, "full"))
    if not args.full_only:
        print("[plot] 每面镜子视图...")
        plot_per_mirror(lens, os.path.join(args.outdir, "per_mirror"))
    print("[plot] 完成")


if __name__ == "__main__":
    main()
