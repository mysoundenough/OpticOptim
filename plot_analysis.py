#!/usr/bin/env python3
"""
plot_analysis.py — 光线遮挡分析绘图模块（通用工具）
=======================================================
专门分析/绘制优化过程中的光线遮挡情况：
- 每视场点单独绘制：主光线 + 边缘（孔径）光线，标注遮挡点与遮挡镜子
- 汇总网格图
- 遮挡统计报告（每视场点到达数、遮挡点明细）

与 plot_structure.py（结构绘制）互补：
- plot_structure：结构本身（镜子/工作区/整体布局）
- plot_analysis：光线路径 + 遮挡诊断（哪个视场、哪条光线、被哪面镜挡）

用法：
    from plot_analysis import plot_fov_rays, occlusion_summary
    lens = LensSystem.from_json('output/xxx.json')
    plot_fov_rays(lens, 'output/analysis/xxx')
    stats = occlusion_summary(lens)   # 打印遮挡统计

CLI：
    python plot_analysis.py --input output/xxx.json --outdir output/analysis/xxx
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lens_schema import LensSystem, SurfaceType
from raytrace import (compute_surface_z, generate_offaxis_rays,
                      generate_rect_offaxis_rays, generate_rect_chief_rays,
                      intersect_surface, surface_normal, reflect, aspheric_sag)
from merit_function import merit_detail

DPI = 200
MIRROR_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c",
                 4: "#d62728", 5: "#9467bd", 6: "#8c564b"}
FOV_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ===========================================================================
# 光线生成与分组
# ===========================================================================

def get_rays(lens):
    """返回 (chief_pos, chief_dir, edge_pos, edge_dir, fov_pts)。"""
    cons = lens.metadata.optical_constraints
    if cons.rectangular_fov:
        cpos, cdirs, _, _ = generate_rect_chief_rays(lens)
        epos, edirs, _, _ = generate_rect_offaxis_rays(lens)
    else:
        cpos, cdirs, _, _ = generate_offaxis_rays(lens)
        epos, edirs, _, _ = generate_offaxis_rays(lens)
        # 非矩形视场：主光线取每视场点第 1 条
    fov_pts = sorted(set(map(lambda p: (round(p[0], 1), round(p[1], 1)), epos)))
    return cpos, cdirs, epos, edirs, fov_pts


def footprint_boxes(lens, epos, edirs):
    """预追迹：每面镜子实体区域（x/y 包围盒）。"""
    surf_z = compute_surface_z(lens.surfaces)
    foot = {}
    p_all, d_all = epos.copy(), edirs.copy()
    for i in range(1, len(lens.surfaces)):
        s = lens.surfaces[i]
        t = intersect_surface(p_all, d_all, surf_z[i], s)
        h = np.isfinite(t)
        if h.any():
            xl = p_all[h, 0] + t[h] * d_all[h, 0] - s.decenter_x
            yl = p_all[h, 1] + t[h] * d_all[h, 1] - s.decenter_y
            foot[i] = (float(xl.min()), float(xl.max()),
                       float(yl.min()), float(yl.max()))
        if not h.any():
            break
        p_all[h] = p_all[h] + t[h, np.newaxis] * d_all[h]
        if s.type == SurfaceType.IMAGE:
            break
        asp = s.aspheric
        norm = surface_normal(p_all[h, 0] - s.decenter_x, p_all[h, 1] - s.decenter_y,
                              p_all[h, 2] - surf_z[i], s.curvature,
                              asp.k if asp else 0.0, asp.A4 if asp else 0.0,
                              asp.A6 if asp else 0.0, asp.A8 if asp else 0.0,
                              asp.A10 if asp else 0.0)
        if s.is_reflective:
            d_all[h] = reflect(d_all[h], norm)
    return foot


def trace_ray(lens, surf_z, foot, p0, d0):
    """
    追迹单条光线。

    返回: (path(N,3), occ[(z,y,blocked_mirror_id)], reached(bool))
    """
    margin = lens.metadata.optical_constraints.obscuration_margin_mm
    p = p0.reshape(1, 3).copy()
    dd = d0.reshape(1, 3).copy()
    path = [p[0].copy()]
    occ = []
    reached = False
    for i in range(1, len(lens.surfaces)):
        s = lens.surfaces[i]
        t = intersect_surface(p, dd, surf_z[i], s)
        if not np.isfinite(t[0]):
            break
        p2 = p + t * dd
        # 遮挡检测：穿过其他镜子实体+margin
        for j in range(1, 7):
            if j == i or j not in foot:
                continue
            other = lens.surfaces[j]
            t_o = intersect_surface(p, dd, surf_z[j], other)
            if np.isfinite(t_o[0]) and t_o[0] > 1e-6 and t_o[0] < t[0] - 1e-6:
                xo = p[0, 0] + t_o[0] * dd[0, 0] - other.decenter_x
                yo = p[0, 1] + t_o[0] * dd[0, 1] - other.decenter_y
                xmin, xmax, ymin, ymax = foot[j]
                if (xmin - margin <= xo <= xmax + margin) and \
                   (ymin - margin <= yo <= ymax + margin):
                    occ.append((p[0, 2] + t_o[0] * dd[0, 2], yo, j))
        path.append(p2[0].copy())
        if s.type == SurfaceType.IMAGE:
            reached = True
            break
        asp = s.aspheric
        norm = surface_normal(p2[:, 0] - s.decenter_x, p2[:, 1] - s.decenter_y,
                              p2[:, 2] - surf_z[i], s.curvature,
                              asp.k if asp else 0.0, asp.A4 if asp else 0.0,
                              asp.A6 if asp else 0.0, asp.A8 if asp else 0.0,
                              asp.A10 if asp else 0.0)
        if s.is_reflective:
            dd = reflect(dd, norm)
        p = p2
    return np.array(path), occ, reached


# ===========================================================================
# 遮挡统计
# ===========================================================================

def occlusion_summary(lens, verbose=True):
    """
    遮挡统计报告：
    - 主光线 / 边缘光线各自到达像面的数量
    - 每视场点的到达数
    - 遮挡点明细（光线 → 被哪面镜挡）

    返回 dict。
    """
    cpos, cdirs, epos, edirs, fov_pts = get_rays(lens)
    surf_z = compute_surface_z(lens.surfaces)
    foot = footprint_boxes(lens, epos, edirs)

    chief_ok = 0
    edge_ok = 0
    per_fov = []
    occ_list = []
    for k in range(len(cpos)):
        _, occ, reached = trace_ray(lens, surf_z, foot, cpos[k], cdirs[k])
        if reached:
            chief_ok += 1
        for o in occ:
            occ_list.append(("chief", k, o))
    for fi, (fx, fy) in enumerate(fov_pts):
        mask = (np.abs(epos[:, 0] - fx) < 0.05) & (np.abs(epos[:, 1] - fy) < 0.05)
        n_ok = 0
        for k in np.where(mask)[0]:
            _, occ, reached = trace_ray(lens, surf_z, foot, epos[k], edirs[k])
            if reached:
                n_ok += 1
                edge_ok += 1
            for o in occ:
                occ_list.append(("edge", k, o))
        per_fov.append((fi, (fx, fy), n_ok, int(mask.sum())))

    stats = dict(chief_ok=chief_ok, chief_total=len(cpos),
                 edge_ok=edge_ok, edge_total=len(epos),
                 per_fov=per_fov, occlusions=occ_list)
    if verbose:
        print("=" * 60)
        print("  遮挡统计")
        print("=" * 60)
        print(f"  主光线到达像面: {chief_ok}/{len(cpos)}")
        print(f"  边缘光线到达像面: {edge_ok}/{len(epos)}")
        for fi, (fx, fy), n_ok, n_tot in per_fov:
            print(f"  FOV{fi} ({fx:6.1f},{fy:6.1f}): {n_ok}/{n_tot} 边缘到达")
        if occ_list:
            print("  遮挡点（前 15 条）:")
            for kind, k, (z, y, j) in occ_list[:15]:
                print(f"    {kind} ray{k}: z={z:.0f} 被 M{j} 挡")
        else:
            print("  无遮挡 ✓")
    return stats


# ===========================================================================
# 每视场点绘图
# ===========================================================================

def _draw_base(ax, lens, surf_z):
    for i in range(1, 7):
        s = lens.surfaces[i]
        z_v = surf_z[i]
        rr = np.linspace(-max(s.semi_aperture, 10.0), max(s.semi_aperture, 10.0), 301)
        asp = s.aspheric
        sag = aspheric_sag(np.abs(rr), s.curvature, asp.k if asp else 0.0,
                           asp.A4 if asp else 0.0, asp.A6 if asp else 0.0,
                           asp.A8 if asp else 0.0, asp.A10 if asp else 0.0)
        ax.plot(z_v + sag, rr, color=MIRROR_COLORS[i], lw=1.5, alpha=0.55)
        ax.text(z_v, max(s.semi_aperture, 10.0) * 0.98, "M%d" % i,
                fontsize=7, color=MIRROR_COLORS[i], ha="center")
    ax.axvline(surf_z[-1], color="orange", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)


def plot_fov_rays(lens, outdir="output/analysis", per_fov=True, verbose=True):
    """
    绘制每视场点光线 + 遮挡标注（主光线黑粗 + 边缘光线彩色），
    并生成汇总网格图。

    输出:
        outdir/all.png          汇总
        outdir/fov0.png ...     每视场点单独
    """
    os.makedirs(outdir, exist_ok=True)
    cpos, cdirs, epos, edirs, fov_pts = get_rays(lens)
    surf_z = compute_surface_z(lens.surfaces)
    foot = footprint_boxes(lens, epos, edirs)
    d = merit_detail(lens)
    res = d["trace_result"]

    # 预计算每条光线
    def ray_groups():
        return (cpos, cdirs, epos, edirs, fov_pts)

    cpos, cdirs, epos, edirs, fov_pts = ray_groups()

    def draw_fov(ax, fi, fx, fy):
        _draw_base(ax, lens, surf_z)
        mask = (np.abs(epos[:, 0] - fx) < 0.05) & (np.abs(epos[:, 1] - fy) < 0.05)
        n_reach = 0
        for k in np.where(mask)[0]:
            path, occ, reached = trace_ray(lens, surf_z, foot, epos[k], edirs[k])
            if reached:
                n_reach += 1
            ax.plot(path[:, 2], path[:, 1], "-" if reached else "--",
                    color=FOV_COLORS[fi % 5], lw=0.9, alpha=0.75)
            for (zo, yo, j) in occ:
                ax.plot(zo, yo, "rx", ms=7, mew=1.8, zorder=10)
        cmask = (np.abs(cpos[:, 0] - fx) < 0.05) & (np.abs(cpos[:, 1] - fy) < 0.05)
        chief_ok = False
        for k in np.where(cmask)[0]:
            path, occ, reached = trace_ray(lens, surf_z, foot, cpos[k], cdirs[k])
            if reached:
                chief_ok = True
            ax.plot(path[:, 2], path[:, 1], "-", color="k", lw=2.2, alpha=0.95)
            for (zo, yo, j) in occ:
                ax.plot(zo, yo, "rx", ms=10, mew=2.5, zorder=10)
                ax.text(zo + 25, yo + 25, "BLOCKED M%d" % j,
                        fontsize=7, color="red", fontweight="bold")
        return n_reach, chief_ok

    # 汇总网格
    n_fov = len(fov_pts)
    ncol = 2
    nrow = (n_fov + 1) // 2
    fig, axes = plt.subplots(nrow + 1, ncol, figsize=(18, 7 * (nrow + 1)), dpi=DPI)
    axes = np.array(axes).flatten()
    for fi, (fx, fy) in enumerate(fov_pts):
        ax = axes[fi]
        n_reach, chief_ok = draw_fov(ax, fi, fx, fy)
        ax.set_title("FOV%d (%.0f,%.0f): edge %d/%d reach, chief %s" % (
            fi, fx, fy, n_reach, int((np.abs(epos[:, 0] - fx) < 0.05).sum() * 0 + 6),
            "OK" if chief_ok else "BLOCKED"), fontsize=10)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlim(-100, surf_z[-1] + 100)
        ax.set_ylim(-1200, 1200)
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.5,
                  "black bold = chief ray\ncolor solid = edge reached\n"
                  "color dashed = edge blocked\nred x = occlusion point\n"
                  "mirror: M1 blue M2 orange M3 green M4 red M5 purple M6 brown",
                  fontsize=11, va="center")
    fig.suptitle("FOV rays occlusion analysis | %s" % _title_str(lens), fontsize=13)
    plt.tight_layout()
    p = os.path.join(outdir, "all.png")
    plt.savefig(p)
    plt.close()
    if verbose:
        print("  saved:", p)

    # 每视场点单独
    if per_fov:
        for fi, (fx, fy) in enumerate(fov_pts):
            fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
            n_reach, chief_ok = draw_fov(ax, fi, fx, fy)
            ax.set_title("FOV%d (%.0f,%.0f): edge %d/6 reach | black=chief" % (
                fi, fx, fy, n_reach), fontsize=11)
            ax.set_xlabel("z (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_xlim(-100, surf_z[-1] + 100)
            ax.set_ylim(-1200, 1200)
            plt.tight_layout()
            p = os.path.join(outdir, "fov%d.png" % fi)
            plt.savefig(p)
            plt.close()
            if verbose:
                print("  saved:", p)


def _title_str(lens):
    cons = lens.metadata.optical_constraints
    return ("obj h=%.1f chief=%.0f img=%.0f mag=%.2f" % (
        cons.object_height_mm, cons.chief_ray_angle_deg,
        cons.image_z_fixed, cons.magnification_target))


# ===========================================================================
# CLI
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="光线遮挡分析绘图")
    parser.add_argument("--input", required=True, help="镜头 JSON")
    parser.add_argument("--outdir", default="output/analysis", help="输出目录")
    parser.add_argument("--no-per-fov", action="store_true", help="不画每视场点单独图")
    parser.add_argument("--summary-only", action="store_true", help="只打印统计")
    args = parser.parse_args()

    lens = LensSystem.from_json(args.input)
    if args.summary_only:
        occlusion_summary(lens)
    else:
        print("[plot_analysis] 遮挡统计:")
        occlusion_summary(lens)
        print("\n[plot_analysis] 绘图:")
        plot_fov_rays(lens, args.outdir, per_fov=not args.no_per_fov)
    print("[plot_analysis] 完成")


if __name__ == "__main__":
    main()
