"""
visualization.py — 光学系统可视化模块
=======================================
提供光学镜头系统的可视化分析图表，全部使用 matplotlib（Agg 后端，无 GUI）。

图表类型：
1. 光斑图 (Spot Diagram)：像面上光线交点散点，按视场分色
2. 系统布局图 (2D Layout)：光学系统二维剖面，显示镜面形状和追迹光线
3. MTF 曲线 (Modulation Transfer Function)：由光斑 PSF 近似计算
4. 光线扇图 (Ray Fan)：横向像差 vs 归一化入瞳坐标
5. 收敛曲线 (Convergence)：优化过程 merit 值变化
6. 综合报告 (Report)：上述图表的组合面板

所有函数保存为 PNG 文件，路径由调用方指定。
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from typing import Optional

from lens_schema import LensSystem, SurfaceType
from raytrace import (
    trace_system, compute_surface_z, aspheric_sag, aspheric_coeffs,
    generate_rays, TraceResult, surface_draw_extents,
)


# 颜色配置
FOV_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SURFACE_COLOR = "#333333"
RAY_COLOR = "#4a90d9"
RAY_ALPHA = 0.4


# ===========================================================================
# 1. 光斑图
# ===========================================================================

def plot_spot_diagram(lens: LensSystem,
                      save_path: str,
                      trace_res: Optional[TraceResult] = None) -> str:
    """
    绘制光斑图：像面上光线交点的散点图。

    每个视场用不同颜色，显示 RMS 光斑半径。
    坐标轴单位与系统一致（mm）。
    """
    if trace_res is None:
        trace_res = trace_system(lens)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    fovs = np.unique(trace_res.fov_labels)
    for i, fov in enumerate(fovs):
        mask = (trace_res.fov_labels == fov) & trace_res.active
        if mask.any():
            xy = trace_res.image_xy[mask]
            color = FOV_COLORS[i % len(FOV_COLORS)]
            ax.scatter(xy[:, 0], xy[:, 1], c=color, s=8, alpha=0.7,
                       label=f"FOV={fov:.2f}°")
            # 画 RMS 圆
            centroid = np.mean(xy, axis=0)
            r = np.sqrt(np.sum((xy - centroid) ** 2, axis=1))
            rms = np.sqrt(np.mean(r ** 2))
            circle = plt.Circle(centroid, rms, color=color, fill=False,
                                linestyle="--", linewidth=1)
            ax.add_patch(circle)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Spot Diagram — {lens.metadata.name}\n"
                 f"RMS Spot = {trace_res.rms_spot_radius:.4e} mm")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ===========================================================================
# 2. 系统布局图（2D Layout）
# ===========================================================================

def plot_layout(lens: LensSystem,
                save_path: str,
                n_rays: int = 15) -> str:
    """
    绘制光学系统二维布局图。

    显示：
    - 每个光学表面的剖面形状（球面/非球面曲线）
    - 追迹光线（从物面到像面，显示折叠光路）
    - 表面标注（ID 和类型）
    - 光轴（虚线）

    对于反射系统，Z 坐标会有往返（折叠光路），布局图真实显示这一点。
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    surf_z = compute_surface_z(lens.surfaces)
    surfaces = lens.surfaces
    # 绘制范围 = max(半口径, 光线足迹)，保证所有光线交点落在画出的镜面上
    draw_extents = surface_draw_extents(lens)

    # 绘制每个表面
    for i, surf in enumerate(surfaces):
        z_v = surf_z[i]
        if surf.type == SurfaceType.OBJECT:
            # 物面：竖线
            ax.plot([z_v, z_v], [-surf.semi_aperture, surf.semi_aperture],
                    color="green", linewidth=2)
            ax.text(z_v, surf.semi_aperture * 1.1, "Object",
                    ha="center", fontsize=8, color="green")
            continue

        if surf.type == SurfaceType.IMAGE:
            # 像面：竖线
            ax.plot([z_v, z_v], [-surf.semi_aperture, surf.semi_aperture],
                    color="red", linewidth=2)
            ax.text(z_v, surf.semi_aperture * 1.1, "Image",
                    ha="center", fontsize=8, color="red")
            continue

        # 绘制曲面剖面（偏心镜子整体偏移 decenter_y）
        ext = max(draw_extents[i], 1e-6)
        y = np.linspace(-ext, ext, 200) + surf.decenter_y
        r = np.abs(y - surf.decenter_y)
        c = surf.curvature
        k = surf.aspheric.k if surf.aspheric else 0.0
        A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)
        z_sag = aspheric_sag(r, c, k, A4, A6, A8, A10, A12, A14)
        # 曲面在全局坐标中的位置
        z_curve = z_v + z_sag
        # 反射面的 sag 方向取决于曲率符号和传播方向
        # 简化：直接画 sag（可能需要翻转，但视觉上可接受）

        color = "#d62728" if surf.is_reflective else SURFACE_COLOR
        ax.plot(z_curve, y, color=color, linewidth=1.5)

        # 标注表面 ID
        label = f"M{surf.surface_id}" if surf.is_reflective else f"S{surf.surface_id}"
        if surf.is_stop:
            label += " (STOP)"
        ax.text(z_v, surf.decenter_y + ext * 1.05, label,
                ha="center", fontsize=7, color=color)

    # 追迹几条代表性光线并绘制
    _plot_layout_rays(lens, ax, surf_z, n_rays)

    # 光轴
    all_z = surf_z
    ax.plot([np.min(all_z) - 50, np.max(all_z) + 50], [0, 0],
            color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Optical Layout — {lens.metadata.name}\n"
                 f"NA={lens.metadata.aperture.value}, "
                 f"λ={lens.metadata.wavelengths[0]*1e6:.1f}nm")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _plot_layout_rays(lens: LensSystem, ax, surf_z: np.ndarray, n_rays: int = 15):
    """在布局图上绘制追迹光线（辅助函数）。

    使用子午面（x=0）光线，保证光线全程位于绘制平面内，
    追迹交点投影到 (z, y) 剖面后精确落在镜面曲线上
    （镜面绘制范围已按光线足迹扩展，见 surface_draw_extents）。
    """
    # 生成子午面光线（x=0，dx=0），避免 3D 光线投影到 2D 剖面时失真
    from raytrace import (intersect_surface, reflect, refract,
                          surface_normal, aspheric_coeffs,
                          generate_meridional_rays)
    positions, directions, _, _ = generate_meridional_rays(lens)
    # 均匀采样 n_rays 条
    if len(positions) > n_rays:
        idx = np.linspace(0, len(positions) - 1, n_rays, dtype=int)
        positions = positions[idx]
        directions = directions[idx]

    surfaces = lens.surfaces
    n_surf = len(surfaces)

    for ray_idx in range(len(positions)):
        pos = positions[ray_idx:ray_idx+1].copy()
        d = directions[ray_idx:ray_idx+1].copy()
        ray_z = [pos[0, 2]]
        ray_y = [pos[0, 1]]

        for i in range(1, n_surf):
            surf = surfaces[i]
            z_v = surf_z[i]
            t = intersect_surface(pos, d, z_v, surf)
            if not np.isfinite(t[0]):
                break
            pos = pos + t[0] * d
            ray_z.append(pos[0, 2])
            ray_y.append(pos[0, 1])

            if surf.type == SurfaceType.IMAGE:
                break

            # 交点超出镜面半口径：镜面绘制范围已按光线足迹扩展
            # （见 surface_draw_extents），此处无需截断，交点必落在画出的曲线上

            # 法线
            c = surf.curvature
            k = surf.aspheric.k if surf.aspheric else 0.0
            A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)
            normals = surface_normal(pos[:, 0] - surf.decenter_x,
                                     pos[:, 1] - surf.decenter_y,
                                     pos[:, 2] - z_v,
                                     c, k, A4, A6, A8, A10, A12, A14)

            if surf.is_reflective:
                d = reflect(d, normals)
            elif surf.type in (SurfaceType.STANDARD, SurfaceType.STOP):
                n1 = lens.get_glass_nd(surfaces[i-1].glass)
                n2 = lens.get_glass_nd(surf.glass)
                d, _ = refract(d, normals, n1, n2)

        ax.plot(ray_z, ray_y, color=RAY_COLOR, linewidth=0.5, alpha=RAY_ALPHA)


# ===========================================================================
# 3. MTF 曲线（近似）
# ===========================================================================

def plot_mtf(lens: LensSystem,
             save_path: str,
             trace_res: Optional[TraceResult] = None,
             max_freq: float = 100.0) -> str:
    """
    绘制近似 MTF 曲线。

    由光斑图构建 PSF（点扩散函数），做傅里叶变换得到 OTF，
    取模得到 MTF。这是几何光学近似（不考虑衍射），
    对于大像差系统是合理的近似。

    参数:
        max_freq: 最大空间频率 (lp/mm)
    """
    if trace_res is None:
        trace_res = trace_system(lens)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    fovs = np.unique(trace_res.fov_labels)
    for i, fov in enumerate(fovs):
        mask = (trace_res.fov_labels == fov) & trace_res.active
        if not mask.any():
            continue
        xy = trace_res.image_xy[mask]

        # 构建 PSF 直方图
        bins = 64
        if np.std(xy[:, 0]) > 0 and np.std(xy[:, 1]) > 0:
            extent = max(np.std(xy[:, 0]), np.std(xy[:, 1])) * 5
            extent = max(extent, 1e-6)
        else:
            extent = 1e-3

        H, xedges, yedges = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=bins,
            range=[[-extent, extent], [-extent, extent]]
        )
        H = H / H.sum() if H.sum() > 0 else H

        # FFT → OTF → MTF
        psf = np.fft.fftshift(H)
        otf = np.fft.fft2(psf)
        mtf = np.abs(np.fft.fftshift(otf))
        mtf = mtf / mtf[bins // 2, bins // 2] if mtf[bins // 2, bins // 2] > 0 else mtf

        # 取水平方向的 MTF 截面
        dx = xedges[1] - xedges[0]
        freq = np.fft.fftfreq(bins, d=dx)
        freq = np.fft.fftshift(freq)
        mtf_slice = mtf[bins // 2, :]

        # 只画正频率到 max_freq
        pos_mask = (freq >= 0) & (freq <= max_freq)
        color = FOV_COLORS[i % len(FOV_COLORS)]
        ax.plot(freq[pos_mask], mtf_slice[pos_mask], color=color,
                linewidth=1.5, label=f"FOV={fov:.2f}°")

    ax.set_xlabel("Spatial Frequency (lp/mm)")
    ax.set_ylabel("MTF")
    ax.set_title(f"MTF (geometric approximation) — {lens.metadata.name}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ===========================================================================
# 4. 光线扇图
# ===========================================================================

def plot_ray_fan(lens: LensSystem,
                 save_path: str,
                 trace_res: Optional[TraceResult] = None) -> str:
    """
    绘制光线扇图：横向像差 vs 归一化入瞳坐标。

    对于每个视场，画出 X 和 Y 方向的横向像差曲线。
    横向像差 = 实际像点 - 主光线像点（归一化到波长）。
    """
    if trace_res is None:
        trace_res = trace_system(lens)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fovs = np.unique(trace_res.fov_labels)
    wl = lens.metadata.wavelengths[0] if len(lens.metadata.wavelengths) > 0 else 0.5876e-3

    for i, fov in enumerate(fovs):
        mask = (trace_res.fov_labels == fov) & trace_res.active
        if not mask.any():
            continue
        xy = trace_res.image_xy[mask]
        centroid = np.mean(xy, axis=0)
        # 横向像差（归一化到波长）
        ex = (xy[:, 0] - centroid[0]) / wl
        ey = (xy[:, 1] - centroid[1]) / wl

        # 用光线的初始 X 位置作为入瞳坐标（简化）
        # 实际上应该用入瞳处的归一化坐标，这里用像点排序近似
        order = np.argsort(ex)
        pupil_norm = np.linspace(-1, 1, len(ex))

        color = FOV_COLORS[i % len(FOV_COLORS)]
        axes[0].plot(pupil_norm, ex[order], color=color, linewidth=1,
                     label=f"FOV={fov:.2f}°")
        axes[1].plot(pupil_norm, ey[order], color=color, linewidth=1,
                     label=f"FOV={fov:.2f}°")

    axes[0].set_xlabel("Normalized Pupil")
    axes[0].set_ylabel("Transverse Aberration X (λ)")
    axes[0].set_title("Ray Fan — X")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Normalized Pupil")
    axes[1].set_ylabel("Transverse Aberration Y (λ)")
    axes[1].set_title("Ray Fan — Y")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Ray Fan Diagram — {lens.metadata.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ===========================================================================
# 5. 收敛曲线
# ===========================================================================

def plot_convergence(merit_history: list[float],
                     save_path: str,
                     title: str = "Optimization Convergence") -> str:
    """
    绘制优化收敛曲线：merit 值 vs 迭代次数。
    支持多条曲线（如 GA 和 DLS 分别绘制）。
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if isinstance(merit_history, dict):
        for label, history in merit_history.items():
            ax.semilogy(range(len(history)), history, linewidth=1.5, label=label)
        ax.legend(fontsize=9)
    else:
        ax.semilogy(range(len(merit_history)), merit_history, linewidth=1.5,
                    color="#1f77b4", label="merit")
        ax.legend(fontsize=9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Merit Function Value (log)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ===========================================================================
# 6. 综合报告
# ===========================================================================

def generate_report(lens: LensSystem,
                    output_dir: str,
                    trace_res: Optional[TraceResult] = None,
                    merit_history: Optional[list[float]] = None,
                    prefix: str = "") -> dict[str, str]:
    """
    生成完整的可视化报告，保存所有图表到指定目录。

    返回图表路径字典。
    """
    os.makedirs(output_dir, exist_ok=True)
    if trace_res is None:
        trace_res = trace_system(lens)

    paths = {}
    p = prefix + "_" if prefix else ""

    paths["spot"] = plot_spot_diagram(lens, os.path.join(output_dir, f"{p}spot_diagram.png"), trace_res)
    paths["layout"] = plot_layout(lens, os.path.join(output_dir, f"{p}layout.png"))
    paths["mtf"] = plot_mtf(lens, os.path.join(output_dir, f"{p}mtf.png"), trace_res)
    paths["rayfan"] = plot_ray_fan(lens, os.path.join(output_dir, f"{p}ray_fan.png"), trace_res)

    if merit_history is not None:
        paths["convergence"] = plot_convergence(
            merit_history, os.path.join(output_dir, f"{p}convergence.png")
        )

    return paths


# ===========================================================================
# 7. 优化过程可视化（快照 + GIF 动画 + 迭代对比网格）
# ===========================================================================

def plot_layout_snapshot(lens: LensSystem,
                         save_path: str,
                         iteration: int = 0,
                         merit: float = 0.0,
                         na: float = 0.0,
                         max_incident: float = 0.0,
                         draw_rays: bool = True,
                         n_rays: int = 8,
                         fixed_limits: Optional[tuple] = None) -> str:
    """
    快速布局快照：用于优化过程中每步迭代保存当前镜头状态。

    与 plot_layout 的区别：
    - 标题包含迭代号、merit、NA、最大入射角，便于观察优化趋势
    - 支持固定坐标范围（fixed_limits），使所有快照坐标系一致，
      动画播放时镜面移动清晰可见
    - draw_rays=False 时只画镜面不追迹光线，速度更快

    参数:
        lens: 当前镜头系统
        save_path: 保存路径
        iteration: 当前迭代号
        merit: 当前 merit 值
        na: 当前实际 NA
        max_incident: 当前最大入射角 (度)
        draw_rays: 是否绘制追迹光线
        n_rays: 绘制光线数量
        fixed_limits: (xmin, xmax, ymin, ymax) 固定坐标范围，None 则自动适配
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    surf_z = compute_surface_z(lens.surfaces)
    surfaces = lens.surfaces
    # 绘制范围 = max(半口径, 光线足迹)，保证光线交点落在画出的镜面上
    draw_extents = surface_draw_extents(lens)

    # 绘制每个表面
    for i, surf in enumerate(surfaces):
        z_v = surf_z[i]
        if surf.type == SurfaceType.OBJECT:
            ax.plot([z_v, z_v], [-surf.semi_aperture, surf.semi_aperture],
                    color="green", linewidth=2)
            continue
        if surf.type == SurfaceType.IMAGE:
            ax.plot([z_v, z_v], [-surf.semi_aperture, surf.semi_aperture],
                    color="red", linewidth=2)
            continue

        ext = max(draw_extents[i], 1e-6)
        y = np.linspace(-ext, ext, 150) + surf.decenter_y
        r = np.abs(y - surf.decenter_y)
        c = surf.curvature
        k = surf.aspheric.k if surf.aspheric else 0.0
        A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)
        z_sag = aspheric_sag(r, c, k, A4, A6, A8, A10, A12, A14)
        z_curve = z_v + z_sag

        color = "#d62728" if surf.is_reflective else SURFACE_COLOR
        lw = 2.0 if surf.is_stop else 1.5
        ax.plot(z_curve, y, color=color, linewidth=lw)

        label = f"M{surf.surface_id}" if surf.is_reflective else f"S{surf.surface_id}"
        if surf.is_stop:
            label += "*"
        ax.text(z_v, surf.decenter_y + ext * 1.08, label,
                ha="center", fontsize=7, color=color)

    # 绘制光线
    if draw_rays:
        _plot_layout_rays(lens, ax, surf_z, n_rays)

    # 光轴
    ax.plot([np.min(surf_z) - 30, np.max(surf_z) + 30], [0, 0],
            color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # 坐标范围
    if fixed_limits is not None:
        ax.set_xlim(fixed_limits[0], fixed_limits[1])
        ax.set_ylim(fixed_limits[2], fixed_limits[3])

    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(
        f"iter {iteration:3d} | merit={merit:.4e} | NA={na:.4f} | "
        f"max_inc={max_incident:.1f}°\n"
        f"{lens.metadata.name}  (λ={lens.metadata.wavelengths[0]*1e6:.1f}nm)",
        fontsize=10
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def compute_fixed_limits(lens: LensSystem, margin: float = 50.0) -> tuple:
    """
    计算固定坐标范围，用于优化动画中所有快照保持一致的坐标系。

    基于初始系统的表面位置和绘制口径（max(半口径, 光线足迹)），
    加上 margin。返回 (xmin, xmax, ymin, ymax)。
    """
    surf_z = compute_surface_z(lens.surfaces)
    draw_extents = surface_draw_extents(lens)
    max_aperture = max(draw_extents)
    xmin = np.min(surf_z) - margin
    xmax = np.max(surf_z) + margin
    ymax = max_aperture * 1.3 + margin * 0.3
    return (xmin, xmax, -ymax, ymax)


def create_optimization_gif(snapshot_paths: list[str],
                            output_path: str,
                            fps: int = 3,
                            loop: int = 0) -> str:
    """
    将布局快照序列合成为 GIF 动画。

    使用 Pillow（matplotlib 的依赖）合成，无需额外安装 ffmpeg。

    参数:
        snapshot_paths: 快照 PNG 路径列表（按迭代顺序）
        output_path: 输出 GIF 路径
        fps: 每秒帧数
        loop: 循环次数，0 表示无限循环
    """
    from PIL import Image

    if len(snapshot_paths) == 0:
        raise ValueError("没有快照可合成")

    images = []
    for path in snapshot_paths:
        img = Image.open(path)
        # 统一尺寸（取第一张的尺寸）
        if len(images) > 0:
            img = img.resize(images[0].size, Image.LANCZOS)
        images.append(img)

    duration = int(1000 / fps)  # 每帧持续时间 (ms)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    return output_path


def plot_optimization_grid(lens_snapshots: list[tuple],
                           save_path: str,
                           cols: int = 4,
                           title: str = "Optimization Progress") -> str:
    """
    将优化过程中的关键迭代步骤放在一张网格图中对比。

    参数:
        lens_snapshots: [(iteration, merit, na, max_incident, lens), ...]
        save_path: 输出路径
        cols: 网格列数
        title: 图表标题
    """
    n = len(lens_snapshots)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    # 计算统一坐标范围（含光线足迹扩展）
    all_lens = [snap[4] for snap in lens_snapshots]
    all_z = []
    max_ap = 0
    for L in all_lens:
        sz = compute_surface_z(L.surfaces)
        all_z.extend(sz)
        max_ap = max(max_ap, max(surface_draw_extents(L)))
    xmin, xmax = min(all_z) - 30, max(all_z) + 30
    ymax = max_ap * 1.3

    for idx, (iteration, merit, na, max_inc, lens) in enumerate(lens_snapshots):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        surf_z = compute_surface_z(lens.surfaces)
        draw_extents = surface_draw_extents(lens)

        for j, surf in enumerate(lens.surfaces):
            z_v = surf_z[j]
            if surf.type in (SurfaceType.OBJECT, SurfaceType.IMAGE):
                c = "green" if surf.type == SurfaceType.OBJECT else "red"
                ax.plot([z_v, z_v], [-surf.semi_aperture, surf.semi_aperture],
                        color=c, linewidth=1.5)
                continue
            ext = max(draw_extents[j], 1e-6)
            y = np.linspace(-ext, ext, 80) + surf.decenter_y
            r = np.abs(y - surf.decenter_y)
            asp = surf.aspheric
            k = asp.k if asp else 0.0
            A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)
            z_sag = aspheric_sag(r, surf.curvature, k, A4, A6, A8, A10, A12, A14)
            color = "#d62728" if surf.is_reflective else SURFACE_COLOR
            ax.plot(z_v + z_sag, y, color=color, linewidth=1.0)

        ax.plot([xmin, xmax], [0, 0], color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"iter {iteration}\nmerit={merit:.2e}\nNA={na:.3f}",
                     fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15)

    # 隐藏多余子图
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


class OptimizationVisualizer:
    """
    优化过程可视化管理器：在 DLS 迭代中自动保存布局快照，
    优化结束后合成 GIF 动画和迭代对比网格图。

    用法：
        viz = OptimizationVisualizer(lens, output_dir)
        # 在 DLS callback 中调用：
        viz.capture(iteration, merit, lens)
        # 优化结束后：
        viz.finalize()
    """

    def __init__(self, lens: LensSystem, output_dir: str,
                 fps: int = 3, draw_rays: bool = True,
                 n_rays: int = 8, snapshot_every: int = 1):
        """
        参数:
            lens: 初始镜头系统（用于计算固定坐标范围）
            output_dir: 快照和动画输出目录
            fps: GIF 帧率
            draw_rays: 快照中是否绘制光线
            n_rays: 快照中光线数量
            snapshot_every: 每 N 次迭代保存一次快照（1=每次都保存）
        """
        import os
        self.output_dir = output_dir
        self.fps = fps
        self.draw_rays = draw_rays
        self.n_rays = n_rays
        self.snapshot_every = max(1, snapshot_every)
        self.snapshot_paths: list[str] = []
        self.snapshot_data: list[tuple] = []  # (iter, merit, na, max_inc, lens_copy)
        self.fixed_limits = compute_fixed_limits(lens)
        os.makedirs(output_dir, exist_ok=True)

    def capture(self, iteration: int, merit: float, lens: LensSystem,
                na: float = 0.0, max_incident: float = 0.0) -> None:
        """
        捕获当前迭代的布局快照。

        应在 DLS 每步迭代接受新解后调用。
        """
        if iteration % self.snapshot_every != 0 and iteration > 0:
            return

        import copy
        path = os.path.join(self.output_dir, f"iter_{iteration:04d}.png")
        plot_layout_snapshot(
            lens, path,
            iteration=iteration, merit=merit,
            na=na, max_incident=max_incident,
            draw_rays=self.draw_rays, n_rays=self.n_rays,
            fixed_limits=self.fixed_limits
        )
        self.snapshot_paths.append(path)
        # 保存深拷贝用于后续网格图（避免后续迭代修改）
        self.snapshot_data.append((iteration, merit, na, max_incident, copy.deepcopy(lens)))

    def finalize(self, gif_name: str = "optimization.gif",
                 grid_name: str = "optimization_grid.png",
                 max_grid_frames: int = 12) -> dict[str, str]:
        """
        优化结束后合成 GIF 动画和迭代对比网格图。

        参数:
            gif_name: GIF 文件名
            grid_name: 网格图文件名
            max_grid_frames: 网格图最多显示的帧数（均匀采样）

        返回:
            {"gif": path, "grid": path, "snapshots": [paths]}
        """
        import os
        result = {"snapshots": self.snapshot_paths}

        if len(self.snapshot_paths) == 0:
            return result

        # 合成 GIF
        gif_path = os.path.join(self.output_dir, gif_name)
        try:
            create_optimization_gif(self.snapshot_paths, gif_path, fps=self.fps)
            result["gif"] = gif_path
        except Exception as e:
            print(f"[WARN] GIF 合成失败: {e}")

        # 生成迭代对比网格图（均匀采样关键帧）
        if len(self.snapshot_data) > 0:
            n = len(self.snapshot_data)
            if n > max_grid_frames:
                indices = np.linspace(0, n - 1, max_grid_frames, dtype=int)
                selected = [self.snapshot_data[i] for i in indices]
            else:
                selected = self.snapshot_data

            grid_path = os.path.join(self.output_dir, grid_name)
            plot_optimization_grid(selected, grid_path,
                                   title="Optimization Progress — Lens Layout")
            result["grid"] = grid_path

        return result

