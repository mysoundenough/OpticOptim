"""
raytrace.py — 光线追迹引擎
===========================
实现序列光线追迹，支持球面折射、球面反射、高阶偶次非球面反射。

核心设计：
- 全向量化：所有光线打包为 (N,3) numpy 数组同时追迹，避免 Python 循环
- 统一 sag 公式：球面 = k=0 且高阶项为 0 的非球面，平面 = curvature=0
- 非球面交点用牛顿迭代，初始猜测取平面交点
- 反射面翻转 Z 传播方向，自动处理折叠光路的顶点坐标
- 表面顶点 Z 坐标在追迹前预计算，反射后累加方向取反

坐标系约定：
  全局右手系，光轴 = Z 轴，物面默认 z=0
  光线初始沿 +Z 传播，遇到反射面后方向 z 分量取反
  表面顶点 z 坐标：z[i+1] = z[i] + sign * thickness[i]
  反射后 sign *= -1
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from lens_schema import LensSystem, Surface, SurfaceType, ApertureType, ASPHERIC_HIGH_ORDER


# ===========================================================================
# 非球面系数提取
# ===========================================================================

def aspheric_coeffs(surf: Surface) -> tuple[float, float, float, float, float, float]:
    """
    提取 Surface 的激活高阶系数 (A4, A6, A8, A10, A12, A14)。

    只返回 asp.high_order_terms 内激活的项（如默认 2 项时 A8/A10/A12/A14 按 0 处理）；
    非非球面（aspheric=None）返回全 0。
    """
    asp = surf.aspheric
    if asp is None:
        return (0.0,) * 6
    act = asp.active_coeffs()
    return tuple(act + [0.0] * (6 - len(act)))  # type: ignore[return-value]


# ===========================================================================
# 非球面 sag 及其导数
# ===========================================================================

def aspheric_sag(r: np.ndarray, c: float, k: float,
                 A4: float = 0.0, A6: float = 0.0,
                 A8: float = 0.0, A10: float = 0.0,
                 A12: float = 0.0, A14: float = 0.0) -> np.ndarray:
    """
    偶次非球面矢高（sag）。

    z(r) = c·r² / [1 + sqrt(1 - (1+k)·c²·r²)] + Σ A_{2m}·r^{2m}

    参数:
        r:  径向距离 sqrt(x²+y²)，形状 (N,)
        c:  曲率 curvature = 1/R
        k:  圆锥常数 conic
        A4..A14: 高阶偶次系数（最多 6 项）

    返回:
        z_sag: 形状 (N,)，顶点处为 0
    """
    r2 = r * r
    # 基础圆锥曲面部分（c=0 时为 0，即平面）
    if abs(c) < 1e-15:
        z_base = np.zeros_like(r)
    else:
        inside = 1.0 - (1.0 + k) * c * c * r2
        # 防止根号内负数（光线超出口径时），截断到小正数
        inside = np.maximum(inside, 1e-20)
        denom = 1.0 + np.sqrt(inside)
        z_base = c * r2 / denom

    # 高阶多项式项（偶次：A_{2m}·r^{2m} = A·(r²)^m）
    z_high = np.zeros_like(r2)
    for order, coef in ((4, A4), (6, A6), (8, A8), (10, A10), (12, A12), (14, A14)):
        if coef != 0.0:
            z_high += coef * r2 ** (order // 2)
    return z_base + z_high


def aspheric_sag_deriv(r: np.ndarray, c: float, k: float,
                        A4: float = 0.0, A6: float = 0.0,
                        A8: float = 0.0, A10: float = 0.0,
                        A12: float = 0.0, A14: float = 0.0) -> np.ndarray:
    """
    非球面 sag 对 r 的一阶导数 dz/dr。

    dz/dr = c·r / sqrt(1 - (1+k)·c²·r²) + Σ (2m)·A_{2m}·r^{2m-1}
    """
    r2 = r * r
    if abs(c) < 1e-15:
        dz_base = np.zeros_like(r)
    else:
        inside = 1.0 - (1.0 + k) * c * c * r2
        inside = np.maximum(inside, 1e-20)
        dz_base = c * r / np.sqrt(inside)

    dz_high = np.zeros_like(r)
    for order, coef in ((4, A4), (6, A6), (8, A8), (10, A10), (12, A12), (14, A14)):
        if coef != 0.0:
            dz_high += order * coef * r ** (order - 1)
    return dz_base + dz_high


def surface_normal(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   c: float, k: float,
                   A4: float, A6: float, A8: float, A10: float,
                   A12: float = 0.0, A14: float = 0.0) -> np.ndarray:
    """
    计算非球面在交点处的单位法向量。

    曲面 F(x,y,z) = z - z_sag(sqrt(x²+y²)) = 0
    梯度 ∇F = (-dz/dx, -dz/dy, 1)
    其中 dz/dx = z'(r)·x/r，dz/dy = z'(r)·y/r

    返回形状 (N,3) 的单位法向量，指向 +Z 侧（入射侧）。
    """
    r = np.sqrt(x * x + y * y)
    r_safe = np.where(r < 1e-15, 1e-15, r)
    dz_dr = aspheric_sag_deriv(r, c, k, A4, A6, A8, A10, A12, A14)

    dz_dx = dz_dr * x / r_safe
    dz_dy = dz_dr * y / r_safe

    # 法向量 = (-dz/dx, -dz/dy, 1)，归一化
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(x)
    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=1)


# ===========================================================================
# 表面顶点 Z 坐标预计算
# ===========================================================================

def compute_surface_z(surfaces: list[Surface]) -> np.ndarray:
    """
    预计算每个表面顶点的全局 Z 坐标。

    反射面会翻转传播方向：
      z[i+1] = z[i] + sign * thickness[i]
      若 surface[i] 是反射面，则 sign *= -1

    物面 z=0。
    """
    n = len(surfaces)
    z = np.zeros(n)
    sign = 1.0
    for i in range(n - 1):
        # 关键：反射面先翻转传播方向，再用新方向计算下一面位置
        if surfaces[i].is_reflective:
            sign *= -1.0
        z[i + 1] = z[i] + sign * surfaces[i].thickness
    return z


# ===========================================================================
# 光线与表面求交（向量化）
# ===========================================================================

def intersect_surface(pos: np.ndarray, dirs: np.ndarray,
                      z_vertex: float, surf: Surface,
                      max_iter: int = 30, tol: float = 1e-12) -> np.ndarray:
    """
    计算光线与光学表面的交点参数 t（向量化）。

    光线 P(t) = pos + t * dirs
    曲面隐式方程 F(t) = z(t) - z_vertex - z_sag(r(t)) = 0

    牛顿迭代：t_{n+1} = t_n - F(t_n) / F'(t_n)
    初始猜测 t0 = (z_vertex - pos_z) / dirs_z （平面交点）

    参数:
        pos:  (N,3) 光线起点
        dirs: (N,3) 光线单位方向
        z_vertex: 表面顶点 Z 坐标
        surf: Surface 对象
        max_iter: 牛顿迭代最大次数
        tol: 收敛容差

    返回:
        t: (N,) 交点参数，无效光线（全反射/超出口径）标记为 inf
    """
    N = pos.shape[0]
    c = surf.curvature
    k = surf.aspheric.k if surf.aspheric else 0.0
    A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)

    # 偏心：在镜面局部坐标系（顶点为原点）中求交，平移不影响参数 t
    dx = surf.decenter_x
    dy = surf.decenter_y

    # 初始猜测：平面交点
    dirs_z = dirs[:, 2]
    dirs_z_safe = np.where(np.abs(dirs_z) < 1e-15, 1e-15, dirs_z)
    t = (z_vertex - pos[:, 2]) / dirs_z_safe

    # 牛顿迭代
    for _ in range(max_iter):
        x = pos[:, 0] + t * dirs[:, 0] - dx
        y = pos[:, 1] + t * dirs[:, 1] - dy
        z_pt = pos[:, 2] + t * dirs[:, 2]
        r = np.sqrt(x * x + y * y)

        z_sag = aspheric_sag(r, c, k, A4, A6, A8, A10, A12, A14)
        F = z_pt - z_vertex - z_sag

        # F'(t) = dz - z_sag'(r) * (x*dx + y*dy)/r
        r_safe = np.where(r < 1e-15, 1e-15, r)
        dz_dr = aspheric_sag_deriv(r, c, k, A4, A6, A8, A10, A12, A14)
        dr_dt = (x * dirs[:, 0] + y * dirs[:, 1]) / r_safe
        Fp = dirs[:, 2] - dz_dr * dr_dt

        Fp_safe = np.where(np.abs(Fp) < 1e-15, 1e-15, Fp)
        dt = F / Fp_safe
        t = t - dt

        if np.max(np.abs(dt)) < tol:
            break

    # 标记无效光线：t 为负（交点在光线后方）或 NaN
    t = np.where(np.isfinite(t) & (t > -1e-6), t, np.inf)
    return t


# ===========================================================================
# 折射 / 反射
# ===========================================================================

def reflect(dirs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    反射定律（向量化）。

    d_reflected = d - 2·(d·n)·n

    参数:
        dirs:    (N,3) 入射方向（单位向量）
        normals: (N,3) 表面法向量（单位向量，指向入射侧）

    返回:
        (N,3) 反射方向（单位向量）
    """
    # 确保法线指向入射侧（与入射方向夹角 > 90°，即点积 < 0）
    dot = np.sum(dirs * normals, axis=1, keepdims=True)
    n = np.where(dot < 0, normals, -normals)
    dot = np.sum(dirs * n, axis=1, keepdims=True)
    reflected = dirs - 2.0 * dot * n
    # 归一化（数值误差修正）
    norm = np.linalg.norm(reflected, axis=1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return reflected / norm


def refract(dirs: np.ndarray, normals: np.ndarray,
            n1: float, n2: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Snell 折射定律（向量化）。

    向量形式：
      t = (n1/n2)·d + [(n1/n2)·(d·n) - sqrt(1 - (n1/n2)²·(1-(d·n)²))]·n

    参数:
        dirs:    (N,3) 入射方向
        normals: (N,3) 表面法向量（指向入射侧）
        n1: 入射侧折射率
        n2: 透射侧折射率

    返回:
        (refracted_dirs, tir_mask):
            refracted_dirs: (N,3) 折射方向
            tir_mask: (N,) bool，True 表示发生全反射
    """
    dot = np.sum(dirs * normals, axis=1, keepdims=True)
    # 确保法线指向入射侧
    n = np.where(dot < 0, normals, -normals)
    dot = np.sum(dirs * n, axis=1, keepdims=True)  # 此时 dot < 0

    eta = n1 / n2
    cos_t_sq = 1.0 - eta * eta * (1.0 - dot * dot)
    tir_mask = cos_t_sq < 0  # 全反射

    cos_t_sq_safe = np.where(tir_mask, 1.0, cos_t_sq)
    cos_t = np.sqrt(cos_t_sq_safe)

    # Snell 向量折射公式：
    #   t = η·d + (η·cos_i - cos_t)·n
    # 其中 n 指向入射侧，cos_i = -d·n（因 d·n < 0），故：
    #   t = η·d + (-η·(d·n) - cos_t)·n
    refracted = eta * dirs + (-eta * dot - cos_t) * n
    # 全反射的光线保留原方向（调用方会根据 tir_mask 处理）
    refracted = np.where(tir_mask, dirs, refracted)

    norm = np.linalg.norm(refracted, axis=1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return refracted / norm, tir_mask[:, 0]


# ===========================================================================
# 光线采样
# ===========================================================================

def generate_rays(lens: LensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成采样光线。

    采样策略：
    - 孔径方向：aperture_rings 个同心环
    - 方位方向：每环 radial_rays 条光线（均匀分布角度）
    - 多视场：每个 fov_deg 生成一组
    - 多波长：每个波长生成一组

    光线从物面 z=0 发出，物点在原点 (0,0,0)。
    入瞳平面设在光阑处（若无光阑则在第一面处），入瞳半径由 NA 确定。
    光线方向 = 归一化(入瞳采样点 - 物点)。

    返回:
        positions: (M,3) 光线起始位置（物面处）
        directions: (M,3) 光线单位方向
        wavelengths: (M,) 波长
        fov_labels: (M,) 视场角标签（度）
    """
    meta = lens.metadata
    rs = meta.ray_sampling
    n_rings = rs.aperture_rings
    n_radial = rs.radial_rays

    # 入瞳位置：光阑面 z 坐标，若无光阑用第一面
    surf_z = compute_surface_z(lens.surfaces)
    stop_idx = 0
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            stop_idx = i
            break
    z_pupil = surf_z[stop_idx] if stop_idx > 0 else surf_z[1]

    # 入瞳半径：根据孔径类型计算
    ap_type = lens.metadata.aperture.type
    ap_value = lens.metadata.aperture.value
    if ap_type == ApertureType.NA:
        # NA = n·sin(θ)，入瞳半径 ≈ NA · |z_pupil|
        pupil_radius = ap_value * abs(z_pupil) * 1.0
    elif ap_type == ApertureType.DIAMETER:
        # 直接指定入瞳直径
        pupil_radius = ap_value / 2.0
    elif ap_type == ApertureType.F_NUMBER:
        # F/# = f / D，需估算焦距（用最后一面到像面距离近似）
        approx_fl = abs(surf_z[-1] - surf_z[-2]) if len(surf_z) >= 2 else 50.0
        pupil_radius = approx_fl / ap_value / 2.0
    else:
        pupil_radius = ap_value * abs(z_pupil) * 1.0

    all_pos = []
    all_dir = []
    all_wl = []
    all_fov = []

    for wl in meta.wavelengths:
        for fov in meta.fov_deg:
            fov_rad = np.deg2rad(fov)
            # 主光线方向（视场角）
            chief_dir = np.array([np.sin(fov_rad), 0.0, np.cos(fov_rad)])

            # 在入瞳平面上环形采样
            for ring in range(1, n_rings + 1):
                rho = ring / n_rings  # 归一化孔径 0..1
                r_phys = rho * pupil_radius
                n_angles = max(n_radial, 1)
                angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
                for ang in angles:
                    px = r_phys * np.cos(ang)
                    py = r_phys * np.sin(ang)
                    # 入瞳处的点（考虑视场偏移：主光线在入瞳处的偏移）
                    pupil_point = np.array([px, py, z_pupil])
                    # 光线从物点(0,0,0)到入瞳点
                    direction = pupil_point / np.linalg.norm(pupil_point)
                    # 叠加视场：将方向绕 Y 轴旋转 fov_rad
                    # 旋转矩阵 Ry(θ): [cosθ, 0, sinθ; 0,1,0; -sinθ,0,cosθ]
                    cos_f = np.cos(fov_rad)
                    sin_f = np.sin(fov_rad)
                    d_rot = np.array([
                        cos_f * direction[0] + sin_f * direction[2],
                        direction[1],
                        -sin_f * direction[0] + cos_f * direction[2]
                    ])
                    all_pos.append([0.0, 0.0, 0.0])
                    all_dir.append(d_rot)
                    all_wl.append(wl)
                    all_fov.append(fov)

    positions = np.array(all_pos, dtype=np.float64)
    directions = np.array(all_dir, dtype=np.float64)
    wavelengths = np.array(all_wl, dtype=np.float64)
    fov_labels = np.array(all_fov, dtype=np.float64)
    return positions, directions, wavelengths, fov_labels


def generate_offaxis_rays(lens: LensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成离轴环形视场光线（EUV 光刻物镜类系统）。

    物理模型：
    - 物方主光线以 chief_ray_angle_deg 角入射（与光轴夹角），
      主光线经过入瞳中心（光轴上）。
    - 物点离轴：位于物面 z=0 上半径 r_obj = |z_pupil|·tan(θ) 处，
      方位角取 φ=0（旋转对称系统，一个方位代表整个环形视场）。
    - 孔径采样：以入瞳中心为圆心环形采样（半径由物方 NA 确定），
      每条光线从离轴物点指向入瞳采样点。
    - 离轴视场使光线只打到镜子的离轴环带，并让光线从镜子间
      的间隙穿过，避免被镜子实体遮挡。

    返回:
        positions: (M,3) 光线起始位置（离轴物点）
        directions: (M,3) 光线单位方向
        wavelengths: (M,) 波长
        fov_labels: (M,) 方位角标签（保持与 fov_deg 兼容，返回 0）
    """
    meta = lens.metadata
    cons = meta.optical_constraints
    rs = meta.ray_sampling
    n_rings = rs.aperture_rings
    n_radial = rs.radial_rays

    theta = np.deg2rad(cons.chief_ray_angle_deg)
    na_obj = cons.na_objective if cons.na_objective is not None else meta.aperture.value
    cobs = cons.central_obscuration_ratio  # 环形孔径中心遮挡比（共轴系统用）

    surf_z = compute_surface_z(lens.surfaces)
    stop_idx = 0
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            stop_idx = i
            break
    z_pupil = surf_z[stop_idx] if stop_idx > 0 else surf_z[1]

    # 入瞳半径（物方孔径）
    ap_type = lens.metadata.aperture.type
    ap_value = lens.metadata.aperture.value
    if ap_type == ApertureType.DIAMETER:
        pupil_radius = ap_value / 2.0
    elif ap_type == ApertureType.F_NUMBER:
        approx_fl = abs(surf_z[-1] - surf_z[-2]) if len(surf_z) >= 2 else 50.0
        pupil_radius = approx_fl / ap_value / 2.0
    else:
        pupil_radius = na_obj * abs(z_pupil)

    # 离轴物点（方位角 φ=0，位于 +y 侧）：物高直接用配置或由角度换算
    r_obj = cons.object_height_mm if cons.object_height_mm > 0 else abs(z_pupil) * np.tan(theta)
    obj_point = np.array([0.0, r_obj, 0.0])
    # 入瞳中心：默认在光轴上；物方远心时移到物点正上方（主光线平行光轴，
    # 光束保持离轴环带穿过共轴镜子）
    if cons.telecentric_objective:
        pupil_center = np.array([0.0, r_obj, z_pupil])
    else:
        pupil_center = np.array([0.0, 0.0, z_pupil])

    all_pos = []
    all_dir = []
    all_wl = []
    all_fov = []

    for wl in meta.wavelengths:
        for ring in range(1, n_rings + 1):
            # 环形孔径：rho ∈ [cobs, 1]，光轴中心（rho<cobs）不追迹
            rho = cobs + (1.0 - cobs) * ring / n_rings
            r_phys = rho * pupil_radius
            for k in range(max(n_radial, 1)):
                psi = 2 * np.pi * k / max(n_radial, 1)
                pupil_pt = pupil_center + np.array([
                    r_phys * np.cos(psi), r_phys * np.sin(psi), 0.0
                ])
                direction = pupil_pt - obj_point
                direction /= np.linalg.norm(direction)
                all_pos.append(obj_point)
                all_dir.append(direction)
                all_wl.append(wl)
                all_fov.append(0.0)

    positions = np.array(all_pos, dtype=np.float64)
    directions = np.array(all_dir, dtype=np.float64)
    wavelengths = np.array(all_wl, dtype=np.float64)
    fov_labels = np.array(all_fov, dtype=np.float64)
    return positions, directions, wavelengths, fov_labels


# ===========================================================================
# 主追迹函数
# ===========================================================================

def generate_rect_offaxis_rays(lens: LensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    矩形视场光线（EUV 物镜实际形态）。

    物面视场 = 环形视场内接的最大矩形 [-w,w]×[-w,w]（w = rect_half_width_mm）。
    在矩形内采样 n×n 个视场点，每点发一组孔径光线（物方远心：主光线平行光轴，
    孔径光线围绕主光线环形采样，半径由物方 NA 决定）。

    镜片只接收矩形视场发出的光打到的部分（单片镜片，非环形）——
    遮挡判定用光束实际区域（footprint 包围盒），不用环形环带。
    """
    meta = lens.metadata
    cons = meta.optical_constraints
    rs = meta.ray_sampling
    n_rings = rs.aperture_rings
    n_radial = rs.radial_rays

    na_obj = cons.na_objective if cons.na_objective is not None else meta.aperture.value
    wx = cons.rect_half_wx_mm if cons.rect_half_wx_mm > 0 else cons.rect_half_width_mm
    wy = cons.rect_half_wy_mm if cons.rect_half_wy_mm > 0 else cons.rect_half_width_mm
    wx = wx if wx > 0 else 50.0
    wy = wy if wy > 0 else 50.0
    cx = cons.fov_center_x
    cy = cons.fov_center_y
    cobs = cons.central_obscuration_ratio
    n_fov = max(cons.fov_grid_n, 1)

    if cons.fov_pattern == "corners_center":
        # 四角 + 中心（5 视场点）
        pts = [(cx-wx, cy-wy), (cx+wx, cy-wy), (cx+wx, cy+wy), (cx-wx, cy+wy), (cx, cy)]
        fov_pts = np.array(pts, dtype=np.float64)
    else:
        g = np.linspace(-1, 1, n_fov)
        fov_pts = np.array([(cx + gx*wx, cy + gy*wy) for gx in g for gy in g])
    surf_z = compute_surface_z(lens.surfaces)
    stop_idx = 0
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            stop_idx = i
            break
    z_pupil = surf_z[stop_idx] if stop_idx > 0 else surf_z[1]
    pupil_radius = na_obj * abs(z_pupil)

    # 物方远心与否：
    #   telecentric_objective=True  → 主光线平行光轴（+z），孔径光线围绕主光线
    #   telecentric_objective=False → 主光线以 chief_ray_angle_deg 倾斜入射（非远心），
    #                                 孔径光线围绕倾斜主光线（NA 锥）
    tele = cons.telecentric_objective
    theta = np.deg2rad(cons.chief_ray_angle_deg) if not tele else 0.0

    all_pos, all_dir, all_wl, all_fov = [], [], [], []
    for (fx, fy) in fov_pts:
            obj_point = np.array([fx, fy, 0.0])
            # 主光线方向：远心=+z；非远心=与光轴成 theta，位于物点-光轴平面内指向光轴
            r_pt = np.hypot(fx, fy)
            if not tele and r_pt > 1e-9:
                d_chief = np.array([-np.sin(theta) * fx / r_pt,
                                    -np.sin(theta) * fy / r_pt,
                                    np.cos(theta)])
            else:
                d_chief = np.array([0.0, 0.0, 1.0])
            # 正交基（孔径环所在平面）
            aux = np.array([0.0, 0.0, 1.0]) if abs(d_chief[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            u = np.cross(aux, d_chief)
            u /= np.linalg.norm(u)
            v = np.cross(d_chief, u)
            for ring in range(1, n_rings + 1):
                rho = cobs + (1.0 - cobs) * ring / n_rings
                half_angle = np.arcsin(np.clip(na_obj * rho, 0.0, 1.0))
                for k in range(max(n_radial, 1)):
                    psi = 2 * np.pi * k / max(n_radial, 1)
                    direction = (np.cos(half_angle) * d_chief +
                                 np.sin(half_angle) * (np.cos(psi) * u + np.sin(psi) * v))
                    direction /= np.linalg.norm(direction)
                    all_pos.append(obj_point)
                    all_dir.append(direction)
                    all_wl.append(meta.wavelengths[0])
                    all_fov.append(0.0)
    return (np.array(all_pos, dtype=np.float64),
            np.array(all_dir, dtype=np.float64),
            np.array(all_wl, dtype=np.float64),
            np.array(all_fov, dtype=np.float64))


def generate_meridional_offaxis_rays(lens: LensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    离轴系统的子午面光线（y-z 平面，x=0），供快速结构搜索 / 布局使用。

    物理模型与 generate_offaxis_rays 相同（离轴环形视场），但只在 y-z 子午面采样：
    - 主光线以 chief_ray_angle_deg 从离轴物点进入入瞳中心（光轴上）
    - 孔径在 y-z 平面采样：每环取 +y / -y 两个子午方位
    - 全程 x=0（光线位于子午面内，离轴系统 decenter 也在 y 方向，
      因此能完整描述遮挡几何）

    光线数 = n_rings × 2（默认 6 条），约为全 3D 环形采样（18 条）的 1/3，
    用于结构搜索阶段大幅加速 GA/merit 评估。
    """
    meta = lens.metadata
    cons = meta.optical_constraints
    rs = meta.ray_sampling
    n_rings = rs.aperture_rings

    theta = np.deg2rad(cons.chief_ray_angle_deg)
    na_obj = cons.na_objective if cons.na_objective is not None else meta.aperture.value
    cobs = cons.central_obscuration_ratio  # 环形孔径中心遮挡比

    surf_z = compute_surface_z(lens.surfaces)
    stop_idx = 0
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            stop_idx = i
            break
    z_pupil = surf_z[stop_idx] if stop_idx > 0 else surf_z[1]

    ap_type = lens.metadata.aperture.type
    ap_value = lens.metadata.aperture.value
    if ap_type == ApertureType.DIAMETER:
        pupil_radius = ap_value / 2.0
    elif ap_type == ApertureType.F_NUMBER:
        approx_fl = abs(surf_z[-1] - surf_z[-2]) if len(surf_z) >= 2 else 50.0
        pupil_radius = approx_fl / ap_value / 2.0
    else:
        pupil_radius = na_obj * abs(z_pupil)

    # 离轴物点（+y 侧，子午面内）：物高直接用配置或由角度换算
    r_obj = cons.object_height_mm if cons.object_height_mm > 0 else abs(z_pupil) * np.tan(theta)
    obj_point = np.array([0.0, r_obj, 0.0])
    # 入瞳中心：默认在光轴上；物方远心时移到物点正上方
    if cons.telecentric_objective:
        pupil_center = np.array([0.0, r_obj, z_pupil])
    else:
        pupil_center = np.array([0.0, 0.0, z_pupil])

    all_pos = []
    all_dir = []
    all_wl = []
    all_fov = []

    for wl in meta.wavelengths:
        for ring in range(1, n_rings + 1):
            # 环形孔径：rho ∈ [cobs, 1]
            rho = cobs + (1.0 - cobs) * ring / n_rings
            r_phys = rho * pupil_radius
            for sy in (1.0, -1.0):
                pupil_pt = pupil_center + np.array([0.0, sy * r_phys, 0.0])
                direction = pupil_pt - obj_point
                direction /= np.linalg.norm(direction)
                all_pos.append(obj_point)
                all_dir.append(direction)
                all_wl.append(wl)
                all_fov.append(0.0)

    return (np.array(all_pos, dtype=np.float64),
            np.array(all_dir, dtype=np.float64),
            np.array(all_wl, dtype=np.float64),
            np.array(all_fov, dtype=np.float64))


def generate_chief_offaxis_rays(lens: LensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    仅主光线（1 条）的超快速结构筛选光线。

    主光线：离轴物点 → 入瞳中心（光轴上）。
    用于 GA 结构搜索时对"不遮挡"做超快速初筛；
    注意单条光线无法计算 NA/光斑，仅在结构项 merit（光斑权重为 0）下使用。
    """
    meta = lens.metadata
    cons = meta.optical_constraints

    theta = np.deg2rad(cons.chief_ray_angle_deg)
    surf_z = compute_surface_z(lens.surfaces)
    stop_idx = 0
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            stop_idx = i
            break
    z_pupil = surf_z[stop_idx] if stop_idx > 0 else surf_z[1]

    r_obj = cons.object_height_mm if cons.object_height_mm > 0 else abs(z_pupil) * np.tan(theta)
    obj_point = np.array([0.0, r_obj, 0.0])
    # 入瞳中心：默认在光轴上；物方远心时移到物点正上方（主光线平行光轴）
    if cons.telecentric_objective:
        pupil_center = np.array([0.0, r_obj, z_pupil])
    else:
        pupil_center = np.array([0.0, 0.0, z_pupil])
    direction = pupil_center - obj_point
    direction /= np.linalg.norm(direction)

    positions = obj_point[np.newaxis, :].astype(np.float64)
    directions = direction[np.newaxis, :].astype(np.float64)
    wavelengths = np.array([meta.wavelengths[0]], dtype=np.float64)
    fov_labels = np.zeros(1, dtype=np.float64)
    return positions, directions, wavelengths, fov_labels


def generate_meridional_rays(lens: LensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成位于 y-z 子午面（x=0）的代表光线，专供 2D 布局图/动画快照使用。

    与 generate_rays 的区别：
    - 只在子午面采样：每环取 +y / -y 两个方位，光线全程 x=0（x 方向分量恒为 0）
    - 视场绕 X 轴旋转（在 y-z 平面内旋转），保持 x=0
    - 用途：2D 布局图 / 动画快照。因光线始终在绘制平面内，
      追迹交点投影到 (z, y) 剖面后能精确落在镜面曲线上，
      不会出现"光线起点悬空在镜面外"的视觉失真

    返回:
        positions: (M,3) 光线起始位置（物面处，x=0）
        directions: (M,3) 光线单位方向（dx=0）
        wavelengths: (M,) 波长
        fov_labels: (M,) 视场角标签（度）
    """
    meta = lens.metadata
    rs = meta.ray_sampling
    n_rings = rs.aperture_rings

    # 入瞳位置与半径（与 generate_rays 相同逻辑）
    surf_z = compute_surface_z(lens.surfaces)
    stop_idx = 0
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            stop_idx = i
            break
    z_pupil = surf_z[stop_idx] if stop_idx > 0 else surf_z[1]

    ap_type = lens.metadata.aperture.type
    ap_value = lens.metadata.aperture.value
    if ap_type == ApertureType.NA:
        pupil_radius = ap_value * abs(z_pupil) * 1.0
    elif ap_type == ApertureType.DIAMETER:
        pupil_radius = ap_value / 2.0
    elif ap_type == ApertureType.F_NUMBER:
        approx_fl = abs(surf_z[-1] - surf_z[-2]) if len(surf_z) >= 2 else 50.0
        pupil_radius = approx_fl / ap_value / 2.0
    else:
        pupil_radius = ap_value * abs(z_pupil) * 1.0

    all_pos = []
    all_dir = []
    all_wl = []
    all_fov = []

    for wl in meta.wavelengths:
        for fov in meta.fov_deg:
            fov_rad = np.deg2rad(fov)
            cos_f = np.cos(fov_rad)
            sin_f = np.sin(fov_rad)
            # 每个环取 +y / -y 两个子午方位
            for ring in range(1, n_rings + 1):
                rho = ring / n_rings
                r_phys = rho * pupil_radius
                for sy in (1.0, -1.0):
                    pupil_point = np.array([0.0, sy * r_phys, z_pupil])
                    direction = pupil_point / np.linalg.norm(pupil_point)
                    # 绕 X 轴旋转视场（在 y-z 平面内旋转，保持 dx=0）：
                    #   d' = [dx, cosθ·dy - sinθ·dz, sinθ·dy + cosθ·dz]
                    d_rot = np.array([
                        0.0,
                        cos_f * direction[1] - sin_f * direction[2],
                        sin_f * direction[1] + cos_f * direction[2],
                    ])
                    all_pos.append([0.0, 0.0, 0.0])
                    all_dir.append(d_rot)
                    all_wl.append(wl)
                    all_fov.append(fov)

    positions = np.array(all_pos, dtype=np.float64)
    directions = np.array(all_dir, dtype=np.float64)
    wavelengths = np.array(all_wl, dtype=np.float64)
    fov_labels = np.array(all_fov, dtype=np.float64)
    return positions, directions, wavelengths, fov_labels


def compute_ray_footprint(lens: LensSystem) -> np.ndarray:
    """
    计算子午面光线在各表面的最大足迹 |y|（用于布局绘制范围）。

    某些 demo 系统的入瞳口径大于镜面半口径，边缘光线会打到镜面
    半口径之外。若镜面只画到 semi_aperture，这些光线的交点就会
    落在画出的镜面之外（看起来"起点悬空"）。
    布局绘制时将镜面范围扩展到 max(semi_aperture, 足迹)，
    保证所有追迹光线的交点都精确落在画出的镜面曲线上。

    返回:
        footprint: (n_surf,) 每个表面（含像面）的最大 |y| 足迹，
                   未命中返回 0
    """
    surf_z = compute_surface_z(lens.surfaces)
    n_surf = len(lens.surfaces)
    footprint = np.zeros(n_surf)

    positions, directions, _, _ = generate_meridional_rays(lens)
    for ray_idx in range(len(positions)):
        p = positions[ray_idx:ray_idx + 1].copy()
        d = directions[ray_idx:ray_idx + 1].copy()
        for i in range(1, n_surf):
            surf = lens.surfaces[i]
            t = intersect_surface(p, d, surf_z[i], surf)
            if not np.isfinite(t[0]):
                break
            p = p + t[0] * d
            # 足迹是相对偏心顶点的局部半径
            footprint[i] = max(footprint[i], abs(p[0, 1] - surf.decenter_y))
            if surf.type == SurfaceType.IMAGE:
                break
            asp = surf.aspheric
            A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)
            normals = surface_normal(
                p[:, 0] - surf.decenter_x, p[:, 1] - surf.decenter_y,
                p[:, 2] - surf_z[i], surf.curvature,
                asp.k if asp else 0.0, A4, A6, A8, A10, A12, A14,
            )
            if surf.is_reflective:
                d = reflect(d, normals)
            elif surf.type in (SurfaceType.STANDARD, SurfaceType.STOP):
                n1 = lens.get_glass_nd(lens.surfaces[i - 1].glass)
                n2 = lens.get_glass_nd(surf.glass)
                d, _ = refract(d, normals, n1, n2)
    return footprint


def surface_draw_extents(lens: LensSystem) -> np.ndarray:
    """
    布局绘制时每个表面的绘制半宽：max(半口径, 光线足迹)。

    保证所有追迹光线的交点都落在画出的镜面曲线范围内。
    """
    footprint = compute_ray_footprint(lens)
    extents = np.array([s.semi_aperture for s in lens.surfaces])
    return np.maximum(extents, footprint)


class TraceResult:
    """光线追迹结果容器。"""

    def __init__(self):
        self.image_positions: np.ndarray = np.array([])  # (M,3) 像面交点
        self.image_xy: np.ndarray = np.array([])          # (M,2) 像面 x,y
        self.wavelengths: np.ndarray = np.array([])
        self.fov_labels: np.ndarray = np.array([])
        self.active: np.ndarray = np.array([])             # (M,) bool 有效光线（未遮挡且追迹成功）
        self.obscured: np.ndarray = np.array([])           # (M,) bool 被镜子实体遮挡的光线
        self.obscuration_depth: float = 0.0                # 连续遮挡度量：最大穿入深度比例
                                                           # 0=无遮挡，1=光线穿过镜面中心
                                                           # （全遮挡个体间也有梯度，GA/DLS 能区分）
        self.incident_angles: dict[int, np.ndarray] = {}  # surface_id -> (M,) 入射角(度)
        self.exit_angles: np.ndarray = np.array([])        # (M,) 像面出射角(度)
        self.image_directions: np.ndarray = np.array([])   # (M,3) 像面处光线方向
        self.ray_segments: list[np.ndarray] = []           # 每光线交点序列 [(n_seg,3), ...]（含像面，供封装盒子/路径检查）
        self.footprint_regions: dict[int, tuple[float, float]] = {}  # 每镜面有效工作环带 [r_min, r_max]
        self.footprint_z: dict[int, tuple[float, float]] = {}            # 每镜面命中位置 z 范围 [z_min, z_max]（镜子实体位置）
        self.na_actual: float = 0.0
        self.working_distance: float = 0.0

    @property
    def obscured_ratio(self) -> float:
        """被遮挡光线比例（0~1）。"""
        if len(self.obscured) == 0:
            return 0.0
        return float(np.mean(self.obscured))

    @property
    def rms_spot_radius(self) -> float:
        """RMS 光斑半径（相对于主光线或质心）。"""
        if not self.active.any():
            return float("inf")
        xy = self.image_xy[self.active]
        if len(xy) == 0:
            return float("inf")
        centroid = np.mean(xy, axis=0)
        r = np.sqrt(np.sum((xy - centroid) ** 2, axis=1))
        return float(np.sqrt(np.mean(r ** 2)))

    @property
    def max_incident_angle(self) -> float:
        """所有反射面的最大入射角。"""
        max_ang = 0.0
        for ang_arr in self.incident_angles.values():
            valid = ang_arr[np.isfinite(ang_arr)]
            if len(valid) > 0:
                max_ang = max(max_ang, float(np.max(valid)))
        return max_ang

    def per_surface_max_incident(self) -> dict[int, float]:
        """每个反射面的最大入射角。"""
        result = {}
        for sid, ang_arr in self.incident_angles.items():
            valid = ang_arr[np.isfinite(ang_arr)]
            if len(valid) > 0:
                result[sid] = float(np.max(valid))
        return result


def trace_system(lens: LensSystem,
                 positions: Optional[np.ndarray] = None,
                 directions: Optional[np.ndarray] = None,
                 wavelengths: Optional[np.ndarray] = None,
                 fov_labels: Optional[np.ndarray] = None,
                 check_obscuration: bool = True,
                 ray_mode: Optional[str] = None) -> TraceResult:
    """
    对整个镜头系统执行序列光线追迹。

    流程：
    1. 预计算各面顶点 Z 坐标
    2. 生成采样光线（若未提供；模式由 ray_mode 或 lens.metadata.ray_mode 决定）
    3. 逐面求交 → 法线 → 折射/反射 → 更新方向
    4. 记录每个反射面的入射角、像面出射角
    5. 遮挡检测：检查每个光线段是否穿过其他镜子实体（口径内）
    6. 计算实际 NA、工作距离

    ray_mode 取值（默认 'auto'）：
      auto        离轴系统（chief_ray_angle_deg>0）用全 3D 离轴环形视场光线，
                  共轴系统用标准环形采样
      offaxis     强制用 generate_offaxis_rays（全 3D 离轴环形）
      meridional  用子午面光线（2×n_rings 条，结构搜索阶段加速）
      chief       仅主光线（1 条，超快速不遮挡筛选，仅限离轴系统）
      full        强制用 generate_rays（标准环形采样）

    参数:
        lens: LensSystem 镜头系统
        positions/directions/wavelengths/fov_labels: 可选的外部光线
        check_obscuration: 是否进行遮挡检测（默认开启）
        ray_mode: 光线采样模式，None 时用 lens.metadata.ray_mode

    返回:
        TraceResult 追迹结果（active 排除被遮挡光线，obscured 单独记录）
    """
    result = TraceResult()

    # 生成光线
    if positions is None:
        if ray_mode is None:
            ray_mode = getattr(lens.metadata, "ray_mode", "auto")
        chief_angle = lens.metadata.optical_constraints.chief_ray_angle_deg
        if ray_mode == "auto":
            if chief_angle > 0:
                positions, directions, wavelengths, fov_labels = generate_offaxis_rays(lens)
            else:
                positions, directions, wavelengths, fov_labels = generate_rays(lens)
        elif ray_mode == "offaxis":
            if lens.metadata.optical_constraints.rectangular_fov:
                positions, directions, wavelengths, fov_labels = generate_rect_offaxis_rays(lens)
            else:
                positions, directions, wavelengths, fov_labels = generate_offaxis_rays(lens)
        elif ray_mode == "meridional":
            if chief_angle > 0:
                positions, directions, wavelengths, fov_labels = generate_meridional_offaxis_rays(lens)
            else:
                positions, directions, wavelengths, fov_labels = generate_meridional_rays(lens)
        elif ray_mode == "chief":
            if chief_angle > 0:
                positions, directions, wavelengths, fov_labels = generate_chief_offaxis_rays(lens)
            else:
                positions, directions, wavelengths, fov_labels = generate_meridional_rays(lens)
        elif ray_mode == "full":
            positions, directions, wavelengths, fov_labels = generate_rays(lens)
        else:
            raise ValueError(f"未知 ray_mode: {ray_mode}")

    M = positions.shape[0]
    result.wavelengths = wavelengths
    result.fov_labels = fov_labels
    active = np.ones(M, dtype=bool)
    obscured = np.zeros(M, dtype=bool)
    max_pen = np.zeros(M, dtype=np.float64)  # 每光线最大穿入深度比例

    # 预计算表面 Z 坐标
    surf_z = compute_surface_z(lens.surfaces)
    surfaces = lens.surfaces
    n_surf = len(surfaces)

    # ------------------------------------------------------------------
    # 预追迹：计算每面镜子的有效工作区域（光线实际命中的足迹环带）
    # 真实 EUV 物镜的镜子不是完整圆镜，而是回转对称镜面的离轴部分——
    # 每面镜子只需"有效工作部分不重叠不遮挡"即可。
    # 遮挡判定用足迹环带 [r_min, r_max]（含裕量）代替整圆半口径：
    # 光线穿过其他镜子的"空余部分"（不在其工作环带内）不算遮挡。
    # ------------------------------------------------------------------
    foot_regions: dict[int, tuple[float, float]] = {}
    foot_z_regions: dict[int, tuple[float, float]] = {}
    foot_xy: dict[int, tuple[float, float, float, float]] = {}  # (xmin,xmax,ymin,ymax)
    if check_obscuration:
        p0 = positions.copy()
        d0 = directions.copy()
        foot_r: dict[int, list[np.ndarray]] = {}
        foot_z: dict[int, list[np.ndarray]] = {}
        foot_x: dict[int, list[np.ndarray]] = {}
        foot_y: dict[int, list[np.ndarray]] = {}
        for i in range(1, n_surf):
            surf = surfaces[i]
            t0 = intersect_surface(p0, d0, surf_z[i], surf)
            h0 = np.isfinite(t0)
            if h0.any():
                xl = p0[h0, 0] + t0[h0] * d0[h0, 0] - surf.decenter_x
                yl = p0[h0, 1] + t0[h0] * d0[h0, 1] - surf.decenter_y
                rl = np.sqrt(xl * xl + yl * yl)
                foot_r[i] = rl
                foot_z[i] = p0[h0, 2] + t0[h0] * d0[h0, 2]  # 命中位置全局 z
                foot_x[i] = p0[h0, 0] + t0[h0] * d0[h0, 0] - surf.decenter_x
                foot_y[i] = p0[h0, 1] + t0[h0] * d0[h0, 1] - surf.decenter_y
            if not h0.any():
                break
            p0[h0] = p0[h0] + t0[h0, np.newaxis] * d0[h0]
            if surf.type == SurfaceType.IMAGE:
                break
            c = surf.curvature
            asp = surf.aspheric
            A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)
            normals = surface_normal(
                p0[h0, 0] - surf.decenter_x, p0[h0, 1] - surf.decenter_y,
                p0[h0, 2] - surf_z[i], c,
                asp.k if asp else 0.0, A4, A6, A8, A10, A12, A14,
            )
            if surf.is_reflective:
                d0[h0] = reflect(d0[h0], normals)
            elif surf.type in (SurfaceType.STANDARD, SurfaceType.STOP):
                n1 = lens.get_glass_nd(surfaces[i - 1].glass)
                n2 = lens.get_glass_nd(surf.glass)
                d0[h0], _ = refract(d0[h0], normals, n1, n2)
        for i, rl in foot_r.items():
            if len(rl) > 0:
                rmin = float(rl.min())
                rmax = float(rl.max())
                margin = 0.1 * (rmax - rmin) + 2.0  # 裕量：10% 足迹宽度 + 2mm
                foot_regions[i] = (max(rmin - margin, 0.0), rmax + margin)
        for i, zl in foot_z.items():
            if len(zl) > 0:
                foot_z_regions[i] = (float(zl.min()), float(zl.max()))
        for i in foot_x:
            if len(foot_x[i]) > 0:
                foot_xy[i] = (float(foot_x[i].min()), float(foot_x[i].max()),
                              float(foot_y[i].min()), float(foot_y[i].max()))

    # 当前光线状态
    pos = positions.copy()
    dirs = directions.copy()
    ray_segments: list[np.ndarray] = []  # 每光线的交点序列
    for k in range(M):
        ray_segments.append(np.array([positions[k]]))

    # 逐面追迹（跳过物面，从第一面开始到像面）
    for i in range(1, n_surf):
        surf = surfaces[i]
        z_v = surf_z[i]

        # 求交
        t = intersect_surface(pos, dirs, z_v, surf)
        hit = np.isfinite(t) & active

        # ---- 命中裁剪：挖孔内(r<inner) → 光线穿过（不反射），保持原方向传播 ----
        # （半口径不裁剪命中——数值稳定；镜子大小通过遮挡判定体现：小镜子不挡别的光路）
        if surf.inner_aperture > 0 and hit.any():
            x_hit = pos[hit, 0] + t[hit] * dirs[hit, 0] - surf.decenter_x
            y_hit = pos[hit, 1] + t[hit] * dirs[hit, 1] - surf.decenter_y
            r_hit = np.sqrt(x_hit * x_hit + y_hit * y_hit)
            hole_rays = np.where(hit)[0][r_hit < surf.inner_aperture]
            if len(hole_rays) > 0:
                hit[hole_rays] = False  # 视为未命中，光线继续原方向传播

        # ---- 遮挡检测：本段光线是否穿过其他镜子的有效工作区域 ----
        if check_obscuration:
            seg_obsc = np.zeros(M, dtype=bool)
            for j in range(1, n_surf):
                if j == i or j not in foot_regions:
                    continue  # 无工作足迹的镜子不算实体遮挡
                other = surfaces[j]
                if not other.is_reflective:
                    continue
                t_o = intersect_surface(pos, dirs, surf_z[j], other)
                # 交点必须在段内：t_o ∈ (0, t) 之间
                valid = hit & np.isfinite(t_o) & (t_o > 1e-6) & (t_o < t - 1e-6)
                if valid.any():
                    xo = pos[:, 0] + t_o * dirs[:, 0] - other.decenter_x
                    yo = pos[:, 1] + t_o * dirs[:, 1] - other.decenter_y
                    ro = np.sqrt(xo * xo + yo * yo)
                    rmin_j, rmax_j = foot_regions[j]
                    inner_j = other.inner_aperture  # 中心挖孔：孔内光线穿过不算遮挡
                    margin_j = getattr(lens.metadata.optical_constraints, "obscuration_margin_mm", 0.0)
                    # 矩形视场（单片镜片）：用 x,y 包围盒；其余（环形/离轴）用环带判定
                    # 注意：foot_xy 对所有系统都会预计算，必须按系统类型选分支——
                    # 环形/离轴系统若误走 box 分支会因未按半口径裁剪而过度判遮挡。
                    is_rect = getattr(lens.metadata.optical_constraints, "rectangular_fov", False)
                    if is_rect and j in foot_xy:
                        (xmin_j, xmax_j, ymin_j, ymax_j) = foot_xy[j]
                        inside = valid & (xo >= xmin_j - margin_j) & (xo <= xmax_j + margin_j) & \
                                 (yo >= ymin_j - margin_j) & (yo <= ymax_j + margin_j)
                        seg_obsc |= inside
                        if inside.any():
                            span = max(xmax_j - xmin_j, ymax_j - ymin_j, 1e-9)
                            pen = 1.0 - np.sqrt(((xo - 0.5*(xmin_j+xmax_j))/span)**2 +
                                                ((yo - 0.5*(ymin_j+ymax_j))/span)**2)
                            pen = np.clip(pen, 0.0, 1.0)
                            max_pen = np.maximum(max_pen, np.where(inside, pen, 0.0))
                    else:  # 环形/离轴视场：环形环带判定（按半口径裁剪）
                        rmin_j, rmax_j = foot_regions[j]
                        inner_j = other.inner_aperture
                        rmax_eff = min(rmax_j, other.semi_aperture) if other.semi_aperture > 0 else rmax_j
                        inside = valid & (ro >= max(rmin_j, inner_j) - margin_j) & (ro <= rmax_eff + margin_j)
                        seg_obsc |= inside
                        if inside.any():
                            span = max(rmax_j - rmin_j, 1e-9)
                            pen = np.clip((rmax_j - ro) / span, 0.0, 1.0)
                            max_pen = np.maximum(max_pen, np.where(inside, pen, 0.0))
            obscured |= seg_obsc

        active = active & hit

        # 更新位置
        pos[hit] = pos[hit] + t[hit, np.newaxis] * dirs[hit]

        # 记录每光线的交点（含像面；未命中的光线保持起点）
        for k in np.where(hit)[0]:
            ray_segments[k] = np.vstack([ray_segments[k], pos[k]])

        if not active.any():
            break

        # 像面：记录交点，不做折射/反射
        if surf.type == SurfaceType.IMAGE:
            result.image_positions = pos.copy()
            result.image_xy = pos[:, :2].copy()
            result.image_directions = dirs.copy()  # 像面处光线方向（供像方远心判定）
            # 出射角：光线方向与 Z 轴夹角
            cos_exit = np.abs(dirs[:, 2])
            cos_exit = np.clip(cos_exit, 0, 1)
            result.exit_angles = np.rad2deg(np.arccos(cos_exit))
            break

        # 计算法线
        c = surf.curvature
        k = surf.aspheric.k if surf.aspheric else 0.0
        A4, A6, A8, A10, A12, A14 = aspheric_coeffs(surf)

        # 交点处的局部坐标（相对于偏心顶点）
        x_local = pos[:, 0] - surf.decenter_x
        y_local = pos[:, 1] - surf.decenter_y
        z_local = pos[:, 2] - z_v

        normals = surface_normal(x_local, y_local, z_local, c, k, A4, A6, A8, A10, A12, A14)

        # 入射角（入射方向与法线夹角，取锐角）
        cos_inc = np.abs(np.sum(dirs * normals, axis=1))
        cos_inc = np.clip(cos_inc, 0, 1)
        inc_angle_deg = np.rad2deg(np.arccos(cos_inc))
        inc_angle_deg[~active] = np.nan
        # 所有光学面（折射+反射）都记录入射角，物面和像面已跳过
        result.incident_angles[surf.surface_id] = inc_angle_deg

        # 反射面
        if surf.is_reflective:
            dirs = reflect(dirs, normals)

        # 折射面
        elif surf.type in (SurfaceType.STANDARD, SurfaceType.STOP):
            # 获取两侧折射率
            prev_glass = surfaces[i - 1].glass if i > 0 else "air"
            n1 = lens.get_glass_nd(prev_glass)
            n2 = lens.get_glass_nd(surf.glass)
            new_dirs, tir = refract(dirs, normals, n1, n2)
            # 全反射的光线标记为无效
            active = active & ~tir
            dirs = new_dirs

        # 其他面（object 已跳过，plane 等）
        else:
            pass  # 平面等不改变方向

    # 被遮挡的光线在物理上到不了像面，从有效光线中排除
    result.obscured = obscured
    result.obscuration_depth = float(np.max(max_pen)) if M > 0 else 0.0
    result.ray_segments = ray_segments
    result.footprint_regions = foot_regions
    result.footprint_z = foot_z_regions
    result.footprint_xy = foot_xy
    active = active & ~obscured
    result.active = active

    # 计算实际 NA：像面处光线的最大 sin(theta)
    if active.any():
        cos_angles = np.abs(dirs[active, 2])
        sin_angles = np.sqrt(np.clip(1 - cos_angles ** 2, 0, 1))
        result.na_actual = float(np.max(sin_angles))

    # 工作距离：最后一个光学面到像面的轴向距离
    opt_surfs = [s for s in surfaces if s.type not in (SurfaceType.IMAGE,)]
    if opt_surfs:
        last_opt_idx = surfaces.index(opt_surfs[-1])
        result.working_distance = abs(surf_z[-1] - surf_z[last_opt_idx])

    return result


# ===========================================================================
# 便捷函数：追迹并返回核心指标
# ===========================================================================

def trace_and_summarize(lens: LensSystem) -> dict:
    """
    追迹系统并返回核心指标字典，用于评价函数和迭代报告。
    """
    res = trace_system(lens)
    return {
        "rms_spot": res.rms_spot_radius,
        "na_actual": res.na_actual,
        "max_incident_angle": res.max_incident_angle,
        "per_surface_max_incident": res.per_surface_max_incident(),
        "max_exit_angle": float(np.nanmax(result_exit_angles(res))) if res.active.any() else 0.0,
        "working_distance": res.working_distance,
        "active_rays": int(np.sum(res.active)),
        "total_rays": len(res.active),
        "trace_result": res,
    }


def result_exit_angles(res: TraceResult) -> np.ndarray:
    """辅助函数：获取有效光线的出射角。"""
    if len(res.exit_angles) == 0:
        return np.array([0.0])
    return res.exit_angles[res.active]
