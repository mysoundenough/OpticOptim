"""
merit_function.py — 评价函数
==============================
定义光学系统优化的评价函数（merit function）。

评价函数由三部分组成：
1. 基础像差项：RMS 光斑、畸变、色差
2. 光学约束惩罚：NA 偏差、入射角超限、出射角超限、工作距离
3. 结构约束惩罚：最小中心厚度、最小空气间隔、系统总长

提供两个接口：
- merit_residuals(lens) -> np.ndarray：残差向量（10 项），供 DLS 最小二乘使用
- merit_scalar(lens) -> float：加权标量值，供 GA/SA 和迭代报告使用

残差项顺序（固定，确保 Jacobian 维度一致）：
  0: rms_spot          RMS 光斑半径 (mm)
  1: distortion        畸变 (mm)
  2: chroma            色差 (mm)
  3: na_penalty        NA 偏差惩罚
  4: angle_incident    入射角超限惩罚 (度)
  5: angle_exit        出射角超限惩罚 (度)
  6: working_distance  工作距离惩罚 (mm)
  7: obscuration       被镜子遮挡的光线比例 (0~1) + 穿入深度（连续）
  8: missing           追迹失败（未遮挡但到不了像面）光线比例 (0~1)
  9: min_thickness     最小中心厚度违反 (mm)
  10: min_air_gap      最小空气间隔违反 (mm)
  11: total_length     系统总长超限 (mm)
  12: telecentricity   像方远心偏差（像面光线平均方向与光轴夹角，度）
  13: package           封装盒子干涉（光路/有效结构超出盒子的最大距离，mm）
  14: mirror_gap        镜子间最小间距违反（径向环带相交时轴向间距不足，mm）
  15: image_z           像面位置偏差（mm）
  16: magnification     放大率偏差（像高/物高）
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from lens_schema import LensSystem
from raytrace import trace_system, TraceResult


# 残差项名称与索引
RESIDUAL_NAMES = [
    "rms_spot",
    "distortion",
    "chroma",
    "na_penalty",
    "angle_incident",
    "angle_exit",
    "working_distance",
    "obscuration",
    "missing",
    "min_thickness",
    "min_air_gap",
    "total_length",
    "telecentricity",
    "package",
    "mirror_gap",
    "image_z",
    "magnification",
    "object_path",
    "mirror_z",
]
NUM_RESIDUALS = len(RESIDUAL_NAMES)


# ===========================================================================
# 单项计算
# ===========================================================================

def compute_rms_spot(res: TraceResult, lost_penalty: float = 50.0) -> float:
    """
    RMS 光斑半径。

    对所有光线计算，其中被遮挡或追迹失败的光线按 lost_penalty (mm)
    计入。这样优化器无法通过"丢弃光线"（让光线被挡/追不到像面）
    来降低 RMS。折射系统（如 Cooke 三分离）边缘光线被口径自然渐晕
    时可在 merit_weights.lost_ray_penalty_mm 调低该惩罚。
    """
    n_total = max(len(res.active), 1)
    n_active = int(np.sum(res.active))
    if n_active == 0:
        return lost_penalty  # 全部失败

    xy = res.image_xy[res.active]
    centroid = np.mean(xy, axis=0)
    r = np.sqrt(np.sum((xy - centroid) ** 2, axis=1))
    rms_active = float(np.sqrt(np.mean(r ** 2)))

    # 未激活光线（被遮挡/失败）按 lost_penalty 计入
    n_lost = n_total - n_active
    if n_lost > 0:
        rms_all = np.sqrt((n_active * rms_active ** 2 + n_lost * lost_penalty ** 2) / n_total)
        return float(rms_all)
    return rms_active


def compute_distortion(res: TraceResult) -> float:
    """
    畸变（简化版）。

    畸变 = |实际像高 - 理想像高|，单位 mm。

    理想像高由视场角正切比例确定：
      ideal_r_max = r_ref * tan(fov_max) / tan(fov_ref)

    用绝对偏差而非相对比值，避免数值爆炸，量级与光斑半径一致。
    """
    if not res.active.any():
        return 1.0  # 大惩罚

    fovs = np.unique(res.fov_labels)
    if len(fovs) < 2:
        return 0.0  # 只有一个视场，无法计算畸变

    # 参考视场：最小的非零视场（轴上视场质心可能为 0，不适合做参考）
    nonzero_fovs = fovs[np.abs(fovs) > 1e-6]
    if len(nonzero_fovs) == 0:
        return 0.0
    fov_ref = nonzero_fovs[0]

    mask_ref = (res.fov_labels == fov_ref) & res.active
    if not mask_ref.any():
        return 0.0
    centroid_ref = np.mean(res.image_xy[mask_ref], axis=0)
    r_ref = np.sqrt(np.sum(centroid_ref ** 2))

    # 最大视场
    fov_max = fovs[-1]
    mask_max = (res.fov_labels == fov_max) & res.active
    if not mask_max.any():
        return 0.0
    centroid_max = np.mean(res.image_xy[mask_max], axis=0)
    r_max = np.sqrt(np.sum(centroid_max ** 2))

    # 理想像高比 = tan(fov_max) / tan(fov_ref)
    tan_ref = np.tan(np.deg2rad(fov_ref))
    tan_max = np.tan(np.deg2rad(fov_max))
    if abs(tan_ref) < 1e-10:
        return 0.0
    ideal_ratio = tan_max / tan_ref
    ideal_r_max = r_ref * ideal_ratio

    # 畸变 = 实际像高与理想像高的绝对偏差 (mm)
    return float(abs(r_max - ideal_r_max))


def compute_chroma(res: TraceResult) -> float:
    """
    色差（简化版）。

    比较不同波长光线在像面上的光斑质心偏移。
    对于全反射系统，色差为 0（反射不依赖波长），
    此项通常权重设为 0。
    """
    if not res.active.any():
        return 1.0

    wls = np.unique(res.wavelengths)
    if len(wls) < 2:
        return 0.0

    centroids = []
    for wl in wls:
        mask = (res.wavelengths == wl) & res.active
        if mask.any():
            centroids.append(np.mean(res.image_xy[mask], axis=0))

    if len(centroids) < 2:
        return 0.0

    # 最大质心间距作为色差度量
    c = np.array(centroids)
    diffs = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
    return float(np.max(diffs))


def compute_na_penalty(res: TraceResult, na_target: float) -> float:
    """
    NA 偏差惩罚。

    实际 NA 与目标 NA 的绝对偏差。
    NA 过小会降低分辨率，NA 过大可能超出设计约束。
    """
    return float(abs(res.na_actual - na_target))


def compute_incident_angle_penalty(res: TraceResult, max_angle_deg: float) -> float:
    """
    入射角超限惩罚。

    对所有反射面，计算超过最大允许入射角的部分，取最大值。
    大入射角会导致偏振效应、阴影遮挡和像差增大。
    """
    max_excess = 0.0
    for ang_arr in res.incident_angles.values():
        valid = ang_arr[np.isfinite(ang_arr)]
        if len(valid) > 0:
            excess = np.maximum(valid - max_angle_deg, 0.0)
            max_excess = max(max_excess, float(np.max(excess)))
    return max_excess


def compute_exit_angle_penalty(res: TraceResult, max_angle_deg: float) -> float:
    """
    出射角超限惩罚。

    像面处光线与光轴夹角超过允许值的部分。
    大出射角会导致像面照度不均匀和探测器耦合困难。
    """
    if not res.active.any() or len(res.exit_angles) == 0:
        return 0.0
    valid = res.exit_angles[res.active]
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return 0.0
    excess = np.maximum(valid - max_angle_deg, 0.0)
    return float(np.max(excess))


def compute_working_distance_penalty(res: TraceResult, min_distance: float) -> float:
    """
    工作距离惩罚。

    工作距离（最后光学面到像面的距离）小于最小值时的超出量。
    工作距离过小会导致机械干涉和装调困难。
    """
    if res.working_distance <= 0:
        return float(abs(min_distance))  # 大惩罚
    return float(max(min_distance - res.working_distance, 0.0))


def compute_telecentricity_penalty(res: TraceResult, max_deg: float) -> float:
    """
    像方远心偏差惩罚。

    像方远心：像面处所有光线的平均方向（光束中心）平行光轴（+z）。
    偏差 = 平均方向与光轴的夹角，超过允许值 max_deg 的部分作为惩罚。
    真实 EUV 物镜像方远心（晶圆侧主光线平行光轴），
    是系统级约束，非远心会导致不同视场照度不均、焦深变化。
    """
    if max_deg <= 0:
        return 0.0  # 未启用
    if len(res.image_directions) == 0 or not res.active.any():
        return 1.0
    dirs = res.image_directions[res.active]
    mean_dir = np.mean(dirs, axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-12:
        return 1.0
    mean_dir /= norm
    # 平均方向与 +z 的夹角（度）
    cos_a = np.clip(mean_dir[2], -1.0, 1.0)
    angle = float(np.rad2deg(np.arccos(cos_a)))
    return float(max(angle - max_deg, 0.0))


def compute_package_penalty(lens: LensSystem, res: TraceResult) -> float:
    """
    封装盒子干涉惩罚。

    整个系统必须装进给定盒子（外包络），盒子有入口（物面）和出口（像面）：
    1. 光路不干涉：所有光线路径（含每段中点）必须在盒子内
    2. 有效结构不干涉：每面镜子的有效工作部分（光线足迹环带）必须在盒子内

    惩罚 = 最大超出量（mm，连续，有梯度）：
      光线穿出盒子的最大距离 / 工作环带外缘超出盒子的最大距离。
    """
    box = lens.metadata.package
    if not box.enabled:
        return 0.0

    # 盒子范围
    z_min, z_max = box.z_min, box.z_max
    if box.shape == "box":
        x_half = box.x_half if box.x_half else box.r_max
        y_half = box.y_half if box.y_half else box.r_max
    else:  # cylinder
        r_max = box.r_max

    pen = 0.0

    # ---- 1) 光路不干涉：交点 + 段中点 ----------------
    for seg in res.ray_segments:
        pts = seg
        if len(pts) >= 2:
            mids = 0.5 * (pts[:-1] + pts[1:])
            check = np.vstack([pts, mids])
        else:
            check = pts
        zz = check[:, 2]
        pen = max(pen, float(np.max(zz - z_max)), float(np.max(z_min - zz)))
        if box.shape == "box":
            xx = np.abs(check[:, 0]); yy = np.abs(check[:, 1])
            pen = max(pen, float(np.max(xx - x_half)), float(np.max(yy - y_half)))
        else:
            rr = np.sqrt(check[:, 0] ** 2 + check[:, 1] ** 2)
            pen = max(pen, float(np.max(rr - r_max)))

    # ---- 2) 有效结构不干涉：工作环带外缘超出盒子 -------
    for i, (rmin, rmax) in res.footprint_regions.items():
        if box.shape == "box":
            # 环带外缘近似为 r_max（保守：方形角点 r√2）
            pen = max(pen, rmax * np.sqrt(2.0) - min(x_half, y_half))
        else:
            pen = max(pen, rmax - r_max)

    # ---- 3) 入口/出口语义：物面在入口侧（盒子外），像面在出口侧（盒子外）----
    #      入射光线经入口（z_min 面）射入，出射光线经出口（z_max 面）射出盒子外。
    if len(res.ray_segments) > 0:
        # 物面 z = 0（起点），应在入口侧/盒外（z_obj ≤ z_min）
        z_obj = float(np.min([seg[0][2] for seg in res.ray_segments]))
        pen = max(pen, max(0.0, z_obj - z_min))
        # 像面 z（最后一段光路终点），应在出口侧/盒外（z_img ≥ z_max）
        z_img = float(np.max([seg[-1][2] for seg in res.ray_segments]))
        pen = max(pen, max(0.0, z_max - z_img))
        # 入射光线经入口射入：第一段光路（物面→第一镜）z 范围必须覆盖 z_min
        seg0 = res.ray_segments[0]
        z_first = float(max(seg0[0][2], seg0[1][2])) if len(seg0) >= 2 else float(seg0[0][2])
        pen = max(pen, max(0.0, z_min - z_first))
        # 出射光线经出口射出：最后一段光路（最后镜→像面）z 范围必须覆盖 z_max
        segl = res.ray_segments[0]
        if len(segl) >= 2:
            z_last = float(min(segl[-2][2], segl[-1][2]))
            pen = max(pen, max(0.0, z_last - z_max))

    return float(max(pen, 0.0))


def compute_mirror_gap_penalty(res: TraceResult, min_gap: float) -> float:
    """
    镜子间最小间距惩罚（物理可制造性）。

    任意两面镜子：若径向工作环带相交（同心环 r 区间重叠），
    则轴向中心距必须 ≥ min_gap（镜子实体+支撑的最小安装间距）。
    违反量 = min_gap - 轴向间距（连续，有梯度）。

    径向环带完全错开的两镜（套娃式同心不同半径）不检查轴向，
    因为物理上互不接触。
    """
    if min_gap <= 0 or len(res.footprint_regions) < 2:
        return 0.0
    pen = 0.0
    ids = sorted(res.footprint_regions.keys())
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if i not in res.footprint_z or j not in res.footprint_z:
                continue
            zmin_i, zmax_i = res.footprint_z[i]
            zmin_j, zmax_j = res.footprint_z[j]
            zc_i = 0.5 * (zmin_i + zmax_i)
            zc_j = 0.5 * (zmin_j + zmax_j)
            gap = abs(zc_i - zc_j)
            pen = max(pen, max(0.0, min_gap - gap))
    return float(pen)


def compute_image_z_penalty(res: TraceResult, z_target: float) -> float:
    """像面位置偏差：|实际像面 z - 目标 z|（mm）。物面固定 z=0。"""
    if z_target <= 0 or len(res.ray_segments) == 0:
        return 0.0
    z_img = float(np.max([seg[-1][2] for seg in res.ray_segments]))
    return float(abs(z_img - z_target))


def compute_magnification_penalty(res: TraceResult, target: float) -> float:
    """
    放大率偏差：|像高/物高 - 目标|。

    像高 = 有效光线像面交点质心的离轴距离；
    物高 = 物点离轴高度（ray_segments 起点）。
    光刻物镜 4:1 缩小 → target=0.25。
    """
    if target <= 0 or len(res.ray_segments) == 0 or not res.active.any():
        return 0.0
    # 物高：优先用配置的 object_height_mm（视场中心离轴量），否则物点离轴距离
    import lens_schema as _ls
    obj_h = 0.0
    obj_xy = res.ray_segments[0][0]
    obj_h = float(np.sqrt(obj_xy[0] ** 2 + obj_xy[1] ** 2))
    if obj_h < 1e-9:
        return 1.0
    xy = res.image_xy[res.active]
    centroid = np.mean(xy, axis=0)
    img_h = float(np.sqrt(centroid[0] ** 2 + centroid[1] ** 2))
    return float(abs(img_h / obj_h - target))


def compute_object_path_penalty(res: TraceResult, lens: LensSystem) -> float:
    """
    物光路穿镜惩罚。

    物面发出的光在到达第一面镜子（M1）之前的传播路径上，
    不能穿过任何其他镜子的实体区域（单片 x/y 包围盒）。

    折返布局中 M2 等镜子 z 常小于 M1，若其实体覆盖物光路横向位置，
    光线会"先穿过 M2 再反射 M1"——布局不合理（用户指出）。
    该约束强制物光路干净（M1 是光路上第一个镜面）。

    返回最大违反量（mm，0=物光路干净）。
    """
    if len(res.ray_segments) < 2 or len(res.ray_segments[0]) < 2:
        return 0.0
    pen = 0.0
    for seg in res.ray_segments:
        if len(seg) < 2:
            continue
        a = seg[0]
        b = seg[1]
        zmin = min(a[2], b[2])
        zmax = max(a[2], b[2])
        for i, s in enumerate(lens.surfaces):
            if not s.is_reflective:
                continue
            if i not in res.footprint_z:
                continue
            zlo, zhi = res.footprint_z[i]
            zc = 0.5 * (zlo + zhi)
            # 该镜面位于物光路区间内
            if zmin < zc < zmax and abs(b[2] - a[2]) > 1e-9:
                f = (zc - a[2]) / (b[2] - a[2])
                if 0 < f < 1:
                    xc = a[0] + f * (b[0] - a[0])
                    yc = a[1] + f * (b[1] - a[1])
                    if i in res.footprint_xy:
                        xmin2, xmax2, ymin2, ymax2 = res.footprint_xy[i]
                        if (xmin2 - 2 <= xc <= xmax2 + 2) and (ymin2 - 2 <= yc <= ymax2 + 2):
                            pen = max(pen, 10.0)
    return float(pen)


def compute_mirror_z_penalty(lens: LensSystem) -> float:
    """所有镜子顶点 z 必须 > 0（物面 z=0，镜子在正侧；之字形布局）。"""
    from raytrace import compute_surface_z
    z = compute_surface_z(lens.surfaces)
    pen = 0.0
    for i in range(1, len(lens.surfaces) - 1):  # 镜子（不含物面/像面）
        if z[i] < 0:
            pen = max(pen, -z[i])
    return float(pen)


def compute_min_thickness_penalty(lens: LensSystem, min_thickness: float) -> float:
    """
    最小中心厚度惩罚。

    遍历所有镜片材料内部的面（glass != "air"），检查其 thickness（中心厚度）
    是否小于最小值。返回最大违反量。

    镜片中心厚度过小会导致加工困难、应力过大、易碎裂。
    """
    if min_thickness <= 0:
        return 0.0
    max_violation = 0.0
    for surf in lens.surfaces:
        # 只检查镜片材料内部的厚度（当前面 glass != air，thickness 是镜片中心厚度）
        if surf.glass != "air" and surf.type not in ("object", "image"):
            violation = max(min_thickness - surf.thickness, 0.0)
            max_violation = max(max_violation, violation)
    return float(max_violation)


def compute_min_air_gap_penalty(lens: LensSystem, min_gap: float) -> float:
    """
    最小空气间隔惩罚。

    遍历所有空气间隔（glass == "air" 的面的 thickness），检查是否小于最小值。
    跳过：
    - 物面（第一个面）：物距不属于"镜片之间"的间隔
    - 像面（最后一个面）：像面没有 thickness
    - 最后一个光学面到像面的距离：即后截距/工作距离，已有单独约束 working_distance_min

    返回最大违反量。

    空气间隔过小会导致相邻镜片机械干涉、装调困难。
    """
    if min_gap <= 0:
        return 0.0
    # 找到最后一个非像面的光学面的索引（其后的 thickness 是工作距离，跳过）
    last_optical_idx = len(lens.surfaces) - 1
    for i in range(len(lens.surfaces) - 1, -1, -1):
        if lens.surfaces[i].type != "image":
            last_optical_idx = i
            break

    max_violation = 0.0
    for i, surf in enumerate(lens.surfaces):
        # 跳过：物面（物距）、最后一个光学面（工作距离/后截距）、像面
        if i == 0 or i == last_optical_idx or surf.type == "image":
            continue
        # 只检查空气间隔（当前面 glass == air，thickness 是到下一面的空气距离）
        if surf.glass == "air":
            violation = max(min_gap - surf.thickness, 0.0)
            max_violation = max(max_violation, violation)
    return float(max_violation)


def compute_total_length_penalty(lens: LensSystem, max_length: float) -> float:
    """
    系统总长惩罚。

    系统总长 = 从物面到像面的所有 thickness 之和。
    超过最大长度时返回超出量。max_length <= 0 表示不约束。

    系统总长过大会导致系统体积过大、不满足安装空间要求。
    """
    if max_length <= 0:
        return 0.0
    total = sum(surf.thickness for surf in lens.surfaces)
    return float(max(total - max_length, 0.0))


# ===========================================================================
# 评价函数主接口
# ===========================================================================

def merit_residuals(lens: LensSystem,
                    trace_res: Optional[TraceResult] = None) -> np.ndarray:
    """
    计算评价函数残差向量（7 项）。

    残差向量供 DLS 阻尼最小二乘使用，DLS 最小化 sum(w_i * r_i^2)。

    参数:
        lens: LensSystem 镜头系统
        trace_res: 可选的已追迹结果，避免重复追迹

    返回:
        residuals: (7,) numpy 数组
    """
    if trace_res is None:
        trace_res = trace_system(lens)

    constraints = lens.metadata.optical_constraints
    spacing = lens.metadata.spacing_constraints
    weights = lens.optimization_config.merit_weights

    r = np.zeros(NUM_RESIDUALS, dtype=np.float64)

    # 基础像差项
    r[0] = compute_rms_spot(trace_res, weights.lost_ray_penalty_mm)  # rms_spot
    r[1] = compute_distortion(trace_res)                   # distortion
    r[2] = compute_chroma(trace_res)                       # chroma

    # 光学约束惩罚项
    r[3] = compute_na_penalty(trace_res, constraints.na_target)
    r[4] = compute_incident_angle_penalty(
        trace_res, constraints.incident_angle_max_deg
    )
    r[5] = compute_exit_angle_penalty(
        trace_res, constraints.exit_angle_max_deg
    )
    r[6] = compute_working_distance_penalty(
        trace_res, constraints.working_distance_min
    )

    # 结构约束惩罚项（间距约束）
    # 遮挡残差 = 0.5·加权遮挡比例(主光线权重大) + 0.5·最大穿透深度(连续)
    #   - 加权遮挡比例：主光线被挡权重 3×，边缘 1× → 优化优先保主光线
    #   - 无遮挡时为 0；全遮挡时穿透深度提供连续梯度
    if len(trace_res.is_chief) == len(trace_res.obscured):
        w_arr = np.where(trace_res.is_chief, 3.0, 1.0)
        weighted_ratio = float(np.sum(w_arr * trace_res.obscured) / max(np.sum(w_arr), 1.0))
    else:
        weighted_ratio = trace_res.obscured_ratio
    r[7] = 0.5 * weighted_ratio + 0.5 * trace_res.obscuration_depth
    # 追迹失败比例：未遮挡但没到像面（防止优化器丢弃光线作弊）
    total_rays = max(len(trace_res.active), 1)
    n_reached = int(np.sum(trace_res.active)) + int(np.sum(trace_res.obscured))
    r[8] = 1.0 - n_reached / total_rays
    r[9] = compute_min_thickness_penalty(lens, spacing.min_center_thickness)
    r[10] = compute_min_air_gap_penalty(lens, spacing.min_air_gap)
    r[11] = compute_total_length_penalty(lens, spacing.max_total_length)
    r[12] = compute_telecentricity_penalty(
        trace_res, constraints.telecentric_image_max_deg
    )
    r[13] = compute_package_penalty(lens, trace_res)
    r[14] = compute_mirror_gap_penalty(trace_res, spacing.min_mirror_gap)
    r[15] = compute_image_z_penalty(trace_res, constraints.image_z_fixed)
    r[16] = compute_magnification_penalty(trace_res, constraints.magnification_target)
    r[17] = compute_object_path_penalty(trace_res, lens)
    r[18] = compute_mirror_z_penalty(lens)
    # 无效光线（全部被遮挡/失败）时给大惩罚，但保留遮挡/缺失比例信息
    # 供全局优化器区分"全遮挡"与"部分遮挡"（否则所有个体 merit 相同，无选择压力）
    if not trace_res.active.any():
        r[0] = 50.0    # rms 大惩罚
        r[1] = 50.0    # distortion 大惩罚
        r[3] = 1.0     # NA 偏差
        r[4] = 30.0    # 入射角
        r[5] = 30.0    # 出射角
        r[6] = 30.0    # 工作距离
        r[12] = 30.0   # 远心偏差
        r[13] = 30.0   # 盒子干涉
        # r[7] obscuration、r[8] missing 保持实际比例

    return r


def merit_scalar(lens: LensSystem,
                 trace_res: Optional[TraceResult] = None) -> float:
    """
    计算加权标量评价函数值。

    merit = sum(w_i * r_i^2)

    供 GA/SA 全局优化和迭代报告使用。
    """
    r = merit_residuals(lens, trace_res)
    weights = lens.optimization_config.merit_weights
    w = np.array([
        weights.rms_spot,
        weights.distortion,
        weights.chroma,
        weights.na_penalty,
        weights.angle_incident_penalty,
        weights.angle_exit_penalty,
        weights.working_distance_penalty,
        weights.obscuration_penalty,
        weights.missing_penalty,
        weights.min_thickness_penalty,
        weights.min_air_gap_penalty,
        weights.total_length_penalty,
        weights.telecentricity_penalty,
        weights.package_penalty,
        weights.mirror_gap_penalty,
        weights.image_z_penalty,
        weights.magnification_penalty,
        weights.object_path_penalty,
        weights.mirror_z_penalty,
    ], dtype=np.float64)
    return float(np.sum(w * r * r))


def merit_detail(lens: LensSystem,
                 trace_res: Optional[TraceResult] = None) -> dict:
    """
    返回评价函数的详细分解，用于迭代报告打印。
    """
    if trace_res is None:
        trace_res = trace_system(lens)

    r = merit_residuals(lens, trace_res)
    weights = lens.optimization_config.merit_weights
    w = np.array([
        weights.rms_spot,
        weights.distortion,
        weights.chroma,
        weights.na_penalty,
        weights.angle_incident_penalty,
        weights.angle_exit_penalty,
        weights.working_distance_penalty,
        weights.obscuration_penalty,
        weights.missing_penalty,
        weights.min_thickness_penalty,
        weights.min_air_gap_penalty,
        weights.total_length_penalty,
        weights.telecentricity_penalty,
        weights.package_penalty,
        weights.mirror_gap_penalty,
        weights.image_z_penalty,
        weights.magnification_penalty,
        weights.object_path_penalty,
        weights.mirror_z_penalty,
    ])

    detail = {name: {"residual": float(r[i]), "weight": float(w[i]),
                      "contribution": float(w[i] * r[i] * r[i])}
              for i, name in enumerate(RESIDUAL_NAMES)}
    detail["total"] = float(np.sum(w * r * r))
    detail["trace_result"] = trace_res
    return detail
