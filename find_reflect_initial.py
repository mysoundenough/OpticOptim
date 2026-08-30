#!/usr/bin/env python3
"""
find_reflect_initial.py — 找反射初始结构（理想反射面主光线路径设计）
====================================================
全新思路（保留 optimize_pipeline / plot_structure / plot_analysis 之前能力）：

想象没有实体镜子，光线按约束从物面出发，到达一个"反射平面位置"被反射，
再到下一个平面位置……依次到达像面。

- 只用中心视场点的中心光线（1 条主光线）
- 反射平面：位置（反射点）+ 放置角度（法向量由反射定律从入射/出射方向自动确定）
- 平面不存在遮挡（理想反射面，法向量控制反射方向）
- 目前面垂直 z-y 平面（所有反射点在 z-y 平面，x=0）

优化变量：6 个反射点的 (y, z) 位置（12 个）。
约束：入射角 ≤20°、镜子 z 全正、间距 ≥20、之字形（z 递增、光一路 +z）、
像点位置（放大率 0.25 → 像高 9.2mm @ z=3500）。

反射定律：给定入射方向 d_in 和出射方向 d_out，
法向量 n ∝ normalize(d_in - d_out)（平面放置角度 = 法向量方向）。

用法：
    python find_reflect_initial.py                  # 默认设计并打印/绘图
    python find_reflect_initial.py --z1 400 --z6 3000  # 指定反射点 z 范围
"""

from __future__ import annotations

import argparse
import copy
import os
import numpy as np
from ga_sa_optimizer import tournament_select, arithmetic_crossover, gaussian_mutate

# ===========================================================================
# 几何常量（约束）
# ===========================================================================
OBJ_POINT = np.array([0.0, 36.8, 0.0])   # 中心视场点（物高 36.8）
CHIEF_ANGLE = 15.0                        # 物方主光线入射角（度）
IMG_POINT = np.array([0.0, 9.2, 3500.0])  # 像点（像高 = 36.8/4 = 9.2，放大率 0.25）
MAX_INCIDENT = 20.0                       # 反射面入射角上限
MIN_GAP = 20.0                            # 反射点最小间距
MIN_CLEAR = 50.0                          # 反射点离非本镜子光线的最小垂直距离约束


def chief_direction(theta: float = CHIEF_ANGLE) -> np.ndarray:
    """物方主光线方向：y-z 面内向下 theta 度（统一，所有视场点同向）。"""
    t = np.deg2rad(theta)
    return np.array([0.0, -np.sin(t), np.cos(t)])


def reflect_normal(d_in: np.ndarray, d_out: np.ndarray) -> np.ndarray:
    """反射定律：法向量 n ∝ d_in - d_out（单位化）。"""
    n = d_in - d_out
    n /= np.linalg.norm(n)
    return n


def incident_angle(d_in: np.ndarray, n: np.ndarray) -> float:
    """入射角 = 光线与法线夹角（锐角，度）。"""
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(d_in, n)), 0, 1))))


def build_path(pts_yz: np.ndarray, telecentric_image: bool = False) -> dict:
    """
    由 6 个反射点 (y, z) 构建完整路径。

    物方主光线入射角固定：P1 强制在 15° 主光线上（P1_y = 36.8 - z1·tan15°），
    第一段方向恒为 15°（约束不可变）。

    telecentric_image=True：像方远心——M6 反射后主光线平行光轴（+z），
    像点在 M6 反射点正上方（z=3500），像高 = M6 反射点 y。

    返回:
        dict(points(7,3), dirs(7,3), normals(6,3), angles(6,),
             metrics{...})
    """
    pts6 = np.column_stack([np.zeros(6), pts_yz[:, 0], pts_yz[:, 1]])
    # 主光线入射角固定 15°：P1 在从物点沿 15° 方向的射线上
    pts6[0, 1] = OBJ_POINT[1] - pts6[0, 2] * np.tan(np.deg2rad(CHIEF_ANGLE))
    if telecentric_image:
        # 像方远心：M6 反射后方向 = +z，像点 = P6 + (0,0,3500-P6z)
        img_y = pts6[5, 1]  # 像高 = M6 反射点 y
        img_pt = np.array([0.0, img_y, 3500.0])
        pts = np.vstack([OBJ_POINT, pts6, img_pt])
    else:
        pts = np.vstack([OBJ_POINT, pts6, IMG_POINT])
    # 各段方向（归一化）
    dirs = np.zeros((7, 3))
    for i in range(7):
        seg = pts[i + 1] - pts[i]
        dirs[i] = seg / np.linalg.norm(seg)
    # 各反射面法向量 + 入射角
    normals = np.zeros((6, 3))
    angles = np.zeros(6)
    for i in range(6):
        if telecentric_image and i == 5:
            # M6：出射方向强制 +z（像方远心）
            normals[i] = reflect_normal(dirs[i], np.array([0.0, 0.0, 1.0]))
        else:
            normals[i] = reflect_normal(dirs[i], dirs[i + 1])
        angles[i] = incident_angle(dirs[i], normals[i])
    z = pts[:, 2]
    gaps = np.diff(z)
    metrics = dict(
        z_all_positive=float(np.min(z[1:7])),      # 镜子 z 全正
        max_incident=float(np.max(angles)),
        min_gap=float(np.min(np.abs(pts[1:7, 2] - pts[0:6, 2]))),
        img_err=float(np.linalg.norm(pts[-1] - IMG_POINT)),
    )
    return dict(points=pts, dirs=dirs, normals=normals, angles=angles, metrics=metrics)


def path_merit(pts_yz: np.ndarray,
               w_ang: float = 1.0, w_pos: float = 1.0, w_gap: float = 0.5) -> float:
    """路径评价：入射角超限 + z 全正 + 间距不足。"""
    path = build_path(pts_yz)
    m = path["metrics"]
    merit = 0.0
    # 入射角超限
    for a in path["angles"]:
        if a > MAX_INCIDENT:
            merit += w_ang * (a - MAX_INCIDENT) ** 2
    # z 全正
    if m["z_all_positive"] < 0:
        merit += w_pos * (-m["z_all_positive"]) ** 2
    # 间距
    if m["min_gap"] < MIN_GAP:
        merit += w_gap * (MIN_GAP - m["min_gap"]) ** 2
    return merit


def optimize_path(z_range: tuple[float, float] = (300.0, 3200.0),
                  init_amp: float = 400.0,
                  max_iter: int = 300,
                  verbose: bool = True) -> tuple[np.ndarray, dict]:
    """
    优化 6 个反射点 (y, z)。

    初始：z 均匀分布在 [z1, z6]，y 交替 ±amp（之字形）。
    用数值梯度 + 阻尼下降（简单 DLS）。
    """
    z1, z6 = z_range
    zs = np.linspace(z1, z6, 6)
    ys = np.array([init_amp, -init_amp * 0.8, init_amp * 0.6,
                   -init_amp * 0.4, init_amp * 0.2, -init_amp * 0.1])
    x = np.column_stack([ys, zs]).flatten()

    def merit(xx):
        return path_merit(xx.reshape(6, 2))

    m0 = merit(x)
    for it in range(max_iter):
        g = np.zeros(12)
        for j in range(12):
            step = 0.5 if j % 2 == 1 else 5.0  # z 步长 0.5mm, y 步长 5mm
            xp = x.copy(); xp[j] += step
            xm = x.copy(); xm[j] -= step
            g[j] = (merit(xp) - merit(xm)) / (2 * step)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        # 线搜索（保持 z 递增）
        done = False
        for mag in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
            xn = x - g / gn * mag
            # z 强制递增
            xn[1::2] = np.sort(xn[1::2])
            mn = merit(xn)
            if mn < m0:
                x, m0 = xn, mn
                done = True
                break
        if not done:
            break
        if verbose and it % 25 == 0:
            print(f"  iter{it}: merit={m0:.4f} max_inc={build_path(x.reshape(6,2))['metrics']['max_incident']:.1f}°")
    pts_yz = x.reshape(6, 2)
    return pts_yz, build_path(pts_yz)


def total_path_angle(path: dict) -> float:
    """6 个相邻光线段方向夹角的和（度）。夹角越大 → 布局越分散。"""
    dirs = path["dirs"]
    total = 0.0
    for i in range(6):
        cos_a = np.clip(np.dot(dirs[i], dirs[i + 1]), -1, 1)
        total += np.degrees(np.arccos(cos_a))
    return total


def min_path_angle(path: dict) -> float:
    """最小相邻光线段夹角（度）。布局分散 = 最小夹角尽可能大（不出现光线贴近平行）。"""
    dirs = path["dirs"]
    mins = 180.0
    for i in range(6):
        cos_a = np.clip(np.dot(dirs[i], dirs[i + 1]), -1, 1)
        mins = min(mins, np.degrees(np.arccos(cos_a)))
    return mins


def optimize_dispersed(z_range: tuple[float, float] = (300.0, 3200.0),
                       n_trials: int = 50,
                       w_angle: float = 3.0,
                       seed: int = 7,
                       verbose: bool = True) -> tuple[np.ndarray, dict]:
    """
    分散布局优化：多初始随机搜索，目标 = 约束满足 + 相邻光线夹角最大化。

    约束：主光线 15° 固定、像方远心（M6 后 +z）、反射面入射角 ≤20°、
    镜子 z 全正、像点误差小。

    目标：奖励相邻光线段总夹角（布局越分散越好）。

    返回 (最优反射点 (y,z), 路径 dict)
    """
    rng = np.random.default_rng(seed)

    def merit(xx):
        p = build_path(xx.reshape(6, 2), telecentric_image=True)
        m = 0.0
        for a in p["angles"]:
            if a > MAX_INCIDENT:
                m += 500.0 * (a - MAX_INCIDENT) ** 2
        m -= w_angle * min_path_angle(p)   # 每面折返角都大（奖励最小折返角）
        if p["metrics"]["z_all_positive"] < 0:
            m += 500.0 * (-p["metrics"]["z_all_positive"]) ** 2
        if p["metrics"]["min_gap"] < MIN_GAP:
            m += 20.0 * (MIN_GAP - p["metrics"]["min_gap"]) ** 2
        m += p["metrics"]["img_err"] * 300.0
        return m, p

    def local_opt(x0, max_iter=300):
        x = x0.copy()
        m0, _ = merit(x)
        for it in range(max_iter):
            g = np.zeros(12)
            for j in range(12):
                step = 0.3 if j % 2 == 1 else 5.0
                xp = x.copy(); xp[j] += step
                xm = x.copy(); xm[j] -= step
                g[j] = (merit(xp)[0] - merit(xm)[0]) / (2 * step)
            gn = np.linalg.norm(g)
            if gn < 1e-10:
                break
            done = False
            for mag in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002):
                xn = x - g / gn * mag
                mn, _ = merit(xn)
                if mn < m0:
                    x, m0 = xn, mn
                    done = True
                    break
            if not done:
                break
        return x, merit(x)

    z1, z6 = z_range
    best = None
    for trial in range(n_trials):
        zs = np.array([rng.uniform(1500, 3200), rng.uniform(300, 1500),
                       rng.uniform(1800, 3300), rng.uniform(600, 1800),
                       rng.uniform(2400, 3400), rng.uniform(900, 2600)])
        amp = rng.uniform(200, 700)
        ys = np.array([amp, -amp * 0.8, amp * 0.6, -amp * 0.4, amp * 0.2, -amp * 0.1])
        x0 = np.column_stack([ys, zs]).flatten()
        x, (m, p) = local_opt(x0)
        ta = total_path_angle(p)
        chief_ok = abs(np.degrees(np.arccos(p["dirs"][0, 2])) - CHIEF_ANGLE) < 0.5
        valid = (chief_ok and p["metrics"]["z_all_positive"] > 0 and
                 p["metrics"]["max_incident"] <= MAX_INCIDENT)
        if valid and (best is None or ta > best[0]):
            best = (ta, x, p)
            if verbose:
                print(f"  trial{trial}: 总夹角={ta:.0f}° 最小夹角={min_path_angle(p):.0f}° "
                      f"max_inc={p['metrics']['max_incident']:.1f}°")
    if best is None:
        raise RuntimeError("未找到满足约束的分散布局")
    return best[1].reshape(6, 2), best[2]


def ga_dls_dispersed(n_gen: int = 60,
                     pop_size: int = 60,
                     seed: int = 7,
                     w_angle: float = 3.0,
                     verbose: bool = True) -> tuple[np.ndarray, dict]:
    """
    分散布局优化（GA 全局搜索 + DLS 局部精修）。

    - GA：12 变量（6 反射点 y,z），锦标赛选择/算术交叉/高斯变异，
      适应度 = 约束满足 + 相邻光线夹角最大化
    - DLS：从 GA 最优出发，数值雅可比 + Marquardt 阻尼精修

    返回 (最优反射点 (y,z), 路径 dict)
    """
    rng = np.random.default_rng(seed)
    # 变量边界：y ±800，z 300~3500
    bounds = []
    for i in range(6):
        bounds.append((-800.0, 800.0))   # y
        bounds.append((300.0, 3500.0))   # z
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def merit(xx):
        p = build_path(xx.reshape(6, 2), telecentric_image=True)
        m = 0.0
        for a in p["angles"]:
            if a > MAX_INCIDENT:
                m += 500.0 * (a - MAX_INCIDENT) ** 2
        # 每面入射/出射夹角（折返角）都足够大：奖励最小折返角
        m -= w_angle * min_path_angle(p)
        # 布局开放：每个反射点离非本镜子其余光线的垂直距离远
        cl = clearance_metrics(p)["min_clearance"]
        if cl < MIN_CLEAR:
            m += 20.0 * (MIN_CLEAR - cl) ** 2
        m -= 0.5 * min(cl, 300.0)
        # 反射点之间互相距离越远越好
        pd = points_dist_metrics(p)["min_dist"]
        m -= 0.8 * min(pd, 1000.0)
        # M6 反射点离像面近（工作距离小，EUV 晶圆侧）
        wd = 3500.0 - p["points"][6, 2]
        m += 0.3 * wd
        if p["metrics"]["z_all_positive"] < 0:
            m += 500.0 * (-p["metrics"]["z_all_positive"]) ** 2
        if p["metrics"]["min_gap"] < MIN_GAP:
            m += 20.0 * (MIN_GAP - p["metrics"]["min_gap"]) ** 2
        m += p["metrics"]["img_err"] * 300.0
        return m, p

    # ---- GA 全局搜索 ----
    def init_pop():
        pop = np.zeros((pop_size, 12))
        pop[0] = np.array([500.0, 3000.0, -420.0, 800.0, 340.0, 2600.0,
                           -260.0, 1300.0, 180.0, 3100.0, -84.0, 2000.0])
        for i in range(1, pop_size):
            pop[i] = lo + rng.random(12) * (hi - lo)
        return pop

    pop = init_pop()
    fit = np.array([merit(x)[0] for x in pop])
    best_idx = np.argmin(fit)
    best_x = pop[best_idx].copy()
    best_fit = fit[best_idx]
    for gen in range(n_gen):
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros(pop_size)
        new_pop[0] = best_x
        new_fit[0] = best_fit
        for i in range(1, pop_size):
            p1 = tournament_select(pop, fit, 3)
            p2 = tournament_select(pop, fit, 3)
            if rng.random() < 0.8:
                c1, c2 = arithmetic_crossover(p1, p2)
                child = c1 if rng.random() < 0.5 else c2
            else:
                child = p1.copy()
            child = gaussian_mutate(child, bounds, mutation_rate=0.3, mutation_scale=0.1)
            child = np.clip(child, lo, hi)
            new_pop[i] = child
            new_fit[i] = merit(child)[0]
        pop, fit = new_pop, new_fit
        gbi = np.argmin(fit)
        if fit[gbi] < best_fit:
            best_x, best_fit = pop[gbi].copy(), fit[gbi]
        if verbose and gen % 10 == 0:
            p = build_path(best_x.reshape(6, 2), telecentric_image=True)
            print(f"  GA gen{gen}: merit={best_fit:.1f} 总夹角={total_path_angle(p):.0f}° "
                  f"max_inc={p['metrics']['max_incident']:.1f}°")

    # ---- DLS 局部精修（数值雅可比 + Marquardt 阻尼）----
    x = best_x.copy()
    m0, _ = merit(x)
    lam = 0.15
    for it in range(100):
        g = np.zeros(12)
        for j in range(12):
            step = 0.3 if j % 2 == 1 else 5.0
            xp = x.copy(); xp[j] += step
            xm = x.copy(); xm[j] -= step
            g[j] = (merit(xp)[0] - merit(xm)[0]) / (2 * step)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        # Marquardt 阻尼：逐步缩小步长
        done = False
        for mag in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002):
            xn = np.clip(x - g / gn * mag, lo, hi)
            mn, _ = merit(xn)
            if mn < m0:
                x, m0 = xn, mn
                done = True
                break
        if not done:
            break
    _, path = merit(x)
    if verbose:
        print(f"  DLS 精修后: merit={m0:.1f} 总夹角={total_path_angle(path):.0f}°")
    return x.reshape(6, 2), path


def point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点 p 到线段 AB 的垂直距离（若垂足在线段外则取到端点距离）。"""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = float(np.clip(t, 0.0, 1.0))
    foot = a + t * ab
    return float(np.linalg.norm(p - foot))


def clearance_metrics(path: dict) -> dict:
    """
    每个反射点离非本镜子其余光线的垂直距离。

    对每个反射点 P_i，计算它到所有"非本镜子光线段"（S_j, j≠i-1 且 j≠i，
    即不包含 P_i 作为端点的段）的垂直距离，取最小为该反射点的安全距离。

    布局开放 = 每个反射点都离其他光线路径远（真实镜面不会挡别的光线）。

    返回 dict(min_clearance, per_mirror_clearance)
    """
    pts = path["points"]  # (8,3): P0=物, P1..P6 反射点, P7=像
    segs = [(pts[j], pts[j + 1]) for j in range(7)]  # S0..S6
    clears = []
    for i in range(1, 7):  # 反射点 P_i
        p = pts[i]
        dmin = 1e9
        for j, (a, b) in enumerate(segs):
            # 跳过本镜子光线段（P_i 是 S_{i-1} 和 S_i 的端点）
            if j == i - 1 or j == i:
                continue
            dmin = min(dmin, point_seg_dist(p, a, b))
        clears.append(dmin)
    return dict(min_clearance=float(min(clears)),
                per_mirror_clearance=[float(c) for c in clears])


def points_dist_metrics(path: dict) -> dict:
    """
    6 个反射点之间的两两距离（镜子之间互相远离）。

    返回 dict(min_dist, mean_dist, pairs)
    min_dist = 最小两两点距（布局分散 = 最小点距尽可能大）
    """
    pts = path["points"]  # P1..P6 反射点
    dists = []
    for i in range(1, 7):
        for j in range(i + 1, 7):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            dists.append(d)
    return dict(min_dist=float(min(dists)),
                mean_dist=float(np.mean(dists)),
                pairs=[float(d) for d in dists])


def print_path(path: dict) -> None:
    """打印反射点、法向量（平面角度）、入射角。"""
    pts = path["points"]
    normals = path["normals"]
    angles = path["angles"]
    m = path["metrics"]
    print("=" * 72)
    print("  理想反射面主光线路径（中心视场点）")
    print("=" * 72)
    print(f"  物点: ({pts[0][1]:.1f}, {pts[0][2]:.1f})  主光线 {CHIEF_ANGLE}° 向下")
    print(f"  像点: ({pts[-1][1]:.1f}, {pts[-1][2]:.1f})  (像高 {pts[-1][1]:.2f}, 放大率 {pts[-1][1]/OBJ_POINT[1]:.3f})")
    print()
    print(f"  {'面':>3s} {'反射点y':>8s} {'z':>8s} {'法向量(ny,nz)':>16s} {'平面角':>7s} {'入射角':>7s}")
    for i in range(6):
        n = normals[i]
        ang_deg = np.degrees(np.arctan2(-n[1], n[2]))  # 平面法向量与 +z 夹角
        print(f"  M{i+1:>3d} {pts[i+1][1]:8.1f} {pts[i+1][2]:8.1f} "
              f"({n[1]:+.3f},{n[2]:+.3f}) {ang_deg:7.1f}° {angles[i]:7.1f}°")
    print()
    print(f"  最大入射角: {m['max_incident']:.1f}° (限 {MAX_INCIDENT}°)")
    print(f"  镜子 z 全正: {m['z_all_positive']:.0f} (>0)")
    print(f"  像点误差: {m['img_err']:.2f} mm")


def plot_path(path: dict, save_path: str = "output/initial_path/reflect_initial.png") -> None:
    """绘制理想路径：反射点、折线、法向量（平面）。文字不遮挡结构。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = path["points"]
    normals = path["normals"]
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    # 折线
    ax.plot(pts[:, 2], pts[:, 1], 'k-o', lw=2, ms=7, zorder=5, label='chief ray path')
    # 反射面（过反射点、沿法向量的短线 = 平面角度示意）
    for i in range(6):
        p = pts[i + 1]
        n = normals[i]
        tvec = np.array([-n[2], n[1]])  # 在 y-z 平面内的切线
        half = 150.0
        ax.plot([p[2] - tvec[1]*half, p[2] + tvec[1]*half],
                [p[1] - tvec[0]*half, p[1] + tvec[0]*half],
                '-', color=plt.cm.tab10(i), lw=3.0, zorder=4,
                label='M%d (angle %.0f deg)' % (i+1, np.degrees(np.arctan2(-n[1], n[2]))))
    ax.plot(pts[0][2], pts[0][1], 'go', ms=12, zorder=6, label='object')
    ax.plot(pts[-1][2], pts[-1][1], 'o', color='orange', ms=12, zorder=6, label='image')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Ideal reflection-plane chief-ray path (center FOV, unified 15 deg)')
    ax.set_xlabel('z (mm)'); ax.set_ylabel('y (mm)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    # 图例移到图外，不挡结构
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, ncol=1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  图: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="理想反射面主光线路径设计")
    parser.add_argument("--z1", type=float, default=300.0, help="第一个反射点 z")
    parser.add_argument("--z6", type=float, default=3200.0, help="最后一个反射点 z")
    parser.add_argument("--out", default="output/initial_path/reflect_initial.png")
    parser.add_argument("--dispersed", action="store_true",
                        help="分散布局优化（相邻光线夹角越大越好，多初始局部优化）")
    parser.add_argument("--ga-dls", action="store_true",
                        help="分散布局优化（GA 全局搜索 + DLS 局部精修）")
    parser.add_argument("--trials", type=int, default=50, help="分散优化的随机初始数")
    parser.add_argument("--pop-size", type=int, default=60, help="GA 种群")
    parser.add_argument("--gen", type=int, default=60, help="GA 代数")
    args = parser.parse_args()
    os.makedirs("output", exist_ok=True)
    if args.ga_dls:
        pts_yz, path = ga_dls_dispersed(n_gen=args.gen, pop_size=args.pop_size)
    elif args.dispersed:
        pts_yz, path = optimize_dispersed((args.z1, args.z6), n_trials=args.trials, verbose=True)
    else:
        pts_yz, path = optimize_path((args.z1, args.z6), verbose=True)
    print_path(path)
    plot_path(path, args.out)


if __name__ == "__main__":
    main()
