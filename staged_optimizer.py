#!/usr/bin/env python3
"""
staged_optimizer.py — 分阶段离轴反射系统优化器
==================================================
按用户策略实现的分阶段优化流程：

  阶段 0（结构搜索）：6 镜 d/r 共 12 参数 + 子午/主光线追迹，
    GA 全局搜索找到"光线不遮挡、全部到达像面"的个体。
    评价函数只用结构项（遮挡/缺失/NA/入射角/出射角/工作距离），不用光斑
    —— 未聚焦结构上的光斑是悬崖型目标，会掩盖光滑的结构下降方向。

  阶段 1..N（逐级解锁高阶项）：d,r → +k → +A4 → +A6 → +A8 → +A10。
    每一级解锁新的自由度后做 DLS（Marquardt 阻尼 + 鞍点逃逸），
    merit 同伦：先结构项（把结构拉回可行域）→ 光斑小权重 → 光斑全权重。
    高阶项边界按半口径物理缩放（r_max^order 处 sag 贡献 ≤ 0.5mm），
    避免系数微小变化导致镜面爆炸（高维维度灾难的根源）。

  卡住恢复（总有下降方向）：
    1. DLS 法方程无下降步 → 内置鞍点逃逸（梯度 / 残差维度 / 随机扰动）
    2. 仍卡住 → 更小初始阻尼重试 + 随机扰动重启
    3. 仍卡住 → 邻域 GA 重启（围绕当前点小邻域搜索），注入 GA 找到的规则

  收尾（弥散斑优化后去除）：
    全 3D 光线 + 全变量 + 全 merit 最终 DLS；
    再把光斑权重归零跑一次 DLS，验证约束（NA/角度/遮挡/工作距离）保持、
    光斑不反弹 —— 证明结构约束本身已锁住像质。

用法：
  python staged_optimizer.py --input examples/offaxis_6mirror_decenter_demo.json \
      --output output/offaxis_6mirror_staged.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import numpy as np

from lens_schema import LensSystem, SurfaceType, Bounds, AsphericVariable
from merit_function import merit_scalar, merit_detail
from raytrace import trace_system
from dls_optimizer import dls_optimize, project_to_bounds
from ga_sa_optimizer import ga_sa_optimize


# ===========================================================================
# 变量集合控制
# ===========================================================================

PARAM_GROUPS = ["curvature", "thickness", "decenter_y", "k", "A4", "A6", "A8", "A10"]


def set_active_vars(lens: LensSystem, names: list[str]) -> None:
    """
    控制优化变量集合（原位修改）。

    names 中的参数名：curvature / thickness / decenter_y / k / A4 / A6 / A8 / A10。
    注意：schema 中 surface.variable 同时控制 curvature+thickness，
    因此两者要么同时在列，要么同时不在。
    """
    want_curv_thick = ("curvature" in names) or ("thickness" in names)
    for s in lens.surfaces:
        if s.type == SurfaceType.OBJECT:
            # 物面位置固定（z=0），物距不作为变量
            s.variable = False
            continue
        if s.type == SurfaceType.IMAGE:
            continue
        s.variable = want_curv_thick
        s.decenter_variable = "decenter_y" in names
        s.size_variable = "semi_aperture" in names
        if s.type == SurfaceType.ASPHERIC_REFLECT:
            flags = {p: (p in names) for p in ("k", "A4", "A6", "A8", "A10", "A12", "A14")}
            if any(flags.values()):
                s.aspheric_variable = AsphericVariable(**flags)
            else:
                s.aspheric_variable = None
        else:
            s.aspheric_variable = None


def ensure_aspheric_bounds(lens: LensSystem, sag_budget: float = 0.5) -> None:
    """
    为每个非球面反射面补充物理合理的边界（原位修改）。

    高阶项边界按半口径物理缩放：每阶在 r_max 处的 sag 贡献 ≤ sag_budget mm。
    这是维度灾难的关键修复：若 A6 用 ±1e-6 边界，在 r=50mm 处 sag 变化
    可达 9.4m，光线全部追不到像面，雅可比爆炸。
    """
    for s in lens.surfaces:
        if s.type != SurfaceType.ASPHERIC_REFLECT:
            continue
        r_max = s.semi_aperture
        if r_max <= 0:
            continue
        b = s.bounds if s.bounds is not None else Bounds()
        defaults = {
            "k": (-2.5, 2.5),
            "A4": (-sag_budget / r_max ** 4, sag_budget / r_max ** 4),
            "A6": (-sag_budget / r_max ** 6, sag_budget / r_max ** 6),
            "A8": (-sag_budget / r_max ** 8, sag_budget / r_max ** 8),
            "A10": (-sag_budget / r_max ** 10, sag_budget / r_max ** 10),
        }
        for p, (lo, hi) in defaults.items():
            if getattr(b, p, None) is None:
                setattr(b, p, (lo, hi))
        s.bounds = b


# ===========================================================================
# 权重调度（merit 同伦）
# ===========================================================================

# 结构项权重档位：
#   balanced — GA 全局搜索档：聚焦"不遮挡"（obs 100 + 穿透深度梯度主导），
#              NA/角度用低权重引导（结构搜索阶段先找通路，约束精修留给 DLS），
#              外加小 RMS 项（rms_spot=0.01）偏向光斑更汇聚的盆地
#   na_first — NA 优先（角度权重归零，先把 NA 推到目标，避免被角度惩罚"挤出"）
#   full     — 全结构约束（NA + 入射/出射角 + 工作距离）
#   hard     — 约束硬化档（verify/收尾用）：NA/角度/工作距离权重极高，
#              确保所有约束精确满足（NA=0.3300、入射≤20°、出射≤22°、WD≥25）
# 关键：missing 权重 20000（比例²）——丢失 1 条光线惩罚 62，丢失 2 条 247，
#       远大于角度/NA 的收益，优化器永远无法靠"丢弃光线"作弊。
# EUV GA 档位（满足 EUV 要求的结构搜索约束）：
#   - obs=200/miss=40000：有效结构不遮挡、光线必须到像面（核心）
#   - mg=50：镜子间轴向间距 ≥20mm（物理可制造）
#   - na=10：像方 NA 引导到 0.33
#   - angle=0.3：入射/出射角软引导（EUV 多层膜镀膜角度限制）
#   - tele=5：像方远心引导（晶圆侧主光线平行光轴）
#   - rms=0.02：在结构可行个体中偏向光斑更汇聚的盆地
# GA 结构档位：聚焦"不遮挡 + 镜间距 + 光线到像面"（核心），
#   NA/角度/远心用极小权重（全遮挡时这些惩罚恒定会稀释 GA 选择压力），
#   像质约束留给 DLS 精修阶段。
STRUCTURE_BALANCED = dict(rms_spot=0.02, na=0.5, angle_inc=0.1, angle_exit=0.1,
                          wd=0.5, obs=300.0, miss=40000.0, tele=0.0, pkg=0.0, mg=50.0,
                          iz=1.0, mag=8.0, op=200.0, mz=200.0)
STRUCTURE_NA_FIRST = dict(rms_spot=0.0, na=50.0, angle_inc=0.0, angle_exit=0.0,
                          wd=2.0, obs=40.0, miss=20000.0, tele=0.0, pkg=10.0, mg=30.0,
                          iz=20.0, mag=50.0, op=200.0, mz=200.0)
STRUCTURE_FULL = dict(rms_spot=0.0, na=50.0, angle_inc=0.5, angle_exit=0.5,
                      wd=2.0, obs=40.0, miss=20000.0, tele=10.0, pkg=10.0, mg=30.0,
                      iz=20.0, mag=50.0, op=200.0, mz=200.0)
STRUCTURE_HARD = dict(rms_spot=0.0, na=500.0, angle_inc=20.0, angle_exit=20.0,
                      wd=50.0, obs=400.0, miss=40000.0, tele=100.0, pkg=20.0, mg=60.0,
                      iz=100.0, mag=200.0, op=200.0, mz=200.0)


def apply_weight_profile(lens: LensSystem, profile: dict) -> None:
    """应用权重档位（原位修改）。profile 键：rms_spot/na/angle_inc/angle_exit/wd/obs/miss。
    distortion/chroma 始终关闭；间距惩罚（min_thickness 等）保持配置值。"""
    w = lens.optimization_config.merit_weights
    w.rms_spot = profile["rms_spot"]
    w.distortion = 0.0
    w.chroma = 0.0
    w.na_penalty = profile["na"]
    w.angle_incident_penalty = profile["angle_inc"]
    w.angle_exit_penalty = profile["angle_exit"]
    w.working_distance_penalty = profile["wd"]
    w.obscuration_penalty = profile["obs"]
    w.missing_penalty = profile["miss"]
    w.telecentricity_penalty = profile.get("tele", 0.0)
    w.package_penalty = profile.get("pkg", 0.0)
    w.mirror_gap_penalty = profile.get("mg", 0.0)
    w.image_z_penalty = profile.get("iz", 0.0)
    w.magnification_penalty = profile.get("mag", 0.0)
    w.object_path_penalty = profile.get("op", 0.0)
    w.mirror_z_penalty = profile.get("mz", 0.0)


# ===========================================================================
# 卡住恢复工具
# ===========================================================================

def perturb_lens(lens: LensSystem,
                 sigma: float = 0.02,
                 rng: np.random.Generator | None = None) -> LensSystem:
    """围绕当前点做高斯扰动（归一化空间），返回新镜头。"""
    x = lens.variables_to_array()
    bounds = lens.get_variable_bounds()
    rng = rng or np.random.default_rng()
    xp = x.copy()
    for j, (lo, hi) in enumerate(bounds):
        xp[j] += rng.normal(0.0, sigma * (hi - lo))
    xp = project_to_bounds(xp, bounds)
    new = copy.deepcopy(lens)
    new.array_to_variables(xp)
    return new


def dls_with_restart(lens: LensSystem,
                     max_iter: int = 60,
                     damp_init: float = 0.15,
                     n_perturb_restarts: int = 3,
                     n_ga_restarts: int = 1,
                     ga_pop: int = 20,
                     ga_iter: int = 15,
                     restart_sigma: float = 0.02,
                     verbose: bool = True) -> tuple[LensSystem, list[float]]:
    """
    DLS + 卡住恢复：
      1. DLS（含内置鞍点逃逸）
      2. 无改善 → 更小初始阻尼重试
      3. 无改善 → 随机扰动重启 DLS（n_perturb_restarts 次）
      4. 无改善 → 邻域 GA 重启（围绕当前点，n_ga_restarts 次）→ DLS

    返回 (最优镜头, 最优 DLS 历史)。
    """
    best = copy.deepcopy(lens)
    best_merit = merit_scalar(best)
    best_hist = [best_merit]

    if verbose:
        print(f"    [恢复] DLS 起点 merit={best_merit:.6e}")

    # 1) 初始 DLS
    cand, hist = dls_optimize(best, max_iter=max_iter, damp_init=damp_init,
                              verbose=verbose)
    if merit_scalar(cand) < best_merit * (1 - 1e-10):
        best, best_merit, best_hist = cand, merit_scalar(cand), hist
        if verbose:
            print(f"    [恢复] DLS 改善 → merit={best_merit:.6e}")

    # 2) 更小初始阻尼
    if len(best_hist) <= 1:
        cand, hist = dls_optimize(best, max_iter=max_iter, damp_init=0.01,
                                  verbose=verbose)
        if merit_scalar(cand) < best_merit * (1 - 1e-10):
            best, best_merit, best_hist = cand, merit_scalar(cand), hist
            if verbose:
                print(f"    [恢复] 小阻尼 DLS 改善 → merit={best_merit:.6e}")

    # 3) 随机扰动重启
    for k in range(n_perturb_restarts):
        if len(best_hist) > 1:
            break
        cand0 = perturb_lens(best, sigma=restart_sigma * (k + 1))
        cand, hist = dls_optimize(cand0, max_iter=max_iter, damp_init=damp_init,
                                  verbose=verbose)
        if merit_scalar(cand) < best_merit * (1 - 1e-10):
            best, best_merit, best_hist = cand, merit_scalar(cand), hist
            if verbose:
                print(f"    [恢复] 扰动重启 #{k+1} 改善 → merit={best_merit:.6e}")

    # 4) 邻域 GA 重启（注入 GA 找到的规则）
    for k in range(n_ga_restarts):
        if len(best_hist) > 1:
            break
        if verbose:
            print(f"    [恢复] 邻域 GA 重启 #{k+1}...")
        cand0, _ = ga_sa_optimize(best, pop_size=ga_pop, ga_iter=ga_iter,
                                  sa_temp_init=20.0, restart_sigma=restart_sigma * 2,
                                  verbose=verbose)
        cand, hist = dls_optimize(cand0, max_iter=max_iter, damp_init=damp_init,
                                  verbose=verbose)
        if merit_scalar(cand) < best_merit * (1 - 1e-10):
            best, best_merit, best_hist = cand, merit_scalar(cand), hist
            if verbose:
                print(f"    [恢复] 邻域 GA + DLS 改善 → merit={best_merit:.6e}")

    if verbose:
        print(f"    [恢复] 结束，best merit={best_merit:.6e}")
    return best, best_hist


# ===========================================================================
# 阶段报告
# ===========================================================================

def print_state(lens: LensSystem, title: str, ray_mode: str | None = None) -> None:
    """打印系统状态：有效/遮挡/失败光线、RMS、NA、入射角、merit。"""
    d = merit_detail(lens)
    res = d["trace_result"]
    n_total = len(res.active)
    n_ok = int(np.sum(res.active))
    n_obs = int(np.sum(res.obscured))
    n_miss = n_total - n_ok - n_obs
    print(f"  [{title}] 有效={n_ok}/{n_total} 遮挡={n_obs} 失败={n_miss} | "
          f"RMS={res.rms_spot_radius:.5f} mm | NA={res.na_actual:.4f} | "
          f"max_inc={res.max_incident_angle:.2f}° | WD={res.working_distance:.1f} | "
          f"merit={d['total']:.6e}")


# ===========================================================================
# 主分阶段流程
# ===========================================================================

def staged_optimize(lens: LensSystem,
                    pop_size: int = 24,
                    ga_iter: int = 40,
                    dls_iter: int = 60,
                    damp_init: float = 0.15,
                    spot_ramp: tuple[float, ...] = (0.1, 0.5, 1.0),
                    unlock_order: tuple[str, ...] = ("k", "A4", "A6", "A8", "A10"),
                    ray_mode_stage0: str = "meridional",
                    ray_mode_final: str = "offaxis",
                    use_decenter: bool = False,
                    n_perturb_restarts: int = 3,
                    seed: int = 0,
                    direct_all: bool = False,
                    save_start: str | None = None,
                    start_from: str | None = None,
                    verbose: bool = True) -> tuple[LensSystem, list[tuple[str, float]]]:
    """
    分阶段优化主流程。

    参数:
        lens:              初始镜头系统（将被原位修改）
        pop_size:          阶段 0 GA 种群规模
        ga_iter:           阶段 0 GA 迭代代数
        dls_iter:          每级 DLS 最大迭代
        damp_init:         DLS 初始阻尼
        spot_ramp:         光斑权重同伦序列
        unlock_order:      逐级解锁的高阶项顺序
        ray_mode_stage0:   阶段 0 光线模式（meridional=子午，chief=仅主光线）
        ray_mode_final:    后续阶段光线模式（offaxis=全3D环形）
        use_decenter:      是否把 decenter_y 纳入优化变量（默认否：12 参数=d,r）
        n_perturb_restarts: 卡住时随机扰动重启次数
        direct_all:         True=GA 后一次性解锁全部高阶项直接 DLS（用户策略：
                           把全局找到的好起点保留，只用 DLS 从该起点出发全变量优化）
        save_start:         保存 GA 找到的结构起点到该 JSON 路径
        start_from:         从该起点 JSON 载入（跳过 GA），只做 DLS 优化

    返回:
        (best_lens, milestones): 最优系统和各阶段 (名称, merit) 里程碑
    """
    t_start = time.time()
    np.random.seed(seed)

    # 物理合理的高阶项边界（维度灾难修复）
    ensure_aspheric_bounds(lens)

    cons = lens.metadata.optical_constraints
    milestones: list[tuple[str, float]] = []

    print("=" * 72)
    print(f"  分阶段优化: {lens.metadata.name}")
    print(f"  物方 NA={cons.na_objective or lens.metadata.aperture.value}, "
          f"像方 NA={cons.na_target}, 主光线角={cons.chief_ray_angle_deg}°")
    print(f"  阶段0光线模式: {ray_mode_stage0}, 后续模式: {ray_mode_final}")
    print(f"  模式: {'direct_all(GA→全变量DLS)' if direct_all else '逐级解锁'}"
          f"{' + 起点保存' if save_start else ''}{' + 从起点载入' if start_from else ''}")
    print("=" * 72)
    print_state(lens, "初始")

    # ------------------------------------------------------------------
    # 阶段 0：结构搜索（12 参数 d,r + 子午/主光线追迹 + GA）
    #         或从保存的起点载入（跳过 GA，只 DLS）
    # ------------------------------------------------------------------
    if start_from is not None:
        print(f"\n[阶段 0] 从起点载入: {start_from}（跳过 GA，只做 DLS）")
        lens = LensSystem.from_json(start_from)
        ensure_aspheric_bounds(lens)
        milestones.append(("S0_load_start", 0.0))
        print_state(lens, "起点")
    else:
        lens.metadata.ray_mode = ray_mode_stage0
        set_active_vars(lens, ["curvature", "thickness", "k", "A4", "A6"] + (["decenter_y"] if use_decenter else []))
        apply_weight_profile(lens, STRUCTURE_BALANCED)  # 结构项 + 小 RMS 引导
        n_vars0 = len(lens.variables_to_array())
        print(f"\n[阶段 0] GA 结构搜索（变量={n_vars0}: d+r），光线={ray_mode_stage0}，"
              f"merit=结构项+小RMS引导 ...")
        t0 = time.time()
        lens, hist0 = ga_sa_optimize(lens, pop_size=pop_size, ga_iter=ga_iter,
                                     sa_temp_init=60.0, verbose=verbose)
        milestones.append(("S0_GA_structure", hist0[-1]))
        print(f"  耗时 {time.time()-t0:.1f}s")
        print_state(lens, "结构搜索后")
        if save_start:
            lens.to_json(save_start)
            print(f"  [起点已保存] {save_start}")

    lens.metadata.ray_mode = ray_mode_final

    # ------------------------------------------------------------------
    # 优化变量与同伦序列：
    #   direct_all → 一次性解锁全部高阶项（d/r + k + A4..A10），只用 DLS
    #   逐级       → 按 unlock_order 逐级解锁，每级 DLS + merit 同伦
    # ------------------------------------------------------------------
    if direct_all:
        all_vars = ["curvature", "thickness"] + \
                   (["decenter_y"] if use_decenter else []) + \
                   ["k", "A4", "A6", "A8", "A10"]
        set_active_vars(lens, all_vars)
        n_vars = len(lens.variables_to_array())
        print(f"\n[全变量] 一次性解锁全部高阶项（变量={n_vars}），光线={ray_mode_final} ...")
        sub_steps = [("结构-NA优先", STRUCTURE_NA_FIRST),
                     ("结构-全约束", STRUCTURE_FULL)] + \
                    [(f"光斑 w={sw}", dict(STRUCTURE_FULL, rms_spot=sw))
                     for sw in spot_ramp]
        for sub_name, prof in sub_steps:
            apply_weight_profile(lens, prof)
            if verbose:
                print(f"  -- 子步骤: {sub_name}")
            lens, hist = dls_with_restart(
                lens, max_iter=dls_iter, damp_init=damp_init,
                n_perturb_restarts=n_perturb_restarts,
                verbose=verbose,
            )
            milestones.append((f"ALL_{sub_name}", hist[-1]))
            print_state(lens, f"ALL.{sub_name}")
    else:
        active = ["curvature", "thickness"] + (["decenter_y"] if use_decenter else [])
        for level, var_name in enumerate(unlock_order, start=1):
            active.append(var_name)
            set_active_vars(lens, active)
            n_vars = len(lens.variables_to_array())
            print(f"\n[阶段 {level}] 解锁 {var_name}（变量={n_vars}），光线={ray_mode_final} ...")

            # 每级子步骤：
            #   a) 结构项 DLS（NA 优先 → 全约束，新自由度先把结构拉回可行域）
            #   b) 光斑权重同伦（小 → 大）
            sub_steps = [("结构-NA优先", STRUCTURE_NA_FIRST),
                         ("结构-全约束", STRUCTURE_FULL)] + \
                        [(f"光斑 w={sw}", dict(STRUCTURE_FULL, rms_spot=sw))
                         for sw in spot_ramp]
            for sub_name, prof in sub_steps:
                apply_weight_profile(lens, prof)
                if verbose:
                    print(f"  -- 子步骤: {sub_name}")
                lens, hist = dls_with_restart(
                    lens, max_iter=dls_iter, damp_init=damp_init,
                    n_perturb_restarts=n_perturb_restarts,
                    verbose=verbose,
                )
                milestones.append((f"S{level}_{var_name}_{sub_name}", hist[-1]))
                print_state(lens, f"S{level}.{sub_name}")

        # 阶段 N+1：最终 DLS（全变量 + 全 merit，全 3D 光线）
        lens.metadata.ray_mode = ray_mode_final
        apply_weight_profile(lens, dict(STRUCTURE_FULL, rms_spot=1.0))
        print(f"\n[最终] 全变量 DLS（变量={len(lens.variables_to_array())}，全 merit）...")
        lens, hist = dls_optimize(lens, max_iter=dls_iter, damp_init=damp_init, verbose=verbose)
        milestones.append(("Final_full_merit", hist[-1]))
        print_state(lens, "最终 DLS 后")

    # ------------------------------------------------------------------
    # 收尾：约束硬化（光斑权重归零，NA/角度/工作距离权重极高）
    # 弥散斑优化后去除，验证全部约束精确满足、光斑不反弹
    # ------------------------------------------------------------------
    print(f"\n[收尾] 约束硬化 DLS（光斑权重归零，NA/角度/工作距离硬化）...")
    apply_weight_profile(lens, STRUCTURE_HARD)
    lens_check, hist_check = dls_optimize(lens, max_iter=dls_iter,
                                          damp_init=damp_init, verbose=verbose)
    milestones.append(("Verify_hard_constraints", hist_check[-1]))

    d = merit_detail(lens_check)
    res = d["trace_result"]
    spot_after = res.rms_spot_radius
    print_state(lens_check, "约束硬化后")
    print(f"\n  约束硬化验证: RMS={spot_after:.5f} mm（光斑项已移除，约束独自锁住像质）")

    # 报告
    print("\n" + "=" * 72)
    print("  分阶段优化完成")
    print("=" * 72)
    print(f"  总耗时: {time.time()-t_start:.1f}s")
    print(f"  有效光线: {int(np.sum(res.active))}/{len(res.active)}"
          f"  (遮挡 {int(np.sum(res.obscured))})")
    print(f"  RMS 光斑: {spot_after:.6e} mm")
    print(f"  实际 NA:  {res.na_actual:.4f} (目标 {cons.na_target})")
    print(f"  工作距离: {res.working_distance:.2f} mm (要求 ≥ {cons.working_distance_min})")
    print("  各反射面最大入射角:")
    for sid, ang in res.per_surface_max_incident().items():
        lim = cons.incident_angle_max_deg
        mark = "OK" if ang <= lim else "OVER"
        print(f"    M{sid}: {ang:7.3f}°  (limit {lim:.1f}) [{mark}]")

    return lens_check, milestones


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="分阶段离轴反射系统优化器（GA结构搜索 → 逐级/全变量DLS → 约束硬化验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 完整流程：GA 找结构起点（并保存）→ 全变量 DLS（所有高阶项一次加齐）→ 约束硬化
  python staged_optimizer.py --input examples/offaxis_6mirror_decenter_demo.json \
      --output output/offaxis_6mirror_staged.json --direct-all \
      --save-start output/offaxis_start.json

  # 分两步：先用多个 seed 跑 GA 保留最好的起点，再只用 DLS 从该起点出发
  python staged_optimizer.py --input ... --output start.json --save-start start.json
  python staged_optimizer.py --start output/offaxis_start.json \
      --output output/offaxis_6mirror_staged.json --direct-all

  # 阶段0 用仅主光线（更快筛选）
  python staged_optimizer.py --input ... --output ... --chief

  # 允许 decenter_y 作为变量（默认固定，用户策略 12 参数 = 6d + 6r）
  python staged_optimizer.py --input ... --output ... --decenter
        """,
    )
    parser.add_argument("--input", required=False, help="输入镜头 JSON（--start 时可不给）")
    parser.add_argument("--output", required=True, help="输出 JSON")
    parser.add_argument("--start", default=None,
                        help="从该起点 JSON 载入（跳过 GA，只做 DLS）")
    parser.add_argument("--save-start", default=None,
                        help="把 GA 找到的结构起点保存到该 JSON 路径")
    parser.add_argument("--direct-all", action="store_true",
                        help="GA 后一次性解锁全部高阶项直接 DLS（默认逐级解锁）")
    parser.add_argument("--pop-size", type=int, default=24, help="阶段0 GA 种群规模")
    parser.add_argument("--ga-iter", type=int, default=40, help="阶段0 GA 迭代代数")
    parser.add_argument("--dls-iter", type=int, default=60, help="每级 DLS 最大迭代")
    parser.add_argument("--chief", action="store_true",
                        help="阶段0 用仅主光线追迹（超快筛选），默认子午面光线")
    parser.add_argument("--decenter", action="store_true",
                        help="把 decenter_y 纳入优化变量（默认固定，12参数=d,r）")
    parser.add_argument("--no-perturb-restarts", action="store_true",
                        help="卡住时不做随机扰动重启")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    if args.start is None and (not args.input or not os.path.exists(args.input)):
        print("[ERROR] 需要 --input（或 --start 指定起点 JSON）", file=sys.stderr)
        sys.exit(1)

    if args.start is not None:
        if not os.path.exists(args.start):
            print(f"[ERROR] 起点文件不存在: {args.start}", file=sys.stderr)
            sys.exit(1)
        lens = LensSystem.from_json(args.start)
    else:
        lens = LensSystem.from_json(args.input)

    lens_final, milestones = staged_optimize(
        lens,
        pop_size=args.pop_size,
        ga_iter=args.ga_iter,
        dls_iter=args.dls_iter,
        ray_mode_stage0="chief" if args.chief else "meridional",
        use_decenter=args.decenter,
        n_perturb_restarts=0 if args.no_perturb_restarts else 3,
        seed=args.seed,
        direct_all=args.direct_all,
        save_start=args.save_start,
        start_from=args.start,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    lens_final.to_json(args.output)
    print(f"\n[INFO] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
