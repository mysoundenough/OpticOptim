#!/usr/bin/env python3
"""
start_dls_optimizer.py — 起点保存 + 从起点全变量 DLS 优化
=============================================================
独立的两步工作流（不依赖 staged_optimize 的逐级解锁流程）：

  第 1 步（find-start）：GA 结构搜索找好起点
    - 只优化 6 镜 d/r 共 12 参数（+可选 decenter_y）
    - 子午/主光线追迹（6~1 条光线/评估，快）
    - merit = 结构项（遮挡/缺失/NA/角度/工作距离）+ 小 RMS 引导
      （约束未满足区约束主导，约束满足区 RMS 主导，
       让 GA 在结构可行个体中偏向光斑更汇聚的盆地）
    - 把找到的最优结构保存为起点 JSON（保留，可多次复用）

  第 2 步（dls）：从保存的起点出发，只用 DLS 优化
    - 一次性解锁全部高阶项（k + A4 + A6 + A8 + A10），所有变量一起优化
    - 全 3D 环形光线（offaxis）
    - merit 同伦：结构-NA优先 → 结构-全约束 → 光斑 0.1→0.5→1.0
    - 收尾：约束硬化（NA/角度/工作距离权重极高），验证全部约束精确满足

用法：
  # 第 1 步：GA 找起点并保存（可用不同 --seed 跑多次，保留结构 merit 最好的）
  python start_dls_optimizer.py find-start \
      --input examples/offaxis_6mirror_decenter_demo.json \
      --save-start output/offaxis_start.json --seed 7

  # 第 2 步：从起点出发，只用 DLS，全高阶项一起优化
  python start_dls_optimizer.py dls \
      --start output/offaxis_start.json \
      --output output/offaxis_final.json --dls-iter 60
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import numpy as np

from lens_schema import LensSystem
from merit_function import merit_scalar, merit_detail
from ga_sa_optimizer import ga_sa_optimize
from dls_optimizer import dls_optimize

from staged_optimizer import (
    set_active_vars,
    ensure_aspheric_bounds,
    apply_weight_profile,
    dls_with_restart,
    print_state,
    STRUCTURE_BALANCED,
    STRUCTURE_NA_FIRST,
    STRUCTURE_FULL,
    STRUCTURE_HARD,
)

ALL_VARS = ["curvature", "thickness", "k", "A4", "A6", "A8", "A10"]


# ===========================================================================
# 第 1 步：GA 结构搜索找起点
# ===========================================================================

def ga_find_start(lens: LensSystem,
                  pop_size: int = 24,
                  ga_iter: int = 40,
                  save_start: str | None = None,
                  use_decenter: bool = False,
                  ray_mode: str = "meridional",
                  seed: int = 0,
                  verbose: bool = True) -> LensSystem:
    """
    GA 结构搜索：只优化 6 镜 d/r（12 参数）+ 子午光线追迹，
    在"不遮挡、全约束满足"的可行个体中偏向光斑更汇聚的盆地。

    参数:
        lens:          初始镜头系统（含 decenter 布局）
        pop_size:      GA 种群规模
        ga_iter:       GA 迭代代数
        save_start:    把找到的最优结构保存到该 JSON（保留起点）
        use_decenter:  是否把 decenter_y 也纳入变量
        ray_mode:      阶段光线模式（meridional=子午 6 条，chief=仅主光线 1 条）
        seed:          随机种子（不同 seed 找不同盆地，可跑多次取最优）

    返回:
        最优结构镜头（d/r 已更新，decenter/高阶项保持初始值）
    """
    t0 = time.time()
    np.random.seed(seed)

    ensure_aspheric_bounds(lens)
    lens.metadata.ray_mode = ray_mode
    # 方案A：GA 阶段非球面系数(k/A4/A6)也作为变量——镜面形状补偿长程像散/遮挡
    set_active_vars(lens, ["curvature", "thickness", "k", "A4", "A6"] + (["decenter_y"] if use_decenter else []))
    apply_weight_profile(lens, STRUCTURE_BALANCED)

    cons = lens.metadata.optical_constraints
    n_vars = len(lens.variables_to_array())
    print("=" * 72)
    print(f"  [find-start] GA 结构搜索: {lens.metadata.name}")
    print(f"  物方 NA={cons.na_objective or lens.metadata.aperture.value}, "
          f"像方 NA={cons.na_target}, 主光线角={cons.chief_ray_angle_deg}°")
    print(f"  变量={n_vars} (d+r{'+decenter' if use_decenter else ''}), "
          f"光线={ray_mode}, merit=结构项+小RMS引导")
    print("=" * 72)
    print_state(lens, "初始")

    lens, hist = ga_sa_optimize(lens, pop_size=pop_size, ga_iter=ga_iter,
                                sa_temp_init=60.0, verbose=verbose)
    print_state(lens, "结构搜索后")

    if save_start:
        lens.to_json(save_start)
        print(f"\n  [起点已保存] {save_start}（decenter/高阶项保持初始值，"
              f"可反复从该起点出发做 DLS）")

    print(f"  [find-start] 耗时 {time.time()-t0:.1f}s")
    return lens


# ===========================================================================
# 第 2 步：从起点出发，只用 DLS，全高阶项一起优化
# ===========================================================================

def dls_from_start(start_json: str,
                   dls_iter: int = 60,
                   damp_init: float = 0.15,
                   spot_ramp: tuple[float, ...] = (0.1, 0.5, 1.0),
                   use_decenter: bool = False,
                   ray_mode: str = "offaxis",
                   n_perturb_restarts: int = 3,
                   verbose: bool = True) -> LensSystem:
    """
    从保存的起点出发，只用 DLS：一次性解锁全部高阶项（k+A4+A6+A8+A10），
    所有变量一起优化（d/r + 高阶项，可选 decenter）。

    merit 同伦序列：
      1. 结构-NA优先    （角度权重归零，先把 NA 推到目标）
      2. 结构-全约束    （NA + 入射/出射角 + 工作距离）
      3. 光斑 0.1→0.5→1.0（逐步引入弥散斑，收敛到约束边界上的最小 RMS）
      4. 约束硬化收尾   （光斑权重归零，NA/角度/工作距离权重极高，
                         验证全部约束精确满足、光斑不反弹）

    参数:
        start_json:       第 1 步保存的起点 JSON
        dls_iter:         每个子步骤 DLS 最大迭代
        damp_init:        DLS 初始阻尼
        spot_ramp:        光斑权重同伦序列
        use_decenter:     是否把 decenter_y 也纳入变量
        ray_mode:         优化光线模式（offaxis=全3D环形）
        n_perturb_restarts: 卡住时随机扰动重启次数

    返回:
        最终镜头系统（全约束满足 + 最小 RMS）
    """
    if not os.path.exists(start_json):
        raise FileNotFoundError(f"起点文件不存在: {start_json}")

    lens = LensSystem.from_json(start_json)
    ensure_aspheric_bounds(lens)
    lens.metadata.ray_mode = ray_mode

    all_vars = ALL_VARS + (["decenter_y"] if use_decenter else [])
    set_active_vars(lens, all_vars)
    n_vars = len(lens.variables_to_array())

    milestones: list[tuple[str, float]] = []
    cons = lens.metadata.optical_constraints
    print("=" * 72)
    print(f"  [dls] 从起点出发: {start_json}")
    print(f"  一次性解锁全部高阶项，变量={n_vars}（d/r + k + A4..A10"
          f"{' + decenter' if use_decenter else ''}），光线={ray_mode}")
    print("=" * 72)
    print_state(lens, "起点")

    # ---- merit 同伦序列：结构 → 光斑 ----
    sub_steps = [("结构-NA优先", STRUCTURE_NA_FIRST),
                 ("结构-全约束", STRUCTURE_FULL)] + \
                [(f"光斑 w={sw}", dict(STRUCTURE_FULL, rms_spot=sw))
                 for sw in spot_ramp]
    for sub_name, prof in sub_steps:
        apply_weight_profile(lens, prof)
        if verbose:
            print(f"\n  -- 子步骤: {sub_name}")
        lens, hist = dls_with_restart(
            lens, max_iter=dls_iter, damp_init=damp_init,
            n_perturb_restarts=n_perturb_restarts, verbose=verbose,
        )
        milestones.append((sub_name, hist[-1]))
        print_state(lens, sub_name)

    # ---- 约束硬化收尾（弥散斑优化后去除，验证约束精确满足）----
    print(f"\n  -- 子步骤: 约束硬化（光斑权重归零）")
    apply_weight_profile(lens, STRUCTURE_HARD)
    lens, hist = dls_optimize(lens, max_iter=dls_iter, damp_init=damp_init,
                              verbose=verbose)
    milestones.append(("约束硬化", hist[-1]))
    print_state(lens, "约束硬化后")

    # ---- 报告 ----
    d = merit_detail(lens)
    res = d["trace_result"]
    print("\n" + "=" * 72)
    print("  [dls] 完成 — 全约束满足下的最小 RMS")
    print("=" * 72)
    print(f"  有效光线: {int(np.sum(res.active))}/{len(res.active)}"
          f"  (遮挡 {int(np.sum(res.obscured))})")
    print(f"  RMS 光斑: {res.rms_spot_radius:.6e} mm")
    print(f"  实际 NA:  {res.na_actual:.4f} (目标 {cons.na_target})")
    print(f"  工作距离: {res.working_distance:.2f} mm (要求 ≥ {cons.working_distance_min})")
    max_exit = (float(np.nanmax(res.exit_angles[res.active]))
                if len(res.exit_angles) > 0 and res.active.any() else 0.0)
    print(f"  最大出射角: {max_exit:.2f}° (要求 ≤ {cons.exit_angle_max_deg})")
    print("  各反射面最大入射角:")
    for sid, ang in res.per_surface_max_incident().items():
        lim = cons.incident_angle_max_deg
        mark = "OK" if ang <= lim else "OVER"
        print(f"    M{sid}: {ang:7.3f}°  (limit {lim:.1f}) [{mark}]")

    return lens


# ===========================================================================
# 第 3 步（可选）：约束贴边精修
# ===========================================================================

def polish_under_constraints(lens: LensSystem,
                             dls_iter: int = 60,
                             damp_init: float = 0.15,
                             spot_ramp: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0),
                             rounds: int = 3,
                             verbose: bool = True) -> LensSystem:
    """
    约束贴边精修：把 RMS 压到约束边界上的最小值。

    策略：用 FULL 约束（na=50/角度0.5/wd=2/远心10——约束贴边但不锁死结构）
    + 递增的光斑权重（0.2→5.0），让 RMS 在评价函数里逐渐主导、
    结构被推向"约束边界上的最小 RMS"；每轮结束用 HARD 纯约束 verify
    把 NA/角度/工作距离/远心精确拉回可行域。多轮交替直到 RMS 收敛。

    每轮轨迹记录 (rms, na, tele, hard_merit)，保留"约束满足(HARD merit 小)
    且 RMS 最小"的最优解（多目标 Pareto 可行端点）。

    参数:
        lens:     已满足约束的镜头（dls_from_start 的输出）
        dls_iter: 每子步骤 DLS 最大迭代
        spot_ramp: 光斑权重递增序列
        rounds:   贴边精修轮数

    返回:
        贴边精修后的镜头（约束满足 + RMS 最小）
    """
    import copy as _copy
    lens = _copy.deepcopy(lens)
    lens.metadata.ray_mode = "offaxis"
    best = lens
    best_rms = merit_detail(lens)["trace_result"].rms_spot_radius
    history: list[dict] = []

    def _hard_merit(l: LensSystem) -> float:
        """HARD 约束档下的 merit（0 = 全部约束精确满足）。"""
        saved = lens.optimization_config.merit_weights
        apply_weight_profile(l, STRUCTURE_HARD)
        m = merit_scalar(l)
        return m

    print("=" * 72)
    print(f"  [polish] 约束贴边精修：FULL约束+光斑递增 压 RMS"
          f"（{rounds} 轮 × {len(spot_ramp)} 档）")
    print("=" * 72)

    for r in range(1, rounds + 1):
        prev_rms = best_rms
        for sw in spot_ramp:
            apply_weight_profile(lens, dict(STRUCTURE_FULL, rms_spot=sw))
            if verbose:
                print(f"\n  -- 轮 {r} · FULL约束+光斑 w={sw}")
            lens, hist = dls_with_restart(
                lens, max_iter=dls_iter, damp_init=damp_init,
                n_perturb_restarts=2, verbose=verbose,
            )
            d = merit_detail(lens)
            res = d["trace_result"]
            hm = _hard_merit(lens)
            apply_weight_profile(lens, dict(STRUCTURE_FULL, rms_spot=sw))
            rec = dict(round=r, spot=sw,
                       rms=res.rms_spot_radius, na=res.na_actual,
                       tele=d["telecentricity"]["residual"],
                       hard_merit=hm)
            history.append(rec)
            print_state(lens, f"polish r{r} w={sw}")
            # 约束满足(HARD merit 小)且 RMS 更小 → 更新最优
            if hm < 5.0 and rec["rms"] < best_rms:
                best = _copy.deepcopy(lens)
                best_rms = rec["rms"]
        # 每轮结束：HARD 纯约束拉回（确保约束精确满足）
        apply_weight_profile(lens, STRUCTURE_HARD)
        lens, hist = dls_optimize(lens, max_iter=dls_iter // 2,
                                  damp_init=damp_init, verbose=verbose)
        print_state(lens, f"polish r{r} 约束拉回")

        d = merit_detail(lens)
        res = d["trace_result"]
        hm = _hard_merit(lens)
        apply_weight_profile(lens, STRUCTURE_FULL)
        rec = dict(round=r, spot="HARD", rms=res.rms_spot_radius, na=res.na_actual,
                   tele=d["telecentricity"]["residual"], hard_merit=hm)
        history.append(rec)
        rms_now = res.rms_spot_radius
        if hm < 5.0 and rms_now < best_rms:
            best = _copy.deepcopy(lens)
            best_rms = rms_now
        if verbose:
            print(f"  [polish] 轮 {r}: RMS {prev_rms:.5f} → {rms_now:.5f} mm | "
                  f"NA={res.na_actual:.4f} max_inc={res.max_incident_angle:.2f}° "
                  f"tele={rec['tele']:.3f}° | HARD merit={hm:.4f}")
        if abs(rms_now - prev_rms) < 1e-3 * max(prev_rms, 1e-6):
            if verbose:
                print(f"  [polish] RMS 不再下降，提前结束。")
            break

    # 恢复最优解的权重为 FULL（约束贴边），便于后续验证
    apply_weight_profile(best, STRUCTURE_FULL)
    print("\n" + "-" * 72)
    print(f"  [polish] RMS-NA 轨迹（round, spot, rms, na, tele, hard_merit）:")
    for rec in history:
        print(f"    r{rec['round']} w={rec['spot']:<5} RMS={rec['rms']:8.4f} "
              f"NA={rec['na']:.4f} tele={rec['tele']:.3f}° hard={rec['hard_merit']:.3f}")
    print(f"  [polish] 最优: RMS={best_rms:.4f} mm")
    print_state(best, "贴边精修最终")
    return best


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="起点保存 + 从起点全变量 DLS 优化（两步独立工作流）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- find-start ----
    p_find = sub.add_parser("find-start", help="GA 结构搜索找好起点并保存")
    p_find.add_argument("--input", required=True, help="输入镜头 JSON")
    p_find.add_argument("--save-start", required=True, help="保存起点的 JSON 路径")
    p_find.add_argument("--pop-size", type=int, default=24)
    p_find.add_argument("--ga-iter", type=int, default=40)
    p_find.add_argument("--chief", action="store_true",
                        help="仅主光线追迹（1 条/评估，超快筛选）")
    p_find.add_argument("--decenter", action="store_true",
                        help="把 decenter_y 纳入变量（默认固定，12参数=d,r）")
    p_find.add_argument("--seed", type=int, default=0)

    # ---- dls ----
    p_dls = sub.add_parser("dls", help="从起点出发，只用 DLS，全高阶项一起优化")
    p_dls.add_argument("--start", required=True, help="起点 JSON（find-start 保存的）")
    p_dls.add_argument("--output", required=True, help="最终结果 JSON")
    p_dls.add_argument("--dls-iter", type=int, default=60)
    p_dls.add_argument("--decenter", action="store_true",
                       help="把 decenter_y 纳入变量")
    p_dls.add_argument("--no-perturb-restarts", action="store_true",
                       help="卡住时不做随机扰动重启")

    # ---- polish ----
    p_polish = sub.add_parser("polish", help="约束贴边精修（HARD 约束下压 RMS）")
    p_polish.add_argument("--input", required=True, help="已满足约束的镜头 JSON")
    p_polish.add_argument("--output", required=True, help="贴边精修后结果 JSON")
    p_polish.add_argument("--dls-iter", type=int, default=60)
    p_polish.add_argument("--rounds", type=int, default=2)

    args = parser.parse_args()

    if args.command == "find-start":
        if not os.path.exists(args.input):
            print(f"[ERROR] 输入文件不存在: {args.input}", file=sys.stderr)
            sys.exit(1)
        lens = LensSystem.from_json(args.input)
        ga_find_start(
            lens,
            pop_size=args.pop_size,
            ga_iter=args.ga_iter,
            save_start=args.save_start,
            use_decenter=args.decenter,
            ray_mode="chief" if args.chief else "meridional",
            seed=args.seed,
        )
    elif args.command == "dls":
        lens_final = dls_from_start(
            args.start,
            dls_iter=args.dls_iter,
            use_decenter=args.decenter,
            n_perturb_restarts=0 if args.no_perturb_restarts else 3,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        lens_final.to_json(args.output)
        print(f"\n[INFO] 结果已保存: {args.output}")
    elif args.command == "polish":
        if not os.path.exists(args.input):
            print(f"[ERROR] 输入文件不存在: {args.input}", file=sys.stderr)
            sys.exit(1)
        lens = LensSystem.from_json(args.input)
        lens_final = polish_under_constraints(lens, dls_iter=args.dls_iter,
                                              rounds=args.rounds)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        lens_final.to_json(args.output)
        print(f"\n[INFO] 贴边精修结果已保存: {args.output}")


if __name__ == "__main__":
    main()
