#!/usr/bin/env python3
"""
optimize_pipeline.py — 三阶段光学优化流水线（可复用模块）
=============================================================
面向离轴反射系统 / EUV 物镜的完整优化流程，三个阶段可独立调用：

  阶段 1  ga_structure_search()   GA 结构搜索
          按约束（遮挡/间距/到像面等）遍历初始结构，找候选起点。

  阶段 2  dls_find_unobscured()   DLS 找不遮挡起点
          关键：遮挡判定含连续穿透深度梯度（obscuration_depth），
          DLS 可直接沿梯度把光线"挤出"镜子实体，无需 GA。
          实测 depth 0.99 → 0.000（30/30 全通）。

  阶段 3  dls_refine()            DLS 约束精修
          从不遮挡起点出发，解锁全部变量（含高阶非球面 A4~A14），
          逐步引入 NA/角度/像方远心/放大率/RMS 约束（保持不遮挡）。

用法（代码）：
    from optimize_pipeline import run_pipeline
    result = run_pipeline('examples/offaxis_6mirror_axisym_demo.json',
                          'output/pipeline_result')

用法（CLI）：
    python optimize_pipeline.py --input examples/offaxis_6mirror_axisym_demo.json \
        --outdir output/pipeline_result
"""

from __future__ import annotations

import argparse
import copy
import os
import time
import numpy as np

from lens_schema import LensSystem
from merit_function import merit_detail
from raytrace import compute_surface_z
from ga_sa_optimizer import ga_sa_optimize
from staged_optimizer import (set_active_vars, ensure_aspheric_bounds,
                              apply_weight_profile, dls_with_restart,
                              STRUCTURE_BALANCED, perturb_lens)

# ===========================================================================
# 打印结构（GA 结构 / DLS 不遮挡结构 / DLS 精修结构）
# ===========================================================================

def print_structure(lens: LensSystem, title: str) -> None:
    """打印结构参数 + 关键指标。"""
    d = merit_detail(lens)
    res = d["trace_result"]
    z = compute_surface_z(lens.surfaces)
    cons = lens.metadata.optical_constraints
    n_ok = int(res.active.sum())
    n_total = len(res.active)
    obj_h = cons.object_height_mm if cons.object_height_mm > 0 else 50.0
    img_h = (float(np.sqrt(np.mean(res.image_xy[res.active], axis=0)[0] ** 2 +
                           np.mean(res.image_xy[res.active], axis=0)[1] ** 2))
             if n_ok else 0.0)
    mag = img_h / obj_h if obj_h > 0 else 0.0
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  有效光线 : {n_ok}/{n_total}   遮挡: {int(res.obscured.sum())}   "
          f"穿透深度: {res.obscuration_depth:.4f}")
    print(f"  RMS 光斑 : {res.rms_spot_radius:.4f} mm")
    print(f"  实际 NA  : {res.na_actual:.4f}  (目标 {cons.na_target})")
    print(f"  放大率   : {mag:.4f}  (目标 {cons.magnification_target})  "
          f"物高 {obj_h:.1f} → 像高 {img_h:.1f}")
    print(f"  像方远心 : {d['telecentricity']['residual']:.3f}°  "
          f"(限 {cons.telecentric_image_max_deg}°)")
    print(f"  最大入射角: {res.max_incident_angle:.2f}°  (限 {cons.incident_angle_max_deg}°)")
    print(f"  工作距离 : {res.working_distance:.1f} mm  (要求 ≥ {cons.working_distance_min})")
    print(f"  像面 z   : {z[-1]:.1f}  (目标 {cons.image_z_fixed})")
    print(f"  评价函数 : {d['total']:.4e}")
    print("  结构:")
    for i in range(1, 7):
        s = lens.surfaces[i]
        asp = s.aspheric
        hdr = "    M%d: z=%7.1f  c=%+.5f  半口径=%5.0f  k=%+.2f" % (
            i, z[i], s.curvature, s.semi_aperture, asp.k if asp else 0.0)
        if asp is not None:
            hdr += "  A4=%+.2e  A6=%+.2e" % (asp.A4, asp.A6)
        print(hdr)
    print(f"  像面     : z={z[-1]:.1f}")


# ===========================================================================
# 阶段 1：GA 结构搜索（按约束遍历初始结构）
# ===========================================================================

DEFAULT_VARS = ["curvature", "thickness", "semi_aperture", "k"]
# 全部参数（60 = c+d+半口径+k+A4~A14 × 6 镜）——所有阶段默认使用
ALL_VARS = ["curvature", "thickness", "semi_aperture", "k",
            "A4", "A6", "A8", "A10", "A12", "A14"]


def ga_structure_search(lens: LensSystem,
                        pop_size: int = 36,
                        ga_iter: int = 200,
                        seed: int = 0,
                        var_names: list[str] | None = None,
                        callback=None,
                        verbose: bool = True) -> LensSystem:
    """
    阶段 1：GA 按约束遍历初始结构，找候选起点。

    变量集默认全部 60 参数（c+d+半口径+k+A4~A14）× 6 镜；
    可传入 var_names 自定义（如 DEFAULT_VARS 24 参数）。

    返回: 优化后的镜头（GA 找到的最优结构）
    """
    if var_names is None:
        var_names = ALL_VARS
    lens = copy.deepcopy(lens)
    ensure_aspheric_bounds(lens)
    lens.metadata.ray_mode = "offaxis"
    set_active_vars(lens, var_names)
    apply_weight_profile(lens, STRUCTURE_BALANCED)
    n_vars = len(lens.variables_to_array())

    if verbose:
        print(f"\n[阶段1 GA结构搜索] 变量={n_vars}, 种群={pop_size}, 代数={ga_iter}")
    np.random.seed(seed)
    lens, hist = ga_sa_optimize(lens, pop_size=pop_size, ga_iter=ga_iter,
                                sa_temp_init=60.0, callback=callback, verbose=verbose)
    return lens


# ===========================================================================
# 阶段 2：DLS 找不遮挡起点
# ===========================================================================

def _dls_unobscured_single(lens, dls_iter, n_rounds, n_perturb_restarts,
                           process_dir, verbose, tag=""):
    """单个起点的 DLS 找不遮挡（内部）。"""
    # 硬约束全满足：所有光学/结构约束权重高（像质 rms=0，留给阶段3）
    PROF = dict(rms_spot=0.0,
                na=50.0, angle_inc=20.0, angle_exit=20.0, wd=50.0,
                obs=2000.0, miss=40000.0, tele=100.0, pkg=0.0, mg=50.0,
                iz=100.0, mag=200.0, op=200.0, mz=200.0)
    for rnd in range(n_rounds):
        apply_weight_profile(lens, PROF)
        lens, h = dls_with_restart(lens, max_iter=dls_iter, damp_init=0.15,
                                   n_perturb_restarts=n_perturb_restarts,
                                   verbose=False)
        d = merit_detail(lens)
        res = d["trace_result"]
        n_ok = int(res.active.sum())
        if verbose:
            print(f"  {tag}轮{rnd+1}: {n_ok}/{len(res.active)} "
                  f"depth={res.obscuration_depth:.4f} merit={d['total']:.1f}")
        if process_dir is not None:
            p = os.path.join(process_dir, "dls_round%s_%02d.json" % (tag, rnd + 1))
            lens.to_json(p)
        if n_ok == len(res.active):
            if verbose:
                print(f"  {tag}>>> 不遮挡达成! depth=0.000")
            break
    return lens


def dls_find_unobscured(lens: LensSystem,
                        dls_iter: int = 80,
                        n_rounds: int = 8,
                        n_perturb_restarts: int = 6,
                        n_seeds: int = 3,
                        var_names: list[str] | None = None,
                        process_dir: str | None = None,
                        verbose: bool = True) -> LensSystem:
    """
    阶段 2：DLS 直接找不遮挡结构（无需 GA，多 seed 取最优）。

    原理：遮挡判定含连续穿透深度（obscuration_depth），
    DLS 沿梯度把光线从镜子实体里"挤出"。实测 depth 0.99 → 0.000。

    多 seed：从原始起点 + 多个随机扰动起点分别跑 DLS，
    取穿透深度最小（优先全通）的结果——对起点敏感，多起点更稳。

    评价函数仅聚焦遮挡/到像面（obs=2000, miss=40000，其余权重 0）。

    返回: 不遮挡（或最深最小）的镜头
    """
    if var_names is None:
        var_names = ALL_VARS
    base = copy.deepcopy(lens)
    ensure_aspheric_bounds(base)
    base.metadata.ray_mode = "offaxis"
    set_active_vars(base, var_names)

    if verbose:
        d0 = merit_detail(base)
        print(f"\n[阶段2 DLS找不遮挡] 初始 depth={d0['trace_result'].obscuration_depth:.3f}, "
              f"多 seed={n_seeds}")

    best_lens = None
    best_depth = None
    for s in range(n_seeds):
        if s == 0:
            cand = copy.deepcopy(base)
            tag = ""
        else:
            # 扰动起点（sigma 递增，探索不同遮挡盆地）
            cand = perturb_lens(base, sigma=0.03 * (s + 1))
            tag = "s%d " % s
        cand = _dls_unobscured_single(cand, dls_iter, n_rounds, n_perturb_restarts,
                                      process_dir, verbose, tag)
        d = merit_detail(cand)
        res = d["trace_result"]
        n_ok = int(res.active.sum())
        depth = res.obscuration_depth
        # 取最优：全通优先，否则 depth 最小
        better = False
        if best_lens is None:
            better = True
        elif n_ok == len(res.active) and best_depth is not None and best_depth > 0:
            better = True
        elif n_ok == len(res.active) and best_depth == 0:
            better = False
        elif depth < best_depth:
            better = True
        if better:
            best_lens = cand
            best_depth = depth
        if verbose:
            print(f"  [seed{s}] 结果: {n_ok}/{len(res.active)} depth={depth:.4f}")
        if n_ok == len(res.active):
            if verbose:
                print(f"  >>> 不遮挡达成!（seed{s}）")
            break
    return best_lens


# ===========================================================================
# 阶段 3：DLS 约束精修（保持不遮挡）
# ===========================================================================

def dls_refine(lens: LensSystem,
               dls_iter: int = 60,
               var_names: list[str] | None = None,
               spot_ramp: tuple[float, ...] = (0.1, 1.0, 5.0, 20.0),
               verbose: bool = True) -> LensSystem:
    """
    阶段 3：从（不遮挡）起点出发，DLS 精修全部约束。

    - 解锁全部变量（默认 60 参数：c/d/半口径/k/A4~A14）
    - 约束权重渐进引入：NA/角度/像方远心/放大率/像面
    - RMS 光斑权重同伦递增（0.1→1→5→20）
    - 遮挡保护保持（obs=400, miss=40000），不破坏不遮挡状态

    返回: 精修后的镜头
    """
    if var_names is None:
        var_names = ALL_VARS
    lens = copy.deepcopy(lens)
    ensure_aspheric_bounds(lens)
    lens.metadata.ray_mode = "offaxis"
    set_active_vars(lens, var_names)

    BASE = dict(na=20.0, angle_inc=0.5, angle_exit=0.5, wd=2.0,
                obs=400.0, miss=40000.0, tele=10.0, pkg=0.0, mg=50.0,
                iz=20.0, mag=50.0)
    for rms_w in spot_ramp:
        PROF = dict(BASE, rms_spot=rms_w)
        if verbose:
            print(f"\n[阶段3 DLS精修] rms_w={rms_w} ...")
        apply_weight_profile(lens, PROF)
        lens, h = dls_with_restart(lens, max_iter=dls_iter, damp_init=0.15,
                                   n_perturb_restarts=3, verbose=False)
        d = merit_detail(lens)
        res = d["trace_result"]
        if verbose:
            print(f"  {int(res.active.sum())}/{len(res.active)} depth={res.obscuration_depth:.4f} "
                  f"RMS={res.rms_spot_radius:.2f} NA={res.na_actual:.3f} merit={d['total']:.1f}")
    return lens


# ===========================================================================
# 完整流水线
# ===========================================================================

def run_pipeline(input_json: str,
                 outdir: str = "output/pipeline_result",
                 pop_size: int = 36,
                 ga_iter: int = 200,
                 ga_seeds: tuple[int, ...] = (0, 5, 11),
                 dls_iter: int = 80,
                 n_seeds: int = 3,
                 save_intermediate: bool = True,
                 visualize: bool = True,
                 realtime: bool = False,
                 viz_every: int = 20,
                 verbose: bool = True) -> dict:
    """
    完整三阶段流水线：GA 结构搜索 → DLS 找不遮挡 → DLS 精修。

    多 GA seed：对 ga_seeds 中每个随机种子跑 GA 结构搜索（默认 0/5/11），
    每个 GA 结构都走阶段2 DLS 找不遮挡，取穿透深度最小（优先全通）的
    作为阶段3 精修的起点——覆盖不同盆地，更可能找到不遮挡结构。

    功能：
    - 打印所有阶段的完整结构
    - save_intermediate：保存各阶段/过程 JSON
    - visualize：完成后自动调用 plot_structure 可视化三个阶段结构
    - realtime：优化过程中实时保存结构快照 + 画布局图（GA 每 viz_every 代、
      DLS 每轮），输出到 outdir/realtime/

    返回:
        dict(ga_lens, unobscured_lens, final_lens, outdir, best_seed)
    """
    t0 = time.time()
    lens0 = LensSystem.from_json(input_json)
    os.makedirs(outdir, exist_ok=True)
    rt_dir = os.path.join(outdir, "realtime")
    if realtime:
        os.makedirs(rt_dir, exist_ok=True)

    def ga_callback(gen, best_lens, merit):
        if realtime and gen % viz_every == 0:
            p = os.path.join(rt_dir, "ga_gen_%04d.json" % gen)
            best_lens.to_json(p)
            try:
                from plot_structure import plot_quick
                plot_quick(best_lens, os.path.join(rt_dir, "ga_gen_%04d.png" % gen),
                           title="GA gen %d merit=%.3e" % (gen, merit))
            except Exception:
                pass

    # ---- 阶段 1+2：多 GA seed，每个都走 DLS 找不遮挡，取全局最优 ----
    best_ga = None
    best_unob = None
    best_depth = None
    best_seed = None
    for seed in ga_seeds:
        ga_lens = ga_structure_search(lens0, pop_size=pop_size, ga_iter=ga_iter,
                                      seed=seed, callback=ga_callback,
                                      verbose=verbose)
        print_structure(ga_lens, f"阶段1: GA 结构搜索结果 (seed {seed})")
        if save_intermediate:
            p = os.path.join(outdir, f"ga_seed{seed}.json")
            ga_lens.to_json(p)
            print(f"  [保存] {p}")

        unob_lens = dls_find_unobscured(ga_lens, dls_iter=dls_iter, n_seeds=n_seeds,
                                        process_dir=rt_dir if realtime else None,
                                        verbose=verbose)
        print_structure(unob_lens, f"阶段2: DLS 不遮挡结构 (seed {seed})")
        if save_intermediate:
            p = os.path.join(outdir, f"dls_seed{seed}.json")
            unob_lens.to_json(p)
            print(f"  [保存] {p}")

        # 取最优：全通优先，否则 depth 最小
        d = merit_detail(unob_lens)
        res = d["trace_result"]
        depth = res.obscuration_depth
        n_ok = int(res.active.sum())
        better = False
        if best_depth is None:
            better = True
        elif n_ok == len(res.active) and best_depth > 0:
            better = True
        elif not (n_ok == len(res.active) and best_depth == 0) and depth < best_depth:
            better = True
        if better:
            best_ga, best_unob, best_depth, best_seed = ga_lens, unob_lens, depth, seed
            print(f"  >>> 当前最优: seed{seed} depth={depth:.4f}")
        if n_ok == len(res.active):
            print(f"  >>> seed{seed} 不遮挡达成! 停止后续 seed")
            break

    print(f"\n[流水线] 最优起点: seed{best_seed}, depth={best_depth:.4f}")

    # ---- 阶段 3：DLS 精修（最优不遮挡起点）----
    final_lens = dls_refine(best_unob, dls_iter=dls_iter, verbose=verbose)
    print_structure(final_lens, "阶段3: DLS 精修最终结构")
    if save_intermediate:
        p = os.path.join(outdir, "final_refined.json")
        final_lens.to_json(p)
        print(f"  [保存] {p}")

    # ---- 完成后可视化：三阶段结构全套图 ----
    if visualize:
        print(f"\n[可视化] 生成三个阶段结构图...")
        from plot_structure import plot_full_views, plot_per_mirror
        for name, l in [("ga", best_ga), ("unobscured", best_unob),
                        ("final", final_lens)]:
            d = os.path.join(outdir, "figures", name)
            os.makedirs(d, exist_ok=True)
            plot_full_views(l, os.path.join(d, "full"))
            plot_per_mirror(l, os.path.join(d, "per_mirror"))
            print(f"  [{name}] 图已生成: {d}")

    print(f"\n[流水线] 总耗时 {time.time()-t0:.1f}s")
    return dict(ga_lens=best_ga, unobscured_lens=best_unob, final_lens=final_lens,
                outdir=outdir, best_seed=best_seed)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="三阶段光学优化流水线")
    parser.add_argument("--input", required=True, help="输入镜头 JSON")
    parser.add_argument("--outdir", default="output/pipeline_result", help="输出目录")
    parser.add_argument("--pop-size", type=int, default=36)
    parser.add_argument("--ga-iter", type=int, default=200)
    parser.add_argument("--ga-seed", type=int, default=0, help="单 seed（兼容）")
    parser.add_argument("--ga-seeds", type=int, nargs="+", default=[0, 5, 11],
                        help="多 GA seed（默认 0 5 11）")
    parser.add_argument("--dls-iter", type=int, default=80)
    parser.add_argument("--no-viz", action="store_true", help="不生成结构图")
    parser.add_argument("--n-seeds", type=int, default=3, help="阶段2 DLS 多起点数")
    parser.add_argument("--realtime", action="store_true", help="实时可视化")
    parser.add_argument("--viz-every", type=int, default=20, help="实时可视化的间隔代数")
    args = parser.parse_args()
    run_pipeline(args.input, args.outdir, args.pop_size, args.ga_iter,
                 tuple(args.ga_seeds), args.dls_iter, n_seeds=args.n_seeds,
                 visualize=not args.no_viz, realtime=args.realtime,
                 viz_every=args.viz_every)


if __name__ == "__main__":
    main()
