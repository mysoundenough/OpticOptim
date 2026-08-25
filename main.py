#!/usr/bin/env python3
"""
main.py — 光学镜头优化 CLI 入口
=================================
纯终端命令行工具，完全离线，不调用 Zemax/CodeV/COM/API。

用法：
  python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json
  python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --global
  python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --viz

参数：
  --input   输入镜头 JSON 文件（必需）
  --output  输出优化后 JSON 文件（必需）
  --global  开启全局优化（GA+SA），否则仅 DLS 局部优化
  --viz     生成可视化图表（光斑图、布局图、MTF、光线扇、收敛曲线）
  --viz-dir 可视化输出目录（默认 output/figures）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

from lens_schema import LensSystem
from raytrace import trace_system
from merit_function import merit_scalar, merit_detail
from dls_optimizer import dls_optimize
from ga_sa_optimizer import ga_sa_optimize


# ===========================================================================
# 报告打印
# ===========================================================================

def print_system_report(lens: LensSystem, title: str = "System Report") -> None:
    """打印系统状态报告：NA、入射角、工作距离等。"""
    res = trace_system(lens)
    detail = merit_detail(lens, res)

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  系统名称:       {lens.metadata.name}")
    print(f"  表面数:         {lens.num_surfaces}")
    print(f"  波长:           {[f'{w*1e6:.2f}nm' for w in lens.metadata.wavelengths]}")
    print(f"  视场:           {lens.metadata.fov_deg} deg")
    print(f"  目标 NA:        {lens.metadata.optical_constraints.na_target}")
    print(f"-" * 70)
    print(f"  实际 NA:        {res.na_actual:.6f}")
    print(f"  RMS 光斑:       {res.rms_spot_radius:.6e} mm")
    print(f"  工作距离:       {res.working_distance:.2f} mm")
    print(f"  有效光线:       {int(np.sum(res.active))}/{len(res.active)}"
          f"   (被遮挡: {int(np.sum(res.obscured))})")
    print(f"  最大出射角:     {np.nanmax(res.exit_angles[res.active]) if res.active.any() else 0:.2f} deg")
    print(f"-" * 70)
    print(f"  各反射面最大入射角:")
    per_surf = res.per_surface_max_incident()
    for sid in sorted(per_surf.keys()):
        ang = per_surf[sid]
        limit = lens.metadata.optical_constraints.incident_angle_max_deg
        status = "OK" if ang <= limit else "OVER"
        print(f"    Surface {sid:2d}: {ang:7.3f} deg  (limit {limit:.1f})  [{status}]")
    print(f"-" * 70)
    print(f"  评价函数分解:")
    for name in ["rms_spot", "distortion", "chroma", "na_penalty",
                 "angle_incident", "angle_exit", "working_distance",
                 "obscuration"]:
        d = detail[name]
        print(f"    {name:20s}: residual={d['residual']:.6e}  "
              f"weight={d['weight']:.3f}  contrib={d['contribution']:.6e}")
    print(f"    {'TOTAL':20s}: {detail['total']:.6e}")
    print(f"{'='*70}\n")


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="光学镜头优化 CLI（离线，DLS 局部 / GA+SA 全局）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 仅 DLS 局部优化
  python main.py --input examples/euv_6mirror_demo.json --output output/opt.json

  # 全局优化（GA+SA）+ DLS 精修
  python main.py --input examples/euv_6mirror_demo.json --output output/opt.json --global

  # 生成可视化图表
  python main.py --input examples/euv_6mirror_demo.json --output output/opt.json --viz
        """,
    )
    parser.add_argument("--input", required=True, help="输入镜头 JSON 文件")
    parser.add_argument("--output", required=True, help="输出优化后 JSON 文件")

    # 优化模式：--local / --global / --staged 互斥，默认局部优化
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--local", action="store_true", dest="use_local",
                            help="仅 DLS 局部优化（默认模式，与 --global/--staged 互斥）")
    mode_group.add_argument("--global", action="store_true", dest="use_global",
                             help="全局优化（GA+SA）+ DLS 精修，与 --local/--staged 互斥")
    mode_group.add_argument("--staged", action="store_true", dest="use_staged",
                            help="分阶段优化（GA结构搜索→逐级解锁高阶项→约束验证），与 --local/--global 互斥")

    parser.add_argument("--viz", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--animate", action="store_true",
                        help="优化过程动画：每步迭代保存镜头布局快照，合成GIF+对比网格图")
    parser.add_argument("--viz-dir", default=None,
                        help="可视化输出目录（默认 <output_dir>/figures）")

    args = parser.parse_args()

    # 确定优化模式（默认局部优化）
    if args.use_staged:
        opt_mode = "STAGED"
        opt_mode_label = "分阶段优化（GA结构→逐级高阶项→约束验证）"
    elif args.use_global:
        opt_mode = "GLOBAL"
        opt_mode_label = "全局优化（GA+SA + DLS精修）"
    else:
        opt_mode = "LOCAL"
        opt_mode_label = "局部优化（DLS 阻尼最小二乘）"

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 加载镜头系统
    print(f"[INFO] 加载镜头系统: {args.input}")
    print(f"[INFO] 优化模式: {opt_mode_label}")
    lens = LensSystem.from_json(args.input)
    print_system_report(lens, "初始系统")

    # 优化配置
    opt_config = lens.optimization_config
    dls_cfg = opt_config.local_dls
    all_merit_history = {}

    # 输出目录（提前定义，供动画和可视化使用）
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================
    # 分阶段优化（GA结构搜索→逐级解锁高阶项→约束验证）
    # ==================================================================
    if args.use_staged:
        print("\n[INFO] 开始分阶段优化...")
        t0 = time.time()
        from staged_optimizer import staged_optimize
        lens, milestones = staged_optimize(
            lens,
            pop_size=opt_config.global_search.pop_size,
            ga_iter=opt_config.global_search.ga_iter,
            dls_iter=dls_cfg.max_iter,
        )
        all_merit_history["STAGED"] = [m for _, m in milestones]
        print(f"[INFO] 分阶段优化完成，耗时 {time.time()-t0:.1f}s")
        print_system_report(lens, f"优化完成 — 最终系统 [{opt_mode}]")
        lens.to_json(args.output)
        print(f"[INFO] 优化结果已保存: {args.output}")
        if args.viz:
            print("\n[INFO] 生成可视化图表...")
            from visualization import generate_report
            viz_dir = args.viz_dir or os.path.join(output_dir, "figures")
            os.makedirs(viz_dir, exist_ok=True)
            paths = generate_report(lens, viz_dir,
                                    merit_history=all_merit_history["STAGED"],
                                    prefix="final")
            print(f"[INFO] 可视化图表已保存到: {viz_dir}")
            for name, path in paths.items():
                print(f"  {name:12s}: {path}")
        print("\n[INFO] 全部完成。")
        return

    # ==================================================================
    # 全局优化（GA+SA）
    # ==================================================================
    if args.use_global:
        print("\n[INFO] 开始全局优化（GA+SA）...")
        t0 = time.time()
        ga_cfg = opt_config.global_search
        lens, ga_history = ga_sa_optimize(
            lens,
            pop_size=ga_cfg.pop_size,
            ga_iter=ga_cfg.ga_iter,
            sa_temp_init=ga_cfg.sa_temp_init,
            verbose=True,
        )
        all_merit_history["GA+SA"] = ga_history
        print(f"[INFO] 全局优化完成，耗时 {time.time()-t0:.1f}s")
        print_system_report(lens, "全局优化后（DLS 精修前）")

    # ==================================================================
    # 局部优化（DLS）
    # ==================================================================
    print("\n[INFO] 开始局部优化（DLS 阻尼最小二乘）...")
    t0 = time.time()
    dls_cfg = opt_config.local_dls

    # 优化过程动画可视化
    opt_viz = None
    dls_callback = None
    if args.animate:
        from visualization import OptimizationVisualizer
        animate_dir = os.path.join(output_dir, "animation")
        opt_viz = OptimizationVisualizer(
            lens, animate_dir, fps=3, draw_rays=True, n_rays=8
        )
        # 保存初始状态（第 0 帧）
        from merit_function import merit_detail
        init_detail = merit_detail(lens)
        init_tr = init_detail["trace_result"]
        opt_viz.capture(0, merit_scalar(lens), lens,
                        na=init_tr.na_actual,
                        max_incident=init_tr.max_incident_angle)

        def dls_callback(iter_num, merit_val, detail):
            tr = detail["trace_result"]
            opt_viz.capture(iter_num, merit_val, lens,
                            na=tr.na_actual,
                            max_incident=tr.max_incident_angle)

    lens, dls_history = dls_optimize(
        lens,
        max_iter=dls_cfg.max_iter,
        damp_init=dls_cfg.damp_init,
        verbose=True,
        callback=dls_callback,
    )
    all_merit_history["DLS"] = dls_history
    print(f"[INFO] 局部优化完成，耗时 {time.time()-t0:.1f}s")

    # 合成优化动画
    if args.animate and opt_viz is not None:
        print("\n[INFO] 合成优化过程动画...")
        anim_result = opt_viz.finalize()
        if "gif" in anim_result:
            print(f"  GIF 动画: {anim_result['gif']}")
        if "grid" in anim_result:
            print(f"  迭代对比网格: {anim_result['grid']}")
        print(f"  快照数量: {len(anim_result['snapshots'])}")

    # 最终报告
    print_system_report(lens, f"优化完成 — 最终系统 [{opt_mode}]")

    # ==================================================================
    # 保存输出
    # ==================================================================
    lens.to_json(args.output)
    print(f"[INFO] 优化结果已保存: {args.output}")

    # ==================================================================
    # 可视化
    # ==================================================================
    if args.viz:
        print("\n[INFO] 生成可视化图表...")
        from visualization import generate_report

        viz_dir = args.viz_dir or os.path.join(output_dir, "figures")
        os.makedirs(viz_dir, exist_ok=True)

        # 合并收敛曲线
        if len(all_merit_history) > 1:
            # GA 和 DLS 串联：GA 的最后一个点作为 DLS 的起点
            combined = []
            if "GA+SA" in all_merit_history:
                combined.extend(all_merit_history["GA+SA"])
            if "DLS" in all_merit_history:
                combined.extend(all_merit_history["DLS"][1:])  # 跳过重复的起点
        else:
            combined = dls_history

        paths = generate_report(
            lens, viz_dir,
            merit_history=combined,
            prefix="final",
        )
        print(f"[INFO] 可视化图表已保存到: {viz_dir}")
        for name, path in paths.items():
            print(f"  {name:12s}: {path}")

    print("\n[INFO] 全部完成。")


if __name__ == "__main__":
    main()
