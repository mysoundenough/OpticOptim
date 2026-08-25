#!/usr/bin/env python3
"""
find_offaxis_structure.py — 离轴反射系统结构搜索工具
=====================================================
用户输入：几块镜子（JSON）、物方 NA、像方 NA、主光线入射角（离轴角）。
全局搜索（GA+SA）自动寻找镜子位置（厚度/偏心）+ 曲率 + 非球面参数，
使光线从物面打到像面且不被镜子实体遮挡；再送入 DLS 局部精修。

分阶段策略：
  阶段 1（结构搜索）：只优化 thickness + decenter_y（固定曲率），
    全局搜索找到"光线不被遮挡、全部到达像面"的几何布局。
  阶段 2（聚焦精修）：恢复曲率/非球面变量，DLS 局部优化
    在"不遮挡"区域内最小化 RMS 光斑。

用法：
  python find_offaxis_structure.py --input examples/offaxis_6mirror_decenter_demo.json \
      --output output/offaxis_6mirror_opt.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import numpy as np

from lens_schema import LensSystem
from merit_function import merit_detail, merit_scalar
from raytrace import trace_system
from ga_sa_optimizer import ga_sa_optimize
from dls_optimizer import dls_optimize


def print_state(lens: LensSystem, title: str) -> None:
    """打印系统状态：有效/遮挡/失败光线、RMS、NA。"""
    d = merit_detail(lens)
    res = d["trace_result"]
    n_total = len(res.active)
    n_ok = int(np.sum(res.active))
    n_obs = int(np.sum(res.obscured))
    n_miss = n_total - n_ok - n_obs
    print(f"  [{title}] 有效={n_ok}/{n_total} 遮挡={n_obs} 失败={n_miss} | "
          f"RMS={res.rms_spot_radius:.6e} mm | NA={res.na_actual:.4f} | "
          f"merit={d['total']:.6e}")


def main():
    parser = argparse.ArgumentParser(description="离轴反射系统结构搜索")
    parser.add_argument("--input", required=True, help="输入镜头 JSON")
    parser.add_argument("--output", required=True, help="输出 JSON")
    parser.add_argument("--pop-size", type=int, default=36, help="GA 种群规模")
    parser.add_argument("--ga-iter", type=int, default=100, help="GA 迭代代数")
    parser.add_argument("--dls-iter", type=int, default=120, help="DLS 最大迭代")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="跳过结构搜索，直接 DLS（起点需已不遮挡）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    np.random.seed(args.seed)
    lens = LensSystem.from_json(args.input)
    cons = lens.metadata.optical_constraints

    print("=" * 70)
    print(f"  离轴结构搜索: {lens.metadata.name}")
    print(f"  物方 NA={cons.na_objective or lens.metadata.aperture.value}, "
          f"像方 NA={cons.na_target}, 主光线角={cons.chief_ray_angle_deg}°")
    print("=" * 70)
    print_state(lens, "初始")

    # 阶段 1：结构搜索（只 thickness + decenter_y，固定曲率/非球面）
    if not args.skip_stage1:
        lens_s1 = copy.deepcopy(lens)
        for s in lens_s1.surfaces:
            if s.surface_id > 0 and s.surface_id < len(lens_s1.surfaces) - 1:
                c = s.curvature
                if s.bounds is not None:
                    s.bounds.curvature = (c - 1e-9, c + 1e-9)  # 固定曲率
                s.aspheric_variable = None  # 关闭非球面变量
        n_vars = len(lens_s1.variables_to_array())
        print(f"\n[阶段 1] 结构搜索（变量={n_vars}: 厚度+偏心）...")
        t0 = time.time()
        lens_s1, hist1 = ga_sa_optimize(
            lens_s1,
            pop_size=args.pop_size,
            ga_iter=args.ga_iter,
            sa_temp_init=60.0,
            verbose=True,
        )
        print(f"  耗时 {time.time()-t0:.1f}s")
        print_state(lens_s1, "结构搜索后")

        # 把找到的结构（厚度/偏心）复制回原系统
        for i in range(1, len(lens.surfaces) - 1):
            lens.surfaces[i].thickness = lens_s1.surfaces[i].thickness
            lens.surfaces[i].decenter_y = lens_s1.surfaces[i].decenter_y
        print_state(lens, "结构已复制")
    else:
        print("\n[阶段 1] 跳过（--skip-stage1）")

    # 阶段 2：DLS 聚焦精修（全变量）
    print(f"\n[阶段 2] DLS 局部精修（变量={len(lens.variables_to_array())}）...")
    t0 = time.time()
    lens_final, hist2 = dls_optimize(lens, max_iter=args.dls_iter, damp_init=0.15,
                                     verbose=True)
    print(f"  耗时 {time.time()-t0:.1f}s")
    print_state(lens_final, "DLS 后")

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    lens_final.to_json(args.output)
    print(f"\n[INFO] 结果已保存: {args.output}")

    # 报告
    d = merit_detail(lens_final)
    res = d["trace_result"]
    print("\n" + "=" * 70)
    print("  最终报告")
    print("=" * 70)
    print(f"  有效光线: {int(np.sum(res.active))}/{len(res.active)}")
    print(f"  被遮挡:   {int(np.sum(res.obscured))}")
    print(f"  RMS 光斑: {res.rms_spot_radius:.6e} mm")
    print(f"  实际 NA:  {res.na_actual:.4f} (目标 {cons.na_target})")
    print(f"  工作距离: {res.working_distance:.2f} mm")
    print("  各面偏心:")
    for s in lens_final.surfaces:
        if s.surface_id > 0 and s.surface_id < len(lens_final.surfaces) - 1:
            print(f"    M{s.surface_id}: z={s.thickness:.1f} decenter_y={s.decenter_y:.1f} "
                  f"c={s.curvature:.6f}")


if __name__ == "__main__":
    main()
