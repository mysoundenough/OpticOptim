"""
dls_optimizer.py — 阻尼最小二乘（DLS）局部优化（Marquardt + 鲁棒差分 + 鞍点逃逸）
==================================================================================
实现光学设计中经典的阻尼最小二乘（Damped Least Squares）局部优化算法。

算法原理：
  评价函数 merit = Σ w_i · r_i(x)²，其中 r 为残差向量。
  在当前点 x 处线性化：r(x+δx) ≈ r(x) + J·δx
  求解带 Marquardt 阻尼的法方程（归一化空间）：
      (J_normᵀ W J_norm + λ·diag(J_normᵀ W J_norm)) · δx_norm = -J_normᵀ W r
  λ 为阻尼因子。Marquardt 形式对每个方向按自身曲率缩放阻尼，
  避免高维空间条件数病态导致法方程数值爆炸（维度灾难）。

  阻尼自适应：
    - 步长被接受（merit 下降）→ λ /= 10（信任更大步长）
    - 步长被拒绝（merit 上升）→ λ *= 10（缩小步长，更接近梯度下降）

变量归一化（scaling）：
  光学变量量级差异极大（曲率~1e-3, 厚度~100, 非球面系数~1e-12）。
  对每个变量除以其特征尺度 scale_j 做归一化：
    scale_j = max(bounds范围, |x_j|·0.1, 1e-10)（普通变量）
    scale_j = max(span, SAG_STEP·10/r_max^order)（高阶非球面系数 A4-A10，
              按 sag 影响缩放，避免 ±1e-6 边界下的镜面爆炸）

鲁棒数值差分：
  前向差分步长若导致"光线断裂跳变"（遮挡/缺失比例突变或 RMS 跳到
  lost_penalty），自动缩小步长重试，使雅可比落在光滑区域。

鞍点逃逸（saddle_escape=True）：
  法方程在所有阻尼下都找不到下降步长时（梯度≈0 的鞍点/平台），依次尝试：
    1. 归一化梯度下降（小步长线搜索）
    2. 残差维度下降：对贡献最大的残差，沿其负梯度方向搜索
       （评价函数的每个维度——光斑/NA/离轴角/遮挡——都独立提供一个下降方向）
    3. 随机扰动逃逸（高斯小扰动，接受更优点）
"""

from __future__ import annotations

import copy
import numpy as np
from typing import Callable, Optional

from lens_schema import LensSystem, ASPHERIC_HIGH_ORDER
from merit_function import merit_residuals, merit_scalar, merit_detail, NUM_RESIDUALS, RESIDUAL_NAMES
from raytrace import trace_system


# ===========================================================================
# 变量尺度与差分步长
# ===========================================================================

SAG_STEP = 1e-3  # 高阶非球面系数前向差分对应的 sag 变化 (mm)


def per_variable_scales(lens: LensSystem,
                        x0: np.ndarray,
                        bounds: list[tuple[float, float]],
                        eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """
    每个变量的 (scale, step)：
      scale: 归一化 D 矩阵对角元（归一化空间 1 单位对应的物理量）
      step:  前向差分步长

    普通变量（curvature/thickness/decenter/k）：
      scale = max(bounds范围, |x|·0.1, 1e-10)，step = eps·scale

    高阶非球面系数 A4-A10：
      差分步长按 sag 影响缩放 step = SAG_STEP / r_max^order，
      保证扰动产生 ~1e-3mm 的镜面 sag 变化（可测量且光滑）。
      若沿用 ±1e-6 边界 + eps·scale，在 r=50mm 处 A6 扰动 2e-13 就
      产生 9.4m 的 sag 变化，光线全部追不到像面——维度灾难的根源。
      scale = max(span, step·10)，使归一化后各列 J_norm ≈ O(1)。
    """
    indices = lens.get_variable_indices()
    surf_map = {s.surface_id: s for s in lens.surfaces}
    n = len(x0)
    scale = np.zeros(n)
    step = np.zeros(n)
    for j, (sid, pname) in enumerate(indices):
        lo, hi = bounds[j]
        if pname in ASPHERIC_HIGH_ORDER:
            order = int(pname[1:])
            r_max = max(surf_map[sid].semi_aperture, 1e-3)
            sag_step = SAG_STEP / r_max ** order
            step[j] = sag_step
            scale[j] = max(hi - lo, sag_step * 10.0)
        else:
            scale[j] = max(hi - lo, abs(x0[j]) * 0.1, 1e-10)
            step[j] = eps * scale[j]
    return scale, step


# ===========================================================================
# 鲁棒数值雅可比
# ===========================================================================

def _ray_jump(r0: np.ndarray, r_pert: np.ndarray) -> bool:
    """
    检测前向差分扰动是否导致"光线断裂跳变"（merit 悬崖）。

    高阶非球面系数扰动过大时，sag 变化可达米级，光线全部追不到像面：
    - 遮挡比例/缺失比例突变（r[7], r[8]）
    - RMS 从小值跳到 lost_penalty（r[0] ≈ 50）

    这类跳变会让雅可比失真（数值巨大但方向无意义），
    DLS 在任何阻尼下都找不到下降步长。
    """
    # 遮挡/缺失比例跳变
    if abs(r_pert[7] - r0[7]) + abs(r_pert[8] - r0[8]) > 0.35:
        return True
    # RMS 跳变（基值小、扰动后接近 lost_penalty）
    if r0[0] < 20.0 and r_pert[0] - r0[0] > 30.0:
        return True
    return False


def numerical_jacobian(lens: LensSystem,
                       x0: np.ndarray,
                       r0: np.ndarray,
                       eps: float = 1e-7,
                       robust: bool = True) -> np.ndarray:
    """
    前向数值差分计算雅可比矩阵 J = ∂r/∂x（鲁棒版本）。

    差分步长由 per_variable_scales 决定：
      - 普通变量：eps · scale_j（与特征尺度匹配）
      - 高阶非球面系数：按 sag 影响缩放（SAG_STEP / r_max^order）

    鲁棒性（robust=True）：
      若扰动后出现"光线断裂跳变"，自动缩小步长 ×10 重试（最多 4 次），
      使差分采样落在光滑区域，雅可比才真正反映局部梯度。

    参数:
        lens: 镜头系统（当前状态对应 x0）
        x0:   (n,) 当前变量向量
        r0:   (m,) 当前残差向量
        eps:  差分步长系数（相对特征尺度）
        robust: 是否启用光线断裂检测与步长自适应

    返回:
        J: (m, n) 雅可比矩阵
    """
    n = len(x0)
    m = len(r0)
    J = np.zeros((m, n), dtype=np.float64)

    bounds = lens.get_variable_bounds()
    _, step_vec = per_variable_scales(lens, x0, bounds, eps)

    for j in range(n):
        step = step_vec[j]
        r_pert = None
        for _ in range(5):
            x_pert = x0.copy()
            x_pert[j] += step

            lens_pert = copy.deepcopy(lens)
            lens_pert.array_to_variables(x_pert)

            r_pert = merit_residuals(lens_pert)

            # 鲁棒检测：光线断裂跳变 → 缩小步长重试
            if robust and _ray_jump(r0, r_pert):
                step *= 0.1
                continue
            break

        J[:, j] = (r_pert - r0) / step

    return J


# ===========================================================================
# 边界投影
# ===========================================================================

def project_to_bounds(x: np.ndarray,
                      bounds: list[tuple[float, float]]) -> np.ndarray:
    """将变量向量 clamp 到上下界。"""
    x_proj = x.copy()
    for j, (lo, hi) in enumerate(bounds):
        x_proj[j] = np.clip(x_proj[j], lo, hi)
    return x_proj


# ===========================================================================
# DLS 主优化
# ===========================================================================

def dls_optimize(lens: LensSystem,
                 max_iter: int = 120,
                 damp_init: float = 0.15,
                 eps: float = 1e-7,
                 tol: float = 1e-10,
                 saddle_escape: bool = True,
                 callback: Optional[Callable[[int, float, dict], None]] = None,
                 verbose: bool = True) -> tuple[LensSystem, list[float]]:
    """
    阻尼最小二乘局部优化（Marquardt 阻尼 + 鲁棒差分 + 鞍点逃逸）。

    详见模块 docstring。
    """
    # 初始化
    best_lens = copy.deepcopy(lens)
    x = best_lens.variables_to_array()
    bounds = best_lens.get_variable_bounds()
    n_vars = len(x)

    if n_vars == 0:
        if verbose:
            print("[DLS] 没有标记为变量的参数，跳过优化。")
        return best_lens, [merit_scalar(best_lens)]

    # 变量尺度：用于归一化（高阶非球面按 sag 影响缩放）
    scale, _ = per_variable_scales(best_lens, x, bounds, eps)
    D = np.diag(scale)  # 缩放矩阵，x = D · x_norm

    # 权重向量
    weights = best_lens.optimization_config.merit_weights
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
    W = np.diag(w)

    # 初始评价
    r0 = merit_residuals(best_lens)
    merit0 = float(np.sum(w * r0 * r0))
    merit_history = [merit0]
    lam = damp_init

    if verbose:
        print(f"[DLS] 变量数: {n_vars}, 残差项: {NUM_RESIDUALS}, "
              f"初始 merit: {merit0:.6e}")
        print(f"[DLS] 初始阻尼 λ: {lam:.6e}")
        print(f"[DLS] 变量尺度范围: [{scale.min():.2e}, {scale.max():.2e}]")
        print("-" * 70)

    prev_merit = merit0

    for iteration in range(1, max_iter + 1):
        # 1. 计算原始空间雅可比矩阵（鲁棒数值差分）
        J = numerical_jacobian(best_lens, x, r0, eps)

        # 2. 归一化空间雅可比：J_norm = J · D（列缩放）
        J_norm = J @ D

        # 3. 高斯-牛顿近似海森矩阵（二阶导数）：H ≈ J_normᵀ W J_norm
        JtWJ = J_norm.T @ W @ J_norm
        JtWr = J_norm.T @ W @ r0

        # 4. 求解带 Marquardt 阻尼的法方程（归一化空间）
        #    A = JᵀWJ + λ·diag(JᵀWJ)：每个方向按自身曲率缩放阻尼，
        #    避免高维空间条件数病态导致法方程数值爆炸（维度灾难）。
        step_accepted = False
        jtwj_diag = np.maximum(np.diag(JtWJ), 1e-30)
        lam_trial = lam
        for _ in range(30):  # 最多调整 30 次阻尼
            A = JtWJ + lam_trial * np.diag(jtwj_diag)
            try:
                delta_x_norm = np.linalg.solve(A, -JtWr)
            except np.linalg.LinAlgError:
                delta_x_norm = np.linalg.lstsq(A, -JtWr, rcond=None)[0]

            # 5. 转换回原始空间并边界投影
            delta_x = D @ delta_x_norm
            x_new = project_to_bounds(x + delta_x, bounds)

            # 6. 评价新点
            trial_lens = copy.deepcopy(best_lens)
            trial_lens.array_to_variables(x_new)
            r_new = merit_residuals(trial_lens)
            merit_new = float(np.sum(w * r_new * r_new))

            if merit_new < prev_merit * (1 + 1e-12):
                # 步长接受
                step_accepted = True
                x = x_new
                best_lens = trial_lens
                r0 = r_new
                prev_merit = merit_new
                merit_history.append(merit_new)
                # 减小阻尼（信任更大步长）
                lam = max(lam_trial / 10.0, 1e-15)
                break
            else:
                # 步长拒绝，增大阻尼
                lam_trial = min(lam_trial * 10.0, 1e15)

        # ----------------------------------------------------------------
        # 鞍点逃逸：法方程在所有阻尼下都找不到下降步长
        # （梯度≈0 的鞍点 / 高维空间局部平台）
        # 每个评价维度（光斑/NA/离轴角/遮挡）都独立提供下降方向，
        # 逐维尝试 + 随机扰动绕过鞍点 —— 总有下降方向。
        # ----------------------------------------------------------------
        if not step_accepted and saddle_escape:
            if verbose:
                print(f"[DLS] 迭代 {iteration}: 法方程无下降步，尝试鞍点逃逸...")

            def _try_step(delta_norm: np.ndarray, tag: str) -> bool:
                nonlocal x, best_lens, r0, prev_merit, merit_history, lam
                trial_lens = copy.deepcopy(best_lens)
                trial_lens.array_to_variables(
                    project_to_bounds(x + D @ delta_norm, bounds)
                )
                r_trial = merit_residuals(trial_lens)
                m_trial = float(np.sum(w * r_trial * r_trial))
                if m_trial < prev_merit * (1 - 1e-12):
                    x = project_to_bounds(x + D @ delta_norm, bounds)
                    best_lens = trial_lens
                    r0 = r_trial
                    prev_merit = m_trial
                    merit_history.append(m_trial)
                    lam = max(damp_init, 1e-15)
                    if verbose:
                        print(f"[DLS]   鞍点逃逸成功: {tag} | merit={m_trial:.6e}")
                    return True
                return False

            # (a) 归一化梯度下降（法方程方向失效时的退路）
            g = J_norm.T @ W @ r0  # 梯度 g = JᵀWr
            gnorm = float(np.linalg.norm(g))
            if gnorm > 1e-12:
                for mag in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                    if _try_step(-mag * g / gnorm, f"gradient mag={mag:.1e}"):
                        step_accepted = True
                        break

            # (b) 残差维度下降：对贡献最大的几个残差，沿 -∇r_i 找下降方向
            if not step_accepted:
                contribs = w * r0 * r0
                order = np.argsort(-contribs)
                for i in order[:min(6, len(r0))]:
                    if contribs[i] <= 0:
                        continue
                    gi = J_norm[i]
                    ginorm = float(np.linalg.norm(gi))
                    if ginorm < 1e-12:
                        continue
                    dir_i = -gi / ginorm  # 沿该残差减小方向
                    for mag in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                        if _try_step(mag * dir_i,
                                     f"residual[{RESIDUAL_NAMES[i]}] mag={mag:.1e}"):
                            step_accepted = True
                            break
                    if step_accepted:
                        break

            # (c) 随机扰动逃逸：小幅度随机步，接受更优点
            if not step_accepted:
                rng = np.random.default_rng(iteration)
                for k in range(40):
                    delta = rng.normal(0.0, 0.01, n_vars)  # 归一化空间
                    if _try_step(delta, f"random perturb #{k}"):
                        step_accepted = True
                        break

        if not step_accepted:
            if verbose:
                print(f"[DLS] 迭代 {iteration}: 无法找到下降步长，提前终止。")
            break

        # 回调
        detail = merit_detail(best_lens)
        if callback is not None:
            callback(iteration, prev_merit, detail)

        # 打印迭代报告
        if verbose:
            trace_res = detail["trace_result"]
            na = trace_res.na_actual
            max_inc = trace_res.max_incident_angle
            wd = trace_res.working_distance
            active = int(np.sum(trace_res.active))
            print(f"[DLS] iter {iteration:3d} | merit={prev_merit:.6e} | "
                  f"λ={lam:.2e} | NA={na:.4f} | "
                  f"max_inc={max_inc:.2f}° | WD={wd:.1f}mm | "
                  f"rays={active}/{len(trace_res.active)}")

        # 收敛检查
        if len(merit_history) >= 2:
            rel_change = abs(merit_history[-1] - merit_history[-2]) / (abs(merit_history[-2]) + 1e-20)
            if rel_change < tol:
                if verbose:
                    print(f"[DLS] 收敛（相对变化 {rel_change:.2e} < {tol:.2e}）。")
                break

    if verbose:
        print("-" * 70)
        print(f"[DLS] 优化完成。最终 merit: {merit_history[-1]:.6e} "
              f"(初始 {merit_history[0]:.6e})")

    return best_lens, merit_history
