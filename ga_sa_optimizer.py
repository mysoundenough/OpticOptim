"""
ga_sa_optimizer.py — 遗传算法 + 模拟退火（GA+SA）混合全局优化
=================================================================
实现遗传算法（GA）与模拟退火（SA）混合的全局优化策略。

策略：
  1. GA 负责全局探索：种群在变量空间中广泛搜索，通过选择-交叉-变异
     不断进化，找到有潜力的区域。
  2. SA 负责局部跳跃：在 GA 每代中对部分个体施加 SA 扰动，以概率
     exp(-ΔE/T) 接受恶化解，帮助跳出局部最优。
  3. 输出：全局优化结束后，将最优候选解送入 DLS 做局部精修。

  温度衰减：T = T0 · α^generation，α=0.95

  变量边界：所有操作后 clamp 到 bounds。

非球面参数（k, A4-A10）作为普通变量参与 GA/SA 迭代，
与曲率、厚度同等对待。
"""

from __future__ import annotations

import copy
import numpy as np
from typing import Optional

from lens_schema import LensSystem
from merit_function import merit_scalar
from dls_optimizer import project_to_bounds


# ===========================================================================
# 适应度评估
# ===========================================================================

def evaluate_individual(lens_template: LensSystem,
                        x: np.ndarray) -> float:
    """评估单个个体的适应度（merit 值，越小越好）。"""
    lens = copy.deepcopy(lens_template)
    lens.array_to_variables(x)
    return merit_scalar(lens)


def evaluate_population(lens_template: LensSystem,
                        population: np.ndarray) -> np.ndarray:
    """评估整个种群，返回适应度数组。"""
    pop_size = population.shape[0]
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = evaluate_individual(lens_template, population[i])
    return fitness


# ===========================================================================
# 种群初始化
# ===========================================================================

def init_population(lens: LensSystem,
                    pop_size: int,
                    bounds: list[tuple[float, float]],
                    restart_sigma: Optional[float] = None) -> np.ndarray:
    """
    初始化种群。

    restart_sigma=None：
      第一个个体为初始设计（x0），其余在 bounds 内均匀随机采样。
      均匀随机（而非围绕 x0 小扰动）能覆盖更大参数空间，
      对"遮挡/通过率"这类硬约束更可能产生部分可行个体，
      避免整个种群 merit 相同导致 GA 无选择压力。

    restart_sigma 不为 None（邻域重启）：
      所有个体围绕 x0 高斯采样，σ = restart_sigma·(hi-lo)。
      用于 DLS 卡住时在局部邻域重新探索（注入 GA 找到的下降方向）。
    """
    x0 = lens.variables_to_array()
    n_vars = len(x0)
    population = np.zeros((pop_size, n_vars))

    population[0] = x0
    for i in range(1, pop_size):
        for j in range(n_vars):
            lo, hi = bounds[j]
            if restart_sigma is not None:
                # 邻域重启：围绕当前点高斯采样
                val = x0[j] + np.random.normal(0, restart_sigma * (hi - lo))
            else:
                # 50% 概率围绕初始值扰动，50% 概率全范围均匀采样
                if np.random.rand() < 0.5:
                    span = (hi - lo) * 0.5
                    val = x0[j] + (np.random.rand() - 0.5) * span
                else:
                    val = lo + np.random.rand() * (hi - lo)
            val = np.clip(val, lo, hi)
            population[i, j] = val

    return population


# ===========================================================================
# GA 操作
# ===========================================================================

def tournament_select(population: np.ndarray,
                      fitness: np.ndarray,
                      tournament_size: int = 3) -> np.ndarray:
    """
    锦标赛选择：随机选 k 个个体，取适应度最好的。
    返回选中的个体向量。
    """
    pop_size = population.shape[0]
    indices = np.random.choice(pop_size, tournament_size, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return population[best_idx].copy()


def arithmetic_crossover(parent1: np.ndarray,
                         parent2: np.ndarray,
                         alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    算术交叉：child = α·p1 + (1-α)·p2
    产生两个子代（α 和 1-α 互换）。
    """
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


def gaussian_mutate(individual: np.ndarray,
                    bounds: list[tuple[float, float]],
                    mutation_rate: float = 0.3,
                    mutation_scale: float = 0.1) -> np.ndarray:
    """
    高斯变异：以 mutation_rate 概率对每个变量加高斯噪声。
    噪声标准差 = mutation_scale · (hi - lo)
    """
    mutant = individual.copy()
    n_vars = len(individual)
    for j in range(n_vars):
        if np.random.rand() < mutation_rate:
            lo, hi = bounds[j]
            sigma = mutation_scale * (hi - lo)
            mutant[j] += np.random.normal(0, sigma)
            mutant[j] = np.clip(mutant[j], lo, hi)
    return mutant


# ===========================================================================
# SA 操作
# ===========================================================================

def sa_perturb(individual: np.ndarray,
               fitness: float,
               lens_template: LensSystem,
               bounds: list[tuple[float, float]],
               temperature: float,
               scale: float = 0.05) -> tuple[np.ndarray, float]:
    """
    模拟退火扰动：在当前解附近随机扰动，以概率 exp(-ΔE/T) 接受恶化解。

    参数:
        individual: 当前解
        fitness: 当前适应度
        temperature: 当前温度
        scale: 扰动幅度（相对于 bounds 范围）

    返回:
        (new_individual, new_fitness): 接受后的解和适应度
    """
    n_vars = len(individual)
    # 随机选一个变量做扰动
    j = np.random.randint(n_vars)
    lo, hi = bounds[j]
    sigma = scale * (hi - lo)

    candidate = individual.copy()
    candidate[j] += np.random.normal(0, sigma)
    candidate[j] = np.clip(candidate[j], lo, hi)

    candidate_fitness = evaluate_individual(lens_template, candidate)
    delta_e = candidate_fitness - fitness

    if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(temperature, 1e-10)):
        return candidate, candidate_fitness
    else:
        return individual, fitness


# ===========================================================================
# GA+SA 主优化
# ===========================================================================

def ga_sa_optimize(lens: LensSystem,
                   pop_size: int = 28,
                   ga_iter: int = 70,
                   sa_temp_init: float = 90.0,
                   sa_cooling: float = 0.95,
                   tournament_size: int = 3,
                   crossover_rate: float = 0.8,
                   mutation_rate: float = 0.3,
                   restart_sigma: Optional[float] = None,
                   verbose: bool = True) -> tuple[LensSystem, list[float]]:
    """
    GA+SA 混合全局优化。

    参数:
        lens:           初始镜头系统
        pop_size:       种群规模
        ga_iter:        GA 迭代代数
        sa_temp_init:   SA 初始温度
        sa_cooling:     温度衰减系数（每代乘以该值）
        tournament_size: 锦标赛选择大小
        crossover_rate: 交叉概率
        mutation_rate:  变异概率
        restart_sigma:  邻域重启标准差系数（相对 bounds 范围）；None=全局探索，
                        非 None=围绕当前点小邻域探索（DLS 卡住时的 GA 重启）
        verbose:        是否打印过程

    返回:
        (best_lens, merit_history): 最优镜头系统和每代最优 merit 历史
    """
    x0 = lens.variables_to_array()
    n_vars = len(x0)

    if n_vars == 0:
        if verbose:
            print("[GA+SA] 没有变量，跳过全局优化。")
        return copy.deepcopy(lens), [merit_scalar(lens)]

    bounds = lens.get_variable_bounds()

    # 初始化种群
    population = init_population(lens, pop_size, bounds, restart_sigma=restart_sigma)
    fitness = evaluate_population(lens, population)

    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    merit_history = [best_fitness]

    if verbose:
        print(f"[GA+SA] 种群: {pop_size}, 变量: {n_vars}, 代数: {ga_iter}")
        print(f"[GA+SA] 初始最优 merit: {best_fitness:.6e}")
        print("-" * 70)

    temperature = sa_temp_init

    for gen in range(1, ga_iter + 1):
        new_population = np.zeros_like(population)
        new_fitness = np.zeros(pop_size)

        # 精英保留：最好的个体直接进入下一代
        new_population[0] = best_x
        new_fitness[0] = best_fitness

        # 产生剩余个体
        for i in range(1, pop_size):
            # 选择
            parent1 = tournament_select(population, fitness, tournament_size)
            parent2 = tournament_select(population, fitness, tournament_size)

            # 交叉
            if np.random.rand() < crossover_rate:
                child1, child2 = arithmetic_crossover(parent1, parent2)
                child = child1 if np.random.rand() < 0.5 else child2
            else:
                child = parent1.copy()

            # 变异
            child = gaussian_mutate(child, bounds, mutation_rate)

            # 边界投影
            child = project_to_bounds(child, bounds)

            new_population[i] = child
            new_fitness[i] = evaluate_individual(lens, child)

        # SA 局部跳跃：对种群中 30% 的个体做 SA 扰动
        n_sa = max(1, int(pop_size * 0.3))
        sa_indices = np.random.choice(pop_size, n_sa, replace=False)
        for idx in sa_indices:
            new_population[idx], new_fitness[idx] = sa_perturb(
                new_population[idx], new_fitness[idx],
                lens, bounds, temperature, scale=0.05
            )

        # 更新种群
        population = new_population
        fitness = new_fitness

        # 更新最优
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_x = population[gen_best_idx].copy()
            best_fitness = fitness[gen_best_idx]

        merit_history.append(best_fitness)

        # 降温
        temperature *= sa_cooling

        if verbose and (gen % 5 == 0 or gen == 1):
            print(f"[GA+SA] gen {gen:3d} | best_merit={best_fitness:.6e} | "
                  f"gen_merit={fitness[gen_best_idx]:.6e} | "
                  f"T={temperature:.2f}")

    if verbose:
        print("-" * 70)
        print(f"[GA+SA] 全局优化完成。最优 merit: {best_fitness:.6e}")

    # 构造最优镜头系统
    best_lens = copy.deepcopy(lens)
    best_lens.array_to_variables(best_x)
    return best_lens, merit_history
