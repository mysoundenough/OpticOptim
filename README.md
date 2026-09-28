# 光学镜头优化 CLI（Lens Optimizer）

使用 Python 3.11+ + numpy + scipy + pydantic + matplotlib 实现。
不调用 Zemax / CodeV / COM / API / 网络，纯终端 CLI 运行。

## 功能特性

- **光线追迹**：球面折射、球面反射、高阶偶次非球面反射，支持折叠光路
- **非球面**：圆锥常数 k + 四阶~十阶偶次系数，支持作为优化变量
- **评价函数**：RMS 光斑、畸变、色差 + NA/入射角/出射角/工作距离约束惩罚
- **优化算法**：
  - DLS 阻尼最小二乘（局部，Levenberg-Marquardt 风格，数值雅可比+高斯-牛顿海森近似，变量归一化）
  - GA+SA 遗传算法+模拟退火（全局探索，输出候选解送入 DLS 精修）CMA-ES（协方差进化算法）
  - 基于人机协作智能体的全局优化器，（工作流：模糊需求 -> 可靠的光学设计）
- **可视化**：光斑图、系统布局图、MTF 曲线、光线扇图、收敛曲线
- **优化过程动画**：每步 DLS 迭代保存镜头布局快照，自动合成 GIF 动画 + 迭代对比网格图
- **多波长多视场**：支持波长列表和视场角列表
- **JSON 读写**：镜头参数、优化目标、约束全部通过 JSON 配置

## 目录结构

```
lens_optimizer/
├── main.py                  # CLI 入口（--local / --global / --staged 三种模式）
├── lens_schema.py           # Pydantic 数据模型（surface/aspheric/constraints）
├── raytrace.py              # 光线追迹引擎（折射/反射/非球面求交，含子午/主光线快速模式）
├── merit_function.py        # 评价函数（像差项+约束惩罚，12 维残差）
├── dls_optimizer.py         # DLS 阻尼最小二乘（Marquardt 阻尼 + 鲁棒差分 + 鞍点逃逸）
├── ga_sa_optimizer.py       # GA+SA 混合全局优化（含邻域重启）
├── staged_optimizer.py      # ★ 分阶段优化器（GA结构搜索→逐级解锁高阶项→约束验证）
├── find_reflect_initial.py  # ★ 找反射初始结构（理想反射面主光线路径设计）
├── build_axisym.py          # ★ 共轴折返系统构造（主光线反射点→共轴镜面方程，decenter=0）
├── find_offaxis_structure.py # 离轴结构搜索（GA + DLS 两阶段）
├── visualization.py         # 可视化模块（光斑/布局/MTF/光线扇/收敛/优化动画）
├── examples/
│   ├── euv_6mirror_demo.json        # EUV 6镜全非球面反射物镜（13.5nm, NA=0.33）
│   ├── euv_6mirror_24param_demo.json # ★ 24参数(c+r+k+d)×6 回转对称离轴6镜（离轴入射15°，矩形视场）
│   ├── example_mixed_aspheric.json  # 3镜混合球面/非球面反射演示（可见光）
│   ├── cooke_triplet_demo.json      # Cooke Triplet 三分离折射物镜（f/4, 3波长消色差）
│   └── offaxis_6mirror_decenter_demo.json  # 离轴 6 镜（d/r/偏心，分阶段优化演示）
└── output/                  # 优化输出目录
```

## 安装依赖

```bash
# Python 3.11+（matplotlib 自带 Pillow，用于 GIF 合成）
pip install numpy scipy pydantic matplotlib
```

## 运行命令

```bash
cd lens_optimizer

# 1. 仅 DLS 局部优化（默认模式，也可显式加 --local）
python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json
python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --local

# 2. 全局优化（GA+SA）+ DLS 精修
python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --global

# 3. 生成最终可视化图表
python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --viz

# 4. 优化过程动画（每步迭代保存布局快照，合成GIF+对比网格图）
python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --animate

# 5. 全局优化 + 动画 + 最终可视化（全功能）
python main.py --input examples/euv_6mirror_demo.json --output output/optimized.json --global --animate --viz

# 6. 使用混合非球面示例
python main.py --input examples/example_mixed_aspheric.json --output output/mixed_opt.json --local --animate --viz

# 7. 分阶段优化（推荐用于离轴反射系统）
#    阶段0: GA 只搜 d/r 12 参数 + 子午光线追迹找不遮挡个体
#    阶段1-5: 逐级解锁 k→A4→A6→A8→A10，每级 DLS + merit 同伦（结构项→光斑）
#    收尾: 光斑权重归零验证约束保持
python main.py --input examples/offaxis_6mirror_decenter_demo.json \
    --output output/offaxis_6mirror_staged.json --staged

# 8. 分阶段优化独立脚本（可调 GA/DLS 预算）
python staged_optimizer.py --input examples/offaxis_6mirror_decenter_demo.json \
    --output output/offaxis_6mirror_staged.json \
    --pop-size 20 --ga-iter 25 --dls-iter 30
```

> **注意**：`--local`、`--global`、`--staged` 互斥，不能同时指定。都不指定时默认为 `--local`（仅 DLS 局部优化）。

## CLI 参数

| 参数 | 说明 |
|---|---|
| `--input` | 输入镜头 JSON 文件（必需） |
| `--output` | 输出优化后 JSON 文件（必需） |
| `--local` | 仅 DLS 局部优化（默认模式，与 `--global` 互斥） |
| `--global` | 全局优化（GA+SA）+ DLS 精修，与 `--local` 互斥 |
| `--viz` | 生成最终可视化图表（光斑/布局/MTF/光线扇/收敛曲线） |
| `--animate` | 优化过程动画：每步 DLS 迭代保存镜头布局快照，合成 GIF + 迭代对比网格图 |
| `--viz-dir` | 可视化输出目录（默认 `<output_dir>/figures`） |

## 非球面方程

偶次非球面矢高（sag）：

```
z(r) = c·r² / [1 + sqrt(1 - (1+k)·c²·r²)] + Σ A_{2m}·r^{2m}   (m=2..7)
```

其中 `c = curvature = 1/R`，`k` 为圆锥常数：
- k = 0：球面
- k = -1：抛物面
- k < -1：双曲面
- -1 < k < 0：椭球面

### 高阶项可配置（high_order_terms）

高阶偶次项数量可配置，默认 **2 项**（A4, A6），最多 **6 项**（A4 ~ A14）：

| `aspheric.high_order_terms` | 参与追迹的高阶项 |
|---|---|
| 2（默认） | A4, A6 |
| 4 | A4, A6, A8, A10 |
| 6 | A4, A6, A8, A10, A12, A14 |

- 超出 `high_order_terms` 的系数在追迹/优化中被忽略（`Aspheric.active_coeffs()` 只返回激活项）
- 优化变量由 `aspheric_variable` 单独控制（每项一个 bool），与 `high_order_terms` 解耦
- 全链路支持：sag/法线（raytrace.py）、雅可比差分（dls_optimizer.py）、绘制（visualization.py / plot_structure.py）均按 6 项通用实现

### 24 参数优化集（crkd × 6 镜子）

离轴 6 镜 EUV 系统的基础优化变量集为 **24 个**：每面镜子 4 个参数 × 6 镜：

| 参数 | 含义 | 变量开关 |
|---|---|---|
| `c`（curvature） | 曲率 = 1/R | `variable: true` |
| `r`（semi_aperture） | **半口径**（镜子物理半径，非曲率半径 R） | `size_variable: true` |
| `k`（conic） | 圆锥常数 | `aspheric_variable.k: true` |
| `d`（thickness） | 到下一面的间距（位置） | `variable: true` |

```
24 = (c + r + k + d) × 6 镜
```

需要更多自由度时，在 `aspheric_variable` 中打开 A4/A6（36 参数）、A8/A10（48 参数）等即可。
演示文件：`examples/euv_6mirror_24param_demo.json`。

## 反射光路符号处理

- 光轴为 Z 轴，光线初始沿 +Z 传播
- 遇到反射面后，方向 Z 分量取反
- 表面顶点 Z 坐标预计算：反射面**先翻转传播方向**，再用新方向计算下一面位置
  `z[i+1] = z[i] + sign · thickness[i]`，其中 sign 在反射后取反
- 非球面交点用牛顿迭代求解，初始猜测取平面交点

## DLS 变量归一化

光学变量量级差异极大（曲率 ~1e-3、厚度 ~100、非球面系数 ~1e-12，跨 13 个数量级），
直接求解法方程会因雅可比条件数极差而无法收敛。

解决方案：每个变量除以其特征尺度 `scale_j`，在归一化空间 `x_norm = D⁻¹x` 中求解法方程，
再转换回原始空间 `δx = D·δx_norm`。

**高阶非球面系数的特殊处理（维度灾难修复）**：
- 若按普通变量用 `eps·scale` 做差分，A6 在 r=50mm 处扰动 2e-13 就产生 9.4m 的 sag 变化，
  光线全部追不到像面，雅可比爆炸（数值 1e8），DLS 在任何阻尼下都找不到下降步长。
- 修复：A4-A10 的差分步长按 sag 影响缩放 `step = 1e-3 / r_max^order`，
  保证扰动产生 ~1e-3mm 的镜面 sag 变化（可测量且光滑）。

## DLS 三大增强（dls_optimizer.py）

1. **Marquardt 阻尼**：法方程 `(JᵀWJ + λ·diag(JᵀWJ))δ = -JᵀWr`，每个方向按自身曲率缩放阻尼，
   避免高维空间条件数病态导致法方程数值爆炸。
2. **鲁棒数值差分**：若扰动后出现"光线断裂跳变"（遮挡/缺失比例突变、RMS 跳到 lost_penalty），
   自动缩小步长 ×10 重试，使雅可比落在光滑区域。
3. **鞍点逃逸**：法方程在所有阻尼下都找不到下降步长时（梯度≈0 的鞍点/平台），依次尝试：
   - 归一化梯度下降（小步长线搜索）
   - 残差维度下降：对贡献最大的残差沿其负梯度搜索（评价函数每个维度——光斑/NA/离轴角/遮挡——
     都独立提供一个下降方向，即使全向量梯度≈0，单个维度仍可能有下降方向）
   - 随机扰动逃逸（高斯小扰动，接受更优点）

## 找反射初始结构（find_reflect_initial.py）

理想反射面主光线路径设计——在优化真实镜子之前，先用**中心视场点的中心光线**
设计一条理想光路（无实体镜子，只有反射平面）：

- 光线按约束从物面出发（主光线入射角固定 15°），依次经 6 个反射平面到达像面
- 每个反射平面：位置（反射点）+ 放置角度（法向量由反射定律从入射/出射方向确定）
- 像方远心：M6 反射后主光线平行光轴（+z）
- 反射面入射角 ≤20°（EUV 镀膜）、镜子 z 全正
- 可选目标：光线段夹角越大越好（布局更分散）

输出：6 个反射点的位置、法向量、平面角度、入射角
（`output/initial_path/initial_chief_path_*.json`）+ 路径图。

```bash
python find_reflect_initial.py            # 默认设计（远心）并打印/绘图
python find_reflect_initial.py --z1 400 --z6 3000   # 指定反射点 z 范围
```

```python
from find_reflect_initial import build_path, print_path, plot_path
path = build_path(pts_yz, telecentric_image=True)   # 构建路径（含法向量/入射角）
print_path(path)                                    # 打印反射面设计
plot_path(path, 'output/initial_path/reflect.png')  # 画图
```

设计结果可作为真实凹面镜/非球面镜的初始结构（镜面顶点位置 = 反射点，
镜面放置角度 = 平面角度）。

## 共轴折返系统构造（build_axisym.py）

由**主光线反射点（锚点）** 直接确定 6 镜**共轴（回转对称）折返系统**的
镜面方程——这是真实 EUV 物镜的形态：

- 6 镜**共轴堆叠**（decenter=0，回转对称），主光线走 **Z 形折返光路**（z 交替）
- 镜面形态：大凹面镜（M1/M3/M5，R 大）+ 强弯小镜（M2/M6，R 小）
- **环形孔径**避遮挡（孔径光线打在镜面离轴环带）

### 镜面方程（k=-1 抛物面，锚点处法线严格对齐设计法线）

```python
ny_raw = sign(nz)·ny/|nz|        # 镜面法线原始 y 分量（取 nz>0 版本）
c      = -ny_raw / y_anchor      # 曲率：c·y = 法线斜率，法线角精确 = 设计角
z_v    = z_anchor - c·y²/2       # 顶点 z：镜面精确过锚点（误差 < 1e-3 mm）
thickness = |z_v 差|             # 全正：折返布局（追迹时反射翻转自动 z 交替）
```

### 命令行用法

```bash
# 由理想路径的锚点（反射点）构造共轴镜面 + 保存 LensSystem + 主光线验证
python build_axisym.py \
    --path output/initial_path/initial_chief_path_all_seed27.json \
    --out  examples/offaxis_6mirror_axisym_seed27.json
```

输出：6 镜曲率/顶点 z 打印 + 主光线验证（6 锚点误差、像面 y、远心）+ JSON 文件。

### Python API

```python
from build_axisym import build_axisym_mirrors, build_axisym_system, verify_chief_ray

# 1) 锚点 (6,2) (y,z) → 6 镜参数（decenter=0, c, k, z_vertex）
mirrors = build_axisym_mirrors(anchors_yz, normals=None)   # normals=None 自动重算

# 2) 完整 LensSystem dict（可直接 json 保存）
sys_dict = build_axisym_system(anchors_yz, normals, name="offaxis_6mirror_axisym")

# 3) 主光线验证：6 锚点全过 + 像面 y=9.2 + 远心
from lens_schema import LensSystem
lens = LensSystem.from_json("examples/offaxis_6mirror_axisym_seed27.json")
v = verify_chief_ray(lens, anchors_yz)   # max_anchor_err / image_y / telecentric_ok
```

### 已验证布局

| 布局 | M1 | M2 | M3 | M4 | M5 | M6 | 验证 |
|---|---|---|---|---|---|---|---|
| seed27 | 凹 827mm | 凸 289mm | 凹 1070mm | 凸 2274mm | 凹 4005mm | 强弯 45mm | 6 锚点 0.000mm，像 y=9.211，远心 OK |
| seed14 | 凹 1478mm | 强弯 51mm | 凹 1653mm | 凸 2448mm | 凹 2249mm | 强弯 38mm | 6 锚点 0.000mm，像 y=9.199，远心 OK |

> 注意：`find_reflect_initial.py` 的自由离轴布局（每镜独立 decenter）与
> `build_axisym.py` 的共轴布局（decenter=0）是两条路线：前者镜子位置自由、
> 遮挡余量大；后者是真实 EUV 物镜形态（共轴 + 环形孔径）。

## 三阶段优化流水线（optimize_pipeline.py）

面向离轴反射 / EUV 物镜的完整优化流程，三个阶段可独立调用：

| 阶段 | 功能 | 说明 |
|---|---|---|
| 1 | `ga_structure_search()` | GA 按约束遍历初始结构，找候选起点（默认 24 参数：c+d+半口径+k）×6 镜 |
| 2 | `dls_find_unobscured()` | **DLS 直接找不遮挡**：遮挡判定含连续穿透深度梯度，DLS 沿梯度把光线"挤出"镜子实体（实测 depth 0.99→0.000，无需 GA） |
| 3 | `dls_refine()` | 从不遮挡起点解锁全变量（含高阶非球面 A4~A14），逐步引入 NA/角度/像方远心/放大率/RMS 约束（保持不遮挡） |

**可视化与过程保存**：
- 优化过程中保存各阶段 / 每轮结构 JSON（`save_intermediate`、`realtime`）
- 实时可视化：GA 每 N 代、DLS 每轮自动保存结构 + 画布局快照（`--realtime --viz-every 10`）
- 完成后自动调用 `plot_structure.py` 对三个阶段结构出全套图（3D + x/y/z 三视角 + 每镜单独）

```bash
# 完整流水线（GA → DLS 不遮挡 → DLS 精修 + 可视化）
python optimize_pipeline.py --input examples/offaxis_6mirror_axisym_demo.json \
    --outdir output/pipeline_result

# 实时可视化（GA 每 10 代、DLS 每轮保存快照）
python optimize_pipeline.py --input ... --outdir ... --realtime --viz-every 10

# 只跑某阶段（代码调用）
from optimize_pipeline import ga_structure_search, dls_find_unobscured, dls_refine
from lens_schema import LensSystem
lens = LensSystem.from_json('examples/xxx.json')
lens = ga_structure_search(lens, pop_size=36, ga_iter=200)      # 阶段1
lens = dls_find_unobscured(lens)                                 # 阶段2（DLS 找不遮挡）
lens = dls_refine(lens)                                          # 阶段3（精修）
```

输出目录结构：
```
output/pipeline_result/
├── ga_structure.json / dls_unobscured.json / final_refined.json   # 三阶段结构
├── realtime/          # 实时快照（ga_gen_XXXX.json/png, dls_round_XX.json）
└── figures/           # 完成后可视化
    ├── ga / unobscured / final
    │   ├── full/      # 整体 3D + x/y/z 三视角 + 合成
    │   └── per_mirror/ # 每镜单独 4 视角（24 张）
```

## 分阶段优化（staged_optimizer.py）

针对离轴反射系统（EUV 类）的专用优化策略：

| 阶段 | 内容 | 目的 |
|---|---|---|
| 0 | GA 只搜 d/r 共 12 参数，子午/主光线追迹（6~1 条光线/评估） | 全局找不遮挡个体（光斑权重=0，避免悬崖型目标掩盖光滑的结构下降方向） |
| 1..N | 逐级解锁 k→A4→A6→A8→A10，每级 DLS + merit 同伦：结构-NA优先 → 结构-全约束 → 光斑 0.1→0.5→1.0 | 新自由度先拉结构回可行域，再逐步引入光斑，避免维度灾难 |
| 收尾 | 光斑权重归零再跑一次 DLS | 验证约束（NA/角度/遮挡/工作距离）独立锁住像质，光斑不反弹 |

**卡住恢复（总有下降方向）**：
1. DLS 法方程无下降步 → 内置鞍点逃逸（梯度/残差维度/随机扰动）
2. 仍卡住 → 更小初始阻尼重试 + 随机扰动重启
3. 仍卡住 → 邻域 GA 重启（围绕当前点小邻域搜索），注入 GA 找到的规则

**实测效果**（`examples/offaxis_6mirror_decenter_demo.json`）：
RMS 6.96 → 0.63 mm（降 91%），NA 0.264 → 0.3300（精确命中），全部入射角 ≤ 20°，
18/18 光线有效；光斑权重归零后约束 merit = 1.8e-12（约束独自锁住像质）。

> 注意：评价函数中的约束必须物理自洽。如 `na_target=0.33` 要求出射锥半角
> asin(0.33)=19.3°，若 `exit_angle_max_deg=12°`（对应 NA 上限 0.208）则两者矛盾，
> 优化器只能退到 Pareto 前沿。demo 已改为 22°。

## 评价函数残差项

| 索引 | 名称 | 说明 |
|---|---|---|
| 0 | rms_spot | RMS 光斑半径 (mm) |
| 1 | distortion | 畸变（绝对偏差 mm） |
| 2 | chroma | 色差 (mm)，反射系统通常为 0 |
| 3 | na_penalty | NA 偏差惩罚 |
| 4 | angle_incident | 入射角超限惩罚 (deg) |
| 5 | angle_exit | 出射角超限惩罚 (deg) |
| 6 | working_distance | 工作距离惩罚 (mm) |
| 7 | min_thickness | 最小中心厚度违反 (mm) |
| 8 | min_air_gap | 最小空气间隔违反 (mm) |
| 9 | total_length | 系统总长超限 (mm) |

`merit = Σ w_i · r_i²`

## 镜面间距约束

在 `metadata.spacing_constraints` 中配置，作为评价函数的软惩罚项（与硬边界 bounds 互补）：

| 字段 | 说明 |
|---|---|
| `min_center_thickness` | 最小镜片中心厚度 (mm)，防止镜片太薄无法加工。仅检查 glass != air 的面 |
| `min_air_gap` | 最小空气间隔 (mm)，防止相邻镜片机械干涉。跳过物距和后截距（工作距离） |
| `max_total_length` | 最大系统总长 (mm)，0 表示不约束。从物面到像面的轴向总距离 |

约束违反量作为残差项并入评价函数，可在 `optimization_config.merit_weights` 中配置权重：
- `min_thickness_penalty`（默认 5.0）
- `min_air_gap_penalty`（默认 5.0）
- `total_length_penalty`（默认 0.5）

## EUV 6 镜系统说明

`examples/euv_6mirror_demo.json` 为 EUV 光刻物镜演示：
- 波长 13.5 nm（EUV）
- 6 片全非球面反射镜，NA = 0.33
- 全反射折反射系统，无色差
- 第 3 面为孔径光阑
- 非球面系数 k/A4/A6 标记为优化变量，A8/A10 固定
- 约束：入射角 ≤ 12°，出射角 ≤ 7.5°，工作距离 ≥ 25 mm

> 注意：该 demo 为演示初始结构，非实际生产级 EUV 物镜。优化结果取决于初始点和变量设置。

## Cooke Triplet 三分离折射物镜说明

`examples/cooke_triplet_demo.json` 为经典 Cooke Triplet 摄影镜头演示：
- 焦距约 50mm，f/4，半视场 10°
- 三片分离薄透镜：第一片 BK7 正透镜 → 光阑 → 第二片 F2 负透镜 → 第三片 BK7 正透镜
- 三个波长：F光(486.1nm)、d光(587.6nm)、C光(656.3nm)，用于消色差设计
- 两种玻璃：BK7（冕牌，nd=1.5168, Vd=64.17）+ F2（火石，nd=1.6200, Vd=36.37）
- 12 个优化变量：6 个折射面的曲率 + 6 个厚度（含后截距）
- 孔径类型：`diameter`（直接指定入瞳直径 12.5mm）

> **注意**：当前玻璃模型仅使用单一 nd 值（d光折射率），未实现 Sellmeier 色散公式，因此三个波长的折射率相同，色差项始终为 0。消色差效果需在引入色散公式后才能体现。

## 可视化输出

### 最终图表（`--viz`，输出到 `output/figures/`）

- `final_spot_diagram.png` — 光斑图（按视场分色，含 RMS 圆）
- `final_layout.png` — 系统二维布局图（镜面剖面+追迹光线）
- `final_mtf.png` — MTF 曲线（PSF 傅里叶几何近似）
- `final_ray_fan.png` — 光线扇图（X/Y 横向像差 vs 归一化入瞳）
- `final_convergence.png` — 优化收敛曲线（对数坐标）

### 优化过程动画（`--animate`，输出到 `output/animation/`）

```
animation/
├── iter_0000.png          # 初始状态快照（第0帧）
├── iter_0001.png          # 第1步DLS迭代快照
├── ...
├── iter_NNNN.png          # 第N步迭代快照
├── optimization.gif       # 合成的GIF动画（3fps，无限循环）
└── optimization_grid.png  # 关键帧对比网格图（均匀采样≤12帧）
```

**动画技术要点：**

- **固定坐标系**：所有快照使用基于初始系统计算的统一坐标范围，动画播放时镜面的移动和形变清晰可见，不会因自动缩放而抖动
- **快照内容**：每帧显示镜面剖面、8条代表性追迹光线、迭代号、merit值、实际NA、最大入射角
- **GIF合成**：使用 Pillow（matplotlib 依赖），无需 ffmpeg，完全离线
- **网格图**：迭代步数超过12步时均匀采样12个关键帧，放在一张网格图中对比镜面形状随优化的演变
- **性能影响**：单帧生成约0.3-0.5秒，对优化总耗时影响很小

### 可视化模块 API（`visualization.py`）

| 函数/类 | 用途 |
|---|---|
| `plot_spot_diagram()` | 光斑图 |
| `plot_layout()` | 系统布局图（完整） |
| `plot_mtf()` | MTF曲线 |
| `plot_ray_fan()` | 光线扇图 |
| `plot_convergence()` | 收敛曲线 |
| `plot_layout_snapshot()` | 快速布局快照（动画用） |
| `compute_fixed_limits()` | 计算固定坐标范围 |
| `create_optimization_gif()` | 快照序列合成GIF |
| `plot_optimization_grid()` | 迭代对比网格图 |
| `OptimizationVisualizer` | 动画管理器（capture/finalize） |
| `generate_report()` | 一键生成全部最终图表 |
