# 项目目标

我要开发一个自主的 Optical Design / Lens Optimization Engine。

目标是基于公开的光学设计理论和数值优化算法，自主实现一个能够在工程上承担大量传统商业光学设计软件优化工作的光学优化内核。

重点研究和实现：

* Sequential Optical System
* Ray Tracing
* Aberration Analysis
* Merit Function
* Sensitivity / Jacobian
* Damped Least Squares
* Levenberg-Marquardt
* SVD
* Variable Scaling
* Constraint Handling
* Step Optimization
* Trust Region
* Local Optimization
* Global / Multi-start Optimization

目标能力是：

```text
Lens Definition
      ↓
Optical Ray Trace
      ↓
Image / Wavefront / Aberration
      ↓
Merit Function
      ↓
Jacobian / Sensitivity
      ↓
SVD-DLS
      ↓
Step Control
      ↓
Constraint Handling
      ↓
Trial Lens
      ↓
Ray Trace
      ↓
Accept / Reject
      ↓
Adaptive Damping
      ↓
Convergence
```

这是一个独立的 Optical Lens Design Optimization Engine。

---

# 1. 开发原则

不要尝试：

* 逆向工程 CODE V 私有源码
* 破解 CODE V license
* 绕过授权
* 提取 CODE V 二进制中的私有算法
* 声称某个实现就是 CODE V 内部实现

而是：

根据公开的：

* Optical Design
* Numerical Optimization
* Damped Least Squares
* Levenberg-Marquardt
* SVD
* Ray Tracing
* Aberration Theory

自主实现。

我们研究的是：

> 为什么工业级光学优化器能够快速、稳定、准确地优化镜头。

---

# 2. 软件总体架构

设计为：

```text
Optical Design Engine
│
├── Optical Model
│   ├── Surface
│   ├── Material
│   ├── Coordinate System
│   ├── Aperture
│   ├── Stop
│   ├── Object
│   └── Image Plane
│
├── Ray Tracing
│   ├── Paraxial Ray Trace
│   ├── Real Ray Trace
│   ├── Refraction
│   ├── Reflection
│   ├── Coordinate Transform
│   └── Vignetting
│
├── Optical Analysis
│   ├── RMS Spot
│   ├── Wavefront
│   ├── OPD
│   ├── MTF
│   ├── PSF
│   ├── Distortion
│   ├── Field Curvature
│   ├── Chromatic Aberration
│   └── Zernike
│
├── Merit Function
│   ├── Operand
│   ├── Target
│   ├── Weight
│   ├── Normalization
│   └── Constraint
│
├── Optimization
│   ├── Gradient
│   ├── Gauss Newton
│   ├── DLS
│   ├── Levenberg Marquardt
│   ├── SVD
│   ├── Scaling
│   ├── Step Control
│   ├── Trust Region
│   └── Constraint Solver
│
├── Global Optimization
│   ├── Multi Start
│   ├── Random Perturbation
│   ├── Simulated Annealing
│   └── Evolutionary Search
│
└── Diagnostics
    ├── Convergence
    ├── Condition Number
    ├── Singular Values
    ├── Parameter Sensitivity
    └── Optimization History
```

---

# 3. 第一阶段：建立 Sequential Optical Model

首先实现一个最小 Sequential Optical System。

例如：

```python
system = OpticalSystem()

system.add_surface(
    radius=100.0,
    thickness=5.0,
    material="N-BK7"
)

system.add_surface(
    radius=-100.0,
    thickness=95.0,
    material="AIR"
)

system.set_aperture(...)
system.set_wavelength(...)
system.set_field(...)
```

每个 Surface 至少支持：

```text
radius
thickness
material
aperture
conic
asphere coefficients
coordinate transform
```

设计：

```text
OpticalSystem
Surface
Material
Wavelength
Field
Ray
RayTraceResult
```

---

# 4. 第二阶段：Real Ray Trace

实现真正的 Sequential Ray Trace。

核心流程：

```text
Object
 ↓
Surface Intersection
 ↓
Surface Normal
 ↓
Snell Law
 ↓
Next Surface
 ↓
...
 ↓
Image Plane
```

必须支持：

* spherical surface
* planar surface
* refractive surface
* reflective surface
* finite object
* infinity object
* multiple wavelengths
* multiple fields

第一阶段不要追求所有复杂 surface。

先完成：

1. Plane
2. Spherical
3. Aspheric

---

# 5. Ray-Surface Intersection

球面：

$$
x^2+y^2+(z-R)^2=R^2
$$

求 ray：

$$
P=P_0+tD
$$

与 surface 的交点。

必须考虑：

* no intersection
* tangent
* wrong intersection
* aperture clipping

返回：

```python
IntersectionResult(
    valid=True,
    point=...,
    normal=...,
    distance=...
)
```

---

# 6. Refraction

实现 Snell Law。

输入：

```text
incident direction
surface normal
n1
n2
```

输出：

```text
refracted direction
```

必须处理：

* normal orientation
* total internal reflection
* sign convention

不要把符号约定散落在代码中。

建立：

```python
OpticalConvention
```

统一管理坐标和符号。

---

# 7. Paraxial Ray Trace

同时实现：

ParaxialRayTracer

用于：

* EFL
* BFL
* principal plane
* paraxial image
* magnification

不要所有东西都通过 real ray trace 计算。

---

# 8. Field / Wavelength

支持：

```text
Field:
    x
    y
    weight

Wavelength:
    wavelength
    weight
```

例如：

```python
fields = [
    Field(0, 0),
    Field(0, 0.7),
    Field(0, 1.0),
]

wavelengths = [
    486.1,
    587.6,
    656.3,
]
```

后续 Merit Function 可以对：

field × wavelength

进行组合。

---

# 9. Merit Function

建立通用：

```python
MeritFunction
Operand
```

不要把 MTF、RMS、EFL 等直接写死。

Operand 结构：

```python
Operand(
    type="RMS_SPOT",
    field=1,
    wavelength=1,
    target=0.0,
    weight=1.0
)
```

支持第一批 Operand：

```text
RMS_SPOT
RMS_WAVEFRONT
OPD
MTF
EFL
BFL
DISTORTION
MAGNIFICATION
CHIEF_RAY
FOCAL_SHIFT
IMAGE_HEIGHT
```

最终：

$$
MF =
\sum_i w_i(r_i-t_i)^2
$$

---

# 10. Merit Function 必须支持 normalization

不同物理量的数量级差异非常大。

例如：

```text
EFL        ~ 100
RMS spot   ~ 0.01
wavefront  ~ 0.001
distortion ~ 0.01
```

不能直接混合。

设计：

$$
r_i =
\frac{measurement_i-target_i}{scale_i}
$$

然后：

$$
MF=\sum_i w_i r_i^2
$$

实现：

```python
Operand(
    target=...,
    weight=...,
    scale=...
)
```

---

# 11. Optimization Variable

所有可优化参数必须统一抽象：

```python
Variable
```

例如：

```text
surface_1.radius
surface_1.thickness
surface_2.radius
surface_2.thickness
surface_3.conic
...
```

支持：

```text
value
lower
upper
scale
enabled
```

例如：

```python
Variable(
    name="S1_RADIUS",
    value=100.0,
    lower=10.0,
    upper=1000.0,
    scale=100.0
)
```

---

# 12. Jacobian

核心：

$$
J_{ij}
=
\frac{\partial r_i}{\partial x_j}
$$

第一阶段允许使用 finite difference。

例如：

$$
J_{ij}
=
\frac{r_i(x+\delta x_j)-r_i(x)}{\delta x_j}
$$

但必须设计：

```python
JacobianCalculator
```

支持：

```text
forward difference
central difference
adaptive difference
```

不要把 finite difference 写死。

---

# 13. Adaptive Derivative Step

这是工业级优化非常重要的一部分。

不能所有变量：

```python
dx = 1e-6
```

必须考虑变量尺度：

$$
\delta x_i
=
\epsilon \max(|x_i|,scale_i)
$$

支持：

```text
absolute_step
relative_step
minimum_step
maximum_step
```

---

# 14. Variable Scaling

必须实现：

```python
VariableScaler
```

例如：

```text
radius
thickness
conic
asphere
```

量纲和数量级完全不同。

优化空间使用：

$$
x_s=Sx
$$

Jacobian：

$$
J_s=J S^{-1}
$$

最终：

$$
\Delta x=S^{-1}\Delta x_s
$$

---

# 15. DLS

核心实现：

$$
(J^TWJ+\lambda D)\Delta x
=
-J^TWr
$$

但是：

## 禁止

```python
np.linalg.inv(J.T @ J + lambda * D)
```

必须使用稳定分解。

优先：

```text
SVD
QR
Cholesky
```

---

# 16. SVD-DLS

实现：

$$
J=U\Sigma V^T
$$

并通过：

$$
\Delta x
=
-V
diag
\left(
\frac{\sigma_i}
{\sigma_i^2+\lambda d_i}
\right)
U^Tr
$$

求解。

输出：

```text
singular_values
rank
condition_number
effective_rank
```

---

# 17. 为什么必须 SVD

光学优化 Jacobian 经常高度相关。

例如：

```text
R1
R2
Thickness1
Thickness2
```

可能产生高度相似的 aberration response。

因此：

$$
J
$$

可能病态。

必须检测：

$$
\kappa(J)
=
\frac{\sigma_{max}}
{\sigma_{min}}
$$

并识别：

```text
weak directions
```

---

# 18. Adaptive Damping

实现：

```python
DampingPolicy
```

核心：

$$
\rho =
\frac{actual\ reduction}
{predicted\ reduction}
$$

如果：

```text
rho 很好
```

降低：

```text
lambda
```

如果：

```text
rho 很差
```

增加：

```text
lambda
```

参数全部配置化。

---

# 19. Trust Region / Step Optimization

DLS 得到：

```text
dx
```

不能无条件使用。

建立：

```python
StepController
```

控制：

```text
maximum step
trust radius
variable limits
relative step
```

如果：

```text
||dx|| > trust_radius
```

缩放：

$$
\Delta x_{new}
=
\alpha\Delta x
$$

其中：

$$
0<\alpha<1
$$

---

# 20. Trial Lens

优化不能：

```text
calculate dx
↓
直接修改 lens
```

必须：

```text
current lens
↓
calculate dx
↓
copy lens
↓
apply dx
↓
ray trace
↓
calculate new MF
↓
accept / reject
```

设计：

```python
TrialDesign
```

支持：

```python
trial = system.copy()
trial.apply_delta(dx)
```

---

# 21. Accept / Reject

如果：

```text
new_MF < old_MF
```

accept。

否则：

```text
reject
increase lambda
reduce step
retry
```

优化器必须拥有明确状态：

```text
ACCEPT
REJECT
RETRY
CONVERGED
FAILED
```

---

# 22. Constraint

第一阶段：

```text
variable lower bound
variable upper bound
```

例如：

```text
radius > 0
thickness > 0
air gap > minimum
```

不要简单：

```python
x = np.clip(x, lower, upper)
```

因为这可能破坏 DLS 的搜索方向。

优先采用：

```text
step scaling
active set
```

架构上为：

```python
ConstraintHandler
```

以后扩展：

```text
equality constraint
inequality constraint
Lagrange multiplier
```

---

# 23. Optimization Loop

最终实现：

```python
optimizer.optimize(system)
```

内部：

for iteration:

```
trace current system

evaluate merit function

calculate residual

calculate jacobian

scale variables

SVD

calculate DLS step

constraint processing

step control

create trial lens

ray trace trial lens

calculate trial merit

predicted reduction

actual reduction

rho

accept / reject

update lambda

check convergence
```

---

# 24. Convergence

不能只检查：

```text
MF 是否下降
```

至少检查：

```text
cost
gradient_norm
step_norm
relative_cost_change
```

例如：

$$
||g|| < \epsilon_g
$$

$$
||\Delta x|| < \epsilon_x
$$

$$
\frac{|MF_{new}-MF_{old}|}
{\max(1,MF_{old})}
<\epsilon_f
$$

---

# 25. Optical Analysis

优化器最终必须能够优化真实 optical quantities。

逐步实现：

## 第一阶段

* paraxial focus
* RMS spot
* centroid
* ray aberration

## 第二阶段

* wavefront
* OPD
* Zernike

## 第三阶段

* PSF
* MTF

## 第四阶段

* distortion
* field curvature
* chromatic aberration

---

# 26. Zernike

实现：

```python
ZernikeAnalyzer
```

输入：

```text
pupil coordinates
wavefront / OPD
```

输出：

```text
Z1
Z2
...
Zn
```

必须明确：

```text
Noll
ANSI
Fringe
```

等编号体系。

不要默认它们相同。

---

# 27. MTF

最终实现：

```text
Wavefront
↓
Complex Pupil Function
↓
PSF
↓
OTF
↓
MTF
```

基本关系：

$$
P(x,y)
=
A(x,y)
e^{i2\pi W(x,y)/\lambda}
$$

然后：

$$
PSF
=
|\mathcal{F}\{P\}|^2
$$

以及：

$$
OTF=\mathcal{F}\{PSF\}
$$

$$
MTF=|OTF|
$$

---

# 28. Optimization Performance

注意：

Real Ray Trace 很昂贵。

因此必须考虑：

```text
cache
parallel field
parallel wavelength
vectorized ray trace
Jacobian parallelization
```

如果 Jacobian：

```text
N residual × M variable
```

可以并行计算不同 variable perturbation。

---

# 29. Optimization Diagnostics

每次 iteration 输出：

```text
iteration
MF
gradient norm
step norm
lambda
trust radius
rho
rank
condition number
accepted/rejected
```

同时记录：

```text
variable values
variable changes
singular values
operand values
```

可以输出：

```text
optimization_history.csv
```

---

# 30. Visualization

实现：

```text
Merit Function vs Iteration

RMS Spot vs Iteration

MTF vs Iteration

Wavefront RMS vs Iteration

Lambda vs Iteration

Step Size vs Iteration

Singular Values

Variable Changes
```

---

# 31. Benchmark

建立标准 optical optimization benchmark。

至少包括：

### Benchmark 1

简单 singlet。

目标：

优化 radius / thickness，使 RMS spot 最小。

### Benchmark 2

Doublet。

优化：

```text
radius
thickness
glass
```

降低 chromatic aberration。

### Benchmark 3

Triplet。

同时优化：

```text
RMS spot
EFL
BFL
```

### Benchmark 4

复杂多变量系统。

验证：

```text
SVD
Scaling
DLS
Adaptive lambda
```

---

# 32. 必须做 Algorithm Ablation

这是非常重要的。

对同一个镜头分别运行：

```text
Gradient Descent

Gauss Newton

DLS

SVD-DLS

SVD-DLS + Scaling

SVD-DLS + Scaling + Adaptive Lambda

SVD-DLS + Scaling + Adaptive Lambda + Trust Region
```

比较：

```text
iterations
ray trace count
final MF
convergence rate
failed steps
condition number
```

最终回答：

> 为什么工业级 DLS 比简单 Gradient Descent 快？

以及：

> 哪一个模块真正贡献了收敛稳定性？

---

# 33. 最终软件接口

目标：

```python
system = OpticalSystem(...)

merit = MeritFunction(...)

optimizer = LensOptimizer(
    system=system,
    merit_function=merit,
    config=config,
)

result = optimizer.optimize()
```

返回：

```python
OptimizationResult(
    success=True,
    converged=True,
    iterations=...,
    initial_merit=...,
    final_merit=...,
    final_system=...,
    history=...,
    singular_values=...,
    condition_number=...,
)
```

---

# 34. 开发阶段

严格按照以下顺序：

## Phase 1

Optical data model

## Phase 2

Paraxial ray tracing

## Phase 3

Real ray tracing

## Phase 4

RMS Spot

## Phase 5

Merit Function

## Phase 6

Optimization Variables

## Phase 7

Finite Difference Jacobian

## Phase 8

Gauss-Newton

## Phase 9

DLS

## Phase 10

SVD-DLS

## Phase 11

Variable Scaling

## Phase 12

Adaptive Lambda

## Phase 13

Trust Region / Step Control

## Phase 14

Constraints

## Phase 15

Wavefront / OPD

## Phase 16

Zernike

## Phase 17

PSF / MTF

## Phase 18

Multi-field / Multi-wavelength

## Phase 19

Optimization benchmark

## Phase 20

Performance optimization

---

# 35. Codex 工作方式

开始修改代码之前，不要直接写代码。

第一步：

扫描整个 repository。

寻找：

* 现有 OpticalSystem
* Surface
* Ray
* Material
* RayTrace
* Merit Function
* Optimization
* Zernike
* MTF
* PSF

然后输出：

1. 当前项目结构
2. 已存在的 optical modules
3. 已存在的 optimization modules
4. 可以复用的代码
5. 缺失模块
6. 建议的架构
7. Phase 1 实施计划

然后等待确认。

每一个 Phase 完成后
