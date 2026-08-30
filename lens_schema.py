"""
lens_schema.py — Pydantic 数据模型层
=====================================
定义光学镜头系统的完整数据契约：表面类型、非球面参数、光学约束、
优化配置等。所有 JSON 读写均通过本模块校验，确保追迹引擎拿到的
参数合法且物理自洽。

设计要点：
- SurfaceType 用 Enum 约束面型，杜绝拼写错误
- Aspheric 仅对 aspheric_reflect 生效，其他面型忽略
- Bounds 全字段可选，兼容 demo 中只给 curvature/thickness 的情况
- 变量标记分两级：surface.variable 控制曲率/厚度，aspheric_variable
  单独控制非球面系数，粒度更细
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 非球面高阶项定义
# ---------------------------------------------------------------------------

# 偶次非球面高阶项名称（按阶次升序）。最多支持 6 项：A4 ~ A14。
ASPHERIC_HIGH_ORDER: tuple[str, ...] = ("A4", "A6", "A8", "A10", "A12", "A14")


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------

class SurfaceType(str, Enum):
    """光学表面类型枚举。"""
    OBJECT = "object"           # 物面
    STANDARD = "standard"       # 球面折射面
    REFLECT = "reflect"         # 球面反射镜
    ASPHERIC_REFLECT = "aspheric_reflect"  # 高阶偶次非球面反射镜
    STOP = "stop"               # 光阑
    IMAGE = "image"             # 像面


class ApertureType(str, Enum):
    """孔径类型。"""
    NA = "na"           # 数值孔径
    F_NUMBER = "fnum"   # F 数
    DIAMETER = "diameter"  # 物理直径


# ---------------------------------------------------------------------------
# 非球面
# ---------------------------------------------------------------------------

class Aspheric(BaseModel):
    """
    偶次非球面参数。

    面型方程（ sag ）：
        z(r) = r² / [ R·(1 + sqrt(1 - (1+k)·r²/R²)) ]
               + Σ A_{2m}·r^{2m}   （m=2..7，即 A4 ~ A14）

    其中 R = 1/curvature，k 为圆锥常数（conic constant）。
    k=0 为球面，k=-1 为抛物面，k<-1 为双曲面，-1<k<0 为椭球面。

    高阶偶次项数量可配置（high_order_terms，默认 2 项 = A4,A6；最多 6 项 = A4..A14）：
        high_order_terms = 2 → 使用 A4, A6
        high_order_terms = 4 → 使用 A4, A6, A8, A10
        high_order_terms = 6 → 使用 A4, A6, A8, A10, A12, A14
    超出该数量的系数在光线追迹/优化中被忽略（active_coeffs() 只返回激活项）。
    """
    k: float = Field(0.0, description="圆锥常数 conic constant")
    high_order_terms: int = Field(
        2, ge=0, le=6,
        description="激活的高阶偶次项数量：2=A4,A6；4=A4..A10；6=A4..A14"
    )
    A4: float = Field(0.0, description="四阶非球面系数")
    A6: float = Field(0.0, description="六阶非球面系数")
    A8: float = Field(0.0, description="八阶非球面系数")
    A10: float = Field(0.0, description="十阶非球面系数")
    A12: float = Field(0.0, description="十二阶非球面系数")
    A14: float = Field(0.0, description="十四阶非球面系数")

    def active_coeffs(self) -> list[float]:
        """按 high_order_terms 返回激活的高阶系数（A4 起，长度 = high_order_terms）。"""
        return [getattr(self, n) for n in ASPHERIC_HIGH_ORDER[: self.high_order_terms]]


class AsphericVariable(BaseModel):
    """非球面系数的优化变量标记。"""
    k: bool = False
    A4: bool = False
    A6: bool = False
    A8: bool = False
    A10: bool = False
    A12: bool = False
    A14: bool = False


# ---------------------------------------------------------------------------
# 边界约束
# ---------------------------------------------------------------------------

class Bounds(BaseModel):
    """
    优化变量上下界。所有字段可选，未提供则不约束该参数。

    兼容 demo 格式：通常只给 curvature 和 thickness；
    非球面系数/偏心的边界可按需补充。
    """
    curvature: Optional[tuple[float, float]] = None
    thickness: Optional[tuple[float, float]] = None
    decenter_x: Optional[tuple[float, float]] = None
    decenter_y: Optional[tuple[float, float]] = None
    k: Optional[tuple[float, float]] = None
    A4: Optional[tuple[float, float]] = None
    A6: Optional[tuple[float, float]] = None
    A8: Optional[tuple[float, float]] = None
    A10: Optional[tuple[float, float]] = None
    A12: Optional[tuple[float, float]] = None
    A14: Optional[tuple[float, float]] = None
    semi_aperture: Optional[tuple[float, float]] = Field(
        None, description="半口径边界（镜子大小作为优化变量时）")


# ---------------------------------------------------------------------------
# 光学表面
# ---------------------------------------------------------------------------

class Surface(BaseModel):
    """
    单个光学表面。

    坐标系：全局右手系，光轴为 Z 轴，光线默认沿 +Z 传播。
    反射面会翻转传播方向（见 raytrace.py 符号处理）。

    thickness 含义：当前面顶点到下一面顶点的轴向距离（沿光轴）。
    curvature = 1/R，R 为曲率半径，curvature=0 表示平面。
    """
    surface_id: int = Field(..., ge=0, description="表面序号")
    type: SurfaceType
    curvature: float = Field(0.0, description="曲率 C=1/R，0 为平面")
    thickness: float = Field(0.0, description="到下一面的轴向距离")
    glass: str = Field("air", description="材料名称，对应 glass_library")
    semi_aperture: float = Field(0.0, ge=0, description="半口径")
    inner_aperture: float = Field(
        0.0, ge=0,
        description="中心挖孔半径 (mm)，0=实心镜。环形镜子（EUV 离轴镜片）："
        "光线穿过内孔区域（r < inner_aperture）不算遮挡，可做很大而挡住其他光路"
    )
    is_stop: bool = Field(False, description="是否为孔径光阑")
    variable: bool = Field(False, description="曲率/厚度是否为优化变量")
    decenter_x: float = Field(0.0, description="偏心：顶点相对光轴 X 方向平移 (mm)")
    decenter_y: float = Field(0.0, description="偏心：顶点相对光轴 Y 方向平移 (mm)")
    decenter_variable: bool = Field(
        False, description="偏心 x/y 是否为优化变量（离轴系统避免遮挡）"
    )
    size_variable: bool = Field(
        False, description="半口径（镜子大小）是否为优化变量"
    )
    bounds: Optional[Bounds] = Field(default=None, description="变量上下界")
    aspheric: Optional[Aspheric] = Field(default=None, description="非球面参数（仅 aspheric_reflect）")
    aspheric_variable: Optional[AsphericVariable] = Field(
        default=None, description="非球面系数变量标记"
    )

    @model_validator(mode="after")
    def _check_aspheric(self) -> "Surface":
        """非球面参数仅对 aspheric_reflect 有意义，其他面型静默忽略。"""
        if self.type != SurfaceType.ASPHERIC_REFLECT:
            if self.aspheric is not None:
                # 不报错，只是提醒：非球面参数将被忽略
                pass
            if self.aspheric_variable is not None:
                pass
        return self

    @property
    def is_reflective(self) -> bool:
        """是否为反射面（反射会翻转 Z 传播方向）。"""
        return self.type in (SurfaceType.REFLECT, SurfaceType.ASPHERIC_REFLECT)

    @property
    def radius(self) -> float:
        """曲率半径 R = 1/C，平面返回 inf。"""
        if abs(self.curvature) < 1e-15:
            return float("inf")
        return 1.0 / self.curvature


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class Aperture(BaseModel):
    """孔径定义。"""
    type: ApertureType = ApertureType.NA
    value: float = Field(..., gt=0, description="孔径值")


class RaySampling(BaseModel):
    """光线采样配置。"""
    aperture_rings: int = Field(7, ge=1, description="孔径方向环数")
    radial_rays: int = Field(14, ge=1, description="每环光线数（径向扇）")


class OpticalConstraints(BaseModel):
    """
    光学约束集合。

    约束违反量会加权并入评价函数 merit，作为惩罚项。
    每个约束都可以在 merit_weights 中配置权重。

    离轴系统（环形视场）参数：
    - na_objective: 物方数值孔径 NA（用于物方孔径采样）。
      None 时回退用 metadata.aperture.value。
    - na_target: 像方数值孔径 NA（用于 NA 偏差惩罚和像方孔径）。
    - chief_ray_angle_deg: 物方主光线入射角（与光轴夹角，度）。
      0 = 共轴视场；>0 = 离轴环形视场（光线从镜子离轴环带通过，
      避免被镜子实体遮挡）。
    """
    na_objective: Optional[float] = Field(
        default=None, description="物方数值孔径 NA，None 时用 aperture.value"
    )
    na_target: float = Field(0.33, gt=0, description="目标像方数值孔径 NA")
    chief_ray_angle_deg: float = Field(
        0.0, ge=0, description="物方主光线入射角（离轴角，度），0=共轴"
    )
    object_height_mm: float = Field(
        0.0, ge=0,
        description="物方视场高度（物点离轴高度，mm）。0=用 z_pupil·tan(chief角) 换算。"
        ">0 时直接指定物点离轴量（物面高 50mm 等），放大率 = 像高/物高"
    )
    central_obscuration_ratio: float = Field(
        0.0, ge=0, lt=1.0,
        description="入瞳中心遮挡比（环形孔径）：0=实心锥，0.5=内半径占半。"
        "共轴回转对称系统（decenter=0）的成像光束是环形光瞳（光轴附近被设计"
        "遮挡，有效光线走镜子环带），必须 >0 才能找到不遮挡布局"
    )
    telecentric_objective: bool = Field(
        False,
        description="物方远心：主光线平行光轴（真实 EUV 物镜的掩模照明方式）。"
        "True 时入瞳中心移到物点正上方（离轴高度 r_obj），光束保持离轴环带"
        "穿过共轴镜子，才能找到回转对称系统的不遮挡布局"
    )
    telecentric_image_max_deg: float = Field(
        0.0, ge=0,
        description="像方远心约束：像面处光线平均方向与光轴的夹角上限（度）。"
        "0=不约束。>0 时作为评价函数惩罚项（真实 EUV 物镜像方远心，"
        "晶圆侧主光线平行光轴）"
    )
    image_z_fixed: float = Field(
        0.0,
        description="固定像面位置（像面目标 z，mm）。0=不约束。"
        "物面固定 z=0；像面固定在该 z 处（光刻系统物面/像面位置是硬指标）"
    )
    magnification_target: float = Field(
        0.0,
        description="放大率目标（像高/物高）。0=不约束。"
        "光刻物镜为 4:1 缩小 → 0.25（像高 = 物高/4）"
    )
    rectangular_fov: bool = Field(
        False,
        description="矩形视场（EUV 物镜实际形态）：物面矩形视场（环形视场内接的最大矩形），"
        "镜片只做矩形视场发出的光打到的部分（单片镜片，非环形）"
    )
    rect_half_width_mm: float = Field(
        0.0, ge=0,
        description="矩形视场半宽 (mm)。>0 时物面矩形 [-w,w]×[-w,w] 内采样多个视场点"
    )
    fov_grid_n: int = Field(
        3, ge=1,
        description="矩形视场采样网格点数（n×n 视场点）。2=4点(快)，3=9点(标准)，5=25点(精细)"
    )
    rect_half_wx_mm: float = Field(
        0.0, ge=0, description="矩形视场 x 半宽 (mm)。0 时用 rect_half_width_mm"
    )
    rect_half_wy_mm: float = Field(
        0.0, ge=0, description="矩形视场 y 半宽 (mm)。0 时用 rect_half_width_mm"
    )
    fov_center_x: float = Field(
        0.0, description="矩形视场中心 x (mm，偏心于光轴，环形视场内接矩形的情况)"
    )
    fov_center_y: float = Field(
        0.0, description="矩形视场中心 y (mm)"
    )
    fov_pattern: str = Field(
        "grid", description="视场采样模式：grid=n×n 网格；corners_center=四角+中心（5 点）"
    )
    incident_angle_max_deg: float = Field(
        12.0, gt=0, description="反射面最大允许入射角（度）"
    )
    exit_angle_max_deg: float = Field(
        8.0, gt=0, description="像面最大允许出射角（度）"
    )
    working_distance_min: float = Field(
        20.0, gt=0, description="最小工作距离（最后一面到像面）"
    )
    obscuration_margin_mm: float = Field(
        0.0, ge=0,
        description="遮挡安全距离 (mm)：光线不仅不能穿过其他镜子有效工作区域，"
        "还要离其环带边界 ≥ 该距离（不擦边，保证装调/加工余量）"
    )


class SpacingConstraints(BaseModel):
    """
    镜面间距约束集合。

    区分两类间距：
    - 中心厚度（center thickness）：镜片材料内部的轴向厚度（glass != air 的面的 thickness）
    - 空气间隔（air gap）：相邻镜片之间的空气距离（glass == air 的面的 thickness）

    约束违反量作为软惩罚项并入评价函数，与硬边界 bounds 互补：
    bounds 防止变量越界，spacing_constraints 确保物理可制造性。
    """
    min_center_thickness: float = Field(
        1.0, ge=0, description="最小镜片中心厚度 (mm)，防止镜片太薄无法加工"
    )
    min_air_gap: float = Field(
        1.0, ge=0, description="最小空气间隔 (mm)，防止相邻镜片机械干涉"
    )
    max_total_length: float = Field(
        0.0, description="最大系统总长 (mm)，0 表示不约束。从物面到像面的轴向总距离"
    )
    min_mirror_gap: float = Field(
        0.0, ge=0,
        description="镜子间最小轴向间距 (mm)，0=不约束。任意两面镜子若径向工作环带相交，"
        "轴向中心距必须 ≥ 该值（否则镜子实体/支撑会碰撞，物理上无法制造安装）"
    )


class PackageBox(BaseModel):
    """
    封装盒子（外包络）约束：整个光学系统必须装进给定体积内。

    盒子形状：
    - cylinder：圆柱，z ∈ [z_min, z_max]，径向 r ≤ r_max（回转对称系统最自然）
    - box：矩形，z ∈ [z_min, z_max]，|x| ≤ x_half，|y| ≤ y_half

    约束内容（违反量作为评价函数惩罚项）：
    1. 光路不干涉：所有光线路径（含段中点）必须在盒子内
    2. 有效结构不干涉：每面镜子的有效工作部分（光线足迹环带）必须在盒子内

    入口/出口：物面（入口）通常位于 z_min 处，像面（出口）位于盒子内或边界。
    """
    enabled: bool = Field(False, description="是否启用封装盒子约束")
    shape: str = Field("cylinder", description="cylinder / box")
    z_min: float = Field(0.0, description="盒子轴向最小 z (mm)，通常为物面/入口")
    z_max: float = Field(600.0, description="盒子轴向最大 z (mm)")
    r_max: float = Field(200.0, description="圆柱半径 / 矩形半宽基准 (mm)")
    x_half: Optional[float] = Field(None, description="矩形盒子 x 半宽")
    y_half: Optional[float] = Field(None, description="矩形盒子 y 半宽")


class Metadata(BaseModel):
    """镜头系统元数据。"""
    name: str = Field("unnamed_lens", description="系统名称")
    wavelengths: list[float] = Field(
        default_factory=lambda: [0.5876],
        description="波长列表，单位 mm（可见光约 0.0005mm，EUV 13.5nm=0.0135mm）",
    )
    aperture: Aperture
    fov_deg: list[float] = Field(
        default_factory=lambda: [0.0],
        description="视场角列表（度），半视场",
    )
    unit: str = Field("mm", description="长度单位")
    ray_mode: str = Field(
        "auto",
        description="光线采样模式：auto/offaxis/meridional/chief/full。"
        "auto=离轴系统用全3D环形视场，共轴系统用标准环形；"
        "meridional=子午面光线（结构搜索加速）；chief=仅主光线（超快筛选）",
    )
    ray_sampling: RaySampling = Field(default_factory=RaySampling)
    optical_constraints: OpticalConstraints = Field(
        default_factory=OpticalConstraints
    )
    spacing_constraints: SpacingConstraints = Field(
        default_factory=SpacingConstraints,
        description="镜面间距约束（最小中心厚度/最小空气间隔/最大总长）"
    )
    package: PackageBox = Field(
        default_factory=PackageBox,
        description="封装盒子约束（外包络：光路与有效结构不干涉）"
    )


# ---------------------------------------------------------------------------
# 玻璃库
# ---------------------------------------------------------------------------

class Glass(BaseModel):
    """玻璃材料光学常数。"""
    nd: float = Field(1.0, description="d 光（587.6nm）折射率")
    vd: float = Field(0.0, description="阿贝数")


# ---------------------------------------------------------------------------
# 优化配置
# ---------------------------------------------------------------------------

class MeritWeights(BaseModel):
    """评价函数各项权重。"""
    rms_spot: float = Field(1.0, description="RMS 光斑半径权重")
    distortion: float = Field(0.2, description="畸变权重")
    chroma: float = Field(0.0, description="色差权重（反射系统通常为 0）")
    na_penalty: float = Field(2.0, description="NA 偏差惩罚权重")
    angle_incident_penalty: float = Field(1.5, description="入射角超限惩罚权重")
    angle_exit_penalty: float = Field(1.0, description="出射角超限惩罚权重")
    working_distance_penalty: float = Field(1.0, description="工作距离惩罚权重")
    obscuration_penalty: float = Field(
        5.0, description="光线被镜子遮挡比例惩罚权重（离轴系统必需）"
    )
    missing_penalty: float = Field(
        10.0, description="追迹失败（未遮挡但到不了像面）光线比例惩罚权重，防止优化器丢弃光线作弊"
    )
    lost_ray_penalty_mm: float = Field(
        50.0, description="被遮挡/追迹失败的光线在 RMS 中的等效误差 (mm)，防止优化器丢弃光线作弊"
    )
    min_thickness_penalty: float = Field(5.0, description="最小中心厚度违反惩罚权重")
    min_air_gap_penalty: float = Field(5.0, description="最小空气间隔违反惩罚权重")
    total_length_penalty: float = Field(0.5, description="系统总长超限惩罚权重")
    telecentricity_penalty: float = Field(
        0.0, description="像方远心偏差惩罚权重（出射光线平均方向与光轴夹角超限）"
    )
    package_penalty: float = Field(
        0.0, description="封装盒子干涉惩罚权重（光路/有效结构超出盒子）"
    )
    mirror_gap_penalty: float = Field(
        0.0, description="镜子间最小间距违反惩罚权重（径向环带相交时轴向间距不足）"
    )
    object_path_penalty: float = Field(
        0.0, description="物光路穿镜惩罚权重：物面→第一镜的传播路径上不能穿过其他镜子实体"
    )
    mirror_z_penalty: float = Field(
        0.0, description="镜子 z 全正惩罚权重：所有镜子顶点 z>0（物面 z=0，镜子应在正侧）"
    )
    image_z_penalty: float = Field(0.0, description="像面位置偏差惩罚权重（mm）")
    magnification_penalty: float = Field(0.0, description="放大率偏差惩罚权重")


class GlobalSearch(BaseModel):
    """全局优化（GA+SA）配置。"""
    pop_size: int = Field(28, ge=4, description="种群规模")
    ga_iter: int = Field(70, ge=1, description="遗传算法迭代代数")
    sa_temp_init: float = Field(90.0, gt=0, description="模拟退火初始温度")


class LocalDLS(BaseModel):
    """局部阻尼最小二乘（DLS）配置。"""
    damp_init: float = Field(0.15, gt=0, description="初始阻尼因子")
    max_iter: int = Field(120, ge=1, description="最大迭代次数")


class OptimizationConfig(BaseModel):
    """优化器总配置。"""
    merit_weights: MeritWeights = Field(default_factory=MeritWeights)
    global_search: GlobalSearch = Field(default_factory=GlobalSearch)
    local_dls: LocalDLS = Field(default_factory=LocalDLS)


# ---------------------------------------------------------------------------
# 顶层镜头系统
# ---------------------------------------------------------------------------

class LensSystem(BaseModel):
    """
    完整光学镜头系统。

    这是 JSON 文件的顶层模型，也是追迹/优化的唯一数据入口。
    """
    metadata: Metadata
    surfaces: list[Surface] = Field(..., min_length=2, description="表面列表（含物面和像面）")
    glass_library: dict[str, Glass] = Field(
        default_factory=lambda: {"air": Glass(nd=1.0, vd=0.0)},
        description="玻璃材料库",
    )
    optimization_config: OptimizationConfig = Field(default_factory=OptimizationConfig)

    # -- 便捷属性 -----------------------------------------------------------

    @property
    def num_surfaces(self) -> int:
        return len(self.surfaces)

    @property
    def object_surface(self) -> Surface:
        return self.surfaces[0]

    @property
    def image_surface(self) -> Surface:
        return self.surfaces[-1]

    @property
    def stop_surface(self) -> Optional[Surface]:
        for s in self.surfaces:
            if s.is_stop:
                return s
        return None

    def get_glass_nd(self, glass_name: str, wavelength: float | None = None) -> float:
        """
        获取指定玻璃在给定波长下的折射率。

        简化模型：仅使用 nd（d 光折射率），不做色散插值。
        反射系统全为 air，此函数主要为折射面预留。
        """
        glass = self.glass_library.get(glass_name)
        if glass is None:
            return 1.0  # 未知材料默认空气
        return glass.nd

    # -- 变量管理 -----------------------------------------------------------

    def get_variable_indices(self) -> list[tuple[int, str]]:
        """
        提取所有优化变量的 (surface_id, param_name) 列表。

        param_name 取值：curvature, thickness, k, A4, A6, A8, A10
        顺序固定：先曲率/厚度，再非球面系数，确保优化向量维度一致。
        """
        indices: list[tuple[int, str]] = []
        for surf in self.surfaces:
            if surf.variable:
                indices.append((surf.surface_id, "curvature"))
                indices.append((surf.surface_id, "thickness"))
            if surf.decenter_variable:
                # 环形视场（离轴）沿 y 方向对称，decenter_x 无梯度，只优化 decenter_y
                indices.append((surf.surface_id, "decenter_y"))
            if surf.size_variable:
                indices.append((surf.surface_id, "semi_aperture"))
            if surf.aspheric_variable is not None:
                av = surf.aspheric_variable
                if av.k:
                    indices.append((surf.surface_id, "k"))
                # 高阶项：按 ASPHERIC_HIGH_ORDER 顺序（A4~A14，最多 6 项）
                for name in ASPHERIC_HIGH_ORDER:
                    if getattr(av, name):
                        indices.append((surf.surface_id, name))
        return indices

    def get_variable_bounds(self) -> list[tuple[float, float]]:
        """获取所有变量的 (lower, upper) 边界，顺序与 get_variable_indices 一致。"""
        bounds_list: list[tuple[float, float]] = []
        surf_map = {s.surface_id: s for s in self.surfaces}
        for sid, pname in self.get_variable_indices():
            surf = surf_map[sid]
            b = surf.bounds
            if b is not None:
                val = getattr(b, pname, None)
                if val is not None:
                    bounds_list.append(val)
                    continue
            # 无显式边界时给宽松默认值
            if pname == "curvature":
                bounds_list.append((-0.1, 0.1))
            elif pname == "thickness":
                bounds_list.append((1.0, 1000.0))
            elif pname in ("decenter_x", "decenter_y"):
                bounds_list.append((-80.0, 80.0))
            elif pname == "semi_aperture":
                bounds_list.append((30.0, 1500.0))
            elif pname == "k":
                bounds_list.append((-5.0, 5.0))
            else:  # A4-A10
                # 物理合理默认边界：在 r_max=100mm 处每阶 sag 贡献 ≤ 1mm
                # （A6 若用 ±1e-6，在 r=50mm 处 sag 变化可达 9.4m，会导致
                #   光线全部追不到像面，雅可比爆炸——维度灾难的根源）
                order = int(pname[1:])
                amp = 1.0 / (100.0 ** order)
                bounds_list.append((-amp, amp))
        return bounds_list

    def variables_to_array(self) -> "np.ndarray":
        """将当前所有优化变量值提取为 1D numpy 数组。"""
        import numpy as np
        surf_map = {s.surface_id: s for s in self.surfaces}
        vals = []
        for sid, pname in self.get_variable_indices():
            surf = surf_map[sid]
            if pname in ("curvature", "thickness", "decenter_x", "decenter_y", "semi_aperture"):
                vals.append(getattr(surf, pname))
            else:
                if surf.aspheric is not None:
                    vals.append(getattr(surf.aspheric, pname))
                else:
                    vals.append(0.0)
        return np.array(vals, dtype=np.float64)

    def array_to_variables(self, x: "np.ndarray") -> None:
        """将 1D numpy 数组写回镜头系统的变量字段。"""
        surf_map = {s.surface_id: s for s in self.surfaces}
        indices = self.get_variable_indices()
        if len(x) != len(indices):
            raise ValueError(f"变量维度不匹配：期望 {len(indices)}，得到 {len(x)}")
        for (sid, pname), val in zip(indices, x):
            surf = surf_map[sid]
            if pname in ("curvature", "thickness", "decenter_x", "decenter_y", "semi_aperture"):
                setattr(surf, pname, float(val))
            else:
                if surf.aspheric is None:
                    surf.aspheric = Aspheric()
                setattr(surf.aspheric, pname, float(val))

    # -- 序列化 -------------------------------------------------------------

    def to_json(self, path: str) -> None:
        """保存为 JSON 文件。"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "LensSystem":
        """从 JSON 文件加载并校验。"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
