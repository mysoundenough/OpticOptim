"""
共轴折返系统布局构造模块
=========================

根据主光线反射点（锚点）确定 6 镜共轴（回转对称）折返系统的镜面方程。

核心思想
--------
真实 EUV 物镜 = 6 镜共轴堆叠（decenter=0，回转对称）+ Z 形折返光路
（z 交替）+ 环形孔径避遮挡。镜子形态：大凹面镜（M1/M3/M5，R 大）
+ 强弯小镜（M2/M6，R 小），反射点可近轴也可离轴。

镜面方程（k=-1 抛物面，锚点处法线严格对齐设计法线）：
    decenter = 0                      # 共轴
    ny_raw = sign(nz)·ny/|nz|         # 镜面法线原始 y 分量（取 nz>0 版本）
    c      = -ny_raw / y_anchor       # 曲率：c·y = 法线斜率，法线角精确
    z_v    = z_anchor - c·y²/2        # 顶点 z：镜面精确过锚点
    thickness = |z_v 差|              # 全正：折返布局（compute_surface_z 反射翻转自动 z 交替）

验证
----
主光线（物点 (0,36.8)，15° 入射，像方远心）应精确通过全部 6 个锚点
（误差 < 1e-3 mm），像面 y=9.2，出射平行 +z。

用法
----
    python build_axisym.py --path output/initial_path/initial_chief_path_all_seed27.json \\
                           --out examples/offaxis_6mirror_axisym_seed27.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from find_reflect_initial import build_path
from lens_schema import LensSystem
from raytrace import (SurfaceType, compute_surface_z, generate_rect_chief_rays,
                      intersect_surface, reflect, surface_normal)

Z_IMAGE = 3500.0          # 像面 z（固定）
OBJ_Y = 36.8              # 物点 y（fov center）
CHIEF_ANGLE = 15.0        # 主光线物方入射角（统一 15° 向下）


# ---------------------------------------------------------------------------
# 共轴镜面构造
# ---------------------------------------------------------------------------

def build_axisym_mirrors(anchors_yz: np.ndarray, normals: np.ndarray | None = None,
                         telecentric_image: bool = True) -> list[dict]:
    """
    由 6 个主光线反射点（锚点）+ 法向量构造共轴镜面。

    参数:
        anchors_yz: (6,2) 锚点 (y, z)
        normals:    (6,3) 设计法向量（None 时由 build_path 重算）
        telecentric_image: 像方远心（M6 出射平行 +z，影响 M6 法线）

    返回:
        每面镜子 dict(curvature, k, decenter_y=0, decenter_x=0, z_vertex,
                       anchor_y, anchor_z, ny_raw, normal)
    """
    pts6 = np.array(anchors_yz, dtype=float)
    if normals is None:
        path = build_path(pts6, telecentric_image=telecentric_image)
        normals = path["normals"]
    normals = np.array(normals, dtype=float)

    mirrors = []
    for i in range(6):
        ny, nz = normals[i, 1], normals[i, 2]
        ay, az = pts6[i, 0], pts6[i, 1]
        # 镜面法线原始 y 分量：取设计法线的 nz>0 版本（surface_normal 约定 nz 恒 +）
        ny_raw = np.sign(nz) * ny / max(abs(nz), 1e-12)
        c = -ny_raw / ay          # k=-1 抛物面：法线 y 分量 = -c·y → c·y = tan(法线角)
        k = -1.0
        z_v = az - c * ay ** 2 / 2.0   # 镜面过锚点
        mirrors.append(dict(
            curvature=float(c), k=float(k),
            decenter_y=0.0, decenter_x=0.0,
            z_vertex=float(z_v),
            anchor_y=float(ay), anchor_z=float(az),
            ny_raw=float(ny_raw),
            normal=[float(v) for v in normals[i]],
        ))
    return mirrors


def axisym_thickness(mirrors: list[dict], z_image: float = Z_IMAGE) -> list[float]:
    """
    共轴折返布局 thickness（全正）：
    折返光路 z 交替（compute_surface_z 对反射面翻转符号 → 自动交替）。
    """
    zv = [m["z_vertex"] for m in mirrors]
    t = [zv[0]]
    for i in range(6):
        if i < 5:
            t.append(abs(zv[i + 1] - zv[i]))
        else:
            t.append(z_image - zv[5])
    return t


def axisym_mirror_surfaces(mirrors: list[dict],
                           thickness: list[float],
                           semi_aperture_margin: float = 200.0) -> list[dict]:
    """由共轴镜面参数构建 surfaces 字典列表（物面 + 6 镜 + 像面）。"""
    surfaces = [{"surface_id": 0, "type": "object",
                 "thickness": thickness[0], "semi_aperture": 600.0}]
    for i, m in enumerate(mirrors):
        sa = max(abs(m["anchor_y"]) + semi_aperture_margin, 150.0)
        surfaces.append({
            "surface_id": i + 1, "type": "aspheric_reflect",
            "curvature": m["curvature"], "thickness": thickness[i + 1],
            "semi_aperture": sa, "variable": True,
            "decenter_y": 0.0, "decenter_x": 0.0, "decenter_variable": False,
            "bounds": {"curvature": [-0.05, 0.05], "thickness": [50.0, 2500.0],
                       "decenter_y": [-300.0, 300.0], "k": [-5.0, 5.0],
                       "A4": [-1e-7, 1e-7], "A6": [-1e-11, 1e-11],
                       "A8": [-1e-15, 1e-15], "A10": [-1e-19, 1e-19],
                       "semi_aperture": [100.0, 3000.0]},
            "aspheric": {"k": m["k"], "A4": 0.0, "A6": 0.0, "A8": 0.0, "A10": 0.0,
                         "A12": 0.0, "A14": 0.0, "high_order_terms": 2},
            "aspheric_variable": {"k": True, "A4": True, "A6": True, "A8": False,
                                  "A10": False, "A12": False, "A14": False},
        })
    surfaces.append({"surface_id": 7, "type": "image",
                     "thickness": 0.0, "semi_aperture": 100.0})
    return surfaces


def build_axisym_system(anchors_yz: np.ndarray,
                        normals: np.ndarray | None = None,
                        name: str = "offaxis_6mirror_axisym",
                        z_image: float = Z_IMAGE) -> dict:
    """完整共轴 LensSystem JSON 字典。"""
    mirrors = build_axisym_mirrors(anchors_yz, normals)
    thick = axisym_thickness(mirrors, z_image)
    surfaces = axisym_mirror_surfaces(mirrors, thick)
    return {
        "metadata": {
            "name": name, "wavelengths": [0.0135],
            "aperture": {"type": "na", "value": 0.33}, "fov_deg": [0.0],
            "unit": "mm", "ray_mode": "offaxis",
            "ray_sampling": {"aperture_rings": 2, "radial_rays": 3},
            "optical_constraints": {
                "na_objective": 0.08, "na_target": 0.33,
                "chief_ray_angle_deg": CHIEF_ANGLE,
                "telecentric_objective": False, "object_height_mm": 36.8,
                "image_z_fixed": z_image, "magnification_target": 0.25,
                "telecentric_image_max_deg": 0.5,
                "incident_angle_max_deg": 20.0, "exit_angle_max_deg": 22.0,
                "working_distance_min": 25.0, "obscuration_margin_mm": 10.0,
                "rectangular_fov": True, "rect_half_wx_mm": 6.8,
                "rect_half_wy_mm": 24.4, "fov_center_x": 0.0,
                "fov_center_y": 36.8, "fov_pattern": "corners_center"},
            "spacing_constraints": {"min_center_thickness": 1.0,
                                    "min_air_gap": 5.0, "max_total_length": 0.0,
                                    "min_mirror_gap": 20.0},
            "package": {"enabled": False},
        },
        "surfaces": surfaces,
        "glass_library": {"air": {"nd": 1.0, "vd": 0.0}},
        "optimization_config": {
            "merit_weights": {"rms_spot": 1.0, "distortion": 0.0, "chroma": 0.0,
                              "na_penalty": 2.0, "angle_incident_penalty": 1.0,
                              "angle_exit_penalty": 1.0,
                              "working_distance_penalty": 1.0,
                              "obscuration_penalty": 20.0,
                              "missing_penalty": 50.0,
                              "lost_ray_penalty_mm": 50.0,
                              "min_thickness_penalty": 0.0,
                              "min_air_gap_penalty": 0.0,
                              "total_length_penalty": 0.0,
                              "telecentricity_penalty": 0.0,
                              "package_penalty": 0.0, "mirror_gap_penalty": 0.0,
                              "image_z_penalty": 0.0,
                              "magnification_penalty": 0.0,
                              "object_path_penalty": 0.0,
                              "mirror_z_penalty": 0.0},
            "global_search": {"pop_size": 30, "ga_iter": 60, "sa_temp_init": 60.0},
            "local_dls": {"damp_init": 0.15, "max_iter": 60},
        },
    }


# ---------------------------------------------------------------------------
# 主光线验证
# ---------------------------------------------------------------------------

def verify_chief_ray(lens: LensSystem, anchors_yz: np.ndarray) -> dict:
    """
    追迹中心视场点主光线，验证是否精确通过 6 个锚点并成像。

    返回 dict(n_anchors_ok, max_anchor_err, image_y, image_z, telecentric_ok)
    """
    surf_z = compute_surface_z(lens.surfaces)
    cpos, cdirs, cwl, cfov = generate_rect_chief_rays(lens)
    ci = np.argmin(np.abs(cpos[:, 0]) + np.abs(cpos[:, 1] - OBJ_Y))
    p = cpos[ci:ci + 1].copy()
    dd = cdirs[ci:ci + 1].copy()
    max_err = 0.0
    image_y = image_z = None
    tele_ok = False
    for i in range(1, len(lens.surfaces)):
        s = lens.surfaces[i]
        t = intersect_surface(p, dd, surf_z[i], s)
        if not np.isfinite(t[0]):
            break
        p2 = p + t * dd
        if s.type == SurfaceType.IMAGE:
            image_y, image_z = float(p2[0, 1]), float(p2[0, 2])
            tele_ok = abs(dd[0, 2] - 1.0) < 0.01
            break
        if s.is_reflective and i <= 6:
            ay, az = anchors_yz[i - 1]
            err = float(np.hypot(p2[0, 1] - ay, p2[0, 2] - az))
            max_err = max(max_err, err)
        A4, A6, A8, A10, A12, A14 = (s.aspheric.A4, s.aspheric.A6, s.aspheric.A8,
                                     s.aspheric.A10, s.aspheric.A12, s.aspheric.A14)
        norm = surface_normal(p2[:, 0] - s.decenter_x, p2[:, 1] - s.decenter_y,
                              p2[:, 2] - surf_z[i], s.curvature,
                              s.aspheric.k if s.aspheric else 0.0, A4, A6, A8, A10, A12, A14)
        if s.is_reflective:
            dd = reflect(dd, norm)
        p = p2
    return dict(n_anchors_ok=int(max_err < 1e-3),
                max_anchor_err=max_err, image_y=image_y, image_z=image_z,
                telecentric_ok=bool(tele_ok))


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="共轴折返系统布局构造（主光线反射点 → 镜面方程）")
    ap.add_argument("--path", required=True,
                    help="理想路径 JSON（含 mirrors y/z 或 points/normals）")
    ap.add_argument("--out", required=True, help="输出 LensSystem JSON")
    ap.add_argument("--name", default=None, help="系统名称")
    args = ap.parse_args()

    d = json.load(open(args.path))
    if "mirrors" in d and all("y" in m for m in d["mirrors"]):
        anchors = np.array([[m["y"], m["z"]] for m in d["mirrors"]])
    else:
        anchors = np.array(d["points"][1:7])[:, [1, 2]]   # points: (7,3) xyz → 取 (y,z)
    normals = d.get("normals")

    name = args.name or f"axisym_from_{args.path.split('/')[-1].replace('.json','')}"
    sys_dict = build_axisym_system(anchors, normals, name=name)
    json.dump(sys_dict, open(args.out, "w"), indent=2, ensure_ascii=False)
    lens = LensSystem.from_json(args.out)

    print("=== 共轴镜面（decenter=0, k=-1）===")
    for i, m in enumerate(build_axisym_mirrors(anchors, normals)):
        print(f"M{i+1}: c={m['curvature']:+.7f} R={abs(1/m['curvature']) if m['curvature'] else 0:8.1f}mm "
              f"锚点y={m['anchor_y']:8.1f} 顶点z={m['z_vertex']:8.1f}")
    v = verify_chief_ray(lens, anchors)
    print("\n=== 主光线验证 ===")
    print(f"6 锚点全过: {'OK' if v['n_anchors_ok'] else 'FAIL'} (最大误差 {v['max_anchor_err']:.3f} mm)")
    print(f"像面: y={v['image_y']:.3f} (目标 9.2)  z={v['image_z']:.1f}  远心: {'OK' if v['telecentric_ok'] else 'NO'}")
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
