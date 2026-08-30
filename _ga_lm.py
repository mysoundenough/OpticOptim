"""GA + LM：24 变量 (c,k,A4,A6)×6，M6 碗口朝+z 固定，中心主光线+边缘光线+约束"""
import numpy as np, copy, time, json
from scipy.optimize import least_squares
from lens_schema import LensSystem
from raytrace import (compute_surface_z, generate_rect_chief_rays, generate_rect_offaxis_rays,
                      intersect_surface, surface_normal, reflect, SurfaceType, aspheric_coeffs)

lens = LensSystem.from_json('examples/offaxis_6mirror_axisym_seed27_m6concave.json')
BASE_T = [lens.surfaces[i].thickness for i in range(7)]
FIELDS = [(sid, p) for sid in range(1,7) for p in ('curvature','k','A4','A6')]
NV = len(FIELDS)
print('变量数:', NV)

def apply_x(l, x):
    for (sid, p), v in zip(FIELDS, x):
        s = l.surfaces[sid]
        if p == 'curvature': s.curvature = float(v)
        elif p == 'k': s.aspheric.k = float(v)
        elif p == 'A4': s.aspheric.A4 = float(v)
        else: s.aspheric.A6 = float(v)
    for i in range(7): l.surfaces[i].thickness = BASE_T[i]

def get_x(l):
    out = []
    for sid, p in FIELDS:
        s = l.surfaces[sid]
        if p == 'curvature': out.append(s.curvature)
        elif p == 'k': out.append(s.aspheric.k)
        elif p == 'A4': out.append(s.aspheric.A4)
        else: out.append(s.aspheric.A6)
    return np.array(out)

cpos, cdirs, cwl, cfov = generate_rect_chief_rays(lens)
ci = np.argmin(np.abs(cpos[:,0])+np.abs(cpos[:,1]-36.8))
apos, adirs, _, _ = generate_rect_offaxis_rays(lens)
n_per = len(apos)//5
pos_all = np.vstack([cpos[ci:ci+1], apos[4*n_per:5*n_per]])
dir_all = np.vstack([cdirs[ci:ci+1], adirs[4*n_per:5*n_per]])
dd = json.load(open('output/initial_path/initial_chief_path_all_seed27.json'))
ay = [m['y'] for m in dd['mirrors']]; az = [m['z'] for m in dd['mirrors']]
BIG = 40.0

def trace_ray(lc, p, ddir):
    sz = compute_surface_z(lc.surfaces)
    p = p.copy(); ddir = ddir.copy()
    hits = []
    for i in range(1, len(lc.surfaces)):
        s = lc.surfaces[i]
        t = intersect_surface(p, ddir, sz[i], s)
        if not np.isfinite(t[0]): break
        p2 = p + t*ddir
        d_in = ddir[0].copy(); d_out = None
        if s.type == SurfaceType.IMAGE:
            hits.append((i, p2[0].copy(), d_in, d_out)); break
        A4,A6,A8,A10,A12,A14 = aspheric_coeffs(s)
        norm = surface_normal(p2[:,0]-s.decenter_x, p2[:,1]-s.decenter_y, p2[:,2]-sz[i],
                              s.curvature, s.aspheric.k if s.aspheric else 0.0, A4,A6,A8,A10,A12,A14)
        if s.is_reflective:
            d_out = reflect(ddir, norm)[0].copy(); ddir = d_out.reshape(1,3)
        hits.append((i, p2[0].copy(), d_in, d_out))
        p = p2
    return hits

def residuals(x, return_rays=False):
    lc = copy.deepcopy(lens); apply_x(lc, x)
    res = []
    h0 = trace_ray(lc, pos_all[0:1], dir_all[0:1])
    for i in range(1, 7):
        hit = next((h for h in h0 if h[0] == i), None)
        if hit is not None:
            res.append(3.0*(hit[1][1]-ay[i-1])); res.append(3.0*(hit[1][2]-az[i-1]))
        else:
            res.append(BIG); res.append(BIG)
    if h0 and h0[-1][0] == 7:
        res.append(1.0*(h0[-1][1][1]-9.2))
        res.append(6.0*h0[-1][2][1])
        res.append(6.0*(h0[-1][2][2]-1.0))
    else:
        res += [BIG]*3
    for k in range(1, 7):
        h = trace_ray(lc, pos_all[k:k+1], dir_all[k:k+1])
        if h and h[-1][0] == 7:
            res.append(1.0*(h[-1][1][1]-9.2))
        else:
            res.append(BIG)
    for k in range(7):
        h = trace_ray(lc, pos_all[k:k+1], dir_all[k:k+1])
        for i in range(1, 7):
            hit = next((hh for hh in h if hh[0] == i), None)
            if hit is not None and hit[3] is not None:
                ang = np.degrees(np.arccos(np.clip(np.dot(hit[2], hit[3]), -1, 1)))/2
                res.append(0.8*max(0.0, ang-20.0))
            else:
                res.append(BIG)
    r = np.array(res)
    if return_rays:
        return float(np.sum(r*r)), h0
    return r

def merit(x):
    return float(np.sum(residuals(x)**2))

# ---- GA ----
lo = np.zeros(NV); hi = np.zeros(NV)
for j, (sid, p) in enumerate(FIELDS):
    if p == 'curvature': lo[j], hi[j] = -0.03, 0.03
    elif p == 'k': lo[j], hi[j] = -3.0, 3.0
    elif p == 'A4': lo[j], hi[j] = -5e-7, 5e-7
    else: lo[j], hi[j] = -1e-10, 1e-10
x0 = get_x(lens)
rng = np.random.default_rng(42)
N = 50; GEN = 120
pop = np.zeros((N, NV))
pop[0] = x0
for i in range(1, N):
    pop[i] = np.clip(x0 + rng.normal(0, 1, NV)*(hi-lo)*0.15, lo, hi)  # 起点附近扰动
fit = np.array([merit(x) for x in pop])
bi = np.argmin(fit); bx = pop[bi].copy(); bf = fit[bi]
t0 = time.time()
print(f'GA gen0: merit={bf:.1f}')
for g in range(GEN):
    np2 = np.zeros_like(pop); nf = np.zeros(N)
    np2[0] = bx; nf[0] = bf
    top = np.argsort(fit)[:12]
    for i in range(1, N):
        p1 = pop[rng.choice(top)]; p2 = pop[rng.choice(top)]
        if rng.random() < 0.7:
            ch = 0.5*(p1+p2) + rng.normal(0,1,NV)*(hi-lo)*0.1
        else:
            ch = p1 + rng.normal(0,1,NV)*(hi-lo)*0.2
        ch = np.clip(ch, lo, hi)
        np2[i] = ch; nf[i] = merit(ch)
    pop, fit = np2, nf
    gbi = np.argmin(fit)
    if fit[gbi] < bf: bx, bf = pop[gbi].copy(), fit[gbi]
    if g % 30 == 0:
        print(f'GA gen{g}: merit={bf:.1f} [{time.time()-t0:.0f}s]')
print(f'GA 结束: merit={bf:.1f} [{time.time()-t0:.0f}s]')

# ---- LM 精修 ----
scale = np.ones(NV)
for j, (sid, p) in enumerate(FIELDS):
    if p == 'curvature': scale[j] = 1e-4
    elif p == 'k': scale[j] = 1e-2
    elif p == 'A4': scale[j] = 1e-8
    else: scale[j] = 1e-11
try:
    sol = least_squares(residuals, bx, method='lm', max_nfev=300, ftol=1e-9, xtol=1e-10)
except Exception as e:
    print('LM fail:', str(e)[:60])
    sol = least_squares(residuals, bx, method='trf', x_scale=scale, max_nfev=300, ftol=1e-9)
x = sol.x
m_f = merit(x)
print(f'LM 精修: {bf:.1f} → {m_f:.1f} (nfev={sol.nfev})')
lc = copy.deepcopy(lens); apply_x(lc, x)
lc.to_json('output/results/axisym_seed27_m6c_ga_lm.json')

# 验证
sz = compute_surface_z(lc.surfaces)
m, h0 = merit(x), trace_ray(lc, pos_all[0:1], dir_all[0:1])
print('\n=== 最终主光线 ===')
for hit in h0:
    i = hit[0]
    if i <= 6:
        err = np.hypot(hit[1][1]-ay[i-1], hit[1][2]-az[i-1])
        print(f'  M{i}: 锚点误差={err:.1f}mm')
    else:
        print(f'  像面: y={hit[1][1]:.1f} 出射({hit[2][1]:+.3f},{hit[2][2]:+.3f}) 远心={"OK" if abs(hit[2][1])<0.01 else "NO"}')
print('=== 最终孔径 ===')
ok = 0
for k in range(1, 7):
    h = trace_ray(lc, pos_all[k:k+1], dir_all[k:k+1])
    if h and h[-1][0] == 7:
        ok += 1
        print(f'  孔径{k}: 像面 y={h[-1][1][1]:8.1f}')
    else:
        print(f'  孔径{k}: 停在面{h[-1][0] if h else 0}')
print(f'孔径到像面: {ok}/6')
