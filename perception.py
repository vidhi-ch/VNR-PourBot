import numpy as np
from scipy import ndimage

class CamModel:
    """Genesis camera + my own pinhole model (look-at basis, vertical fov) for depth back-projection."""
    def __init__(self, scene, res, pos, lookat, fov, up=(0, 0, 1)):
        self.res, self.pos, self.fov = res, np.array(pos, float), fov
        f = np.array(lookat, float) - self.pos; self.f = f/np.linalg.norm(f)
        r = np.cross(self.f, up); self.r = r/np.linalg.norm(r); self.u = np.cross(self.r, self.f)
        self.cam = scene.add_camera(res=res, pos=pos, lookat=lookat, up=up, fov=fov, GUI=False)

    def points(self):
        out = self.cam.render(depth=True)
        d = out[1] if isinstance(out, (tuple, list)) else out
        d = (d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)).astype(float)
        H, W = d.shape[:2]; fx = (H/2)/np.tan(np.deg2rad(self.fov)/2)
        px, py = np.arange(W) - (W-1)/2, np.arange(H) - (H-1)/2
        D = self.f + (px/fx)[None, :, None]*self.r - (py/fx)[:, None, None]*self.u
        P = self.pos + d[..., None]*D
        return P, np.isfinite(d) & (d > 0) & (d < 50)

def locate(P, valid, top, rmin=0.38, rmax=0.95, hmin=0.04):
    h = P[..., 2] - top; r = np.hypot(P[..., 0], P[..., 1])
    lab, n = ndimage.label(valid & (h > hmin) & (r > rmin) & (r < rmax))
    objs = []
    for k in range(1, n+1):
        m = lab == k
        if m.sum() < 120: continue
        hm = h[m].max(); rim = P[m & (h > hm - 0.012)]          # top ring -> object axis
        objs.append(dict(x=float(rim[:, 0].mean()), y=float(rim[:, 1].mean()), h=float(hm)))
    return objs

def classify(P, valid, o, top, r_side):
    m = valid & (np.hypot(P[..., 0]-o["x"], P[..., 1]-o["y"]) < 0.08) & (P[..., 2] > top + 0.012)
    if m.sum() > 40:
        pts = P[m]; z = pts[:, 2] - top; lat = np.abs((pts - [o["x"], o["y"], 0]) @ r_side); Hh = z.max()
        lo, hi = lat[(z > .25*Hh) & (z < .6*Hh)], lat[z > .82*Hh]
        if len(lo) > 5 and len(hi) > 5:
            return "bottle" if hi.max() < 0.7*lo.max() else "glass"
    return "bottle" if o["h"] > 0.18 else "glass"               # fallback: height from top camera

def perceive(cam_top, cam_side, top):
    Pt, vt = cam_top.points(); Ps, vs = cam_side.points()
    objs = locate(Pt, vt, top)
    for o in objs: o["type"] = classify(Ps, vs, o, top, cam_side.r)
    objs.sort(key=lambda o: np.arctan2(o["y"], o["x"]))
    for i, o in enumerate(objs): o["id"] = i + 1
    return objs