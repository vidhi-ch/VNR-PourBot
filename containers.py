import numpy as np
from main import pad_pose, PAD_T, _f

class Spec:
    def __init__(self, kind, R, H_body, wall, bottom, grasp_z, fill, tilt,
                 neck_r=None, neck_h=0.0, neck_wall=0.006):
        self.kind, self.R, self.H_body, self.wall, self.bottom = kind, R, H_body, wall, bottom
        self.neck_r, self.neck_h, self.neck_wall = neck_r, neck_h, neck_wall
        self.H = H_body + neck_h
        self.open_r = (neck_r - neck_wall) if neck_r else (R - wall)    # liquid opening radius
        self.lip = np.array([0.0, neck_r or R, self.H])                 # pour pivot (+y side)
        self.grasp = np.array([0.0, 0.0, grasp_z])
        self.fill, self.tilt = fill, tilt                               # liquid box (x,y,z), max tilt deg
        target, lo, hi = 2*R + PAD_T + 0.001, 0.0, 0.7                  # finger angle closing on the body
        for _ in range(50):
            mid = 0.5*(lo+hi)
            sep = abs(pad_pose(mid, "L")[1] - pad_pose(mid, "R")[1])
            lo, hi = (mid, hi) if sep > target else (lo, mid)
        self.fing_closed = 0.5*(lo+hi)
        self.pad_d = pad_pose(self.fing_closed, "L")[2]

SPECS = {
    "glass":  Spec("glass",  R=0.045, H_body=0.12, wall=0.008, bottom=0.010, grasp_z=0.060,
                   fill=(0.050, 0.050, 0.058), tilt=115),
    "bottle": Spec("bottle", R=0.050, H_body=0.15, wall=0.008, bottom=0.010, grasp_z=0.075,
                   fill=(0.050, 0.050, 0.070), tilt=140, neck_r=0.030, neck_h=0.07),
}

def write_urdf(path, S, n=20, rgba="0.7 0.9 1 1", name=None):
    def geo(g, xyz, rp):
        o = f'<origin xyz="{_f(xyz)}" rpy="{_f(rp)}"/>'
        return (f'<visual>{o}<geometry>{g}</geometry><material name="m"><color rgba="{rgba}"/></material></visual>'
                f'<collision>{o}<geometry>{g}</geometry></collision>')
    def ring(r, wall, h, z, m):   # thin box staves around a circle
        out = ""
        for a in 2*np.pi*np.arange(m)/m:
            w = 2*r*np.tan(np.pi/m) + 0.003
            out += geo(f'<box size="{_f((wall, w, h))}"/>', ((r-wall/2)*np.cos(a), (r-wall/2)*np.sin(a), z), (0, 0, a))
        return out
    s = geo(f'<cylinder radius="{S.R}" length="{S.bottom}"/>', (0, 0, S.bottom/2), (0, 0, 0))
    s += ring(S.R, S.wall, S.H_body, S.H_body/2, n)
    if S.neck_r:                                                    # shoulder slabs + neck
        ri = S.neck_r - S.neck_wall; L = S.R - ri; rc = 0.5*(S.R + ri)
        for a in 2*np.pi*np.arange(n)/n:
            w = 2*S.R*np.tan(np.pi/n) + 0.003
            s += geo(f'<box size="{_f((L, w, 0.008))}"/>', (rc*np.cos(a), rc*np.sin(a), S.H_body - 0.004), (0, 0, a))
        s += ring(S.neck_r, S.neck_wall, S.neck_h, S.H_body + S.neck_h/2, 16)
    inert = (f'<inertial><origin xyz="0 0 {S.H/3:.4f}"/><mass value="0.25"/>'
             '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>')
    robot_name = name or S.kind
    path.write_text(f'<?xml version="1.0"?>\n<robot name="{robot_name}"><link name="base">{inert}{s}</link></robot>\n')