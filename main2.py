"""
UR5 + Robotiq 2F-140: pick a source container, pour liquid (SPH) into a target container,
put the source container back.  Circular ring table, robot on a central pedestal (Genesis).

Containers: BOTTLE (source) -> GLASS (target)

Run:   python main.py                      (viewer window, default layout)
       python main.py --check              (only plan + reachability check, no Genesis needed)
       python main.py --src 0.55 -22 --tgt 0.55 22      (polar: radius [m], angle [deg])
       python main.py --tune --record      (auto-tune pour offset, save pour.mp4)
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from controller import JointPID

# ============================ USER SETTINGS (metres / degrees) ============================
SOURCE_POLAR = (0.55, -22.0)   # (radius from robot axis, angle)  source container on the table
TARGET_POLAR = (0.55,  22.0)   # target container
POUR_DY = -0.05                # spout-lip offset from target centre along pour direction (neg = towards source)
POUR_CLEAR = 0.16              # lip height above target rim while pouring
SHOW_VIEWER, RECORD = True, False
# ==========================================================================================
HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "urdf" / "ur5_robotiq_140.urdf"
GEN = HERE / "generated"

# ---- timing / speed profile (module level: used by plan() and check_limits()) ----
SIM_DT = 4e-3
# per phase: (v_max, a_max). Linear phases: m/s, m/s^2.  Tilt phases: rad/s, rad/s^2.
SPEED = dict(
    approach  = (0.15, 0.6),    # slow, careful near the cup
    lift      = (0.25, 1.0),
    carry     = (0.60, 1.5),    # fast transfer source -> target (keep a_max low: liquid inside)
    tilt_fast = (0.60, 2.0),    # quick first part of the tilt (no liquid at the lip yet)
    tilt_slow = (0.45, 0.3),    # slow pour (57 deg at 0.15 rad/s ~ 12 s); lower = gentler flow
    back      = (0.50, 1.5),    # untilt after pouring
    ret       = (0.60, 1.5),    # fast return target -> source
    lower     = (0.25, 1.0),
    retreat   = (0.20, 0.8),
)
FIXED_T = dict(settle=0.4, to_pre=1.2, close=0.32, hold=1.2, open=0.32)   # seconds
META = {}
def n_frames(L, v, a):
    """minimum-jerk duration of a move of length L limited by speed v and accel a -> frame count"""
    T = max(1.875*L/v, np.sqrt(5.77*L/a)) if L > 1e-9 else 0.0
    return max(int(np.ceil(T/SIM_DT)), 10)

# ---- PID (arm) ----
TAU_MAX = np.array([150, 150, 150, 28, 28, 28], float)        # UR5 joint torque limits [Nm]
PID_KP = [4000, 4000, 2500, 500, 300, 200]
PID_KI = [100, 100, 60, 10, 5, 3]
PID_KD = [400, 400, 250, 40, 25, 15]

# ---- workspace (same names/values as scene_setup.py) ----
PEDESTAL_HEIGHT, PEDESTAL_RADIUS = 0.80, 0.18   # raised from 0.65 so the whole pour is reachable
TABLE_HEIGHT, TABLE_THICKNESS = 0.85, 0.06
TABLE_INNER_RADIUS, TABLE_OUTER_RADIUS = 0.45, 0.80
LEG_RADIUS, LEG_DISTANCE, NUMBER_OF_LEGS = 0.04, 0.68, 4
BASE = np.array([0.0, 0.0, PEDESTAL_HEIGHT])      # UR5 base_link (top of pedestal)
TOP = TABLE_HEIGHT                                  # table top surface z

# ---- containers (hollow cylinders, built from thin box staves so liquid can enter) ----
CUP_R, CUP_WALL, CUP_BOT, CUP_H = 0.055, 0.010, 0.012, 0.12    # source (grasped) container
TGT_R, TGT_WALL, TGT_BOT, TGT_H = 0.095, 0.010, 0.012, 0.14    # target container (fixed)
CUP_GAP = 0.001
LIP_LOCAL = np.array([0.0, CUP_R, CUP_H])          # spout lip (+y side of cup) = pour pivot
GRASP_LOCAL = np.array([0.0, 0.0, 0.075])          # pad centre in cup frame

# ---- UR5 kinematics (identical to the URDF) ----
H = np.pi / 2
UR5 = [((0,0,0.089159),(0,0,0),(0,0,1)), ((0,0.13585,0),(0,H,0),(0,1,0)),
       ((0,-0.1197,0.425),(0,0,0),(0,1,0)), ((0,0,0.39225),(0,H,0),(0,1,0)),
       ((0,0.093,0),(0,0,0),(0,0,1)), ((0,0,0.09465),(0,0,0),(0,1,0))]
LO, HI = np.full(6, -np.pi), np.full(6, np.pi)

def rpy(r): return R.from_euler("xyz", r).as_matrix()
def Tm(xyz, r):
    M = np.eye(4); M[:3,:3] = rpy(r); M[:3,3] = xyz; return M
T_W3_G = Tm((0,0.0823,0),(0,0,H)) @ Tm((0,0,0),(-1.57,1.57,-1.57))   # wrist_3 -> robotiq base (z = approach)

def fk(q):
    Rm = np.eye(3); p = BASE.copy(); pts = [p.copy()]
    for (xyz, rp, ax), qi in zip(UR5, q):
        p = p + Rm @ np.array(xyz); Rm = Rm @ rpy(rp) @ R.from_rotvec(np.array(ax,float)*qi).as_matrix(); pts.append(p.copy())
    T = np.eye(4); T[:3,:3] = Rm; T[:3,3] = p; T = T @ T_W3_G
    return T[:3,3], T[:3,:3], pts

def ik(Rt, pt, q0):
    def res(q):
        p, Rm, _ = fk(q); return np.concatenate([p - pt, 0.2*R.from_matrix(Rt @ Rm.T).as_rotvec()])
    s = least_squares(res, np.clip(q0, LO, HI), bounds=(LO, HI), xtol=1e-10, ftol=1e-10, gtol=1e-10)
    return s.x, np.abs(s.fun[:3]).max(), np.abs(s.fun[3:]).max()/0.2

# ---- Robotiq 140 finger geometry (from URDF) -> pad centre vs finger_joint angle ----
def _T(xyz, r, ax=None, q=0.0):
    M = Tm(xyz, r)
    if ax is not None: M[:3,:3] = M[:3,:3] @ R.from_rotvec(np.array(ax,float)*q).as_matrix()
    return M
def pad_pose(th, side):
    s = 1 if side == "L" else -1
    A = _T((0,-s*0.030601,0.054905), (2.2957963267948966,0,0 if s==1 else np.pi), (-s,0,0), s*th)
    B = _T((0,0.01821998610742,0.0260018192872234), (0,0,0))
    C = _T((0,0.0817554015893473,-0.0282203446692936), (-0.725,0,0), (1,0,0), th)
    D = _T((0,0.0457554015893473,-0.0272203446692936), (0,0,0))
    return (A @ B @ C @ D)[:3,3]
def _sep(th): return abs(pad_pose(th,"L")[1] - pad_pose(th,"R")[1])
PAD_T = 0.0075
_target_sep = 2*CUP_R + PAD_T + 0.001
lo_, hi_ = 0.0, 0.7
for _ in range(50):
    mid = 0.5*(lo_+hi_); lo_, hi_ = (mid, hi_) if _sep(mid) > _target_sep else (lo_, mid)
FING_OPEN, FING_CLOSED = 0.0, 0.5*(lo_+hi_)            # finger_joint angle [rad]
PAD_D = pad_pose(FING_CLOSED, "L")[2]                  # gripper base -> pad centre along approach axis
def gripper_q(th):  # finger_joint + its 5 mimic joints (set explicitly)
    return [th, -th, th, -th, -th, th]

# default container specs (containers.py provides richer ones for glass/bottle)
DEF_SRC = SimpleNamespace(lip=LIP_LOCAL, grasp=GRASP_LOCAL, pad_d=PAD_D, fing_closed=FING_CLOSED, tilt=115, H=CUP_H, open_r=CUP_R-CUP_WALL)
DEF_TGT = SimpleNamespace(lip=None, grasp=None, pad_d=None, fing_closed=None, tilt=None, H=TGT_H, open_r=TGT_R-TGT_WALL)

def smooth(s): return s**3*(10 - 15*s + 6*s*s)

def _cup_points(S, n=14):
    """points on the outer surface of the source container (cup frame) - used for collision checks"""
    R_ = getattr(S, "R", CUP_R); Hb = getattr(S, "H_body", S.H); nr = getattr(S, "neck_r", None)
    pts = [(0.0, 0.0, 0.0)]
    for ph in np.linspace(0, 2*np.pi, n, endpoint=False):
        for z in np.linspace(0, Hb, 5): pts.append((R_*np.cos(ph), R_*np.sin(ph), z))
        if nr:
            for z in np.linspace(Hb, S.H, 3): pts.append((nr*np.cos(ph), nr*np.sin(ph), z))
    return np.array(pts)

def hits_target(O, Rc, S, T, tgt_xy, margin=0.015):
    """True if the posed source container intersects the target container's solid body (body, plus neck for a bottle)"""
    P = _cup_points(S) @ Rc.T + O
    d = np.hypot(P[:, 0]-tgt_xy[0], P[:, 1]-tgt_xy[1]); z = P[:, 2] - TOP
    Hb = getattr(T, "H_body", T.H); nr = getattr(T, "neck_r", None)
    hit = (d < getattr(T, "R", TGT_R) + margin) & (z < Hb + margin)
    if nr: hit |= (d < nr + margin) & (z < T.H + margin)
    return bool(np.any(hit))

def plan(src_xy, tgt_xy, S=None, T=None, flip=1, tilt_deg=None, tilt_fast_deg=50, lift=0.12, pour_dy=None,
         pre_back=0.13, pour_clear=None, rise=0.07, carry_tilt_deg=30):
    S = S or DEF_SRC; T = T or DEF_TGT
    lip, grasp, pad_d, fing_closed = S.lip, S.grasp, S.pad_d, S.fing_closed
    tilt_deg = S.tilt if tilt_deg is None else tilt_deg
    if pour_dy is None: pour_dy = -0.6*T.open_r
    if pour_clear is None: pour_clear = 0.10 if (T and T.H > 0.15) else POUR_CLEAR
    src_xy, tgt_xy = np.asarray(src_xy, float), np.asarray(tgt_xy, float)
    u = (tgt_xy - src_xy) / np.linalg.norm(tgt_xy - src_xy)
    I = R.from_euler("z", np.arctan2(u[1], u[0]) - np.pi/2).as_matrix()
    R_CUP_G = np.column_stack([[0,0,flip], [flip,0,0], [0,1,0]])
    O_src = np.array([*src_xy, TOP + CUP_GAP])
    th_max, th_f = np.deg2rad(tilt_deg), np.deg2rad(tilt_fast_deg)
    th_c = np.deg2rad(min(carry_tilt_deg, tilt_fast_deg - 5))     # cup arrives already tilted a little (clears the rim)
    def tilt_pose_c(th, pc):
        P = np.array([*(tgt_xy + u*pour_dy), TOP + T.H + pc])
        Rc = I @ R.from_euler("x", -th).as_matrix(); return P - Rc @ lip, Rc
    def path_ok(pc):                                              # whole carry + tilt must stay clear of the target
        O_c, _ = tilt_pose_c(th_c, pc); O_l = O_src + [0, 0, max(lift, O_c[2] - O_src[2])]
        for sv in np.linspace(0, 1, 12):
            Rc = I @ R.from_euler("x", -th_c*sv).as_matrix()
            if hits_target(O_l + sv*(O_c - O_l), Rc, S, T, tgt_xy): return False
        return not any(hits_target(*tilt_pose_c(th, pc), S, T, tgt_xy) for th in np.linspace(th_c, th_max, 15))
    pc = pour_clear
    while pc < 0.45 and not path_ok(pc): pc += 0.01               # raise the pour height only as far as needed
    pour_clear = pc
    tilt_pose = lambda th: tilt_pose_c(th, pour_clear)
    O_pour, _ = tilt_pose(th_c)
    lift = max(lift, O_pour[2] - O_src[2])                        # rise straight up to carry height, then move level
    O_lift = O_src + [0, 0, lift]
    d_carry = float(np.linalg.norm(O_pour - O_lift))
    n = {k: int(round(t/SIM_DT)) for k, t in FIXED_T.items()}
    n["approach"]  = n_frames(pre_back,    *SPEED["approach"])
    n["lift"]      = n_frames(lift,        *SPEED["lift"])
    n["carry"]     = n_frames(d_carry,     *SPEED["carry"])
    n["tilt_fast"] = n_frames(th_f-th_c,   *SPEED["tilt_fast"])
    n["tilt_slow"] = n_frames(th_max-th_f, *SPEED["tilt_slow"])
    n["back"]      = n_frames(th_max-th_c, *SPEED["back"])
    n["ret"]       = n_frames(d_carry,     *SPEED["ret"])
    n["lower"]     = n_frames(lift,        *SPEED["lower"])
    n["retreat"]   = n_frames(pre_back,    *SPEED["retreat"])
    n["rise"]      = n_frames(rise,        *SPEED["lift"])
    F = []
    def add(phase, O, Rc, fing, att, back=0.0, dz=0.0):
        RG = Rc @ R_CUP_G; pG = O + Rc @ grasp
        F.append(dict(Rt=RG, pt=pG - RG @ np.array([0,0,pad_d+back]) + np.array([0,0,dz]), fing=fing, attached=att, cup=(O,Rc), phase=phase))
    for _ in range(n["settle"]): F.append(dict(phase="settle", jq=0.0))
    for i in range(n["to_pre"]): F.append(dict(phase="to_pre", jq=smooth((i+1)/n["to_pre"])))
    for i in range(n["approach"]): add("approach", O_src, I, FING_OPEN, False, pre_back*(1-smooth((i+1)/n["approach"])))
    for i in range(n["close"]): add("close", O_src, I, FING_OPEN + smooth((i+1)/n["close"])*(fing_closed-FING_OPEN), False)
    for i in range(n["lift"]): add("lift", O_src + smooth((i+1)/n["lift"])*(O_lift-O_src), I, fing_closed, True)
    for i in range(n["carry"]):
        sv = smooth((i+1)/n["carry"]); add("carry", O_lift + sv*(O_pour-O_lift), I @ R.from_euler("x", -th_c*sv).as_matrix(), fing_closed, True)
    for i in range(n["tilt_fast"]):
        O, Rc = tilt_pose(th_c + (th_f-th_c)*smooth((i+1)/n["tilt_fast"])); add("tilt", O, Rc, fing_closed, True)
    for i in range(n["tilt_slow"]):
        O, Rc = tilt_pose(th_f + (th_max-th_f)*smooth((i+1)/n["tilt_slow"])); add("tilt", O, Rc, fing_closed, True)
    for _ in range(n["hold"]):
        O, Rc = tilt_pose(th_max); add("hold", O, Rc, fing_closed, True)
    for i in range(n["back"]):
        O, Rc = tilt_pose(th_c + (th_max-th_c)*(1-smooth((i+1)/n["back"]))); add("back", O, Rc, fing_closed, True)
    for i in range(n["ret"]):
        sv = smooth((i+1)/n["ret"]); add("ret", O_pour + sv*(O_lift-O_pour), I @ R.from_euler("x", -th_c*(1-sv)).as_matrix(), fing_closed, True)
    for i in range(n["lower"]): add("lower", O_lift + smooth((i+1)/n["lower"])*(O_src-O_lift), I, fing_closed, True)
    for i in range(n["open"]): add("open", O_src, I, fing_closed + smooth((i+1)/n["open"])*(FING_OPEN-fing_closed), False)
    for i in range(n["retreat"]): add("retreat", O_src, I, FING_OPEN, False, pre_back*smooth((i+1)/n["retreat"]))
    for i in range(n["rise"]): add("rise", O_src, I, FING_OPEN, False, pre_back, rise*smooth((i+1)/n["rise"]))
    META.clear(); META.update(I=I, lip=lip, grasp=grasp, pad_d=pad_d, R_CUP_G=R_CUP_G, u=u, pour_dy=pour_dy,
                              pc=pour_clear, th_c=th_c, th_max=th_max, tgt_xy=tgt_xy)
    return F

def _best_ik(Rt, pt, seeds=40):
    best = None
    for k in range(seeds):
        q0 = np.random.default_rng(k).uniform(-3, 3, 6)
        q, ep, er = ik(Rt, pt, q0)
        if ep < 1e-5 and er < 1e-4:
            _, _, pts = fk(q)
            score = np.abs(q).sum() + (5.0 if pts[3][2] < pts[1][2] + 0.1 else 0)   # prefer small motion, elbow up
            if best is None or score < best[0]: best = (score, q)
    return None if best is None else best[1]

def solve_plan(F, stride=4):
    """IK every stride-th frame (warm-started), joint interpolation between. Returns Q (N,12), errors, q_home, jump."""
    first = next(f for f in F if f["phase"] == "approach")
    q_pre = _best_ik(first["Rt"], first["pt"])
    if q_pre is None: raise RuntimeError("pre-grasp pose unreachable")
    q_home = _best_ik(first["Rt"], first["pt"] + [0,0,0.22])
    if q_home is None: q_home = q_pre
    N = len(F); Q = np.zeros((N, 12))
    ik_idx = [i for i, f in enumerate(F) if f["phase"] not in ("settle", "to_pre")]
    solved = ik_idx[::stride] + ([ik_idx[-1]] if (len(ik_idx)-1) % stride else [])
    qs, q_prev = [], q_pre
    for i in solved:
        q_prev, _, _ = ik(F[i]["Rt"], F[i]["pt"], q_prev); qs.append(q_prev.copy())
    qs = np.array(qs)
    for i, f in enumerate(F):
        ph = f["phase"]
        if ph == "settle": Q[i,:6] = q_home; Q[i,6:] = gripper_q(FING_OPEN)
        elif ph == "to_pre": Q[i,:6] = q_home + f["jq"]*(q_pre - q_home); Q[i,6:] = gripper_q(FING_OPEN)
        else:
            for j in range(6): Q[i,j] = np.interp(i, solved, qs[:,j])
            Q[i,6:] = gripper_q(f["fing"])
    errs = []
    for i in ik_idx[::2]:
        p, Rm, _ = fk(Q[i,:6]); errs.append((np.abs(p - F[i]["pt"]).max(), np.abs(R.from_matrix(F[i]["Rt"] @ Rm.T).as_rotvec()).max()))
    jump = np.abs(np.diff(qs, axis=0)).max()
    return Q, np.array(errs), q_home, jump

def make_plan(src_xy, tgt_xy, S=None, T=None, pour_dy=None, pour_clear=None):
    """try both gripper roll variants, return the first that is reachable -> (F, Q, q_home)"""
    last = ""
    for flip in (1, -1):
        try:
            F = plan(src_xy, tgt_xy, S, T, flip=flip, pour_dy=pour_dy, pour_clear=pour_clear)
            Q, errs, q_home, jump = solve_plan(F)
        except RuntimeError as e:
            last = str(e); continue
        if errs[:,0].max() < 0.005 and errs[:,1].max() < 0.05 and jump < 0.6:
            return F, Q, q_home
        last = f"IK error {1000*errs[:,0].max():.1f} mm / {errs[:,1].max():.2f} rad, max joint jump {jump:.2f} rad"
    raise RuntimeError(f"Layout not reachable ({last}). Move containers closer / change angle between them "
                       f"(pour direction should not point back towards the robot).")

def check_limits(Q, vmax=np.pi):
    qd = np.diff(Q[:, :6], axis=0) / SIM_DT
    qdd = np.diff(qd, axis=0) / SIM_DT
    print(f"plan: {len(Q)} frames ({len(Q)*SIM_DT:.0f} s) | peak joint speed {np.abs(qd).max():.2f} rad/s | peak joint accel {np.abs(qdd).max():.1f} rad/s^2")
    if np.abs(qd).max() > vmax:
        print("warning: joint speed above limit -> lower SPEED['carry'][0] / SPEED['ret'][0]")

def polar(r, deg): a = np.deg2rad(deg); return np.array([r*np.cos(a), r*np.sin(a)])

# ============================ model files ============================
def _f(v): return " ".join(f"{x:.6f}" for x in v)
def container_urdf(path, name, r, h, wall, bot, rgba, n=20, mass=0.2):
    inert = (f'<inertial><origin xyz="0 0 {h/3:.4f}"/><mass value="{mass}"/>'
             '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>')
    def geo(g, xyz, rp):
        o = f'<origin xyz="{_f(xyz)}" rpy="{_f(rp)}"/>'
        return (f'<visual>{o}<geometry>{g}</geometry><material name="m"><color rgba="{rgba}"/></material></visual>'
                f'<collision>{o}<geometry>{g}</geometry></collision>')
    s = geo(f'<cylinder radius="{r}" length="{bot}"/>', (0,0,bot/2), (0,0,0))
    w = 2*r*np.tan(np.pi/n) + 0.003
    for k in range(n):
        a = 2*np.pi*k/n; rc = r - wall/2
        s += geo(f'<box size="{_f((wall, w, h))}"/>', (rc*np.cos(a), rc*np.sin(a), h/2), (0,0,a))
    path.write_text(f'<?xml version="1.0"?>\n<robot name="{name}"><link name="base">{inert}{s}</link></robot>\n')

def table_urdf(path, n=48):
    ri, ro, t = TABLE_INNER_RADIUS, TABLE_OUTER_RADIUS, TABLE_THICKNESS
    L, rc, w = ro - ri, 0.5*(ri+ro), 2*ro*np.tan(np.pi/n) + 0.004
    s = ""
    for k in range(n):
        a = 2*np.pi*k/n
        o = f'<origin xyz="{_f((rc*np.cos(a), rc*np.sin(a), TABLE_HEIGHT - t/2))}" rpy="0 0 {a:.6f}"/>'
        g = f'<geometry><box size="{_f((L, w, t))}"/></geometry>'
        s += f'<visual>{o}{g}<material name="t"><color rgba="0.62 0.45 0.30 1"/></material></visual><collision>{o}{g}</collision>'
    path.write_text('<?xml version="1.0"?>\n<robot name="ring_table"><link name="base"><inertial><mass value="10"/>'
                    '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>' + s + '</link></robot>\n')

# ============================ simulation ============================
def main():
    ap = argparse.ArgumentParser(description="UR5 Container Pouring Simulation")
    ap.add_argument("--src", nargs=2, type=float, metavar=("R", "DEG"))
    ap.add_argument("--tgt", nargs=2, type=float, metavar=("R", "DEG"))
    ap.add_argument("--pair", type=str, choices=["bottle_to_glass", "glass_to_bottle", "bottle_to_bottle", "glass_to_glass"],
                    help="Container pair: bottle_to_glass, glass_to_bottle, bottle_to_bottle, glass_to_glass")
    ap.add_argument("--src-type", type=str, choices=["bottle", "glass"], help="Source container type")
    ap.add_argument("--tgt-type", type=str, choices=["bottle", "glass"], help="Target container type")
    ap.add_argument("--pour-dy", type=float, default=None, help="Spout-lip offset from target center (default: auto)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    a = ap.parse_args()

    # ---- Determine container pair ----
    from containers import SPECS, write_urdf

    src_kind, tgt_kind = None, None
    if a.src_type and a.tgt_type:
        src_kind, tgt_kind = a.src_type, a.tgt_type
    elif a.pair:
        src_kind, tgt_kind = a.pair.split("_to_")
    elif sys.stdin.isatty():
        print("\n" + "="*50)
        print("  Select Pouring Configuration:")
        print("  1) Bottle -> Glass")
        print("  2) Glass -> Bottle")
        print("  3) Bottle -> Bottle")
        print("  4) Glass -> Glass")
        print("="*50)
        choice = input("Enter choice [1-4] (default 1): ").strip().lower()
        pair_map = {
            "1": ("bottle", "glass"),
            "2": ("glass", "bottle"),
            "3": ("bottle", "bottle"),
            "4": ("glass", "glass"),
            "bottle to glass": ("bottle", "glass"),
            "glass to bottle": ("glass", "bottle"),
            "bottle to bottle": ("bottle", "bottle"),
            "glass to glass": ("glass", "glass"),
            "bottle_to_glass": ("bottle", "glass"),
            "glass_to_bottle": ("glass", "bottle"),
            "bottle_to_bottle": ("bottle", "bottle"),
            "glass_to_glass": ("glass", "glass"),
        }
        src_kind, tgt_kind = pair_map.get(choice, ("bottle", "glass"))
    else:
        src_kind, tgt_kind = "bottle", "glass"

    S_spec = SPECS[src_kind]
    T_spec = SPECS[tgt_kind]

    src = polar(*(a.src or SOURCE_POLAR)); tgt = polar(*(a.tgt or TARGET_POLAR))
    for nm, p, sp in (("source", src, S_spec), ("target", tgt, T_spec)):
        r = np.hypot(*p)
        if not (TABLE_INNER_RADIUS + sp.R < r < TABLE_OUTER_RADIUS - sp.R + 0.02):
            print(f"warning: {nm} at r={r:.2f} m is near the table edge / hole")
    if not URDF_PATH.exists(): sys.exit(f"URDF not found: {URDF_PATH}")

    # Color themes
    colors = {
        "bottle": {"rgba": "0.2 0.5 0.9 0.8", "rgb": (0.2, 0.5, 0.9)},       # blue
        "glass":  {"rgba": "0.85 0.9 0.95 0.5", "rgb": (0.85, 0.9, 0.95)},   # glass
    }
    s_rgba = colors[src_kind]["rgba"]
    s_rgb = colors[src_kind]["rgb"]
    if src_kind == tgt_kind:
        # Give target a contrasting green tint if same type
        if tgt_kind == "bottle":
            t_rgba, t_rgb = "0.2 0.7 0.3 0.8", (0.2, 0.7, 0.3)
        else:
            t_rgba, t_rgb = "0.95 0.8 0.6 0.5", (0.95, 0.8, 0.6)
    else:
        t_rgba = colors[tgt_kind]["rgba"]
        t_rgb = colors[tgt_kind]["rgb"]

    print(f"source: {src_kind.upper()} at {np.round(src,3)} | target: {tgt_kind.upper()} at {np.round(tgt,3)}")
    print(f"finger_joint closed = {S_spec.fing_closed:.3f} rad, pad depth {S_spec.pad_d:.3f} m, max tilt = {S_spec.tilt} deg")
    print("planning + IK (can take up to a minute) ...")
    pour_dy = a.pour_dy if a.pour_dy is not None else -0.6 * T_spec.open_r
    F, Q, q_home = make_plan(src, tgt, S=S_spec, T=T_spec, pour_dy=pour_dy)
    print(f"plan OK: all poses reachable"); check_limits(Q)
    if a.check: return

    import genesis as gs
    GEN.mkdir(exist_ok=True)
    write_urdf(GEN/"source_container.urdf", S_spec, rgba=s_rgba, name="source_container")
    write_urdf(GEN/"target_container.urdf", T_spec, rgba=t_rgba, name="target_container")
    table_urdf(GEN/"ring_table.urdf")
    try: gs.init(backend=gs.gpu)
    except Exception as e: print("gs.init note:", e)

    lo = np.minimum(src, tgt) - 0.30; hi = np.maximum(src, tgt) + 0.30
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=10),
        sph_options=gs.options.SPHOptions(lower_bound=(lo[0], lo[1], TOP - 0.05), upper_bound=(hi[0], hi[1], TOP + 0.75), particle_size=0.005),
        viewer_options=gs.options.ViewerOptions(res=(1280, 800), camera_pos=(1.6, -1.9, 1.7), camera_lookat=(0.3, 0.0, 0.9),
                                                camera_fov=45, refresh_rate=60),
        show_viewer=SHOW_VIEWER and not a.no_viewer)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Cylinder(radius=PEDESTAL_RADIUS, height=PEDESTAL_HEIGHT, pos=(0,0,PEDESTAL_HEIGHT/2), fixed=True))
    scene.add_entity(gs.morphs.URDF(file=str(GEN/"ring_table.urdf"), fixed=True))
    leg_h = TABLE_HEIGHT - TABLE_THICKNESS
    for i in range(NUMBER_OF_LEGS):
        ang = 2*np.pi*i/NUMBER_OF_LEGS
        scene.add_entity(gs.morphs.Cylinder(radius=LEG_RADIUS, height=leg_h, pos=(LEG_DISTANCE*np.cos(ang), LEG_DISTANCE*np.sin(ang), leg_h/2), fixed=True))
    # target container (fixed on table)
    scene.add_entity(gs.morphs.URDF(file=str(GEN/"target_container.urdf"), pos=(*tgt, TOP), fixed=True), surface=gs.surfaces.Default(color=t_rgb))
    robot = scene.add_entity(gs.morphs.URDF(file=str(URDF_PATH), pos=tuple(BASE), fixed=True))
    # source container (picked and poured)
    cup = scene.add_entity(gs.morphs.URDF(file=str(GEN/"source_container.urdf"), pos=(*src, TOP + CUP_GAP), fixed=False),
                           surface=gs.surfaces.Default(color=s_rgb))
    fx, fy, fh = S_spec.fill
    liquid = scene.add_entity(morph=gs.morphs.Box(pos=(*src, TOP + CUP_GAP + S_spec.bottom + 0.004 + fh/2), size=(fx, fy, fh)),
                              material=gs.materials.SPH.Liquid(), surface=gs.surfaces.Default(color=(0.4,0.8,1.0), vis_mode="particle"))
    cam = None
    if RECORD or a.record:
        cam = scene.add_camera(res=(1280, 720), pos=(1.4, -1.6, 1.6), lookat=(0.35, 0.0, 0.95), fov=45, GUI=False)
    scene.build()

    names = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint",
             "finger_joint","left_inner_knuckle_joint","left_inner_finger_joint","right_outer_knuckle_joint",
             "right_inner_knuckle_joint","right_inner_finger_joint"]
    def _dof(n):
        d = getattr(robot.get_joint(n), "dofs_idx_local", None)
        if d is None: d = robot.get_joint(n).dof_idx_local
        return int(d[0] if hasattr(d, "__len__") else d)
    dofs = [_dof(n) for n in names]
    arm_dofs, grip_dofs = dofs[:6], dofs[6:]
    def set_q(q): robot.set_dofs_position(np.asarray(q, dtype=np.float32), dofs_idx_local=dofs)
    def set_cup(O, Rc):
        x, y, z, w = R.from_matrix(Rc).as_quat()
        cup.set_pos(np.asarray(O, dtype=np.float32), zero_velocity=True)
        cup.set_quat(np.array([w, x, y, z], dtype=np.float32), zero_velocity=True)
    def _np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    def particles():
        for nm in ("get_particles_pos", "get_particles"):
            fn = getattr(liquid, nm, None)
            if fn is None: continue
            try: return _np(fn()).reshape(-1, 3)
            except Exception: pass
        return None

    # ---- PID for the arm, simulator PD for the gripper ----
    pid = JointPID(kp=PID_KP, ki=PID_KI, kd=PID_KD, dt=SIM_DT, tau_max=TAU_MAX)
    robot.set_dofs_kp(np.full(6, 500.0), dofs_idx_local=grip_dofs)
    robot.set_dofs_kv(np.full(6, 30.0), dofs_idx_local=grip_dofs)
    robot.set_dofs_force_range(-TAU_MAX, TAU_MAX, dofs_idx_local=arm_dofs)
    def drive(Qp, i):
        q  = _np(robot.get_dofs_position(dofs_idx_local=arm_dofs)).astype(float).ravel()
        qd = _np(robot.get_dofs_velocity(dofs_idx_local=arm_dofs)).astype(float).ravel()
        qd_ref = (Qp[i, :6] - Qp[i-1, :6]) / SIM_DT if i > 0 else np.zeros(6)   # velocity feedforward
        robot.control_dofs_force(pid(Qp[i, :6], q, qd, qd_ref), dofs_idx_local=arm_dofs)
        robot.control_dofs_position(np.asarray(Qp[i, 6:], dtype=np.float32), dofs_idx_local=grip_dofs)
        return Qp[i, :6] - q                                                      # tracking error

    u = (tgt - src)/np.linalg.norm(tgt - src)
    in_tgt = lambda p: (np.hypot(p[:,0]-tgt[0], p[:,1]-tgt[1]) < T_spec.open_r) & (p[:,2] > TOP - 0.01) & (p[:,2] < TOP + T_spec.H + 0.02)
    def cup_mask(p, O, Rc):
        l = (p - O) @ Rc; return (np.hypot(l[:,0], l[:,1]) < S_spec.R) & (l[:,2] > 0) & (l[:,2] < S_spec.H + 0.03)

    idx = lambda ph: [i for i, f in enumerate(F) if f["phase"] == ph]
    carry_end, hold_end = idx("carry")[-1], idx("hold")[-1]

    def run_once(Fp, Qp, record=False):
        scene.reset(); res = {}
        set_q(Qp[0]); pid.reset()                      # start exactly on the plan
        max_err = 0.0
        if record and cam is not None: cam.start_recording()
        for i, f in enumerate(Fp):
            err = drive(Qp, i); max_err = max(max_err, float(np.abs(err).max()))
            if f.get("attached"): set_cup(*f["cup"])    # virtual weld while the fingers hold the cup
            scene.step()
            if record and cam is not None and i % 4 == 0: cam.render()
            if i == carry_end and particles() is not None: res["kept"] = cup_mask(particles(), *f["cup"]).mean()
            if i == hold_end and particles() is not None: res["p_hold"], res["cup_hold"] = particles(), f["cup"]
        if record and cam is not None: cam.stop_recording(save_to_filename=str(HERE/"pour.mp4"), fps=30); print("saved pour.mp4")
        res["p_end"] = particles()
        print(f"max joint tracking error: {np.degrees(max_err):.2f} deg")
        return res

    def summarize(res):
        pe, ph = res["p_end"], res["p_hold"]
        t = in_tgt(pe); c = (np.hypot(pe[:,0]-src[0], pe[:,1]-src[1]) < S_spec.R) & (pe[:,2] > TOP - 0.01) & ~t
        out = (np.hypot(ph[:,0]-tgt[0], ph[:,1]-tgt[1]) < 0.30) & ~cup_mask(ph, *res["cup_hold"])
        err_u = float(((ph[out,:2] - tgt) @ u).mean()) if out.sum() > 20 else 0.0
        return dict(tgt=100*t.mean(), cup=100*c.mean(), spill=100*(1-t.mean()-c.mean()), kept=100*res.get("kept", np.nan), err_u=err_u)
    def show(dy, s): print(f"POUR_DY={dy:+.3f} | in TARGET {s['tgt']:.0f}% | left in source {s['cup']:.0f}% | spilled {s['spill']:.0f}% | "
                           f"kept during carry {s['kept']:.0f}% | landing offset along pour dir {1000*s['err_u']:+.0f} mm")

    dy = pour_dy
    if a.tune and particles() is not None:
        for k in range(3):
            res = run_once(F, Q)
            if res.get("p_hold") is None: break
            s = summarize(res); print(f"trial {k+1}:", end=" "); show(dy, s)
            if s["tgt"] >= 95: break
            dy = float(np.clip(dy - 0.8*s["err_u"], -0.15, 0.05))
            F, Q, q_home = make_plan(src, tgt, S=S_spec, T=T_spec, pour_dy=dy); check_limits(Q)
            carry_end, hold_end = idx("carry")[-1], idx("hold")[-1]
        print("using POUR_DY =", round(dy, 3))
    res = run_once(F, Q, record=True)
    if res.get("p_end") is not None and "p_hold" in res: print("FINAL:", end=" "); show(dy, summarize(res))
    print("done. Close the viewer window to exit.")
    while SHOW_VIEWER and not a.no_viewer:
        drive(Q, len(Q) - 1); scene.step()

if __name__ == "__main__":
    main()