import argparse, os, sys, numpy as np, tkinter as tk, torch
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
import main3 as M
from containers import SPECS, write_urdf

# ---- pouring tuning (these edit main2.py's tables in place) ----
M.SPEED["tilt_fast"] = (0.90, 3.0)
M.SPEED["tilt_slow"] = (0.40, 0.60)
M.SPEED["carry"]     = (0.45, 1.00)
M.SPEED["lift"]      = (0.25, 0.80)
M.SPEED["back"]      = (0.60, 1.50)
M.SPEED["ret"]       = (0.60, 1.50)
M.FIXED_T["hold"]    = 1.5
POUR_CLEAR_NEW = 0.008  # pivot line clearance above target rim [m]
POUR_SIDE_GAP  = 0.012  # horizontal gap between source pivot and target outer wall [m]

# ---- arm control: Genesis PD (kp, kv) + integral action done here  => PID ----
CONTROL = "pid"                       # "pid" | "kinematic" (teleport joints, for debugging)
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000] + [500]*6, float)
KV = np.array([450, 450, 350, 350, 200, 200] + [30]*6, float)
KI = np.array([5.0]*6 + [0.0]*6)      # integral gain [1/s]: target offset = KI * integral(error); gripper has none
I_LIM = 0.05                          # max integral offset [rad]
FMAX = np.array([150, 150, 150, 28, 28, 28] + [60]*6, float)   # torque limits [Nm]

# ---- sensor-driven pouring (spill avoidance) ----
ADAPTIVE_POUR = True
UPD_STEPS   = 12                      # controller update every 12 sim steps (48 ms)
POUR_RATE   = 0.22                    # rad/s tilt rate once liquid is flowing
FAST_RATE   = 0.80                    # rad/s before any liquid has left the source
BACK_DEG    = 6.0                     # tilt back by this much when the stream touches / nears the wall
WAIT_S      = 0.7                     # pause after a back-off before pouring again [s]
SHIFT       = 0.012                   # hand shift per back-off [m] (towards the target centre)
SHIFT_MAX   = 0.04
SHIFT_RATE  = 0.05                    # m/s
WALL_N      = 3                       # stream particles near the wall that trigger a back-off
WALL_BAND   = 0.012                   # "close to the wall" distance [m]
AIR_MAX     = 30                      # more than this many particles in the air -> pause tilting (flow limiter)
MAX_BACKOFFS = 8
DONE_FRAC   = 0.04                    # pour finished when less than this fraction is left in the source
MAX_POUR_S, HOLD_END_S = 45.0, 4.0
PRED_MARGIN = 0.012             # predicted landing must stay this far inside the target opening [m]
PRED_N      = 3                       # particles predicted to miss the opening that trigger a back-off
TAIL_FRAC   = 0.25                    # below this fraction left in the source the tilt rate is halved
TAIL_REPOSITION_FRAC = 0.10
TAIL_BACK_M = 0.025
TAIL_UNTILT_DEG = 8.0
TAIL_WAIT = 0.7
CLEAR_S     = 0.8                     # stream must be gone for this long before the container is untilted [s]
UNTILT_SLOW = 0.35                    # rad/s, first 25 deg of untilting

SRC_POLAR, TGT_POLAR = (0.55, -22.0), (0.55, 22.0)
RMIN, RMAX, MIN_SEP = 0.50, 0.74, 0.17
Q_PARK = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0.0])
COLOR = {"glass": (0.75, 0.9, 1.0, 0.4), "bottle": (0.2, 0.6, 0.3, 0.5)}
COLOR_TGT_SAME = {"glass": (0.95, 0.8, 0.6, 0.5), "bottle": (0.2, 0.4, 0.9, 0.6)}
PAIRS = [("Glass  ->  Bottle", ("glass", "bottle")), ("Bottle  ->  Glass", ("bottle", "glass")),
         ("Bottle  ->  Bottle", ("bottle", "bottle")), ("Glass  ->  Glass", ("glass", "glass"))]
STEPS_PER_TICK = UPD_STEPS

def ask_pair(root):
    var = tk.StringVar()
    fr = ttk.Frame(root); fr.pack(padx=24, pady=16)
    ttk.Label(fr, text="Select the pouring process", font=("TkDefaultFont", 12, "bold")).pack(pady=(0, 10))
    for txt, kinds in PAIRS:
        ttk.Button(fr, text=txt, width=26, command=lambda k=kinds: var.set("_".join(k))).pack(pady=3)
    root.protocol("WM_DELETE_WINDOW", lambda: var.set("quit"))
    root.wait_variable(var); fr.destroy()
    if var.get() == "quit": sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    return var.get().split("_")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-viewer", action="store_true"); a = ap.parse_args()
    root = tk.Tk(); root.title("Pouring control"); root.attributes("-topmost", True)
    src_kind, tgt_kind = ask_pair(root)
    wait = ttk.Label(root, text=f"Building scene: {src_kind} -> {tgt_kind} ...", padding=30); wait.pack(); root.update()

    import genesis as gs
    M.GEN.mkdir(exist_ok=True)
    for role, kind in (("src", src_kind), ("tgt", tgt_kind)):
        write_urdf(
            M.GEN/f"{role}_{kind}.urdf",
            SPECS[kind],
            marker_role="source" if role == "src" else "target"
        )
    M.table_urdf(M.GEN/"ring_table.urdf")
    try: gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    except Exception as e: print("gs.init:", e)
    TOP = M.TOP; DT = M.SIM_DT
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=14),
        rigid_options=gs.options.RigidOptions(max_collision_pairs=5000),
        sph_options=gs.options.SPHOptions(lower_bound=(-0.95, -0.95, 0.0), upper_bound=(0.95, 0.95, TOP+0.7), particle_size=0.006),
        viewer_options=gs.options.ViewerOptions(res=(1280, 800), camera_pos=(1.6, -1.9, 1.7), camera_lookat=(0.3, 0, 0.9), camera_fov=45, refresh_rate=60),
        show_viewer=not a.no_viewer)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Cylinder(radius=M.PEDESTAL_RADIUS, height=M.PEDESTAL_HEIGHT, pos=(0, 0, M.PEDESTAL_HEIGHT/2), fixed=True))
    scene.add_entity(gs.morphs.URDF(file=str(M.GEN/"ring_table.urdf"), fixed=True))
    lh = M.TABLE_HEIGHT - M.TABLE_THICKNESS
    for i in range(M.NUMBER_OF_LEGS):
        t = 2*np.pi*i/M.NUMBER_OF_LEGS
        scene.add_entity(gs.morphs.Cylinder(radius=M.LEG_RADIUS, height=lh, pos=(M.LEG_DISTANCE*np.cos(t), M.LEG_DISTANCE*np.sin(t), lh/2), fixed=True))
    robot_morph = gs.morphs.URDF(file=str(M.URDF_PATH), pos=tuple(M.BASE), fixed=True)
    try: robot = scene.add_entity(robot_morph, material=gs.materials.Rigid(gravity_compensation=1.0))   # arm does not sag under gravity
    except Exception as e:
        print("gravity_compensation not available, using default material:", e)
        robot = scene.add_entity(gs.morphs.URDF(file=str(M.URDF_PATH), pos=tuple(M.BASE), fixed=True))

    items = []                                                # [source, target]
    for role, kind, (r, deg) in (("src", src_kind, SRC_POLAR), ("tgt", tgt_kind, TGT_POLAR)):
        xy = M.polar(r, deg)
        col = COLOR_TGT_SAME[kind] if (role == "tgt" and src_kind == tgt_kind) else COLOR[kind]
        ent = scene.add_entity(gs.morphs.URDF(file=str(M.GEN/f"{role}_{kind}.urdf"), pos=(*xy, TOP+M.CUP_GAP), fixed=False),
                               surface=gs.surfaces.Default(color=col))
        S_ = SPECS[kind]; fx, fy, fh = S_.fill
        liq = scene.add_entity(morph=gs.morphs.Box(pos=(*xy, TOP+M.CUP_GAP+S_.bottom+0.004+fh/2), size=(fx, fy, fh)),
                               material=gs.materials.SPH.Liquid(),
                               surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0) if role == "src" else (0.3, 0.55, 1.0), vis_mode="particle"))
        items.append(dict(role=role, kind=kind, S=S_, ent=ent, liq=liq, xy=xy.copy(), goal=xy.copy(), xy0=xy.copy()))
    src_it, tgt_it = items

    scene.build()
    wait.destroy()

    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
             "finger_joint", "left_inner_knuckle_joint", "left_inner_finger_joint", "right_outer_knuckle_joint",
             "right_inner_knuckle_joint", "right_inner_finger_joint"]
    def _dof(n):
        d = getattr(robot.get_joint(n), "dofs_idx_local", None)
        if d is None: d = robot.get_joint(n).dof_idx_local
        return int(d[0] if hasattr(d, "__len__") else d)
    dofs = [_dof(n) for n in names]
    _np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    set_q = lambda q: robot.set_dofs_position(np.asarray(q, dtype=np.float32), dofs_idx_local=dofs)
    Q_PARK12 = np.concatenate([Q_PARK, M.gripper_q(0.0)])

    if CONTROL == "pid":
        robot.set_dofs_kp(KP, dofs_idx_local=dofs); robot.set_dofs_kv(KV, dofs_idx_local=dofs)
        robot.set_dofs_force_range(-FMAX, FMAX, dofs_idx_local=dofs)
    integ = np.zeros(12); INTEG_MAX = I_LIM/np.maximum(KI, 1e-9)

    def command(qref):
        """PID: Genesis PD (kp,kv) toward (qref + KI*integral(error)). Returns arm tracking error."""
        qref = np.asarray(qref, float)
        if CONTROL != "pid": set_q(qref); return np.zeros(6)
        q = _np(robot.get_dofs_position(dofs_idx_local=dofs)).astype(float).ravel()
        e = qref - q
        integ[:] = np.clip(integ + e*DT, -INTEG_MAX, INTEG_MAX)
        robot.control_dofs_position((qref + KI*integ).astype(np.float32), dofs_idx_local=dofs)
        return e[:6]

    def put(it, O, Rc=np.eye(3)):
        x, y, z, w = R.from_matrix(Rc).as_quat()
        it["ent"].set_pos(np.asarray(O, dtype=np.float32), zero_velocity=True)
        it["ent"].set_quat(np.array([w, x, y, z], dtype=np.float32), zero_velocity=True)
    def particles(e):
        for nm in ("get_particles_pos", "get_particles"):
            fn = getattr(e, nm, None)
            if fn is None: continue
            try: return _np(fn()).reshape(-1, 3)
            except Exception: pass
        return None
    def set_particles(liq, P):
        err = None
        for nm in ("set_position", "set_pos"):
            fn = getattr(liq, nm, None)
            if fn is None: continue
            for arg in (np.asarray(P, dtype=np.float32), torch.as_tensor(P, dtype=torch.float32, device=gs.device)):
                try: fn(arg); return True
                except Exception as e: err = e
        print("could not move particles:", err); return False
    def shift_water(it, d):
        p = particles(it["liq"])
        if p is not None: set_particles(it["liq"], p + np.array([d[0], d[1], 0.0]))

    # ---------------- GUI ----------------
    st = dict(mode="idle", seg="plan", maxerr=0.0)
    status = tk.StringVar(value="Move the containers if you like, then press START SIMULATION")
    ttk.Label(root, text=f"{src_kind.upper()} (source)   ->   {tgt_kind.upper()} (target)", font=("TkDefaultFont", 11, "bold")).pack(padx=10, pady=(10, 4))
    mv_v = tk.StringVar(value="src")
    r0 = ttk.Frame(root); r0.pack(pady=2)
    ttk.Label(r0, text="Move:").pack(side="left", padx=4)
    ttk.Radiobutton(r0, text=f"Source ({src_kind})", variable=mv_v, value="src").pack(side="left", padx=4)
    ttk.Radiobutton(r0, text=f"Target ({tgt_kind})", variable=mv_v, value="tgt").pack(side="left", padx=4)

    def nudge(dr, ddeg):
        if st["mode"] != "idle": return
        it = src_it if mv_v.get() == "src" else tgt_it; other = tgt_it if it is src_it else src_it
        r, ang = np.hypot(*it["goal"]), np.arctan2(it["goal"][1], it["goal"][0])
        new = M.polar(r + dr, np.degrees(ang) + ddeg)
        if not (RMIN <= np.hypot(*new) <= RMAX): status.set("outside allowed ring"); return
        if np.linalg.norm(other["goal"] - new) < MIN_SEP: status.set("too close to the other container"); return
        it["goal"] = new; status.set("moving ...")

    mrow = ttk.Frame(root); mrow.pack(pady=3)
    for txt, dr, dg in (("in", -0.02, 0), ("out", 0.02, 0), ("CW", 0, -3), ("CCW", 0, 3)):
        ttk.Button(mrow, text=txt, width=5, command=lambda dr=dr, dg=dg: nudge(dr, dg)).pack(side="left", padx=2)

    # ---------------- sensor-driven pour ----------------
    def cup_mask(p, O, Rc, S_):
        l = (p - O) @ Rc
        return (np.hypot(l[:, 0], l[:, 1]) < S_.R + 0.005) & (l[:, 2] > -0.005) & (l[:, 2] < S_.H + 0.02)

    def pose(th, s):
        """source container pose when its spout lip is at (planned pour point + s) and tilt = th; also gripper target"""
        m = st["meta"]; Ts = tgt_it["S"]
        P = np.array([*(m["tgt_xy"] + m["u"]*m["pour_dy"] + s), TOP + Ts.H + m["pc"]])
        Rc = m["I"] @ R.from_euler("x", -th).as_matrix()
        O = P - Rc @ m["lip"]
        RG = Rc @ m["R_CUP_G"]
        pt = O + Rc @ m["grasp"] - RG @ np.array([0, 0, m["pad_d"]])
        return O, Rc, RG, pt

    grip_q = M.gripper_q(src_it["S"].fing_closed)

    def start_pour():
        m = st["meta"]; qa = np.array(st["Q"][st["i"]-1][:6], float)
        st["ad"] = dict(seg="pour", th=m["th_c"], th_tgt=m["th_c"], s=np.zeros(2), s_tgt=np.zeros(2), th0=m["th_c"], s0=np.zeros(2),
                        q0=qa.copy(), q_g=qa.copy(), k=0, wait=0.0, nb=0, t=0.0, t_max=0.0, blocked=0, spill_prev=0,
                        p_prev=None, started=False,clear_t=0.0,log=[], tail_repositioned=False)
        st["seg"] = "pour"

    G = 9.81
    def sense(ad, m, O, Rc):
        """read the source liquid and classify the particles (stream, near wall, predicted landing, spilled)"""
        p = particles(src_it["liq"])
        if p is None: return None
        Ts, Ss = tgt_it["S"], src_it["S"]; dt = UPD_STEPS*DT
        pp = ad["p_prev"] if (ad["p_prev"] is not None and ad["p_prev"].shape == p.shape) else p
        vel = (p - pp)/dt; speed = np.linalg.norm(vel, axis=1); ad["p_prev"] = p
        tx, ty = m["tgt_xy"]; rim_z = TOP + Ts.H
        Rb = getattr(Ts, "R", Ts.open_r); Ro = getattr(Ts, "neck_r", None) or Rb
        d = np.hypot(p[:, 0]-tx, p[:, 1]-ty); z = p[:, 2]
        inside = cup_mask(p, O, Rc, Ss)
        out = ~inside & (z > TOP - 0.005)
        fast = speed > 0.25
        band = (np.abs(d - Ts.open_r) < WALL_BAND) | ((d > Ts.open_r) & (d < Ro + 0.015) & (z > rim_z - 0.03))
        res = dict(n_left=float(inside.mean()), started=bool(out.any()))
        res["near"] = int((out & fast & (z > TOP + 0.02) & (z < rim_z + 0.04) & band).sum())      # stream touching / near the wall or rim
        res["spill_n"] = int((out & (d > Rb + 0.005) & (z < rim_z)).sum())                        # liquid outside the target body
        res["n_air"] = int((out & (z > rim_z) & (speed > 0.15)).sum())                            # stream in the air
        stream = out & fast & (z > TOP + 0.02) & (z < rim_z + 0.10) & (d < 0.25)
        res["stream_c"] = p[stream][:, :2].mean(axis=0) if stream.sum() >= 3 else None
        fall = out & (vel[:, 2] < -0.05) & (z > rim_z + 0.005) & (z < rim_z + 0.25)               # predicted landing (ballistic) at rim height
        res["n_pred"], res["land_c"] = 0, None
        if fall.any():
            vz = vel[fall, 2]; hh = z[fall] - rim_z
            tt = (vz + np.sqrt(vz*vz + 2*G*hh))/G
            land = p[fall, :2] + vel[fall, :2]*tt[:, None]
            bad = np.hypot(land[:, 0]-tx, land[:, 1]-ty) > Ts.open_r - PRED_MARGIN
            res["n_pred"] = int(bad.sum())
            if bad.any(): res["land_c"] = land[bad].mean(axis=0)
        return res

    def adaptive_update():
        ad, m = st["ad"], st["meta"]; dt = UPD_STEPS*DT
        ad["th0"], ad["s0"], ad["q0"] = ad["th"], ad["s"].copy(), ad["q_g"].copy()
        th_c, th_max = m["th_c"], m["th_max"]
        O, Rc, _, _ = pose(ad["th"], ad["s"])
        sn = sense(ad, m, O, Rc)
        if sn is not None and sn["spill_n"] > ad["spill_prev"]:                                   # remember when spills happen
            ad["log"].append((ad["seg"], round(ad["t"], 1), int(round(np.degrees(ad["th"]))), round(sn["n_left"], 2), sn["spill_n"] - ad["spill_prev"]))
        if ad["seg"] == "pour":
            ad["t"] += dt
            if sn is not None:
                if sn["started"]: ad["started"] = True

                # At 10% remaining, move away from the target and reduce tilt once.
                if (ad["started"] and sn["n_left"] <= TAIL_REPOSITION_FRAC
                        and not ad["tail_repositioned"]):
                    ad["th_tgt"] = max(
                        th_c,
                        ad["th_tgt"] - np.deg2rad(TAIL_UNTILT_DEG)
                    )
                    ad["s_tgt"] = ad["s"] - m["u"] * TAIL_BACK_M
                    ad["tail_repositioned"] = True
                    ad["wait"] = TAIL_WAIT

                ad["clear_t"] = ad["clear_t"] + dt if (ad["started"] and sn["n_air"] == 0) else 0.0   # time since the last airborne particle
                if ad["wait"] > 0:
                    ad["wait"] -= dt  # retain the short pause used by the 10%-remaining reposition
                elif sn["n_air"] <= AIR_MAX:
                    # Open-loop tilt progression: do not tilt back or shift based on wall/spill sensing.
                    fast_phase = (not ad["started"]) and ad["th_tgt"] < th_max - np.deg2rad(40)
                    rate = FAST_RATE if fast_phase else POUR_RATE
                    if ad["started"] and sn["n_left"] < TAIL_FRAC:
                        rate *= 0.5  # retain slower flow near the end
                    ad["th_tgt"] = min(th_max, ad["th_tgt"] + rate*dt)
                if ad["th"] >= th_max - 1e-3: ad["t_max"] += dt
                drained = ad["started"] and sn["n_left"] < DONE_FRAC and ad["clear_t"] > CLEAR_S   # source empty AND the stream has landed
                stuck = ad["t_max"] > HOLD_END_S and ad["clear_t"] > CLEAR_S
                if (drained and ad["t"] > 1.0) or stuck or ad["t"] > MAX_POUR_S: ad["seg"] = "untilt"
                ad["spill_prev"] = sn["spill_n"]
                status.set(f'pouring | tilt {np.degrees(ad["th"]):.0f} deg | back-offs {ad["nb"]} | left in source {100*sn["n_left"]:.0f}% | spilled {sn["spill_n"]}')
            else:
                ad["th_tgt"] = min(th_max, ad["th_tgt"] + POUR_RATE*dt)
        elif sn is not None:
            ad["spill_prev"] = sn["spill_n"]
        if ad["seg"] == "untilt":
            ad["th_tgt"] = th_c; ad["s_tgt"] = np.zeros(2); status.set("pour done - returning the container")
        dth = ad["th_tgt"] - ad["th"]
        down = UNTILT_SLOW if (ad["th"] - th_c) > np.deg2rad(25) else M.SPEED["back"][0]        # untilt gently at first (no slosh)
        ad["th"] += float(np.clip(dth, -down*dt, 1.0*dt))
        ds = ad["s_tgt"] - ad["s"]; n = np.linalg.norm(ds)
        ad["s"] = ad["s_tgt"].copy() if n <= SHIFT_RATE*dt else ad["s"] + ds/n*SHIFT_RATE*dt
        O, Rc, RG, pt = pose(ad["th"], ad["s"])
        q, ep, er = M.ik(RG, pt, ad["q0"])
        if ep > 0.006 or er > 0.06:                                                             # pose out of reach: hold, stop pouring if it keeps happening
            ad["th"], ad["s"] = ad["th0"], ad["s0"].copy(); ad["th_tgt"] = ad["th"]; ad["s_tgt"] = ad["s"].copy()
            ad["blocked"] += 1
            if ad["blocked"] >= 4 and ad["seg"] == "pour": ad["seg"] = "untilt"
            ad["q_g"] = ad["q0"].copy()
        else:
            ad["q_g"] = q

    def adaptive_step():
        ad = st["ad"]
        if ad["k"] == 0: adaptive_update()
        a_ = (ad["k"] + 1)/UPD_STEPS
        th = ad["th0"] + (ad["th"] - ad["th0"])*a_
        s = ad["s0"] + (ad["s"] - ad["s0"])*a_
        q = ad["q0"] + (ad["q_g"] - ad["q0"])*a_
        O, Rc, _, _ = pose(th, s)
        err = command(np.concatenate([q, grip_q])); st["maxerr"] = max(st["maxerr"], float(np.abs(err).max()))
        put(src_it, O, Rc)                                                                # virtual weld
        scene.step(); ad["k"] += 1
        if ad["k"] >= UPD_STEPS:
            ad["k"] = 0
            if ad["seg"] == "untilt" and abs(ad["th"] - st["meta"]["th_c"]) < 2e-3 and np.linalg.norm(ad["s"]) < 1e-3:
                st["i"] = st["i_ret0"]; st["seg"] = "plan"                                # rejoin the planned return / put-down

    def plan_step():
        i = st["i"]; f = st["F"][i]
        err = command(st["Q"][i]); st["maxerr"] = max(st["maxerr"], float(np.abs(err).max()))
        if f.get("attached"): put(src_it, *f["cup"])
        scene.step(); st["i"] += 1
        if i % 100 == 0 and st["seg"] == "plan": status.set(f'{f["phase"]}  {100*i//len(st["Q"])}%')
        if st["adaptive"] and st["i"] == st["i_tilt0"]: start_pour()
        elif st["i"] >= len(st["Q"]): finish()

    def step_run():
        if st["seg"] == "plan": plan_step()
        else: adaptive_step()

    def on_start():
        if st["mode"] != "idle": return
        if any(np.linalg.norm(i["goal"] - i["xy"]) > 2e-3 for i in items): status.set("containers still moving - try again in a moment"); return
        sxy, txy = src_it["xy"].copy(), tgt_it["xy"].copy()
        status.set("planning (IK, may take up to a minute) ..."); root.update_idletasks()
        try:
            kw = M.pour_setup(src_it["S"], tgt_it["S"], sxy, txy)
            F, Q, _ = M.make_plan(sxy, txy, src_it["S"], tgt_it["S"], **kw)
        except RuntimeError as e: status.set(str(e)[:300]); return
        n = 100; s = M.smooth(np.linspace(1/n, 1, n))[:, None]
        Qf = np.vstack([Q_PARK12 + s*(Q[0]-Q_PARK12), Q, Q[-1] + s*(Q_PARK12-Q[-1])])
        Ff = [dict(phase="unpark", attached=False)]*n + F + [dict(phase="park", attached=False)]*n
        i_tilt0 = n + next(i for i, f in enumerate(F) if f["phase"] == "tilt")
        i_ret0 = n + next(i for i, f in enumerate(F) if f["phase"] == "ret")
        adaptive = ADAPTIVE_POUR and particles(src_it["liq"]) is not None
        st.update(mode="run", seg="plan", i=0, Q=Qf, F=Ff, meta=dict(M.META), i_tilt0=i_tilt0, i_ret0=i_ret0, adaptive=adaptive, maxerr=0.0)
        set_q(Qf[0]); integ[:] = 0

    def finish():
        p = particles(src_it["liq"]); st["mode"] = "idle"; st["seg"] = "plan"
        nb = st["ad"]["nb"] if st.get("adaptive") and "ad" in st else 0
        print(f"max joint tracking error {np.degrees(st['maxerr']):.2f} deg | back-offs {nb}")
        if st.get("adaptive") and "ad" in st: print("spill events (segment, t[s], tilt[deg], left-in-source, +particles):", st["ad"]["log"])
        if p is None: status.set("done - press Reset to pour again"); return
        T, S_ = tgt_it, src_it
        tx = (np.hypot(p[:, 0]-T["xy"][0], p[:, 1]-T["xy"][1]) < T["S"].open_r+0.005) & (p[:, 2] > TOP-0.01) & (p[:, 2] < TOP+T["S"].H+0.03)
        sx = (np.hypot(p[:, 0]-S_["xy"][0], p[:, 1]-S_["xy"][1]) < S_["S"].R) & (p[:, 2] > TOP-0.01) & ~tx
        status.set(f"done | in target {100*tx.mean():.0f}% | left in source {100*sx.mean():.0f}% | spilled {100*(1-tx.mean()-sx.mean()):.0f}% | back-offs {nb} | press Reset to pour again")

    def on_reset():
        scene.reset(); st.update(mode="idle", seg="plan"); integ[:] = 0
        for it in items:
            it["xy"], it["goal"] = it["xy0"].copy(), it["xy0"].copy()
            put(it, (*it["xy0"], TOP+M.CUP_GAP))
        status.set("reset - both containers back in their original place, water refilled")

    def new_pair():
        os.execv(sys.executable, [sys.executable] + sys.argv)

    brow = ttk.Frame(root); brow.pack(pady=4)
    for txt, fn in (("START SIMULATION", on_start), ("Reset", on_reset), ("New pair", new_pair)): ttk.Button(brow, text=txt, command=fn).pack(side="left", padx=4)
    ttk.Label(root, textvariable=status, wraplength=420, justify="left").pack(padx=8, pady=6)

    def step_idle():
        set_q(Q_PARK12)
        for it in items:
            d = it["goal"] - it["xy"]; dist = np.linalg.norm(d)
            if dist > 1e-4:
                step = d/dist*min(dist, 0.0008); it["xy"] = it["xy"] + step
                put(it, (*it["xy"], TOP+M.CUP_GAP)); shift_water(it, step)
        scene.step()
    def tick():
        for _ in range(STEPS_PER_TICK): step_run() if st["mode"] == "run" else step_idle()
        root.after(1, tick)
    root.after(100, tick); root.mainloop()

if __name__ == "__main__":
    main()