import argparse, os, sys, numpy as np, tkinter as tk, torch
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
import main2 as M
from containers import SPECS, write_urdf
# ---- pouring accuracy tuning (these edit main.py's tables in place) ----
M.SPEED["tilt_fast"] = (0.90, 3.0)
M.SPEED["tilt_slow"] = (0.40, 0.60)
M.SPEED["carry"]     = (0.45, 1.00)
M.SPEED["lift"]      = (0.25, 0.80)
M.SPEED["back"]      = (0.60, 1.50)
M.SPEED["ret"]       = (0.60, 1.50)
M.FIXED_T["hold"]    = 1.5
POUR_CLEAR_NEW = 0.06
POUR_DY_K      = 0.6
# ... keep your M.SPEED[...] / M.FIXED_T / POUR_CLEAR_NEW / POUR_DY_K tuning block here ...

SRC_POLAR, TGT_POLAR = (0.55, -22.0), (0.55, 22.0)    # start positions (radius m, angle deg)
RMIN, RMAX, MIN_SEP = 0.50, 0.74, 0.17
Q_PARK = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0.0])
COLOR = {"glass": (0.75, 0.9, 1.0, 0.4), "bottle": (0.2, 0.6, 0.3, 0.5)}
COLOR_TGT_SAME = {"glass": (0.95, 0.8, 0.6, 0.5), "bottle": (0.2, 0.4, 0.9, 0.6)}   # used when source and target are the same type
PAIRS = [("Glass  ->  Bottle", ("glass", "bottle")), ("Bottle  ->  Glass", ("bottle", "glass")),
         ("Bottle  ->  Bottle", ("bottle", "bottle")), ("Glass  ->  Glass", ("glass", "glass"))]
STEPS_PER_TICK = 12

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
    src_kind, tgt_kind = ask_pair(root)                       # <-- asked BEFORE anything is built
    wait = ttk.Label(root, text=f"Building scene: {src_kind} -> {tgt_kind} ...", padding=30); wait.pack(); root.update()

    import genesis as gs
    M.GEN.mkdir(exist_ok=True)
    for k in {src_kind, tgt_kind}: write_urdf(M.GEN/f"{k}.urdf", SPECS[k])
    M.table_urdf(M.GEN/"ring_table.urdf")
    try: gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    except Exception as e: print("gs.init:", e)
    TOP = M.TOP
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
    robot = scene.add_entity(gs.morphs.URDF(file=str(M.URDF_PATH), pos=tuple(M.BASE), fixed=True))

    items = []                                                # exactly two containers: [source, target]
    for role, kind, (r, deg) in (("src", src_kind, SRC_POLAR), ("tgt", tgt_kind, TGT_POLAR)):
        xy = M.polar(r, deg)
        col = COLOR_TGT_SAME[kind] if (role == "tgt" and src_kind == tgt_kind) else COLOR[kind]
        ent = scene.add_entity(gs.morphs.URDF(file=str(M.GEN/f"{kind}.urdf"), pos=(*xy, TOP+M.CUP_GAP), fixed=False),
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
    set_q = lambda q: robot.set_dofs_position(np.asarray(q, dtype=np.float32), dofs_idx_local=dofs)
    Q_PARK12 = np.concatenate([Q_PARK, M.gripper_q(0.0)])
    def put(it, O, Rc=np.eye(3)):
        x, y, z, w = R.from_matrix(Rc).as_quat()
        it["ent"].set_pos(np.asarray(O, dtype=np.float32), zero_velocity=True)
        it["ent"].set_quat(np.array([w, x, y, z], dtype=np.float32), zero_velocity=True)
    def particles(e):
        for nm in ("get_particles_pos", "get_particles"):
            fn = getattr(e, nm, None)
            if fn is None: continue
            try:
                p = fn(); return (p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)).reshape(-1, 3)
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

    def shift_water(it, d):                                   # container ke saath uska paani bhi move karo
        p = particles(it["liq"])
        if p is not None: set_particles(it["liq"], p + np.array([d[0], d[1], 0.0]))

    # ---------------- GUI (same window the pair was chosen in) ----------------
    st = dict(mode="idle")
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

    def on_start():
        if st["mode"] != "idle": return
        if any(np.linalg.norm(i["goal"] - i["xy"]) > 2e-3 for i in items): status.set("containers still moving - try again in a moment"); return
        sxy, txy = src_it["xy"].copy(), tgt_it["xy"].copy()
        status.set("planning (IK, may take up to a minute) ..."); root.update_idletasks()
        try: F, Q, _ = M.make_plan(sxy, txy, src_it["S"], tgt_it["S"], pour_dy=-POUR_DY_K*tgt_it["S"].open_r, pour_clear=POUR_CLEAR_NEW)
        except RuntimeError as e: status.set(str(e)[:300]); return
        n = 100; s = M.smooth(np.linspace(1/n, 1, n))[:, None]
        Qf = np.vstack([Q_PARK12 + s*(Q[0]-Q_PARK12), Q, Q[-1] + s*(Q_PARK12-Q[-1])])
        Ff = [dict(phase="unpark", attached=False)]*n + F + [dict(phase="park", attached=False)]*n
        st.update(mode="run", i=0, Q=Qf, F=Ff)

    def finish():
        p = particles(src_it["liq"]); st["mode"] = "idle"
        if p is None: status.set("done - press Reset to pour again"); return
        T, S_ = tgt_it, src_it
        tx = (np.hypot(p[:, 0]-T["xy"][0], p[:, 1]-T["xy"][1]) < T["S"].open_r+0.005) & (p[:, 2] > TOP-0.01) & (p[:, 2] < TOP+T["S"].H+0.03)
        sx = (np.hypot(p[:, 0]-S_["xy"][0], p[:, 1]-S_["xy"][1]) < S_["S"].R) & (p[:, 2] > TOP-0.01) & ~tx
        status.set(f"done | in target {100*tx.mean():.0f}% | left in source {100*sx.mean():.0f}% | spilled {100*(1-tx.mean()-sx.mean()):.0f}% | press Reset to pour again")

    def on_reset():
        scene.reset(); st["mode"] = "idle"
        for it in items:
            it["xy"], it["goal"] = it["xy0"].copy(), it["xy0"].copy()
            put(it, (*it["xy0"], TOP+M.CUP_GAP))
        status.set("reset - dono containers apni original jagah par, paani wapas")

    def new_pair():                                           # relaunch the app to choose another process
        os.execv(sys.executable, [sys.executable] + sys.argv)

    brow = ttk.Frame(root); brow.pack(pady=4)
    for txt, fn in (("START SIMULATION", on_start), ("Reset", on_reset), ("New pair", new_pair)): ttk.Button(brow, text=txt, command=fn).pack(side="left", padx=4)
    ttk.Label(root, textvariable=status, wraplength=420, justify="left").pack(padx=8, pady=6)

    # ---------------- stepping ----------------
    def step_run():
        i = st["i"]; f = st["F"][i]; set_q(st["Q"][i])
        if f.get("attached"): put(src_it, *f["cup"])            # virtual weld while gripped
        scene.step(); st["i"] += 1
        if i % 100 == 0: status.set(f'{f["phase"]}  {100*i//len(st["Q"])}%')
        if st["i"] >= len(st["Q"]): finish()
    def step_idle():
        set_q(Q_PARK12)
        for it in items:
            d = it["goal"] - it["xy"]; dist = np.linalg.norm(d)  # glide to goal at <= 0.8 mm per step
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