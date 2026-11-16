import argparse, numpy as np, tkinter as tk, torch
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
import main as M
from containers import SPECS, write_urdf
from perception import CamModel, perceive
# ---- pouring accuracy tuning (these edit main.py's tables in place) ----
M.SPEED["tilt_slow"] = (0.20, 0.30)    # slow pour: ~11 s for a glass (was ~4 s)
M.SPEED["carry"]     = (0.35, 0.80)    # gentler transfer so water doesn't slosh out
M.SPEED["lift"]      = (0.20, 0.60)
M.SPEED["back"]      = (0.30, 0.80)    # untilt gently, no flinging drips
M.FIXED_T["hold"]    = 2.5             # let the cup drain (was 1.2 s)
POUR_CLEAR_NEW = 0.06                  # lip height above target rim (m)
POUR_DY_K      = 0.6                   # lip offset = -POUR_DY_K * target opening radius


LAYOUT = [("glass", 0.55, -45), ("bottle", 0.55, -15), ("glass", 0.55, 15), ("bottle", 0.55, 45)]
RMIN, RMAX, MIN_SEP = 0.50, 0.74, 0.17
Q_PARK = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0.0])           # arm stowed upright (clear of the cameras)
COLOR = {"glass": (0.75, 0.9, 1.0, 0.4), "bottle": (0.2, 0.6, 0.3, 0.5)}
STEPS_PER_TICK = 6

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-viewer", action="store_true"); a = ap.parse_args()
    # Initialize Tkinter root first so macOS NSApplication is properly configured before Genesis/GLFW
    root = tk.Tk(); root.title("Pouring control"); root.attributes("-topmost", True)
    import genesis as gs
    M.GEN.mkdir(exist_ok=True)
    for k, S in SPECS.items(): write_urdf(M.GEN/f"{k}.urdf", S)
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
    items = []
    items = []
    for kind, r, deg in LAYOUT:
        S, xy = SPECS[kind], M.polar(r, deg)
        ent = scene.add_entity(gs.morphs.URDF(file=str(M.GEN/f"{kind}.urdf"), pos=(*xy, TOP+M.CUP_GAP), fixed=False),
                               surface=gs.surfaces.Default(color=COLOR[kind]))
        items.append(dict(S=S, ent=ent, xy=xy.copy(), goal=xy.copy(), xy0=xy.copy(), hidden=False))
    # ONE liquid, built parked under the table ring (out of sight from above) until a source is chosen
    FILL = SPECS["glass"].fill
    PARK = np.array([0.90*np.cos(np.deg2rad(135)), 0.90*np.sin(np.deg2rad(135)), FILL[2]/2 + 0.01])
    liquid = scene.add_entity(morph=gs.morphs.Box(pos=tuple(PARK), size=FILL), material=gs.materials.SPH.Liquid(),
                              surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"))
    cam_top = CamModel(scene, (640, 640), (0, 0, 2.4), (0, 0, 0), 60, up=(1, 0, 0))
    cam_side = CamModel(scene, (640, 480), (1.6, 0, TOP+0.55), (0.45, 0, TOP+0.08), 50)
    scene.build()

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

    nearest = lambda xy: min([it for it in items if not it["hidden"]], key=lambda it: np.linalg.norm(it["xy"] - xy))

    P0 = particles(liquid)                                    # sampled particle layout (parked position)

    def hide(it, j):
        it["hidden"] = True
        put(it, np.array([4.0 + 0.6*j, 4.0, 0.3]))            # container vanishes (far away)

    def restore_hidden():                                     # used when planning fails
        for it in items:
            if it["hidden"]: put(it, (*it["xy"], TOP + M.CUP_GAP)); it["hidden"] = False
        if P0 is not None: set_particles(liquid, P0)

    def fill_source(it):                                      # move the liquid into the chosen source container
        if P0 is None: return
        z = TOP + M.CUP_GAP + it["S"].bottom + 0.004 + FILL[2]/2
        set_particles(liquid, P0 - P0.mean(axis=0) + np.array([*it["xy"], z]))

    def clear_stray(src_it):              # any water not in the source cup goes back to the parking spot
        p = particles(liquid)
        if p is None or P0 is None: return
        d = np.hypot(p[:, 0] - src_it["xy"][0], p[:, 1] - src_it["xy"][1])
        stray = d > src_it["S"].R + 0.02
        print(f"fill check: {len(p)} particles, {stray.sum()} stray, centroid {np.round(p.mean(axis=0), 3)}, source {np.round(src_it['xy'], 3)}")
        if stray.any():
            p = p.copy()
            p[stray] = P0[stray] - P0.mean(axis=0) + PARK
            set_particles(liquid, p)
    # ---------------- GUI ----------------
    status = tk.StringVar(value="Ready"); st = dict(mode="idle", dets=[], dirty=True)
    auto_v = tk.BooleanVar(value=True)
    lb = tk.Listbox(root, width=52, height=5); lb.pack(padx=8, pady=4)
    row = ttk.Frame(root); row.pack()
    src_v, tgt_v, mv_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
    boxes = []
    for lab, var in (("Pour from #", src_v), ("to #", tgt_v), ("Move #", mv_v)):
        ttk.Label(row, text=lab).pack(side="left"); cb = ttk.Combobox(row, textvariable=var, width=3, state="readonly"); cb.pack(side="left", padx=3); boxes.append(cb)
    ttk.Checkbutton(root, text="Auto-start when both are selected", variable=auto_v).pack()

    def scan():
        st["dets"] = perceive(cam_top, cam_side, TOP); lb.delete(0, "end")
        for d in st["dets"]:
            lb.insert("end", f'#{d["id"]}  {d["type"]:6s}  r={np.hypot(d["x"], d["y"]):.2f} m  ang={np.degrees(np.arctan2(d["y"], d["x"])):.0f} deg  h={100*d["h"]:.0f} cm')
        ids = [str(d["id"]) for d in st["dets"]]
        for cb in boxes: cb["values"] = ids
        status.set(f'{len(ids)} objects detected'); st["dirty"] = False

    def nudge(dr, ddeg):
        if st["mode"] != "idle" or not mv_v.get(): return
        d = st["dets"][int(mv_v.get()) - 1]; it = nearest(np.array([d["x"], d["y"]]))
        r, ang = np.hypot(*it["goal"]), np.arctan2(it["goal"][1], it["goal"][0])
        new = M.polar(r + dr, np.degrees(ang) + ddeg)
        if not (RMIN <= np.hypot(*new) <= RMAX): status.set("outside allowed ring"); return
        if any(o is not it and np.linalg.norm(o["goal"] - new) < MIN_SEP for o in items): status.set("too close to another object"); return
        it["goal"] = new; st["dirty"] = True

    mrow = ttk.Frame(root); mrow.pack(pady=3)
    for txt, dr, dg in (("in", -0.02, 0), ("out", 0.02, 0), ("CW", 0, -3), ("CCW", 0, 3)):
        ttk.Button(mrow, text=txt, width=5, command=lambda dr=dr, dg=dg: nudge(dr, dg)).pack(side="left", padx=2)

    def on_start():
        if st["mode"] != "idle": return
        if any(i["hidden"] for i in items): status.set("press Reset first (other containers are hidden)"); return
        if any(np.linalg.norm(i["goal"] - i["xy"]) > 2e-3 for i in items): status.set("objects still moving - try again in a moment"); return
        scan()
        if not (src_v.get() and tgt_v.get()) or src_v.get() == tgt_v.get(): status.set("pick two different objects"); return
        ds, dt = st["dets"][int(src_v.get())-1], st["dets"][int(tgt_v.get())-1]
        S, T = SPECS[ds["type"]], SPECS[dt["type"]]
        sxy, txy = np.array([ds["x"], ds["y"]]), np.array([dt["x"], dt["y"]])
        src_it, tgt_it = nearest(sxy), nearest(txy)
        j = 0
        for it in items:                                      # the other two vanish immediately
            if it is not src_it and it is not tgt_it: hide(it, j); j += 1
        status.set(f'planning {ds["type"]} -> {dt["type"]} ...'); root.update_idletasks()
        try: F, Q, _ = M.make_plan(sxy, txy, S, T, pour_dy=-POUR_DY_K*T.open_r, pour_clear=POUR_CLEAR_NEW)
        except RuntimeError as e:
            restore_hidden(); status.set(str(e)[:300]); return
        fill_source(src_it)                                   # liquid appears only now, only in the source
        clear_stray(src_it)
        for _ in range(30): scene.step()
        clear_stray(src_it)
        n = 100; s = M.smooth(np.linspace(1/n, 1, n))[:, None]
        Qf = np.vstack([Q_PARK12 + s*(Q[0]-Q_PARK12), Q, Q[-1] + s*(Q_PARK12-Q[-1])])
        Ff = [dict(phase="unpark", attached=False)]*n + F + [dict(phase="park", attached=False)]*n
        st.update(mode="run", i=0, Q=Qf, F=Ff, src=src_it, tgt=tgt_it)

    def on_select(_=None):
        if auto_v.get() and src_v.get() and tgt_v.get():
            if src_v.get() == tgt_v.get(): status.set("pick two different objects"); return
            on_start()
    boxes[0].bind("<<ComboboxSelected>>", on_select); boxes[1].bind("<<ComboboxSelected>>", on_select)

    def finish():
        s_it, t_it = st["src"], st["tgt"]; p = particles(liquid); st["mode"] = "idle"; st["dirty"] = False
        if p is None: status.set("done - press Reset to restore all containers"); return
        tx = (np.hypot(p[:, 0]-t_it["xy"][0], p[:, 1]-t_it["xy"][1]) < t_it["S"].open_r+0.005) & (p[:, 2] > TOP-0.01) & (p[:, 2] < TOP+t_it["S"].H+0.03)
        sx = (np.hypot(p[:, 0]-s_it["xy"][0], p[:, 1]-s_it["xy"][1]) < s_it["S"].R) & (p[:, 2] > TOP-0.01) & ~tx
        status.set(f"done | in target {100*tx.mean():.0f}% | left in source {100*sx.mean():.0f}% | spilled {100*(1-tx.mean()-sx.mean()):.0f}%  | press Reset to restore all containers")

    def on_reset():
        scene.reset(); st["mode"] = "idle"; st["dirty"] = True
        if P0 is not None: set_particles(liquid, P0)
        for it in items:
            it["hidden"] = False; it["psave"] = None
            it["xy"], it["goal"] = it["xy0"].copy(), it["xy0"].copy()
            put(it, (*it["xy0"], TOP+M.CUP_GAP))
        src_v.set(""); tgt_v.set(""); mv_v.set("")
        status.set("scene reset - all containers restored")

    brow = ttk.Frame(root); brow.pack(pady=4)
    for txt, fn in (("Scan", scan), ("START SIMULATION", on_start), ("Reset", on_reset)): ttk.Button(brow, text=txt, command=fn).pack(side="left", padx=4)
    ttk.Label(root, textvariable=status, wraplength=420, justify="left").pack(padx=8, pady=6)

    # ---------------- stepping ----------------
    def step_run():
        i = st["i"]; f = st["F"][i]; set_q(st["Q"][i])
        if f.get("attached"): put(st["src"], *f["cup"])          # virtual weld while gripped
        scene.step(); st["i"] += 1
        if i % 100 == 0: status.set(f'{f["phase"]}  {100*i//len(st["Q"])}%')
        if st["i"] >= len(st["Q"]): finish()
    def step_idle():
        set_q(Q_PARK12); moving = False
        for it in items:
            if it["hidden"]: continue
            d = it["goal"] - it["xy"]; dist = np.linalg.norm(d)  # glide to goal at <= 0.8 mm per step
            if dist > 1e-4:
                it["xy"] = it["xy"] + d/dist*min(dist, 0.0008); put(it, (*it["xy"], TOP+M.CUP_GAP)); moving = True
        scene.step()
        if st["dirty"] and not moving: scan()
    running = True
    def on_close():
        nonlocal running
        running = False
        try: root.destroy()
        except Exception: pass
    root.protocol("WM_DELETE_WINDOW", on_close)

    scan()
    while running:
        try:
            root.update()
        except tk.TclError:
            break
        for _ in range(STEPS_PER_TICK):
            step_run() if st["mode"] == "run" else step_idle()

if __name__ == "__main__":
    main()