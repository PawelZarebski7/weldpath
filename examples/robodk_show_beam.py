"""Demo w RoboDK: robot na torze jezdnym spawa belkę dwuteową w czterech ustawieniach.

Wymaga otwartego RoboDK z robotem "ABB IRB 2400/16" (np. data/weldcell.rdk).
Planowanie trwa ok. 4 minut, potem animacja:
  - wózek toru jedzie do spoiny (robot zaparkowany),
  - robot podjeżdża bez kolizji (RRT), spawa, odjeżdża i parkuje,
  - po wszystkich spoinach z danego ustawienia suwnica obraca belkę.

Uruchom:  poetry run python examples/robodk_show_beam.py
"""
from itertools import pairwise
from pathlib import Path

import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from plan_beam import (
    BEAM_CENTER,
    SEARCH,
    SUPPORTS_WORLD,
    T_WORLD_BEAM,
    TRACK,
    make_checker,
)

from weldpath.assembly import find_weld_seams, solids
from weldpath.cad import load_step
from weldpath.orientation import homogeneous, rotation_about_axis
from weldpath.planner import RRTConnect, retract, shortcut
from weldpath.robodk_bridge import RoboDKCell, to_mat, write_torch_step
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION
from weldpath.track import plan_part_on_track, setups_for_long_part
from weldpath.weld_job import DEFAULT_TORCH, IRB2400_LIMITS

ROOT = Path(__file__).resolve().parent.parent
BEAM = ROOT / "data" / "beam.step"
# Parkowanie (osie ABB): robot obrócony do belki, ramię odchylone do tyłu i w górę.
PARK = ABB_IRB2400_CONVENTION.to_opw([90, -50, 40, 0, 50, 0])
RETRACT_MM = 60.0


def write_box_step(path: Path, corner, size) -> Path:
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape(), STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


# --- planowanie ---------------------------------------------------------------
bodies = solids(load_step(str(BEAM)))
seams = [w.seam for w in find_weld_seams(bodies)]
setups = setups_for_long_part(T_WORLD_BEAM, BEAM_CENTER)
checkers = [make_checker(T, bodies) for _, T in setups]
print(f"Planowanie {len(seams)} spoin w {len(setups)} ustawieniach (ok. 3 min)...")
done, failed = plan_part_on_track(ABB_IRB2400, seams, TRACK, setups,
                                  lambda T: make_checker(T, bodies), log=lambda m: None,
                                  offsets=(0.0, 250.0, -250.0), search=SEARCH)
print(f"Spoiny: {len(done)}/{len(seams)} wykonalne")

# Program: (opis, rodzaj, ustawienie, [(q robota, s wózka), ...])
program = []
s_now = 0.0
for k, (name, T_world_part) in enumerate(setups, 1):
    welds = sorted((w for w in done if w.setup == k), key=lambda w: w.s)
    if not welds:
        continue
    checker = checkers[k - 1]
    program.append((f"ustawienie {name}", "setup", k, [(PARK, s_now)]))
    for w in welds:
        for part, plan in enumerate(w.plans):
            # 1. wózek jedzie, robot zaparkowany – sprawdzamy kolizje co 20 mm jazdy
            for s in np.linspace(s_now, w.s, max(2, int(abs(w.s - s_now) / 20) + 1)):
                checker.set_part_pose(TRACK.T_base_part(s, T_world_part))
                if checker.in_collision(PARK):
                    raise RuntimeError(f"Kolizja przy jeździe wózka do s = {s:.0f}")
            program.append(("wózek", "track", k, [(PARK, s_now), (PARK, w.s)]))
            s_now = w.s
            # 2. podjazd (RRT), spawanie, odjazd, powrót na parking (RRT)
            checker.set_part_pose(TRACK.T_base_part(w.s, T_world_part))
            q_in = retract(ABB_IRB2400, plan.joints[0], plan.config, DEFAULT_TORCH, RETRACT_MM)
            q_out = retract(ABB_IRB2400, plan.joints[-1], plan.config, DEFAULT_TORCH, RETRACT_MM)
            if q_in is None or checker.in_collision(q_in):
                q_in = plan.joints[0]
            if q_out is None or checker.in_collision(q_out):
                q_out = plan.joints[-1]
            rrt = RRTConnect(checker, IRB2400_LIMITS, seed=len(program))
            to_seam, back = rrt.plan(PARK, q_in), rrt.plan(q_out, PARK)
            if not (to_seam.success and back.success):
                raise RuntimeError("Brak bezkolizyjnego podjazdu/odjazdu")
            path = (shortcut(to_seam.path, checker) + list(plan.joints)
                    + shortcut(back.path, checker))
            program.append((f"spoina (wózek {w.s:.0f} mm)", "weld", k, [(q, w.s) for q in path]))
print(f"Program: {sum(1 for p in program if p[1] == 'weld')} spoin, "
      f"{sum(1 for p in program if p[1] == 'setup')} ustawienia belki")

# --- RoboDK -------------------------------------------------------------------
cell = RoboDKCell()
cell.remove("Rama", "Stół", "Pozycjoner", "Hala", "Wózek toru")
hall = cell.add_frame("Hala", np.eye(4), parent=0)                 # układ świata (stacja)
rail = write_box_step(ROOT / "data" / "rail.step",
                      (TRACK.s_min - 500, -350, -60), (TRACK.s_max - TRACK.s_min + 1000, 700, 60))
cell.add_part(rail, np.eye(4), name="Tor jezdny", parent=hall)
for i, (_, T_support) in enumerate(SUPPORTS_WORLD):
    path = write_box_step(ROOT / "data" / f"support{i}.step", (-100, -150, -250), (200, 300, 500))
    cell.add_part(path, T_support, name=f"Stojak {i + 1}", parent=hall)
beam_frame = cell.add_frame("Belka (suwnica)", T_WORLD_BEAM, parent=hall)
beam = cell.add_part(BEAM, np.eye(4), name="Belka", parent=beam_frame)
torch_step = write_torch_step(ROOT / "data" / "torch.step", DEFAULT_TORCH.flange_to_tcp())
tool = cell.set_tool(DEFAULT_TORCH.flange_to_tcp(), name="Palnik MIG", geometry_step=torch_step)
try:
    beam.setColor([0.75, 0.80, 0.90, 1.0])
    tool.setColor([0.85, 0.55, 0.20, 1.0])
except Exception as e:  # noqa: BLE001
    print(f"(kolory pominięte: {e})")


def show(q, s, T_world_part) -> None:
    cell.set_base_pose(TRACK.T_world_base(s))
    beam_frame.setPose(to_mat(T_world_part))
    cell.set_joints(q)


def crane(T_from, T_to, s, steps=40) -> None:
    """Suwnica: podnieś belkę o 400 mm, obróć, opuść (tylko animacja, robot zaparkowany)."""
    R_rel = T_from[:3, :3].T @ T_to[:3, :3]
    axis_angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    w = np.array([R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1]])
    axis = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-9 else np.array([0, 0, 1.0])
    if abs(axis_angle - 180) < 1e-6:                       # obrót o 180°: oś z macierzy R + I
        M = R_rel + np.eye(3)
        axis = M[:, np.argmax(np.linalg.norm(M, axis=0))]
        axis /= np.linalg.norm(axis)
    c = homogeneous(np.eye(3), BEAM_CENTER)
    for t in np.linspace(0, 1, steps):
        h = np.sin(np.pi * t)                              # w górę i z powrotem w dół
        R_t = homogeneous(rotation_about_axis(axis, axis_angle * t), np.zeros(3))
        T = (homogeneous(np.eye(3), np.array([0, 0, 400.0 * h])) @ T_from @ c @ R_t
             @ np.linalg.inv(c))
        show(PARK, s, T)


show(PARK, 0.0, setups[0][1])
input("\nUstaw widok w RoboDK i naciśnij Enter, żeby zacząć... ")
T_current = setups[0][1]
for label, kind, k, pts in program:
    T_setup = setups[k - 1][1]
    if kind == "setup":
        print(f"  > {label}")
        if not np.allclose(T_current, T_setup):
            crane(T_current, T_setup, pts[0][1])
        T_current = T_setup
        continue
    for (qa, sa), (qb, sb) in pairwise(pts):
        span = max(np.degrees(np.max(np.abs(qb - qa))), abs(sb - sa) / 10.0)
        for t in np.linspace(0, 1, max(2, int(span / 2))):
            show(qa + (qb - qa) * t, sa + (sb - sa) * t, T_setup)
show(PARK, 0.0, T_current)
cell.set_base_pose(np.eye(4))
print("Koniec.")