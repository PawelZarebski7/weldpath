"""Planuje wszystkie spoiny ramy wielobryłowej z dobieraniem parametrów spawania.

Uruchom:  poetry run python examples/make_frame.py
          poetry run python examples/plan_frame.py data/frame.step
"""
import sys
import time

import numpy as np

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400
from weldpath.weld_job import plan_weld

np.set_printoptions(precision=0, suppress=True)

T_BASE_PART = homogeneous(np.eye(3), np.array([800.0, -150.0, 300.0]))
BODY_NAMES = {0: "płyta", 1: "żebro A", 2: "żebro B", 3: "żebro poprz."}

bodies = solids(load_step(sys.argv[1] if len(sys.argv) > 1 else "data/frame.step"))
seams = find_weld_seams(bodies)
checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), T_BASE_PART, table_top_z=300.0)
print(f"Brył: {len(bodies)}, spoin: {len(seams)}\n")

t0 = time.perf_counter()
feasible = 0
for i, ws in enumerate(seams):
    a, b = (BODY_NAMES.get(k, str(k)) for k in ws.bodies)
    print(f"S{i:02d} {a} + {b}: {ws.seam.start} -> {ws.seam.end} ({ws.seam.length:.0f} mm)")
    plans = plan_weld(ABB_IRB2400, ws.seam, T_BASE_PART, checker)
    if not plans:
        print("     NIEWYKONALNA")
        continue
    feasible += 1
    for p in plans:
        work = "dwusieczna" if p.work_angle is None else f"{p.work_angle:.0f}°"
        notes = []
        if p.reversed:
            notes.append("odwrócony kierunek")
        if len(plans) == 2:
            notes.append("od środka na zewnątrz")
        if any(p.trim):
            notes.append(f"skrócenie {p.trim[0]:.0f}/{p.trim[1]:.0f} mm")
        print(f"     {p.seam.start} -> {p.seam.end}  kąt roboczy {work}, "
              f"prowadzenia {p.travel_angle:+.0f}°, obrót {p.spin:+.0f}°, "
              f"|q5| ≥ {p.min_wrist_q5:.0f}°  {', '.join(notes)}")

print(f"\nWykonalne: {feasible}/{len(seams)} spoin, "
      f"{time.perf_counter() - t0:.0f} s, {checker.checks} sprawdzeń kolizji")