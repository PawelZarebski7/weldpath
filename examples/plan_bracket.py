"""Wspornik z ramieniem pod 60°: spoiny 60°, 90° i 120°, palnik dobierany do kąta.

Uruchom:  poetry run python examples/make_bracket.py
          poetry run python examples/plan_bracket.py
"""
import time
from pathlib import Path

import fcl
import numpy as np

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous, torch_frame
from weldpath.robot import ABB_IRB2400
from weldpath.weld_job import plan_weld

np.set_printoptions(precision=0, suppress=True)

ROOT = Path(__file__).resolve().parent.parent
T_BASE_PART = homogeneous(np.eye(3), np.array([800.0, -75.0, 300.0]))
# Stół 900 x 700 pod wspornikiem (250 x 150), blat tuż pod płytą montażową.
TABLE = (fcl.Box(900.0, 700.0, 50.0), homogeneous(np.eye(3), np.array([125.0, 75.0, -25.0])))
NAMES = {0: "płyta", 1: "ramię", 2: "żebro"}


def angle_to_face(z: np.ndarray, normal: np.ndarray) -> float:
    """Kąt między osią palnika a płaszczyzną ściany [°]."""
    return float(np.degrees(np.arcsin(abs(z @ normal))))


if __name__ == "__main__":
    bodies = solids(load_step(str(ROOT / "data" / "bracket.step")))
    welds = find_weld_seams(bodies)
    checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), T_BASE_PART, attachments=(TABLE,))
    print(f"Wspornik: {len(bodies)} blachy, {len(welds)} spoin\n")
    print("  spoina              kąt między   palnik do    palnik do    kąt          uwagi")
    print("                      blachami     1. blachy    2. blachy    prowadzenia")

    t0 = time.perf_counter()
    for i, ws in enumerate(welds):
        a, b = (NAMES[k] for k in ws.bodies)
        plans = plan_weld(ABB_IRB2400, ws.seam, T_BASE_PART, checker)
        if not plans:
            print(f"  S{i} {a}+{b:7s}  {ws.seam.opening_angle:5.0f}°       NIEWYKONALNA")
            continue
        p = plans[0]
        z = torch_frame(p.seam, p.seam.start, p.work_angle, p.travel_angle)[:3, 2]
        n1, n2 = p.seam.normals
        notes = []
        if abs(p.seam.tangent[2]) > 0.3:
            notes.append("spoina skośna w górę")
        if p.reversed:
            notes.append("odwrócony kierunek")
        if len(plans) == 2:
            notes.append("od środka")
        if any(p.trim):
            notes.append(f"skrócenie {p.trim[0]:.0f}/{p.trim[1]:.0f} mm")
        print(f"  S{i} {a}+{b:7s}  {ws.seam.opening_angle:5.0f}°       "
              f"{angle_to_face(z, n1):5.1f}°       {angle_to_face(z, n2):5.1f}°       "
              f"{p.travel_angle:+4.0f}°        {', '.join(notes)}")
    print(f"\nCzas: {time.perf_counter() - t0:.0f} s, sprawdzeń kolizji: {checker.checks}")
    print("Przy kącie prowadzenia ≠ 0 palnik jest też pochylony wzdłuż spoiny,")
    print("dlatego kąty do blach są trochę mniejsze niż połowa kąta między blachami.")