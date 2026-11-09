"""Rama na pozycjonerze obrotowo-uchylnym: każda spoina ustawiona w pozycji podolnej (PA).

Uruchom:  poetry run python examples/plan_positioner.py data/frame.step
"""
import sys
import time
from collections import Counter

import fcl
import numpy as np

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.positioner import Positioner, plan_weld_with_positioner, welding_position
from weldpath.robot import ABB_IRB2400

np.set_printoptions(precision=0, suppress=True)

# Rama (300 x 200) leży na tarczy środkiem na osi obrotu.
T_TABLE_PART = homogeneous(np.eye(3), np.array([-150.0, -100.0, 0.0]))
POSITIONER = Positioner(tilt_axis_origin=np.array([1000.0, 0.0, 500.0]),
                        T_table_part=T_TABLE_PART)
# Tarcza pozycjonera 500 x 500 x 30 mm pod detalem – rusza się razem z nim.
FACEPLATE = (fcl.Box(500.0, 500.0, 30.0),
             np.linalg.inv(T_TABLE_PART) @ homogeneous(np.eye(3), np.array([0, 0, -15.0])))

bodies = solids(load_step(sys.argv[1] if len(sys.argv) > 1 else "data/frame.step"))
seams = find_weld_seams(bodies)
checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), POSITIONER.T_base_part(0, 0),
                     attachments=(FACEPLATE,))

flat = POSITIONER.T_base_part(0.0, 0.0)  # pozycjoner w zerze = detal leży płasko
before, after = Counter(), Counter()
t0 = time.perf_counter()

for i, ws in enumerate(seams):
    pos_flat = welding_position(ws.seam, flat)
    before[pos_flat] += 1
    result = plan_weld_with_positioner(ABB_IRB2400, ws.seam, POSITIONER, checker)
    if result is None:
        print(f"S{i:02d} {pos_flat} -> NIEWYKONALNA")
        continue
    after[result.position] += 1
    p = result.plans[0]
    work = "dwusieczna" if p.work_angle is None else f"{p.work_angle:.0f}°"
    print(f"S{i:02d} {pos_flat} -> {result.position}  pozycjoner: uchylenie {result.tilt:+.0f}°, "
          f"obrót {result.rot:+.0f}°  |  palnik: kąt roboczy {work}, "
          f"prowadzenia {p.travel_angle:+.0f}°, odcinki: {len(result.plans)}")

print(f"\nPozycje spawania bez pozycjonera: {dict(before)}")
print(f"Pozycje spawania z pozycjonerem:  {dict(after)}")
print(f"Czas: {time.perf_counter() - t0:.0f} s, sprawdzeń kolizji: {checker.checks}")