"""Sprawdza, czy program stanowiska mieści się w limitach osi robota ZAPISANYCH W RoboDK.

RoboDK pokazuje "The robot joints reached the limit", gdy setJoints dostaje kąt spoza
jego limitów (i przycina go). Nasz planer sprawdza limity z karty katalogowej ABB –
ten skrypt porównuje oba źródła i wypisuje, w którym odcinku i na której osi jest różnica.

Uruchom (RoboDK otwarty):  poetry run python examples/robodk_check_limits.py tee
"""
import sys

import numpy as np
from positioner_jobs import plan_job

from weldpath.coordinated import build_cell_program
from weldpath.robodk_bridge import RoboDKCell
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION
from weldpath.weld_job import IRB2400_LIMITS

NAME = sys.argv[1] if len(sys.argv) > 1 else "tee"
HOME = ABB_IRB2400_CONVENTION.to_opw([0, -50, 40, 0, 50, 0])

job, positioner, checker, samples, plan = plan_job(NAME)
program = build_cell_program(ABB_IRB2400, plan, positioner, checker, HOME)

cell = RoboDKCell()
lower, upper, _ = cell.robot.JointLimits()
lower, upper = np.array(lower.list()[:6]), np.array(upper.list()[:6])
print("\nLimity w RoboDK      :", np.round(lower), np.round(upper))
print("Limity w weldpath    :", IRB2400_LIMITS.lower_deg, IRB2400_LIMITS.upper_deg)

problems = 0
for seg in program:
    q_robot = np.array([ABB_IRB2400_CONVENTION.to_robot(q) for q in seg.robot])
    lo = q_robot.min(axis=0)
    hi = q_robot.max(axis=0)
    bad = [k for k in range(6) if lo[k] < lower[k] - 1e-6 or hi[k] > upper[k] + 1e-6]
    mark = "  <-- POZA LIMITEM RoboDK" if bad else ""
    print(f"\n{seg.label}{mark}")
    print(f"  min: {np.round(lo, 1)}")
    print(f"  max: {np.round(hi, 1)}")
    for k in bad:
        problems += 1
        print(f"  oś {k + 1}: {lo[k]:.1f}..{hi[k]:.1f}°, RoboDK pozwala {lower[k]:.1f}..{upper[k]:.1f}°")
    # sprzężenie osi 2-3 sprawdza tylko weldpath; RoboDK może mieć własne – pokazujemy sumę
    print(f"  oś2+oś3: {np.round((q_robot[:, 1] + q_robot[:, 2]).min(), 1)}"
          f"..{np.round((q_robot[:, 1] + q_robot[:, 2]).max(), 1)}")

print(f"\nPrzekroczeń: {problems}")