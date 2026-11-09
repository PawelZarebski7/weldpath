"""Planuje spawanie wszystkich spoin detalu robotem ABB IRB 2400.

Uruchom:  poetry run python examples/plan_welds.py data/tjoint.step
"""
import sys

import numpy as np

from weldpath.cad import load_step
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400
from weldpath.seams import find_seams
from weldpath.weld_job import plan_seam

np.set_printoptions(precision=1, suppress=True)

# Detal leży na stole: 900 mm przed robotem, 100 mm w prawo, 300 mm nad podstawą.
T_BASE_PART = homogeneous(np.eye(3), np.array([900.0, -100.0, 300.0]))

CONFIG_NAMES = {
    0: "ramię przód, łokieć A, bez flipa", 1: "ramię przód, łokieć A, flip",
    2: "ramię przód, łokieć B, bez flipa", 3: "ramię przód, łokieć B, flip",
    4: "ramię tył, łokieć A, bez flipa",   5: "ramię tył, łokieć A, flip",
    6: "ramię tył, łokieć B, bez flipa",   7: "ramię tył, łokieć B, flip",
}

shape = load_step(sys.argv[1] if len(sys.argv) > 1 else "data/tjoint.step")

for i, seam in enumerate(find_seams(shape)):
    print(f"\n=== Spoina S{i}: {seam.start} -> {seam.end} ({seam.length:.0f} mm) ===")
    plan = plan_seam(ABB_IRB2400, seam, T_BASE_PART)
    if plan is None:
        print("  NIEOSIĄGALNA: żadna konfiguracja ani obrót palnika nie pasuje")
        continue

    print(f"  konfiguracja   : {plan.config} ({CONFIG_NAMES[plan.config]})")
    print(f"  obrót palnika  : {plan.spin:+.0f}°")
    print(f"  punktów        : {len(plan.joints)}")
    print(f"  min |q5|       : {plan.min_wrist_q5:.1f}°  (daleko od 0 = dobrze)")
    print(f"  max skok osi   : {plan.max_joint_step:.2f}° na krok")
    print(f"  q start [°]    : {np.degrees(plan.joints[0])}")
    print(f"  q koniec [°]   : {np.degrees(plan.joints[-1])}")
    for w in plan.warnings:
        print(f"  {w}")
    print(f"  status         : {'OK' if plan.feasible else 'NIEWYKONALNA'}")