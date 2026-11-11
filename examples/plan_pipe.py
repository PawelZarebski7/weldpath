"""Spawanie doczołowe dwóch rur: pozycjoner obraca rurę, robot trzyma palnik na górze.

Uruchom:  poetry run python examples/make_pipes.py
          poetry run python examples/plan_pipe.py
"""
import time
from pathlib import Path

import fcl
import numpy as np

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.coordinated import plan_coordinated, sample_seam
from weldpath.orientation import homogeneous
from weldpath.positioner import Positioner
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION, forward_kinematics
from weldpath.weld_job import DEFAULT_TORCH

np.set_printoptions(precision=1, suppress=True)

ROOT = Path(__file__).resolve().parent.parent
# Oś uchylania na wysokości 700 mm, 1.25 m przed robotem. Rura stoi na tarczy w osi obrotu.
POSITIONER = Positioner(tilt_axis_origin=np.array([1250.0, 0.0, 700.0]))
FACEPLATE = (fcl.Box(400.0, 400.0, 30.0), homogeneous(np.eye(3), np.array([0, 0, -15.0])))

bodies = solids(load_step(str(ROOT / "data" / "pipes.step")))
weld = find_weld_seams(bodies)[0]
seam = weld.seam
checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), POSITIONER.T_base_part(0, 0),
                     attachments=(FACEPLATE,))

print(f"Spoina: rura {weld.bodies[0]} + rura {weld.bodies[1]}, obwód {seam.length:.1f} mm, "
      f"rowek V {seam.opening_angle:.0f}°, krzywa: {'prosta' if seam.is_straight else 'okrąg'}")
print(f"Punkty spoiny co 10 mm (z zakładką 15 mm): {len(sample_seam(seam, 10, 15))}")

t0 = time.perf_counter()
plan = plan_coordinated(ABB_IRB2400, seam, POSITIONER, checker)
if plan is None:
    raise SystemExit("Nie znaleziono planu")

print(f"\nPlan ({time.perf_counter() - t0:.1f} s): konfiguracja {plan.config}, "
      f"obrót palnika {plan.spin:+.0f}°")
print(f"Pozycjoner: uchylenie {plan.passes[0].positioner[0, 0]:+.0f}° (rura poziomo), "
      f"obrót tarczy o {plan.positioner_rotation:.0f}° na ścieg")
print(f"Robot: największy ruch osi w czasie spawania {plan.robot_motion:.2f}°  "
      f"(spawa pozycjoner, robot trzyma palnik)")

print("\nŚciegi:")
for p in plan.passes:
    tcp = np.array([(forward_kinematics(ABB_IRB2400, q) @ DEFAULT_TORCH.flange_to_tcp())[:3, 3]
                    for q in p.joints])
    q_abb = ABB_IRB2400_CONVENTION.to_robot(p.joints[0])
    print(f"  {p.name:12s} +{p.offset:.0f} mm od dna rowka:  TCP {tcp.mean(axis=0)} "
          f"(rozrzut {np.ptp(tcp, axis=0).max():.2f} mm),  osie ABB {np.array(q_abb)}°")