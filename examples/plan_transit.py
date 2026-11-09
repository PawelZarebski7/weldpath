"""Pełne zadanie spawalnicze: spoina S0 -> bezkolizyjny przejazd -> spoina S1.

Uruchom:  poetry run python examples/plan_transit.py data/tjoint.step
"""
import sys
from itertools import pairwise

import numpy as np

from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.planner import RRTConnect, motion_valid, path_length, retract, shortcut
from weldpath.robot import ABB_IRB2400, forward_kinematics
from weldpath.seams import find_seams
from weldpath.weld_job import DEFAULT_TORCH, IRB2400_LIMITS, plan_seam

np.set_printoptions(precision=1, suppress=True)

T_BASE_PART = homogeneous(np.eye(3), np.array([900.0, -100.0, 300.0]))
RETRACT_MM = 60.0     # odjazd/dojazd wzdłuż osi palnika
MARGIN_MM = 5.0       # zapas bezpieczeństwa przy ruchach jałowych


def tcp_in_part(q: np.ndarray) -> np.ndarray:
    """Pozycja końca drutu w układzie detalu (łatwiej ją porównać z rysunkiem)."""
    T_tcp = forward_kinematics(ABB_IRB2400, q) @ DEFAULT_TORCH.flange_to_tcp()
    return (np.linalg.inv(T_BASE_PART) @ T_tcp)[:3, 3]


part = load_step(sys.argv[1] if len(sys.argv) > 1 else "data/tjoint.step")
s0, s1 = find_seams(part)[:2]
weld0 = plan_seam(ABB_IRB2400, s0, T_BASE_PART)
weld1 = plan_seam(ABB_IRB2400, s1, T_BASE_PART)

checker = FCLChecker(ABB_IRB2400, part, T_BASE_PART, table_top_z=300.0, margin=MARGIN_MM)

# Odjazd z końca S0 i dojazd do początku S1 (ruchy liniowe wzdłuż osi palnika).
q_retract = retract(ABB_IRB2400, weld0.joints[-1], weld0.config, DEFAULT_TORCH, RETRACT_MM)
q_approach = retract(ABB_IRB2400, weld1.joints[0], weld1.config, DEFAULT_TORCH, RETRACT_MM)

print(f"Prosta droga w przestrzeni przegubów wolna? "
      f"{motion_valid(checker, q_retract, q_approach)}")

result = RRTConnect(checker, IRB2400_LIMITS, seed=0).plan(q_retract, q_approach)
if not result.success:
    sys.exit("Nie znaleziono bezkolizyjnej drogi")

smooth = shortcut(result.path, checker)
print(f"RRT-Connect: {result.iterations} iteracji, {result.tree_nodes} węzłów, "
      f"{result.collision_checks} sprawdzeń kolizji, {result.seconds * 1000:.0f} ms")
print(f"Wygładzanie: {len(result.path)} -> {len(smooth)} punktów, długość "
      f"{path_length(result.path):.2f} -> {path_length(smooth):.2f} rad")

# Najwyższy punkt przejazdu (gęsto próbkowany) – powinien być nad żebrem (z = 90).
dense = [a + (b - a) * t for a, b in pairwise(smooth) for t in np.linspace(0, 1, 30)]
heights = [tcp_in_part(q)[2] for q in dense]
print(f"Najwyższy punkt TCP nad detalem: z = {max(heights):.0f} mm (żebro kończy się na 90)")
print(f"Najmniejszy luz na przejeździe: {min(checker.clearance(q) for q in dense):.1f} mm "
      f"(ponad margines {MARGIN_MM:.0f} mm)")

# Program w stylu sterownika robota.
print("\n--- PROGRAM ---")
print(f"MoveL  spoina S0: {len(weld0.joints)} punktów, TCP {tcp_in_part(weld0.joints[0])} "
      f"-> {tcp_in_part(weld0.joints[-1])}")
print(f"MoveL  odjazd {RETRACT_MM:.0f} mm  -> TCP {tcp_in_part(q_retract)}")
for q in smooth[1:-1]:
    print(f"MoveJ  przez    q = {np.degrees(q)}°  (TCP {tcp_in_part(q)})")
print(f"MoveJ  do       q = {np.degrees(q_approach)}°  (TCP {tcp_in_part(q_approach)})")
print(f"MoveL  dojazd   -> TCP {tcp_in_part(weld1.joints[0])}")
print(f"MoveL  spoina S1: {len(weld1.joints)} punktów, TCP {tcp_in_part(weld1.joints[0])} "
      f"-> {tcp_in_part(weld1.joints[-1])}")