"""Porównuje naszą kinematykę OPW z kinematyką RoboDK dla ABB IRB 2400/16.

Wymaga otwartego RoboDK z robotem "ABB IRB 2400/16" w stacji.
Uruchom:  poetry run python examples/robodk_check_fk.py

Krok 1: pozycja zerowa.
Krok 2: dla każdej osi osobno sprawdzamy, czy jej kierunek obrotu jest taki sam
        w RoboDK i w OPW (znak +1) czy przeciwny (−1). Dla osi 2 dodatkowo
        sprawdzamy sprzężenie z osią 3 (robot z równoległowodem).
Krok 3: 200 losowych ułożeń – największy błąd pozycji i orientacji.
"""
import numpy as np
from robodk.robolink import TargetReachError

from weldpath.robodk_bridge import JointConvention, RoboDKCell
from weldpath.robot import ABB_IRB2400, forward_kinematics

np.set_printoptions(precision=2, suppress=True)


def pose_error(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(błąd pozycji [mm], błąd orientacji [°]) między dwiema macierzami 4x4."""
    pos = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    R = Ta[:3, :3].T @ Tb[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return pos, ang


cell = RoboDKCell()
conv = cell.convention

# --- krok 1: zero ABB ---------------------------------------------------------
q0 = conv.to_opw([0, 0, 0, 0, 0, 0])
pos, ang = pose_error(forward_kinematics(ABB_IRB2400, q0), cell.fk_robot([0] * 6))
print(f"Zero ABB: błąd pozycji {pos:.3f} mm, orientacji {ang:.3f}°")

# --- krok 2: znak każdej osi (i sprzężenie osi 2-3) ---------------------------
# Ruszamy JEDNĄ osią o 30° z pozycji zerowej – pozostałe są w zerze, więc ich
# znaki nie wpływają na wynik i test rozstrzyga tylko o tej osi.
signs = list(conv.signs)
coupling = 0.0
for axis in range(6):
    q_robot = np.zeros(6)
    q_robot[axis] = 30.0
    T_rdk = cell.fk_robot(q_robot)
    options = [(s, coupling) for s in (1, -1)]
    if axis == 1:  # oś 2: sprawdzamy też sprzężenie z osią 3
        options = [(s, c) for s in (1, -1) for c in (0.0, -1.0, 1.0)]
    errors = {}
    for s, c in options:
        trial = list(signs)
        trial[axis] = s
        conv_try = JointConvention(signs=tuple(trial), offsets_deg=conv.offsets_deg,
                                   j23_coupling=c)
        T_opw = forward_kinematics(ABB_IRB2400, conv_try.to_opw(q_robot))
        errors[(s, c)] = sum(pose_error(T_opw, T_rdk))
    (best_sign, best_c) = min(errors, key=errors.get)
    signs[axis] = best_sign
    if axis == 1:
        coupling = best_c
    listing = ", ".join(f"znak {s:+d}" + (f"/sprz. {c:+.0f}" if axis == 1 else "")
                        + f": {e:.2f}" for (s, c), e in errors.items())
    print(f"Oś {axis + 1}: {listing}  ->  znak {best_sign:+d}"
          + (f", sprzężenie {best_c:+.0f}" if axis == 1 else ""))

# --- krok 3: losowe ułożenia z wykrytą konwencją -------------------------------
found = JointConvention(signs=tuple(signs), offsets_deg=conv.offsets_deg,
                        j23_coupling=coupling)
cell.convention = found
rng = np.random.default_rng(0)
worst_pos = worst_ang = 0.0
tested = skipped = 0
while tested < 200 and skipped < 2000:
    q_robot = rng.uniform([-150, -60, -50, -150, -100, -150], [150, 90, 60, 150, 100, 150])
    try:
        T_rdk = cell.fk_robot(q_robot)
    except TargetReachError:  # poza limitami RoboDK (np. limit różnicy osi 2 i 3)
        skipped += 1
        continue
    pos, ang = pose_error(forward_kinematics(ABB_IRB2400, found.to_opw(q_robot)), T_rdk)
    worst_pos, worst_ang = max(worst_pos, pos), max(worst_ang, ang)
    tested += 1

print(f"\nWykryta konwencja: znaki {found.signs}, przesunięcia {found.offsets_deg}, "
      f"sprzężenie osi 2-3: {found.j23_coupling:+.0f}")
print(f"{tested} losowych ułożeń (pominięto {skipped} poza limitami RoboDK): "
      f"największy błąd pozycji {worst_pos:.3f} mm, orientacji {worst_ang:.3f}°")
print("ZGODNE" if worst_pos < 0.1 and worst_ang < 0.01 else "NIEZGODNE – wklej wynik")