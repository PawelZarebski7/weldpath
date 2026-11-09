"""Krok 4b: od trajektorii palnika do kątów robota w każdym punkcie spoiny.

Łańcuch transformacji (wszystko macierze 4x4):

    T_baza→kołnierz = T_baza→detal · T_detal→TCP · (T_kołnierz→TCP)⁻¹
                       gdzie leży      z orientation   geometria palnika
                       detal           .py

Dla każdego punktu IK daje do 8 rozwiązań. Zasady wyboru:
  1. odrzucamy nieosiągalne (NaN) i poza limitami osi,
  2. przez całą spoinę trzymamy JEDNĄ konfigurację (ten sam indeks rozwiązania:
     ramię / łokieć / nadgarstek) – to eliminuje wrist flip,
  3. sprawdzamy skoki między punktami i odległość od osobliwości nadgarstka (q5≈0),
  4. wykorzystujemy wolny obrót palnika wokół jego osi (spin): próbujemy kilku
     wartości i bierzemy tę, przy której nadgarstek jest najdalej od osobliwości.
"""
from dataclasses import dataclass, field

import numpy as np

from weldpath.orientation import homogeneous, rotation_about_axis, torch_path
from weldpath.robot import OPWParams, inverse_kinematics
from weldpath.seams import Seam


@dataclass(frozen=True)
class Torch:
    """Palnik robotyczny: prosta lanca pochylona o kąt szyjki względem kołnierza."""
    length: float = 350.0      # odległość kołnierz -> koniec drutu [mm]
    neck_angle: float = 22.0   # kąt szyjki [°] (typowo 22° albo 45°)

    def flange_to_tcp(self) -> np.ndarray:
        R = rotation_about_axis(np.array([0.0, 1.0, 0.0]), self.neck_angle)
        return homogeneous(R, self.length * R[:, 2])


DEFAULT_TORCH = Torch()


@dataclass(frozen=True)
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray

    def ok(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))


# Limity ABB IRB 2400 z karty katalogowej, przeliczone na konwencję OPW.
# U ABB zero osi 3 to przedramię POZIOMO, w OPW – pionowo w górę, więc
# q3_OPW = q3_ABB + 90°  (zakres ABB −60°…+65°  ->  OPW 30°…155°).
# Osie 4 i 6 mają u ABB ±200° i ±400°, ale nasze IK zawija kąty do ±180°.
IRB2400_LIMITS = JointLimits(
    lower=np.radians([-180, -100, 30, -180, -120, -180]),
    upper=np.radians([180, 110, 155, 180, 120, 180]),
)


@dataclass
class SeamPlan:
    seam: Seam
    spin: float                     # obrót palnika wokół osi Z [°]
    config: int                     # indeks rozwiązania IK (0..7)
    joints: np.ndarray              # (N, 6) kąty w każdym punkcie [rad]
    min_wrist_q5: float             # najmniejsze |q5| na spoinie [°]
    max_joint_step: float           # największy skok osi między punktami [°]
    warnings: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not any(w.startswith("BŁĄD") for w in self.warnings)


def flange_targets(seam: Seam, T_base_part: np.ndarray, torch: Torch, spin: float,
                   step: float, work_angle: float | None,
                   travel_angle: float) -> list[np.ndarray]:
    """Macierze kołnierza w układzie bazy robota dla każdego punktu spoiny."""
    T_tcp_flange = np.linalg.inv(torch.flange_to_tcp())
    R_spin = homogeneous(rotation_about_axis(np.array([0.0, 0.0, 1.0]), spin), np.zeros(3))
    return [T_base_part @ T @ R_spin @ T_tcp_flange
            for T in torch_path(seam, step, work_angle, travel_angle)]


def solve_seam(robot: OPWParams, seam: Seam, T_base_part: np.ndarray,
               torch: Torch = DEFAULT_TORCH, limits: JointLimits = IRB2400_LIMITS,
               spin: float = 0.0, step: float = 10.0, work_angle: float | None = None,
               travel_angle: float = 10.0, singular_q5: float = 10.0,
               max_step: float = 15.0) -> SeamPlan | None:
    """Najlepsza konfiguracja dla jednej spoiny przy zadanym spinie (albo None)."""
    targets = flange_targets(seam, T_base_part, torch, spin, step, work_angle, travel_angle)
    all_sols = np.array([inverse_kinematics(robot, T) for T in targets])  # (N, 8, 6)

    best: SeamPlan | None = None
    for config in range(8):
        q = all_sols[:, config, :]
        if np.isnan(q).any() or not all(limits.ok(qi) for qi in q):
            continue
        steps = np.abs((np.diff(q, axis=0) + np.pi) % (2 * np.pi) - np.pi)
        plan = SeamPlan(
            seam=seam, spin=spin, config=config, joints=q,
            min_wrist_q5=float(np.degrees(np.min(np.abs(q[:, 4])))),
            max_joint_step=float(np.degrees(steps.max())) if len(steps) else 0.0,
        )
        if plan.max_joint_step > max_step:
            plan.warnings.append(f"BŁĄD: skok osi {plan.max_joint_step:.0f}° "
                                 f"między punktami (wrist flip?)")
        if plan.min_wrist_q5 < singular_q5:
            plan.warnings.append(f"UWAGA: |q5| = {plan.min_wrist_q5:.1f}° "
                                 f"– blisko osobliwości nadgarstka")
        if best is None or _score(plan) > _score(best):
            best = plan
    return best


def plan_seam(robot: OPWParams, seam: Seam, T_base_part: np.ndarray,
              spins=range(-90, 91, 15), **kwargs) -> SeamPlan | None:
    """Przeszukuje wolny obrót palnika i zwraca najlepszy wykonalny plan."""
    plans = [p for s in spins if (p := solve_seam(robot, seam, T_base_part,
                                                  spin=float(s), **kwargs))]
    return max(plans, key=_score, default=None)


def _score(plan: SeamPlan) -> tuple:
    # Najpierw wykonalność, potem odległość od osobliwości, potem mały spin.
    return (plan.feasible, min(plan.min_wrist_q5, 45.0), -abs(plan.spin))