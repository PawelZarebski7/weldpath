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
     wartości i bierzemy tę, przy której nadgarstek jest najdalej od osobliwości,
  5. opcjonalnie sprawdzamy kolizje podczas spawania i przeszukujemy parametry
     technologiczne: kąt roboczy, kąt prowadzenia, kierunek spawania i skrócenie
     końców spoiny (dysza nie wejdzie w narożnik z trzech ścian).
     Kandydatów sprawdzamy od "najbardziej standardowych" (najniższy koszt).
"""
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from weldpath.orientation import homogeneous, rotation_about_axis, torch_path
from weldpath.robot import OPWParams, inverse_kinematics
from weldpath.seams import Seam

if TYPE_CHECKING:  # tylko dla podpowiedzi typów – unikamy importu cyklicznego
    from weldpath.collision import CollisionChecker


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
    seam: Seam                      # spoina faktycznie spawana (może być odwrócona/skrócona)
    spin: float                     # obrót palnika wokół osi Z [°]
    work_angle: float | None        # kąt roboczy [°] (None = dwusieczna)
    config: int                     # indeks rozwiązania IK (0..7)
    joints: np.ndarray              # (N, 6) kąty w każdym punkcie [rad]
    min_wrist_q5: float             # najmniejsze |q5| na spoinie [°]
    max_joint_step: float           # największy skok osi między punktami [°]
    travel_angle: float = 10.0      # kąt prowadzenia [°] (+ push, − drag)
    reversed: bool = False          # spawamy od końca do początku krawędzi
    trim: tuple[float, float] = (0.0, 0.0)  # skrócenie początku i końca spoiny [mm]
    cost: float = 0.0               # koszt technologiczny (0 = ustawienia standardowe)
    warnings: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not any(w.startswith("BŁĄD") for w in self.warnings)


def reversed_seam(seam: Seam) -> Seam:
    """Ta sama spoina spawana w przeciwną stronę."""
    return replace(seam, start=seam.end, end=seam.start, tangent=-seam.tangent)


def trimmed_seam(seam: Seam, start_trim: float, end_trim: float) -> Seam:
    """Spoina skrócona o `start_trim` mm na początku i `end_trim` mm na końcu."""
    return replace(seam, start=seam.start + start_trim * seam.tangent,
                   end=seam.end - end_trim * seam.tangent)


def split_center_out(seam: Seam) -> tuple[Seam, Seam]:
    """Dwie połówki spawane od środka na zewnątrz (każda zaczyna się w środku)."""
    mid = (seam.start + seam.end) / 2
    return (replace(seam, start=mid, end=seam.end),
            replace(seam, start=mid, end=seam.start, tangent=-seam.tangent))


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
               max_step: float = 15.0,
               checker: "CollisionChecker | None" = None) -> SeamPlan | None:
    """Najlepsza konfiguracja dla jednej spoiny przy zadanych parametrach (albo None)."""
    targets = flange_targets(seam, T_base_part, torch, spin, step, work_angle, travel_angle)
    all_sols = np.array([inverse_kinematics(robot, T) for T in targets])  # (N, 8, 6)

    candidates = []
    for config in range(8):
        q = all_sols[:, config, :]
        if np.isnan(q).any() or not all(limits.ok(qi) for qi in q):
            continue
        steps = np.abs((np.diff(q, axis=0) + np.pi) % (2 * np.pi) - np.pi)
        plan = SeamPlan(
            seam=seam, spin=spin, work_angle=work_angle, config=config, joints=q,
            min_wrist_q5=float(np.degrees(np.min(np.abs(q[:, 4])))),
            max_joint_step=float(np.degrees(steps.max())) if len(steps) else 0.0,
            travel_angle=travel_angle,
        )
        if plan.max_joint_step > max_step:
            plan.warnings.append(f"BŁĄD: skok osi {plan.max_joint_step:.0f}° "
                                 f"między punktami (wrist flip?)")
        if plan.min_wrist_q5 < singular_q5:
            plan.warnings.append(f"UWAGA: |q5| = {plan.min_wrist_q5:.1f}° "
                                 f"– blisko osobliwości nadgarstka")
        candidates.append(plan)
    if not candidates:
        return None

    # Kolizje są najdroższe, więc sprawdzamy je na końcu i tylko tyle, ile trzeba:
    # od najlepszej konfiguracji, aż trafimy na bezkolizyjną.
    candidates.sort(key=_score, reverse=True)
    if checker is None:
        return candidates[0]
    for plan in candidates:
        if not plan.feasible:
            continue
        if any(checker.in_collision(qi) for qi in plan.joints):
            plan.warnings.append("BŁĄD: kolizja robota lub palnika podczas spawania")
        else:
            return plan
    return candidates[0]


def plan_seam(robot: OPWParams, seam: Seam, T_base_part: np.ndarray,
              spins=range(-90, 91, 15), work_angles=(None,), travel_angles=(10.0,),
              allow_reverse: bool = False, trims=(0.0,), trim_start: bool = True,
              **kwargs) -> SeamPlan | None:
    """Przeszukuje parametry spawania i zwraca najlepszy plan (albo None).

    Kandydaci idą od najniższego kosztu technologicznego. Gdy znajdziemy plan
    wykonalny z nadgarstkiem daleko od osobliwości (|q5| >= 45°), żaden kolejny
    (droższy) nie może być lepszy – przerywamy.
    """
    nominal_work = seam.opening_angle / 2
    nominal_travel = travel_angles[0]
    trim_pairs = [(ts, te) for ts in (trims if trim_start else (0.0,)) for te in trims]
    candidates = []
    for rev in ((False, True) if allow_reverse else (False,)):
        for trim in trim_pairs:
            for w in work_angles:
                for ta in travel_angles:
                    for s in spins:
                        work = nominal_work if w is None else w
                        cost = (abs(work - nominal_work) / 5 + abs(ta - nominal_travel) / 5
                                + abs(s) / 45 + 2 * rev + sum(trim) / 5)
                        candidates.append((cost, rev, trim, w, ta, float(s)))
    candidates.sort(key=lambda c: c[0])

    best: SeamPlan | None = None
    for cost, rev, trim, w, ta, s in candidates:
        variant = trimmed_seam(reversed_seam(seam) if rev else seam, *trim)
        plan = solve_seam(robot, variant, T_base_part, spin=s, work_angle=w,
                          travel_angle=ta, **kwargs)
        if plan is None:
            continue
        plan.reversed, plan.trim, plan.cost = rev, trim, cost
        if best is None or _score(plan) > _score(best):
            best = plan
        if best.feasible and best.min_wrist_q5 >= 45.0:
            break
    return best


def _score(plan: SeamPlan) -> tuple:
    # Najpierw wykonalność, potem odległość od osobliwości (powyżej 45° każda jest
    # "wystarczająco dobra"), na końcu jak najbardziej standardowe ustawienia.
    return (plan.feasible, min(plan.min_wrist_q5, 45.0), -plan.cost)


# Zakres przeszukiwania "jak spawacz": standardowe ustawienia najpierw, potem odstępstwa.
TECHNOLOGY_SEARCH = {
    "spins": range(-90, 91, 30),
    "work_angles": (None, 35.0, 55.0),
    "travel_angles": (10.0, 0.0, -10.0, 20.0, 30.0, 45.0, -45.0),
    "trims": (0.0, 5.0, 10.0, 15.0),
}


def plan_weld(robot: OPWParams, seam: Seam, T_base_part: np.ndarray,
              checker: "CollisionChecker", search: dict | None = None,
              **kwargs) -> list[SeamPlan]:
    """Plan jednej spoiny: najpierw w całości, a jeśli się nie da – od środka na zewnątrz.

    Zwraca listę planów (1 albo 2 odcinki). Pusta lista = spoina niewykonalna.
    """
    search = TECHNOLOGY_SEARCH if search is None else search
    whole = plan_seam(robot, seam, T_base_part, checker=checker, allow_reverse=True,
                      **search, **kwargs)
    if whole is not None and whole.feasible:
        return [whole]
    halves = [plan_seam(robot, half, T_base_part, checker=checker, trim_start=False,
                        **search, **kwargs)
              for half in split_center_out(seam)]
    if all(h is not None and h.feasible for h in halves):
        return halves
    return []