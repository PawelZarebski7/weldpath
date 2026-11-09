"""Krok 6: pozycjoner obrotowo-uchylny (2 osie) i pozycje spawania.

Budowa (widok z boku, robot stoi po lewej):

          tarcza ═══════  ← oś OBROTU (rot): prostopadła do tarczy
                   ║
      oś UCHYLANIA ●──── (tilt): pozioma, wzdłuż osi Y robota
                   ║
               ════╩════  podstawa pozycjonera

    T_baza→detal(tilt, rot) = T_baza→oś_uchylania · Ry(tilt) · Rz(rot) · T_tarcza→detal

Po co: spoinę pachwinową najlepiej spawa się w pozycji PODOLNEJ (PA) – dwusieczna
doliny skierowana pionowo w górę, ciekłe jeziorko leży na dnie "rynny".
Pozycjoner obraca detal tak, żeby kolejne spoiny trafiały w PA.

Pozycje spoin pachwinowych wg ISO 6947 (w uproszczeniu):
    PA  podolna            – dwusieczna pionowo w górę
    PB  naboczna           – dwusieczna pod ~45° (typowe T leżące na stole)
    PD  okapowa            – dwusieczna skierowana w dół
    PF  pionowa z dołu do góry,  PG  pionowa z góry na dół
"""
from dataclasses import dataclass, field

import numpy as np

from weldpath.orientation import homogeneous, rotation_about_axis
from weldpath.seams import Seam

Z_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Positioner:
    tilt_axis_origin: np.ndarray            # punkt na osi uchylania (układ bazy robota) [mm]
    table_offset: float = 150.0             # odległość tarczy od osi uchylania [mm]
    tilt_limits: tuple[float, float] = (-135.0, 135.0)   # [°]
    T_table_part: np.ndarray = field(default_factory=lambda: np.eye(4))  # detal na tarczy

    def T_base_table(self, tilt: float, rot: float) -> np.ndarray:
        """Układ tarczy w bazie robota dla kątów osi [°]."""
        R_tilt = homogeneous(rotation_about_axis(np.array([0.0, 1.0, 0.0]), tilt), np.zeros(3))
        R_rot = homogeneous(rotation_about_axis(Z_UP, rot), np.zeros(3))
        lift = homogeneous(np.eye(3), np.array([0.0, 0.0, self.table_offset]))
        return homogeneous(np.eye(3), self.tilt_axis_origin) @ R_tilt @ lift @ R_rot

    def T_base_part(self, tilt: float, rot: float) -> np.ndarray:
        return self.T_base_table(tilt, rot) @ self.T_table_part

    def angles_pointing_up(self, d_part: np.ndarray) -> list[tuple[float, float]]:
        """Kąty (tilt, rot) [°], przy których kierunek z układu detalu wskazuje w górę.

        Najpierw obrót tarczy (rot) sprowadza poziomą składową kierunku na oś X tarczy,
        potem uchylenie (tilt) wokół Y stawia kierunek pionowo. Są dwa rozwiązania:
        "przechyl do siebie" i "obróć o 180° i przechyl od siebie".
        """
        d = self.T_table_part[:3, :3] @ (d_part / np.linalg.norm(d_part))  # w układzie tarczy
        horizontal = np.hypot(d[0], d[1])
        rot = -np.degrees(np.arctan2(d[1], d[0])) if horizontal > 1e-9 else 0.0
        tilt = -np.degrees(np.arctan2(horizontal, d[2]))
        solutions = [(tilt, rot), (-tilt, _wrap(rot + 180.0))]
        lo, hi = self.tilt_limits
        return [(t, r) for t, r in solutions if lo <= t <= hi]


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def seam_opening(seam: Seam) -> np.ndarray:
    """Dwusieczna doliny ("skąd przychodzi palnik") w układzie detalu."""
    v = seam.normals[0] + seam.normals[1]
    return v / np.linalg.norm(v)


def gravity_angle(seam: Seam, T_base_part: np.ndarray) -> float:
    """Kąt między dwusieczną doliny a pionem [°]. 0° = pozycja podolna PA."""
    opening = T_base_part[:3, :3] @ seam_opening(seam)
    return float(np.degrees(np.arccos(np.clip(opening @ Z_UP, -1.0, 1.0))))


def welding_position(seam: Seam, T_base_part: np.ndarray) -> str:
    """Pozycja spawania spoiny pachwinowej (uproszczone ISO 6947)."""
    tangent = T_base_part[:3, :3] @ seam.tangent
    if abs(tangent[2]) > np.sin(np.radians(45)):       # spoina bardziej pionowa niż pozioma
        return "PF" if tangent[2] > 0 else "PG"
    angle = gravity_angle(seam, T_base_part)
    if angle < 20:
        return "PA"
    if angle < 70:
        return "PB"
    if angle > 110:
        return "PD"
    return "PB"  # dwusieczna poziomo – traktujemy jak naboczną


def candidate_poses(positioner: Positioner, seam: Seam,
                    deviations=(0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0),
                    rot_deviations=(0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 180.0)
                    ) -> list[tuple[float, float, float]]:
    """Ustawienia pozycjonera (koszt, tilt, rot) posortowane od najlepszej pozycji spawania.

    Zaczynamy od dokładnego PA, potem coraz większe odchylenia uchylenia i obrotu
    (bo dokładne PA może być nieosiągalne dla robota albo kolidować).
    Koszt = kąt od pionu [°] + mała kara za duże uchylenie.
    """
    poses = {}
    for tilt0, rot0 in positioner.angles_pointing_up(seam_opening(seam)):
        for dt in deviations:
            for dr in rot_deviations:
                tilt, rot = tilt0 + dt, _wrap(rot0 + dr)
                lo, hi = positioner.tilt_limits
                if not lo <= tilt <= hi:
                    continue
                T = positioner.T_base_part(tilt, rot)
                cost = gravity_angle(seam, T) + abs(tilt) / 30
                key = (round(tilt, 3), round(rot, 3))
                poses[key] = min(cost, poses.get(key, np.inf))
    return sorted((c, t, r) for (t, r), c in poses.items())


# --- planowanie spoiny z pozycjonerem -------------------------------------

@dataclass
class PositionedWeld:
    seam: Seam
    tilt: float                 # ustawienie osi uchylania [°]
    rot: float                  # ustawienie osi obrotu [°]
    position: str               # pozycja spawania przy tym ustawieniu (PA, PB, ...)
    gravity_angle: float        # odchylenie dwusiecznej od pionu [°]
    plans: list                 # list[SeamPlan] – 1 albo 2 odcinki
    poses_tried: int


def plan_weld_with_positioner(robot, seam: Seam, positioner: Positioner, checker,
                              max_poses: int = 6, **kwargs) -> PositionedWeld | None:
    """Szuka ustawienia pozycjonera (od najlepszej pozycji spawania) i planu robota.

    UWAGA: pozycjoner obraca detal razem z sąsiednimi blachami, więc położenie
    palnika WZGLĘDEM detalu (dostęp do kieszeni) się nie zmienia – zmienia się
    grawitacja i zasięg robota. Dlatego dalej potrzebne jest pełne przeszukiwanie
    parametrów spawania z plan_weld.

    `checker` musi mieć metodę set_part_pose() – detal rusza się razem z tarczą.
    """
    from weldpath.weld_job import plan_weld  # import lokalny: weld_job nie zna pozycjonera

    for i, (_, tilt, rot) in enumerate(candidate_poses(positioner, seam)[:max_poses], 1):
        T = positioner.T_base_part(tilt, rot)
        checker.set_part_pose(T)
        plans = plan_weld(robot, seam, T, checker, **kwargs)
        if plans:
            return PositionedWeld(seam, tilt, rot, welding_position(seam, T),
                                  gravity_angle(seam, T), plans, i)
    return None