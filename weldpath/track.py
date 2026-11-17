"""Krok 8: robot na torze jezdnym (7. oś) i długie detale spawane w kilku ustawieniach.

Tor jezdny przesuwa CAŁEGO robota wzdłuż osi X świata:

    T_świat→baza(s) = Trans(s, 0, h)          s – położenie wózka [mm], h – wysokość wózka

Robot z torem ma 7 osi na 6 stopni swobody palnika (REDUNDANCJA): ten sam punkt
spoiny da się spawać z nieskończenie wielu położeń wózka. Tu wybieramy je prosto
i przewidywalnie: wózek "naprzeciw" środka spoiny, a jeśli tam się nie da (kolizja,
zasięg, osobliwość) – próbujemy przesunięć ±200, ±400 mm.

Z jednej strony belki robot nie sięgnie spoin po drugiej stronie środnika, a spoin
pod pasem górnym (pozycja okapowa w kieszeni) nie spawa się wcale – belkę się
OBRACA (suwnicą) i spawa w kolejnym ustawieniu. Szybki test dostępności odrzuca
takie spoiny bez kosztownego planowania.
"""
from dataclasses import dataclass, field

import numpy as np

from weldpath.orientation import homogeneous, rotation_about_axis
from weldpath.positioner import seam_opening
from weldpath.robot import OPWParams
from weldpath.seams import Seam
from weldpath.weld_job import plan_weld


@dataclass(frozen=True)
class Track:
    s_min: float = -300.0          # zakres jazdy wózka [mm]
    s_max: float = 2300.0
    carriage_height: float = 0.0   # wysokość podstawy robota nad osią toru [mm]

    def T_world_base(self, s: float) -> np.ndarray:
        return homogeneous(np.eye(3), np.array([s, 0.0, self.carriage_height]))

    def T_base_part(self, s: float, T_world_part: np.ndarray) -> np.ndarray:
        """Detal widziany z bazy robota, gdy wózek stoi w s."""
        return np.linalg.inv(self.T_world_base(s)) @ T_world_part

    def clamp(self, s: float) -> float:
        return float(np.clip(s, self.s_min, self.s_max))


def setups_for_long_part(T_world_part: np.ndarray, center: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Cztery ustawienia długiego detalu: jak leży, obrócony w poziomie, odwrócony, oba.

    Obroty wokół środka detalu `center` (układ detalu), więc detal zostaje na stojakach:
      2 – obrót o 180° wokół pionu (druga strona środnika do robota),
      3 – obrót o 180° wokół osi belki (pas górny na dół),
      4 – oba obroty.
    """
    def about_center(R):
        return (homogeneous(np.eye(3), center) @ homogeneous(R, np.zeros(3))
                @ homogeneous(np.eye(3), -center))

    Rz = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 180.0)
    Rx = rotation_about_axis(np.array([1.0, 0.0, 0.0]), 180.0)
    return [
        ("1: jak leży", T_world_part),
        ("2: obrócona w poziomie", T_world_part @ about_center(Rz)),
        ("3: odwrócona (pas górny na dół)", T_world_part @ about_center(Rx)),
        ("4: odwrócona i obrócona", T_world_part @ about_center(Rz @ Rx)),
    ]


def accessible_from_track(seam: Seam, T_world_part: np.ndarray, min_dot: float = -0.2,
                          max_overhead: float = -0.5) -> bool:
    """Czy palnik może przyjść do spoiny z tej strony i nie od dołu?

    Dwusieczna doliny mówi, skąd musi przyjść palnik:
      - wskazuje wyraźnie OD toru (np. spoina za środnikiem) -> robot nie dosięgnie,
      - wskazuje w dół (spoina okapowa, np. pod pasem górnym) -> nie spawamy, obracamy detal.
    """
    opening = T_world_part[:3, :3] @ seam_opening(seam)
    if opening[2] < max_overhead:
        return False
    mid = seam.midpoint if seam.midpoint is not None else (seam.start + seam.end) / 2
    p = (T_world_part @ np.append(mid, 1.0))[:3]
    to_track = np.array([0.0, -p[1], 0.0])
    to_track /= np.linalg.norm(to_track)
    return float(opening @ to_track) > min_dot


@dataclass
class TrackWeld:
    seam: Seam
    setup: int                  # numer ustawienia detalu (1, 2, ...)
    s: float                    # położenie wózka [mm]
    plans: list = field(default_factory=list)   # list[SeamPlan] (1 albo 2 odcinki)


def plan_weld_on_track(robot: OPWParams, seam: Seam, track: Track, T_world_part: np.ndarray,
                       checker, setup: int = 1,
                       offsets=(0.0, 200.0, -200.0, 400.0, -400.0),
                       **kwargs) -> TrackWeld | None:
    """Spoina przy stałym położeniu wózka – szukamy położenia, z którego się da.

    `checker` musi mieć set_part_pose(): przesunięcie wózka = przesunięcie detalu
    względem bazy robota.
    """
    mid = seam.midpoint if seam.midpoint is not None else (seam.start + seam.end) / 2
    x_mid = float((T_world_part @ np.append(mid, 1.0))[0])
    tried = set()
    for off in offsets:
        s = track.clamp(x_mid + off)
        if s in tried:
            continue
        tried.add(s)
        T_base_part = track.T_base_part(s, T_world_part)
        checker.set_part_pose(T_base_part)
        plans = plan_weld(robot, seam, T_base_part, checker, **kwargs)
        if plans:
            return TrackWeld(seam, setup, s, plans)
    return None


def world_attachments(T_world_part: np.ndarray, objects_world) -> tuple:
    """Nieruchome przeszkody świata (np. stojaki) wyrażone w układzie detalu.

    FCLChecker przesuwa "przyczepione" przeszkody razem z detalem. Gdy detal ma kilka
    ustawień, przeszkody stojące na podłodze trzeba przeliczyć dla każdego ustawienia.
    """
    inv = np.linalg.inv(T_world_part)
    return tuple((geometry, inv @ T_world) for geometry, T_world in objects_world)


def plan_part_on_track(robot: OPWParams, seams: list[Seam], track: Track,
                       setups: list[tuple[str, np.ndarray]], make_checker,
                       log=print, **kwargs) -> tuple[list[TrackWeld], list[Seam]]:
    """Wszystkie spoiny detalu: dla każdej pierwsze ustawienie, z którego się da.

    make_checker(T_world_part) -> checker z przeszkodami dla danego ustawienia.
    Zwraca (zaplanowane spoiny, spoiny niewykonalne w żadnym ustawieniu).
    """
    checkers = [make_checker(T) for _, T in setups]
    done, failed = [], []
    for i, seam in enumerate(seams):
        result = None
        for k, ((name, T), checker) in enumerate(zip(setups, checkers), 1):
            if not accessible_from_track(seam, T):
                continue
            result = plan_weld_on_track(robot, seam, track, T, checker, setup=k, **kwargs)
            if result:
                break
        if result:
            done.append(result)
            log(f"  S{i:02d}: ustawienie {result.setup}, wózek s = {result.s:.0f} mm")
        else:
            failed.append(seam)
            log(f"  S{i:02d}: NIEWYKONALNA")
    return done, failed