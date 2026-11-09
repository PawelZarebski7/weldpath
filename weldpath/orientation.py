"""Krok 3: orientacja palnika – od geometrii spoiny do macierzy TCP 4x4.

Konwencja układu narzędzia (taka jak w RoboDK i u większości producentów):
    Z – oś palnika, od palnika W STRONĘ materiału (kierunek wylotu drutu),
    X – w przybliżeniu kierunek jazdy wzdłuż spoiny,
    Y – dopełnia układ prawoskrętny: Y = Z × X.

Parametry technologiczne:
    work_angle   – kąt roboczy: kąt między palnikiem a ścianą 1, mierzony
                   w płaszczyźnie prostopadłej do spoiny. None = dwusieczna
                   (np. 45° dla złącza 90°).
    travel_angle – kąt prowadzenia: pochylenie palnika wzdłuż spoiny.
                   > 0 = push (palnik "pcha", celuje do przodu),
                   < 0 = drag/pull (palnik "ciągnie", celuje do tyłu).
"""
import numpy as np

from weldpath.seams import Seam


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Macierz obrotu 3x3 wokół osi (wzór Rodriguesa)."""
    k = _unit(axis)
    a = np.radians(angle_deg)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Składa macierz jednorodną 4x4 z obrotu 3x3 i pozycji."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def torch_frame(seam: Seam, point: np.ndarray, work_angle: float | None = None,
                travel_angle: float = 0.0) -> np.ndarray:
    """Macierz 4x4 pozycji i orientacji TCP palnika w punkcie spoiny."""
    t = seam.tangent
    n1, n2 = seam.normals

    # 1) Pozycja wyjściowa: palnik na dwusiecznej doliny.
    #    n1 + n2 wskazuje "wylot" doliny, więc oś palnika (Z) to minus to.
    z = -_unit(n1 + n2)

    # 2) Kąt roboczy: obrót wokół osi spoiny.
    #    Na dwusiecznej kąt do ściany 1 = opening/2. Odchylamy o różnicę.
    if work_angle is not None:
        z = _apply_work_angle(seam, z, work_angle)

    # 3) Kąt prowadzenia: pochylenie w płaszczyźnie (oś palnika, kierunek jazdy).
    #    Push: koniec palnika celuje do przodu, czyli Z dostaje składową +t.
    y_axis = _unit(np.cross(z, t))
    z = rotation_about_axis(y_axis, travel_angle) @ z

    # 4) Uzupełniamy układ: X jak najbliżej kierunku jazdy, ale prostopadle do Z.
    x = _unit(t - np.dot(t, z) * z)
    y = np.cross(z, x)

    R = np.column_stack([x, y, z])
    return homogeneous(R, point)


def _apply_work_angle(seam: Seam, z: np.ndarray, work_angle: float) -> np.ndarray:
    """Ustawia kąt między palnikiem a ścianą 1 na work_angle (w przekroju spoiny)."""
    t = seam.tangent
    n1 = seam.normals[0]
    # Kierunek "w głąb" ściany 1, w przekroju: prostopadły do t i do n1.
    # Znak wybieramy tak, żeby wskazywał od krawędzi w stronę ściany 1,
    # czyli przeciwnie do strony, z której przychodzi palnik.
    d1 = _unit(np.cross(n1, t))
    if np.dot(d1, -z) < 0:
        d1 = -d1
    # Palnik pod kątem w do ściany 1: wektor od materiału do palnika
    # to cos(w)·d1 + sin(w)·n1, a oś Z palnika to minus ten wektor.
    w = np.radians(work_angle)
    return -_unit(np.cos(w) * d1 + np.sin(w) * n1)


def sample_seam(seam: Seam, step: float = 10.0) -> list[np.ndarray]:
    """Punkty co `step` mm od początku do końca spoiny (prostej)."""
    n = max(1, int(np.ceil(seam.length / step)))
    return [seam.start + (seam.end - seam.start) * i / n for i in range(n + 1)]


def torch_path(seam: Seam, step: float = 10.0, work_angle: float | None = None,
               travel_angle: float = 0.0) -> list[np.ndarray]:
    """Lista macierzy TCP wzdłuż całej spoiny."""
    return [torch_frame(seam, p, work_angle, travel_angle) for p in sample_seam(seam, step)]