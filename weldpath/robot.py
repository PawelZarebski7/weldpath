"""Krok 4: kinematyka robota 6-osiowego z nadgarstkiem sferycznym (model OPW).

Model OPW (Ortho-Parallel basis + spherical Wrist), Brandstötter i in. 2014,
opisuje większość robotów przemysłowych (KUKA, ABB, Fanuc, Yaskawa) siedmioma
wymiarami zamiast tabeli Denavita-Hartenberga:

        c4 ┃ ← kołnierz (flange)
           ◯  ← środek nadgarstka (osie 4, 5, 6 przecinają się w jednym punkcie)
       c3 ╱  a2 (przesunięcie łokcia)
         ◯  ← oś 3 (łokieć)
     c2  ┃
         ◯  ← oś 2 (ramię)   a1 = wysunięcie osi 2 od osi 1
     c1  ┃
    ═════╧═════ podstawa, oś 1 pionowa   (b = boczne przesunięcie, zwykle 0)

Kąty w radianach, wymiary w mm. Konwencja kątów to "natywna" konwencja OPW –
przy podpinaniu konkretnego robota w RoboDK dochodzą znaki i przesunięcia zer osi.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OPWParams:
    a1: float
    a2: float
    b: float
    c1: float
    c2: float
    c3: float
    c4: float


# Wymiary robotów z pracy Brandstöttera i z pakietu ROS-Industrial opw_kinematics.
KUKA_KR6_R700 = OPWParams(a1=25, a2=-35, b=0, c1=400, c2=315, c3=365, c4=80)
ABB_IRB2400 = OPWParams(a1=100, a2=-135, b=0, c1=615, c2=705, c3=755, c4=85)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def forward_kinematics(p: OPWParams, q: np.ndarray) -> np.ndarray:
    """FK: kąty przegubów -> macierz 4x4 kołnierza w układzie podstawy."""
    q1, q2, q3, q4, q5, q6 = q
    psi3 = np.arctan2(p.a2, p.c3)
    k = np.hypot(p.a2, p.c3)

    # Środek nadgarstka w płaszczyźnie ramienia, potem obrót osią 1.
    cx1 = p.c2 * np.sin(q2) + k * np.sin(q2 + q3 + psi3) + p.a1
    cy1 = p.b
    cz1 = p.c2 * np.cos(q2) + k * np.cos(q2 + q3 + psi3)
    wrist = np.array([cx1 * np.cos(q1) - cy1 * np.sin(q1),
                      cx1 * np.sin(q1) + cy1 * np.cos(q1),
                      cz1 + p.c1])

    R = _rz(q1) @ _ry(q2 + q3) @ _rz(q4) @ _ry(q5) @ _rz(q6)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = wrist + p.c4 * R[:, 2]
    return T


def inverse_kinematics(p: OPWParams, T: np.ndarray) -> np.ndarray:
    """IK: macierz 4x4 kołnierza -> do 8 rozwiązań (wiersze 8x6).

    Rozwiązania nieosiągalne (poza zasięgiem) mają wartości NaN.
    Kolejność: [ramię przód/tył] x [łokieć góra/dół] x [nadgarstek bez/z flipem].
    """
    R = T[:3, :3]
    c = T[:3, 3] - p.c4 * R[:, 2]          # środek nadgarstka
    cx, cy, cz = c

    nx1 = np.sqrt(max(cx**2 + cy**2 - p.b**2, 0.0)) - p.a1
    s1_2 = nx1**2 + (cz - p.c1) ** 2
    s2_2 = (nx1 + 2 * p.a1) ** 2 + (cz - p.c1) ** 2
    k_2 = p.a2**2 + p.c3**2
    s1, s2 = np.sqrt(s1_2), np.sqrt(s2_2)

    def acos_or_nan(x):
        return np.arccos(x) if -1.0 <= x <= 1.0 else np.nan

    t1_i = np.arctan2(cy, cx) - np.arctan2(p.b, nx1 + p.a1)
    t1_ii = np.arctan2(cy, cx) + np.arctan2(p.b, nx1 + p.a1) - np.pi

    a_s1 = acos_or_nan((s1_2 + p.c2**2 - k_2) / (2 * s1 * p.c2))
    a_s2 = acos_or_nan((s2_2 + p.c2**2 - k_2) / (2 * s2 * p.c2))
    t2 = [-a_s1 + np.arctan2(nx1, cz - p.c1),
          a_s1 + np.arctan2(nx1, cz - p.c1),
          -a_s2 - np.arctan2(nx1 + 2 * p.a1, cz - p.c1),
          a_s2 - np.arctan2(nx1 + 2 * p.a1, cz - p.c1)]

    b_s1 = acos_or_nan((s1_2 - p.c2**2 - k_2) / (2 * p.c2 * np.sqrt(k_2)))
    b_s2 = acos_or_nan((s2_2 - p.c2**2 - k_2) / (2 * p.c2 * np.sqrt(k_2)))
    psi = np.arctan2(p.a2, p.c3)
    t3 = [b_s1 - psi, -b_s1 - psi, b_s2 - psi, -b_s2 - psi]

    t1 = [t1_i, t1_i, t1_ii, t1_ii]
    solutions = []
    for i in range(4):
        sq1, cq1 = np.sin(t1[i]), np.cos(t1[i])
        s23, c23 = np.sin(t2[i] + t3[i]), np.cos(t2[i] + t3[i])
        m = R[0, 2] * s23 * cq1 + R[1, 2] * s23 * sq1 + R[2, 2] * c23
        q4 = np.arctan2(R[1, 2] * cq1 - R[0, 2] * sq1,
                        R[0, 2] * c23 * cq1 + R[1, 2] * c23 * sq1 - R[2, 2] * s23)
        q5 = np.arctan2(np.sqrt(max(1 - m**2, 0.0)), m)
        q6 = np.arctan2(R[0, 1] * s23 * cq1 + R[1, 1] * s23 * sq1 + R[2, 1] * c23,
                        -R[0, 0] * s23 * cq1 - R[1, 0] * s23 * sq1 - R[2, 0] * c23)
        solutions.append([t1[i], t2[i], t3[i], q4, q5, q6])
        solutions.append([t1[i], t2[i], t3[i], q4 + np.pi, -q5, q6 - np.pi])  # wrist flip

    q = np.array(solutions)
    return (q + np.pi) % (2 * np.pi) - np.pi  # zawijamy do (-pi, pi]