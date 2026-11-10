"""Most do RoboDK: konwencje osi, kinematyka, detal w stacji, kolizje i animacja.

RoboDK musi być uruchomione (albo Robolink() je uruchomi). Moduł działa także
z darmową licencją – część funkcji (np. kolizje) może być wtedy ograniczona.

Konwencje kątów. Nasz model OPW i sterownik/RoboDK mogą inaczej definiować zero
i kierunek osi:

    q_OPW [°] = znak · q_robot [°] + przesunięcie          (osie 1, 2, 4, 5, 6)
    q3_OPW    = znak · q3_robot + przesunięcie + sprzężenie · q2_robot

Sprzężenie osi 2-3: ABB IRB 2400 ma RÓWNOLEGŁOWÓD – silnik osi 3 napędza
przedramię przez dźwignię, więc sterownik mierzy kąt osi 3 względem POZIOMU,
a OPW – względem ramienia. Obrót osi 2 w sterowniku nie zmienia nachylenia
przedramienia, stąd sprzężenie −1. Do tego zero osi 3 w ABB to przedramię
poziomo, a w OPW pionowo – przesunięcie +90°.
"""
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from robodk import robolink, robomath


@dataclass(frozen=True)
class JointConvention:
    signs: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    offsets_deg: tuple[float, ...] = (0.0, 0.0, 90.0, 0.0, 0.0, 0.0)
    j23_coupling: float = 0.0   # q3_OPW += j23_coupling · q2_robot

    def to_robot(self, q_opw: np.ndarray) -> list[float]:
        """Kąty OPW [rad] -> kąty sterownika [°]."""
        deg = np.degrees(np.asarray(q_opw, float))
        q = [(d - o) * s for d, o, s in zip(deg, self.offsets_deg, self.signs)]
        # Oś 3: zdejmujemy sprzężenie z osią 2 (q[1] to już kąt sterownika osi 2).
        q[2] = (deg[2] - self.offsets_deg[2] - self.j23_coupling * q[1]) * self.signs[2]
        return [float(v) for v in q]

    def to_opw(self, q_robot_deg) -> np.ndarray:
        """Kąty sterownika [°] -> kąty OPW [rad]."""
        q = np.asarray(list(q_robot_deg)[:6], float)
        deg = q * np.array(self.signs) + np.array(self.offsets_deg)
        deg[2] += self.j23_coupling * q[1]
        return np.radians(deg)


ABB_IRB2400_CONVENTION = JointConvention(j23_coupling=-1.0)


def to_mat(T: np.ndarray) -> robomath.Mat:
    return robomath.Mat(np.asarray(T, float).tolist())


def to_numpy(M: robomath.Mat) -> np.ndarray:
    # Uwaga: Mat.tolist() w RoboDK zwraca tylko jedną kolumnę – pełna macierz jest w .rows
    return np.array(M.rows, float)


class RoboDKCell:
    """Połączenie z otwartą stacją RoboDK i robotem w niej."""

    def __init__(self, robot_name: str = "ABB IRB 2400/16",
                 convention: JointConvention = ABB_IRB2400_CONVENTION,
                 rdk: robolink.Robolink | None = None):
        self.rdk = rdk or robolink.Robolink()
        self.robot = self.rdk.Item(robot_name, robolink.ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise LookupError(f"Nie ma robota '{robot_name}' w otwartej stacji RoboDK")
        self.convention = convention

    # --- kinematyka RoboDK (nie rusza robota, tylko liczy) --------------------
    def fk(self, q_opw: np.ndarray) -> np.ndarray:
        """Kołnierz względem bazy robota według kinematyki RoboDK (kąty OPW)."""
        return self.fk_robot(self.convention.to_robot(q_opw))

    def fk_robot(self, q_robot_deg) -> np.ndarray:
        """Kołnierz względem bazy robota według kinematyki RoboDK (kąty sterownika [°])."""
        return to_numpy(self.robot.SolveFK([float(v) for v in q_robot_deg]))

    # --- stan stacji ----------------------------------------------------------
    def set_joints(self, q_opw: np.ndarray) -> None:
        self.robot.setJoints(self.convention.to_robot(q_opw))

    def joints(self) -> np.ndarray:
        return self.convention.to_opw(self.robot.Joints().list())

    def set_tool(self, flange_to_tcp: np.ndarray, name: str = "Palnik"):
        """Dodaje do robota narzędzie (sam układ TCP, bez geometrii)."""
        tool = self.robot.AddTool(to_mat(flange_to_tcp), name)
        self.robot.setPoseTool(tool)
        return tool

    def add_part(self, step_path: str, T_base_part: np.ndarray, name: str = "Detal"):
        """Wczytuje STEP do stacji i ustawia go względem bazy robota."""
        base = self.robot.Parent()  # układ "ABB IRB 2400/16 Base"
        part = self.rdk.AddFile(str(step_path), base)
        if not part.Valid():
            raise OSError(f"RoboDK nie wczytało {step_path}")
        part.setName(name)
        part.setPose(to_mat(T_base_part))
        return part

    def play(self, path: list[np.ndarray], steps_per_segment: int = 10) -> None:
        """Animacja: przejazd po kolejnych ułożeniach z interpolacją w przestrzeni osi."""
        for a, b in pairwise(path):
            for t in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
                self.set_joints(a + (b - a) * t)
        self.set_joints(path[-1])


class RoboDKChecker:
    """CollisionChecker oparty na silniku kolizji RoboDK (ten sam interfejs co FCLChecker)."""

    def __init__(self, cell: RoboDKCell):
        self.cell = cell
        self.checks = 0
        cell.rdk.setCollisionActive(robolink.COLLISION_ON)

    def in_collision(self, q: np.ndarray) -> bool:
        self.checks += 1
        self.cell.set_joints(q)
        return self.cell.rdk.Collisions() > 0