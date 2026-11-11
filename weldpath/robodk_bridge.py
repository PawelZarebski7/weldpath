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
from itertools import pairwise
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound
from robodk import robolink, robomath

from weldpath.collision import DEFAULT_GEOMETRY, RobotGeometry

# Konwencja osi to czysta kinematyka – mieszka w robot.py, tu ją tylko udostępniamy.
from weldpath.robot import ABB_IRB2400_CONVENTION, JointConvention

__all__ = ["ABB_IRB2400_CONVENTION", "JointConvention", "RoboDKCell", "RoboDKChecker"]


def write_torch_step(path: Path, flange_to_tcp: np.ndarray,
                     geometry: RobotGeometry = DEFAULT_GEOMETRY) -> Path:
    """Bryła palnika w układzie KOŁNIERZA – te same wymiary co model kolizji.

    Lanca leży na osi Z narzędzia (od kołnierza do końca drutu):
      korpus  Ø 2·torch   od kołnierza do początku dyszy,
      dysza   Ø 2·nozzle  do 'torch_clearance' przed końcem drutu,
      drut    Ø 1.2 mm    wolny wylot aż do TCP.
    """
    tcp = flange_to_tcp[:3, 3]
    axis = flange_to_tcp[:3, 2]
    length = float(np.linalg.norm(tcp))
    g = geometry
    pieces = [  # (od [mm wzdłuż osi], do, promień)
        (0.0, length - g.nozzle_length, g.torch),
        (length - g.nozzle_length, length - g.torch_clearance, g.nozzle),
        (length - g.torch_clearance, length, 0.6),
    ]
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for start, end, radius in pieces:
        origin = gp_Pnt(*(axis * start))
        frame = gp_Ax2(origin, gp_Dir(*axis))
        builder.Add(compound, BRepPrimAPI_MakeCylinder(frame, radius, end - start).Shape())

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise OSError(f"Nie udało się zapisać {path}")
    return path


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

    def find(self, name: str, item_type: int) -> list:
        """Wszystkie elementy stacji danego typu o DOKŁADNIE tej nazwie."""
        return [item for item in self.rdk.ItemList(item_type) if item.Name() == name]

    def remove(self, *names: str) -> int:
        """Usuwa ze stacji obiekty i układy odniesienia o podanych nazwach – wszystkie kopie.

        Usunięcie układu usuwa też wszystko, co jest do niego podpięte.
        Narzędzi celowo nie usuwamy: usunięcie aktywnego narzędzia robota przez API
        potrafi zerwać połączenie z RoboDK. Narzędzia używamy ponownie (set_tool).
        """
        removed = 0
        for name in names:
            for item_type in (robolink.ITEM_TYPE_OBJECT, robolink.ITEM_TYPE_FRAME):
                for item in self.find(name, item_type):
                    item.Delete()
                    removed += 1
        return removed

    def add_frame(self, name: str, T_base_frame: np.ndarray):
        """Nowy układ odniesienia względem bazy robota (np. tarcza pozycjonera)."""
        self.remove(name)
        frame = self.rdk.AddFrame(name, self.robot.Parent())
        frame.setPose(to_mat(T_base_frame))
        return frame

    def set_tool(self, flange_to_tcp: np.ndarray, name: str = "Palnik",
                 geometry_step: Path | None = None):
        """Ustawia robotowi narzędzie: układ TCP i (opcjonalnie) bryłę z pliku STEP.

        Jeśli narzędzie o tej nazwie już jest w stacji, używamy go ponownie (aktualizujemy
        tylko TCP) – dzięki temu skrypt można odpalać wiele razy bez duplikatów.
        Bryła musi być zapisana w układzie kołnierza (np. przez write_torch_step).
        """
        existing = self.find(name, robolink.ITEM_TYPE_TOOL)
        if existing:
            tool = existing[0]
            tool.setPoseTool(to_mat(flange_to_tcp))
            self.robot.setPoseTool(tool)
            return tool
        tool = self.robot.AddTool(to_mat(flange_to_tcp), name)
        self.robot.setPoseTool(tool)
        if geometry_step is not None:
            shape = self.rdk.AddFile(str(geometry_step))  # wczytujemy tymczasowo...
            tool.AddGeometry(shape, robomath.eye(4))      # ...kopiujemy do narzędzia...
            shape.Delete()                                # ...i usuwamy kopię ze stacji
        return tool

    def add_part(self, step_path: str, T_parent_part: np.ndarray, name: str = "Detal",
                 parent=None):
        """Wczytuje STEP do stacji i ustawia go względem rodzica (domyślnie bazy robota).

        Poprzedni obiekt o tej samej nazwie jest usuwany – bez duplikatów przy kolejnych uruchomieniach.
        """
        self.remove(name)
        parent = parent if parent is not None else self.robot.Parent()  # "ABB IRB 2400/16 Base"
        part = self.rdk.AddFile(str(step_path), parent)
        if not part.Valid():
            raise OSError(f"RoboDK nie wczytało {step_path}")
        part.setName(name)
        part.setPose(to_mat(T_parent_part))
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