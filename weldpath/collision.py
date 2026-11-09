"""Krok 5a: model kolizyjny robota i otoczenia (FCL).

Robot  -> kilka KAPSUŁ (walec z półkulami na końcach) rozpiętych między punktami
          charakterystycznymi z robot.joint_positions + palnik z dwóch kapsuł
          (gruby korpus i wąska dysza – jeden gruby walec nie wejdzie w ciasny narożnik).
          Kapsuła to najszybsza bryła do testów kolizji: odległość od kapsuły
          to odległość od odcinka minus promień.
Detal  -> siatka trójkątów z modelu STEP (BRepMesh) w drzewie BVH.
          Detal może się ruszać (pozycjoner): set_part_pose() przestawia go razem
          z przymocowanymi do niego elementami (np. tarczą pozycjonera).
Stół   -> prostopadłościan (nieruchomy, opcjonalny).

Planer pyta tylko przez interfejs CollisionChecker. Dzięki temu silnik FCL
można później podmienić na silnik kolizji RoboDK bez zmiany planera.
"""
from dataclasses import dataclass
from typing import Protocol

import fcl
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape

from weldpath.cad import faces
from weldpath.robot import OPWParams, forward_kinematics, joint_positions
from weldpath.weld_job import DEFAULT_TORCH, Torch


class CollisionChecker(Protocol):
    """Wszystko, czego planer potrzebuje od silnika kolizji."""

    def in_collision(self, q: np.ndarray) -> bool: ...


# --- geometria pomocnicza -------------------------------------------------

def mesh_from_shape(shape: TopoDS_Shape, deflection: float = 0.5):
    """Triangulacja bryły B-Rep -> (wierzchołki Nx3, trójkąty Mx3)."""
    BRepMesh_IncrementalMesh(shape, deflection)  # liczy siatkę i zapisuje ją w ścianach
    verts, tris = [], []
    for face in faces(shape):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        offset = len(verts)
        for i in range(1, tri.NbNodes() + 1):       # indeksy od 1 (OpenCASCADE)
            p = tri.Node(i).Transformed(trsf)
            verts.append([p.X(), p.Y(), p.Z()])
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            tris.append([offset + a - 1, offset + b - 1, offset + c - 1])
    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def _transform(T: np.ndarray) -> fcl.Transform:
    return fcl.Transform(T[:3, :3], T[:3, 3])


def _segment_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Macierz 4x4 ustawiająca kapsułę FCL (oś Z, środek w 0) na odcinku a-b."""
    z = (b - a) / np.linalg.norm(b - a)
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = (a + b) / 2
    return T


# --- model robota ---------------------------------------------------------

@dataclass(frozen=True)
class RobotGeometry:
    """Promienie kapsuł wokół członów robota [mm] (z zapasem względem korpusu)."""
    upper_arm: float = 110.0
    forearm: float = 90.0
    wrist: float = 70.0
    torch: float = 25.0            # korpus palnika i szyjka
    nozzle: float = 10.0           # dysza gazowa (Ø 20 mm)
    nozzle_length: float = 70.0    # długość dyszy liczona od TCP
    torch_clearance: float = 15.0  # dysza kończy się 15 mm przed TCP (wolny wylot drutu)


DEFAULT_GEOMETRY = RobotGeometry()


class FCLChecker:
    """Sprawdzanie kolizji robota z detalem i stołem w bibliotece FCL."""

    def __init__(self, robot: OPWParams, part: TopoDS_Shape, T_base_part: np.ndarray,
                 torch: Torch = DEFAULT_TORCH, geometry: RobotGeometry = DEFAULT_GEOMETRY,
                 table_top_z: float | None = None, margin: float = 0.0,
                 attachments: tuple = ()):
        """attachments: (geometria FCL, T_detal→element) – rzeczy poruszające się z detalem."""
        self.robot = robot
        self.torch = torch
        self.geometry = geometry
        self.margin = margin
        self.checks = 0  # licznik wywołań – przyda się do pomiaru wydajności

        verts, tris = mesh_from_shape(part)
        bvh = fcl.BVHModel()
        bvh.beginModel(len(verts), len(tris))
        bvh.addSubModel(verts, tris)
        bvh.endModel()
        # Obiekty związane z detalem: (obiekt FCL, jego pozycja w układzie detalu).
        self._part_objects = [(fcl.CollisionObject(bvh), np.eye(4))]
        for attached, T_local in attachments:
            self._part_objects.append((fcl.CollisionObject(attached), T_local))
        self.obstacles = [obj for obj, _ in self._part_objects]
        self.set_part_pose(T_base_part)

        if table_top_z is not None:  # stół 2x2 m, 50 mm grubości, pod detalem
            table = fcl.Box(2000.0, 2000.0, 50.0)
            T = np.eye(4)
            T[:3, 3] = [T_base_part[0, 3], T_base_part[1, 3], table_top_z - 25.0]
            self.obstacles.append(fcl.CollisionObject(table, _transform(T)))

        g = geometry
        self._radii = [g.upper_arm, g.forearm, g.wrist, g.torch, g.nozzle]
        self._links: list[fcl.CollisionObject] = []

    def set_part_pose(self, T_base_part: np.ndarray) -> None:
        """Przestawia detal (i elementy z nim związane) w nowe miejsce."""
        self.T_base_part = T_base_part
        for obj, T_local in self._part_objects:
            obj.setTransform(_transform(T_base_part @ T_local))

    def link_segments(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Odcinki osi kapsuł: ramię, przedramię, nadgarstek, korpus palnika, dysza.

        Kapsuła sięga o promień poza swój odcinek, więc odcinki skracamy o promień,
        żeby kapsuła kończyła się dokładnie tam, gdzie chcemy.
        """
        g = self.geometry
        _, shoulder, elbow, wrist, flange = joint_positions(self.robot, q)
        T_tcp = forward_kinematics(self.robot, q) @ self.torch.flange_to_tcp()
        tcp, axis = T_tcp[:3, 3], T_tcp[:3, 2]
        nozzle_tip = tcp - (g.torch_clearance + g.nozzle) * axis
        nozzle_back = tcp - g.nozzle_length * axis
        body_end = tcp - (g.nozzle_length + g.torch) * axis
        return [(shoulder, elbow), (elbow, wrist), (wrist, flange),
                (flange, body_end), (nozzle_back, nozzle_tip)]

    def _update_links(self, q: np.ndarray) -> None:
        # FCL w Pythonie nie pozwala podmienić geometrii obiektu, a długość kapsuły
        # zależy od q (np. palnik), więc za każdym razem budujemy obiekty od nowa.
        # Kapsuła w FCL: lz = długość części walcowej, półkule dokładają r z obu stron.
        self._links = [
            fcl.CollisionObject(fcl.Capsule(r + self.margin, max(np.linalg.norm(b - a), 1e-3)),
                                _transform(_segment_pose(a, b)))
            for r, (a, b) in zip(self._radii, self.link_segments(q))
        ]

    def in_collision(self, q: np.ndarray) -> bool:
        self.checks += 1
        self._update_links(q)
        request = fcl.CollisionRequest()
        for link in self._links:
            for obstacle in self.obstacles:
                if fcl.collide(link, obstacle, request, fcl.CollisionResult()):
                    return True
        return False

    def clearance(self, q: np.ndarray) -> float:
        """Najmniejsza odległość robota od przeszkód [mm] (0 = styk/kolizja)."""
        self._update_links(q)
        request = fcl.DistanceRequest()
        return min(fcl.distance(link, obstacle, request, fcl.DistanceResult())
                   for link in self._links for obstacle in self.obstacles)