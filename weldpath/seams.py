"""Krok 2: wykrywanie spoin pachwinowych na krawędziach wklęsłych.

Dla każdej krawędzi mamy dwie sąsiednie ściany. Dla każdej z nich liczymy
w punkcie środkowym krawędzi:
    n  – normalną zewnętrzną ściany (na zewnątrz materiału),
    d  – kierunek "w głąb ściany": leży w ścianie, prostopadły do krawędzi,
         skierowany od krawędzi w stronę reszty ściany.

Krawędź jest wklęsła (dolina -> kandydat na spoinę), gdy n1 · d2 > 0,
czyli normalna jednej ściany "patrzy" na drugą ścianę.
Kąt otwarcia doliny to kąt między d1 i d2 (90° dla złącza teowego).
"""
from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as EdgeFaceMap,
)
from OCP.GeomAbs import GeomAbs_Line
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shape


@dataclass
class Seam:
    edge: TopoDS_Edge
    faces: tuple[TopoDS_Face, TopoDS_Face]
    start: np.ndarray        # punkt początkowy [mm]
    end: np.ndarray          # punkt końcowy [mm]
    tangent: np.ndarray      # kierunek spoiny (jednostkowy)
    normals: tuple[np.ndarray, np.ndarray]  # normalne zewnętrzne obu ścian (w midpoint)
    opening_angle: float     # kąt otwarcia doliny [°]
    arc_length: float | None = None        # długość krzywej (tylko spoiny krzywoliniowe)
    midpoint: np.ndarray | None = None     # punkt w połowie krzywej (tam liczone normalne)

    @property
    def length(self) -> float:
        # Dla prostej: odległość końców. Dla okręgu początek = koniec, więc liczy
        # się długość łuku zapisana przy wykrywaniu.
        if self.arc_length is not None:
            return self.arc_length
        return float(np.linalg.norm(self.end - self.start))

    @property
    def is_straight(self) -> bool:
        return self.arc_length is None


def _np(v) -> np.ndarray:
    return np.array([v.X(), v.Y(), v.Z()])


def edge_face_map(shape: TopoDS_Shape) -> list[tuple[TopoDS_Edge, list[TopoDS_Face]]]:
    """Każda krawędź (raz) razem z listą ścian, do których należy."""
    m = EdgeFaceMap()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    result = []
    for i in range(1, m.Size() + 1):  # mapy OpenCASCADE są indeksowane od 1
        edge = TopoDS.Edge(m.FindKey(i))
        faces = [TopoDS.Face(f) for f in m.FindFromIndex(i)]
        result.append((edge, faces))
    return result


def face_normal(face: TopoDS_Face, point: np.ndarray) -> np.ndarray:
    """Normalna zewnętrzna ściany w punkcie (uwzględnia orientację ściany)."""
    surface = BRep_Tool.Surface_s(face)
    uv = ShapeAnalysis_Surface(surface).ValueOfUV(gp_Pnt(*point), 1e-6)
    p, n = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face).Normal(uv.X(), uv.Y(), p, n)
    v = _np(n)
    return v / np.linalg.norm(v)


def _edge_orientation_in_face(edge: TopoDS_Edge, face: TopoDS_Face):
    """Ta sama krawędź ma w każdej ścianie inną orientację (FORWARD/REVERSED)."""
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        if exp.Current().IsSame(edge):
            return exp.Current().Orientation()
        exp.Next()
    raise ValueError("Krawędź nie należy do tej ściany")


def inward_direction(edge: TopoDS_Edge, face: TopoDS_Face, point: np.ndarray,
                     tangent: np.ndarray) -> np.ndarray:
    """Kierunek leżący w ścianie, prostopadły do krawędzi, skierowany w głąb ściany.

    OpenCASCADE obiega granicę ściany tak, że materiał jest po LEWEJ stronie
    (patrząc od strony normalnej). "W lewo" od kierunku t to n × t.
    """
    t = tangent.copy()
    if _edge_orientation_in_face(edge, face) == TopAbs_REVERSED:
        t = -t
    d = np.cross(face_normal(face, point), t)
    return d / np.linalg.norm(d)


def find_seams(shape: TopoDS_Shape, min_angle: float = 30.0,
               max_angle: float = 150.0) -> list[Seam]:
    """Znajduje krawędzie wklęsłe o kącie otwarcia w zadanym zakresie."""
    seams = []
    for edge, faces in edge_face_map(shape):
        if len(faces) != 2:
            continue  # krawędź swobodna albo nie-manifold: na razie pomijamy

        curve = BRepAdaptor_Curve(edge)
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        p_mid, v_mid = gp_Pnt(), gp_Vec()
        curve.D1((u0 + u1) / 2, p_mid, v_mid)
        mid = _np(p_mid)
        t = _np(v_mid)
        t /= np.linalg.norm(t)

        f1, f2 = faces
        n1, n2 = face_normal(f1, mid), face_normal(f2, mid)
        d1 = inward_direction(edge, f1, mid, t)
        d2 = inward_direction(edge, f2, mid, t)

        if np.dot(n1, d2) <= 1e-9:
            continue  # grzbiet (wypukła) albo płasko

        angle = float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))
        if not (min_angle <= angle <= max_angle):
            continue

        arc = None
        if curve.GetType() != GeomAbs_Line:
            props = GProp_GProps()
            BRepGProp.LinearProperties_s(edge, props)
            arc = props.Mass()
        seams.append(Seam(
            edge=edge,
            faces=(f1, f2),
            start=_np(curve.Value(u0)),
            end=_np(curve.Value(u1)),
            tangent=t,
            normals=(n1, n2),
            opening_angle=angle,
            arc_length=arc,
            midpoint=mid,
        ))
    return seams