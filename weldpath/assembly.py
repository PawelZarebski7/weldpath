"""Spoiny w detalu wielobryłowym (każda blacha to osobna bryła w STEP).

Pomysł:
  1. Sumujemy (fuse) wszystkie bryły w jedną. Tam, gdzie blachy się stykały,
     powstają krawędzie wklęsłe – dokładnie te, które umie znaleźć find_seams.
  2. Dla każdej znalezionej spoiny sprawdzamy, do których ORYGINALNYCH brył
     należą jej dwie ściany: robimy mały krok od ściany w głąb materiału
     i pytamy, w której bryle leży ten punkt.
  3. Spoina łączy dwie RÓŻNE bryły -> prawdziwa spoina.
     Obie ściany z tej samej bryły -> to wewnętrzny narożnik jednej blachy
     (np. gięty profil), nie spoina – odrzucamy.

Ograniczenie: blachy muszą się stykać. Szczelina montażowa (np. 0.5 mm) sprawi,
że fuse ich nie połączy – wtedy trzeba szukać par ścian "prawie się stykających".
"""
from dataclasses import dataclass

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_IN, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Solid

from weldpath.seams import Seam, find_seams


@dataclass
class WeldSeam:
    seam: Seam
    bodies: tuple[int, int]   # indeksy łączonych brył (w kolejności ścian seam.faces)


def solids(shape: TopoDS_Shape) -> list[TopoDS_Solid]:
    """Wszystkie bryły z modelu (np. z kompaundu wczytanego ze STEP)."""
    result = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        result.append(TopoDS.Solid(exp.Current()))
        exp.Next()
    return result


def fuse_bodies(bodies: list[TopoDS_Shape]) -> TopoDS_Shape:
    """Suma wszystkich brył + scalenie ścian leżących w jednej płaszczyźnie."""
    shape = bodies[0]
    for body in bodies[1:]:
        shape = BRepAlgoAPI_Fuse(shape, body).Shape()
    unify = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unify.Build()
    return unify.Shape()


def body_index(point: np.ndarray, bodies: list[TopoDS_Shape], tol: float = 1e-6) -> int:
    """Indeks bryły zawierającej punkt (albo -1)."""
    for i, body in enumerate(bodies):
        classifier = BRepClass3d_SolidClassifier(body, gp_Pnt(*point), tol)
        if classifier.State() == TopAbs_IN:
            return i
    return -1


def find_weld_seams(bodies: list[TopoDS_Shape], min_length: float = 15.0,
                    probe: float = 0.5, **seam_kwargs) -> list[WeldSeam]:
    """Spoiny między różnymi bryłami złożenia."""
    fused = fuse_bodies(bodies)
    result = []
    for seam in find_seams(fused, **seam_kwargs):
        if seam.length < min_length:
            continue  # np. krótkie przejścia na czołach żeber – zwykle "spawanie dookoła"
        # Punkt na spoinie, w którym policzono normalne (dla prostej to środek odcinka).
        mid = seam.midpoint if seam.midpoint is not None else (seam.start + seam.end) / 2
        n1, n2 = seam.normals
        owners = []
        for n, other in ((n1, n2), (n2, n1)):
            # Kierunek w płaszczyźnie ściany, od krawędzi w głąb ściany: składowa
            # normalnej DRUGIEJ ściany prostopadła do tej (w dolinie to "od narożnika").
            along = other - np.dot(other, n) * n
            along /= np.linalg.norm(along)
            # 2*probe wzdłuż ściany (z dala od narożnika) i probe pod powierzchnię.
            point = mid + 2 * probe * along - probe * n
            owners.append(body_index(point, bodies))
        i, j = owners
        if i >= 0 and j >= 0 and i != j:
            result.append(WeldSeam(seam=seam, bodies=(i, j)))
    return result