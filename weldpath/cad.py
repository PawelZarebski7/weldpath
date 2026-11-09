"""Wczytanie STEP i przegląd topologii B-Rep.

B-Rep (Boundary Representation) opisuje bryłę przez jej granice:
    Solid -> Shell -> Face (ściana) -> Wire -> Edge (krawędź) -> Vertex

Dla spawania najważniejsze są:
    Edge  – linia, wzdłuż której idzie spoina,
    Face  – dwie ściany sąsiadujące z krawędzią; ich normalne mówią,
            jak ustawić palnik.
"""
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape


def load_step(path: str) -> TopoDS_Shape:
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise OSError(f"Nie udało się wczytać {path}")
    reader.TransferRoots()
    return reader.OneShape()


def faces(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield TopoDS.Face(exp.Current())
        exp.Next()


def edges(shape):
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        yield TopoDS.Edge(exp.Current())
        exp.Next()


def face_area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def edge_length(edge) -> float:
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    return props.Mass()


def surface_type(face) -> str:
    # GeomAbs_Plane, GeomAbs_Cylinder, ... -> "Plane", "Cylinder"
    return BRepAdaptor_Surface(face).GetType().name.removeprefix("GeomAbs_")


def curve_type(edge) -> str:
    return BRepAdaptor_Curve(edge).GetType().name.removeprefix("GeomAbs_")