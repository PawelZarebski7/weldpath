"""Generuje wspornik z pochylonym ramieniem i trójkątnym żebrem (data/bracket.step).

Typowa konsola maszynowa (mocowanie siłownika, osprzęt): płyta montażowa,
ramię pochylone pod kątem 60° i trójkątne żebro podpierające ramię.
Spoiny mają kąty 60°, 90° i 120° – palnik nie może stać "na sztywno" pod 45°.

Widok z boku (oś X w prawo, Z w górę), żebro w środku szerokości wspornika:

                       ╱╱  ← ramię 12 mm, pochylone 60° do płyty
                      ╱╱
           żebro →  ╱│╱╱
                  ╱  ╱╱
                ╱   ╱╱  60°
    ═══════════════════════════  ← płyta montażowa 250 x 150 x 15
            120° po lewej, 60° po prawej stronie ramienia

Każda blacha to osobna bryła.
"""
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound

BASE = (250.0, 150.0, 15.0)        # płyta montażowa
ARM_T, ARM_L = 12.0, 200.0         # grubość i długość ramienia
ALPHA = np.radians(60.0)           # pochylenie ramienia względem płyty
ARM_X0 = 110.0                     # gdzie ramię stoi na płycie (lewa krawędź stopy)
GUSSET_T = 10.0                    # grubość żebra
GUSSET_BASE, GUSSET_UP = 80.0, 120.0   # długość żebra na płycie i na ramieniu


def prism_xz(points, y0: float, width: float):
    """Bryła z wielokąta w płaszczyźnie XZ, wyciągnięta wzdłuż Y."""
    poly = BRepBuilderAPI_MakePolygon()
    for x, z in points:
        poly.Add(gp_Pnt(x, y0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, width, 0)).Shape()


def plates():
    z0 = BASE[2]
    yield BRepPrimAPI_MakeBox(*BASE).Shape()

    # Ramię: równoległobok, stopa leży poziomo na płycie, ścianki pochylone o ALPHA.
    u = np.array([np.cos(ALPHA), np.sin(ALPHA)])        # kierunek wzdłuż ramienia
    foot = ARM_T / np.sin(ALPHA)                        # szerokość stopy pochylonej blachy
    p0 = np.array([ARM_X0, z0])
    p1 = p0 + [foot, 0.0]
    arm = [p0, p1, p1 + ARM_L * u, p0 + ARM_L * u]
    yield prism_xz(arm, 0.0, BASE[1])

    # Żebro: trójkąt po stronie kąta rozwartego (lewa ścianka ramienia), w środku szerokości.
    gusset = [p0 - [GUSSET_BASE, 0.0], p0, p0 + GUSSET_UP * u]
    yield prism_xz(gusset, BASE[1] / 2 - GUSSET_T / 2, GUSSET_T)


if __name__ == "__main__":
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for plate in plates():
        builder.Add(compound, plate)
    out = Path(__file__).resolve().parent.parent / "data" / "bracket.step"
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(out)) == IFSelect_RetDone
    print(f"Zapisano {out}: wspornik z ramieniem pod {np.degrees(ALPHA):.0f}° i żebrem")