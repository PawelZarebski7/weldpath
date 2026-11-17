"""Generuje spawaną belkę dwuteową z żebrami usztywniającymi (data/beam.step).

Belka 2000 mm, leży pasem dolnym na stojakach (oś belki = oś X):
    pas dolny   2000 x 200 x 12   (z 0..12)
    środnik     2000 x   8 x 276  (z 12..288, środek y = 0)
    pas górny   2000 x 200 x 12   (z 288..300)
    żebra       10 x 92 x 276, po obu stronach środnika, w x = 700 i x = 1300

Przekrój (widok wzdłuż belki):

    ┌──────────┬┬──────────┐  ← pas górny
    │  żebro   ││  żebro   │
    │          ││          │  ← środnik w środku
    │          ││          │
    └──────────┴┴──────────┘  ← pas dolny
      y=-100   y=0     y=+100

Każda blacha to osobna bryła. Spoiny: środnik z pasami (przerywane żebrami)
oraz żebra ze środnikiem i z obydwoma pasami.
"""
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound

LENGTH, FLANGE_W, FLANGE_T, WEB_T, HEIGHT = 2000.0, 200.0, 12.0, 8.0, 300.0
STIFFENER_T = 10.0
STIFFENERS_X = (700.0, 1300.0)
WEB_H = HEIGHT - 2 * FLANGE_T


def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def plates():
    yield box(0, -FLANGE_W / 2, 0, LENGTH, FLANGE_W, FLANGE_T)                   # pas dolny
    yield box(0, -WEB_T / 2, FLANGE_T, LENGTH, WEB_T, WEB_H)                     # środnik
    yield box(0, -FLANGE_W / 2, HEIGHT - FLANGE_T, LENGTH, FLANGE_W, FLANGE_T)  # pas górny
    depth = FLANGE_W / 2 - WEB_T / 2 - 4.0      # żebro kończy się 4 mm przed krawędzią pasa
    for x in STIFFENERS_X:
        x0 = x - STIFFENER_T / 2
        yield box(x0, WEB_T / 2, FLANGE_T, STIFFENER_T, depth, WEB_H)            # strona +Y
        yield box(x0, -WEB_T / 2 - depth, FLANGE_T, STIFFENER_T, depth, WEB_H)   # strona -Y


if __name__ == "__main__":
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for plate in plates():
        builder.Add(compound, plate)
    out = Path(__file__).resolve().parent.parent / "data" / "beam.step"
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(out)) == IFSelect_RetDone
    print(f"Zapisano {out}: belka {LENGTH:.0f} mm, {2 * len(STIFFENERS_X)} żeber")