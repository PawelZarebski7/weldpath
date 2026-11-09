"""Generuje detal wielobryłowy: płyta + 2 żebra podłużne + żebro poprzeczne.

Każda blacha to OSOBNA bryła (tak jak w modelu złożeniowym z CAD) – blachy tylko
się stykają, nie są połączone. Zapis do data/frame.step.

Widok z góry (oś X w prawo, Y w górę):

    y=160 ┌──────────────────────────────┐  żebro B (y 150..160, wys. 80)
    y=150 ├──────────────┬─┬─────────────┤
          │              │ │  żebro      │
          │              │ │  poprzeczne │  (x 145..155, wys. 60)
    y=50  ├──────────────┴─┴─────────────┤
    y=40  └──────────────────────────────┘  żebro A (y 40..50, wys. 80)
          x=0                          x=300
    Pod spodem płyta 300 x 200 x 10.
"""
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound


def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


plates = [
    box(0, 0, 0, 300, 200, 10),      # płyta bazowa
    box(0, 40, 10, 300, 10, 80),     # żebro A
    box(0, 150, 10, 300, 10, 80),    # żebro B
    box(145, 50, 10, 10, 100, 60),   # żebro poprzeczne (między A i B)
]

compound = TopoDS_Compound()
builder = BRep_Builder()
builder.MakeCompound(compound)
for plate in plates:
    builder.Add(compound, plate)

out = Path(__file__).resolve().parent.parent / "data" / "frame.step"
writer = STEPControl_Writer()
writer.Transfer(compound, STEPControl_AsIs)
assert writer.Write(str(out)) == IFSelect_RetDone
print(f"Zapisano {out} ({len(plates)} bryły)")