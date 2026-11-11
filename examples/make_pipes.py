"""Generuje dwie rury do spawania doczołowego z ukosowaniem V (data/pipes.step).

Rura Ø168.3 x 10 mm (typowy rozmiar DN150), dwa odcinki po 250 mm.
Ukosowanie: rowek V 60° (po 30° na każdą rurę), próg (land) 2 mm, bez szczeliny.

Przekrój ścianki w miejscu złącza (oś rury pionowo, r = promień):

        r_out ┐       ┌  ← powierzchnia zewnętrzna
               ╲  V  ╱   ← ukosowanie 30° na każdej rurze (rowek 60°)
                ╲   ╱
     r_in+2 ─────●───── ← próg 2 mm: tu rury się stykają, dno rowka = spoina
     r_in   ─────┴─────  ← powierzchnia wewnętrzna
            rura 1 | rura 2

Każda rura to osobna bryła (jak w złożeniu z CAD).
"""
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound

R_OUT = 168.3 / 2
WALL = 10.0
R_IN = R_OUT - WALL
LAND = 2.0
LENGTH = 250.0
BEVEL = np.radians(30.0)                       # połowa kąta rowka
BEVEL_HEIGHT = (WALL - LAND) * np.tan(BEVEL)   # o ile ukos "zjada" długość rury


def pipe(z0: float, z_joint: float) -> object:
    """Rura od z0 do złącza w z_joint, ukosowana od strony złącza (obrót profilu wokół Z)."""
    s = 1.0 if z_joint > z0 else -1.0           # w którą stronę idzie rura
    profile = [
        (R_IN, z0),
        (R_OUT, z0),
        (R_OUT, z_joint - s * BEVEL_HEIGHT),     # początek ukosu na powierzchni zewnętrznej
        (R_IN + LAND, z_joint),                  # dno ukosu = krawędź progu
        (R_IN, z_joint),
    ]
    poly = BRepBuilderAPI_MakePolygon()
    for r, z in profile:
        poly.Add(gp_Pnt(r, 0.0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                 2 * np.pi).Shape()


if __name__ == "__main__":
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, pipe(0.0, LENGTH))             # rura dolna, złącze na górze
    builder.Add(compound, pipe(2 * LENGTH, LENGTH))      # rura górna, złącze na dole

    out = Path(__file__).resolve().parent.parent / "data" / "pipes.step"
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(out)) == IFSelect_RetDone
    print(f"Zapisano {out}: 2 rury Ø{2 * R_OUT:.1f} x {WALL:.0f} mm, rowek V 60°")