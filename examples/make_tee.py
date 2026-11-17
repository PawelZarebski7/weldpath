"""Stelaż z rurek: rura Ø42.4 x 3 dospawana prostopadle do rury Ø60.3 x 3 (data/tee.step).

Typowy węzeł stelaża (meble, wózki, poręcze). Spoina leży na przecięciu dwóch walców –
to krzywa SIODŁOWA w 3D: z boku rury głównej schodzi w dół, a kąt między ściankami
zmienia się wzdłuż spoiny (ok. 90° na grzbiecie, ok. 135° na bokach).

    rura główna: oś X, długość 300 mm, środek w (0, 0, 0)
    odgałęzienie: oś Z, od powierzchni rury głównej w górę, 150 mm
"""
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound

MAIN_D, MAIN_T, MAIN_L = 60.3, 3.0, 300.0
BRANCH_D, BRANCH_T, BRANCH_L = 42.4, 3.0, 150.0


def tube(origin, direction, diameter, wall, length):
    ax = gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction))
    outer = BRepPrimAPI_MakeCylinder(ax, diameter / 2, length).Shape()
    inner = BRepPrimAPI_MakeCylinder(ax, diameter / 2 - wall, length).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


def branch():
    """Odgałęzienie przycięte do kształtu rury głównej ("wyprofilowane", jak z wycinarki)."""
    raw = tube((0, 0, 0), (0, 0, 1), BRANCH_D, BRANCH_T, MAIN_D / 2 + BRANCH_L)
    main_solid = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-MAIN_L / 2, 0, 0), gp_Dir(1, 0, 0)),
                                          MAIN_D / 2, MAIN_L).Shape()
    return BRepAlgoAPI_Cut(raw, main_solid).Shape()


if __name__ == "__main__":
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, tube((-MAIN_L / 2, 0, 0), (1, 0, 0), MAIN_D, MAIN_T, MAIN_L))
    builder.Add(compound, branch())
    out = Path(__file__).resolve().parent.parent / "data" / "tee.step"
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(out)) == IFSelect_RetDone
    print(f"Zapisano {out}: rura Ø{BRANCH_D} na rurze Ø{MAIN_D}")