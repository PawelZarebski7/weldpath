"""Węzeł ramy przyczepy: profil zamknięty 80x80x5 dospawany do płyty (data/rhs.step).

Profil prostokątny (RHS) ma zaokrąglone naroża (promień zewnętrzny 10 mm), więc spoina
dookoła profilu to łańcuch: 4 odcinki proste + 4 łuki. Spawa się ją jednym ciągłym
ruchem – palnik obraca się płynnie na narożach, łuk się nie przerywa.

    płyta 300 x 300 x 15, profil stoi pionowo na środku, wysokość 250 mm
"""
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound

PLATE = (300.0, 300.0, 15.0)
SIZE, WALL, HEIGHT = 80.0, 5.0, 250.0
R_OUT = 10.0
R_IN = R_OUT - WALL


def rounded_box(x0, y0, z0, size, height, radius):
    """Prostopadłościan z zaokrąglonymi pionowymi krawędziami."""
    box = BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), size, size, height).Shape()
    fillet = BRepFilletAPI_MakeFillet(box)
    exp = TopExp_Explorer(box, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge(exp.Current())
        c = BRepAdaptor_Curve(edge)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        if abs(p0.X() - p1.X()) < 1e-6 and abs(p0.Y() - p1.Y()) < 1e-6:   # krawędź pionowa
            fillet.Add(radius, edge)
        exp.Next()
    return fillet.Shape()


def profile():
    cx, cy = PLATE[0] / 2, PLATE[1] / 2
    outer = rounded_box(cx - SIZE / 2, cy - SIZE / 2, PLATE[2], SIZE, HEIGHT, R_OUT)
    inner = rounded_box(cx - SIZE / 2 + WALL, cy - SIZE / 2 + WALL, PLATE[2] - 1,
                        SIZE - 2 * WALL, HEIGHT + 2, R_IN)
    return BRepAlgoAPI_Cut(outer, inner).Shape()


if __name__ == "__main__":
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, BRepPrimAPI_MakeBox(*PLATE).Shape())
    builder.Add(compound, profile())
    out = Path(__file__).resolve().parent.parent / "data" / "rhs.step"
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(out)) == IFSelect_RetDone
    print(f"Zapisano {out}: profil {SIZE:.0f}x{SIZE:.0f}x{WALL:.0f} na płycie")