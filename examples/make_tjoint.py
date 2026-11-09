"""Generuje detal testowy: złącze teowe (T-joint) i zapisuje do data/tjoint.step.

Blacha bazowa: 200 x 100 x 10 mm (leży na płasko, z = 0..10)
Żebro:         200 x 10 x 80 mm  (stoi na środku bazy, z = 10..90)

Po zsumowaniu (fuse) powstaje jedna bryła. Wzdłuż linii styku żebra z bazą
są dwie krawędzie wklęsłe — to są nasze przyszłe spoiny pachwinowe.
"""
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

base = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 200, 100, 10).Shape()
rib = BRepPrimAPI_MakeBox(gp_Pnt(0, 45, 10), 200, 10, 80).Shape()

fused = BRepAlgoAPI_Fuse(base, rib).Shape()

# Fuse zostawia "szwy" na ścianach leżących w tej samej płaszczyźnie.
# UnifySameDomain je scala, żeby górna ściana bazy nie była pocięta bez potrzeby.
unify = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
unify.Build()
shape = unify.Shape()

out = Path(__file__).resolve().parent.parent / "data" / "tjoint.step"
writer = STEPControl_Writer()
writer.Transfer(shape, STEPControl_AsIs)
assert writer.Write(str(out)) == IFSelect_RetDone
print(f"Zapisano {out}")