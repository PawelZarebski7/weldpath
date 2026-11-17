"""Demo w RoboDK: spoina dookoła profilu (rhs) albo siodło rury na rurze (tee) na pozycjonerze.

Jak w robodk_show_pipe.py pozycjoner jest symulowany układem "Pozycjoner": detal,
tarcza z przyrządem i kawałki ściegu wiszą pod nim, a w każdym kroku ustawiamy
go tymi samymi macierzami co w planowaniu.

Wymaga otwartego RoboDK z robotem "ABB IRB 2400/16" (np. data/weldcell.rdk).
Uruchom:  poetry run python examples/robodk_show_job.py rhs
          poetry run python examples/robodk_show_job.py tee
"""
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound
from positioner_jobs import FACEPLATE_SIZE, ROOT, plan_job

from weldpath.coordinated import build_cell_program
from weldpath.robodk_bridge import RoboDKCell, to_mat, write_torch_step
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION
from weldpath.weld_job import DEFAULT_TORCH

NAME = sys.argv[1] if len(sys.argv) > 1 else "rhs"
if NAME not in ("rhs", "tee"):
    raise SystemExit("Użycie: robodk_show_job.py rhs|tee  (rury: robodk_show_pipe.py)")
CHUNK = 4                                    # punktów spoiny na jeden kawałek ściegu
BEAD_DIR = ROOT / "data" / "bead"
FIXTURE_STEP = ROOT / "data" / f"{NAME}_fixture.step"
TORCH_STEP = ROOT / "data" / "torch.step"
HOME = ABB_IRB2400_CONVENTION.to_opw([0, -50, 40, 0, 50, 0])


def write_step(shapes, path: Path) -> Path:
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def write_fixture_step(job) -> Path:
    """Tarcza 400 x 400 x 30 i przyrząd (dociski / pryzmy) – w układzie tarczy."""
    sx, sy, sz = FACEPLATE_SIZE
    shapes = [BRepPrimAPI_MakeBox(gp_Pnt(-sx / 2, -sy / 2, -sz), sx, sy, sz).Shape()]
    shapes += [BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape() for corner, size in job.fixtures]
    return write_step(shapes, FIXTURE_STEP)


def write_bead_chunks(samples, radius: float) -> list[Path]:
    """Ścieg jako łańcuch cienkich walców w pachwinie, pocięty na kawałki do animacji."""
    BEAD_DIR.mkdir(exist_ok=True)
    for old in BEAD_DIR.glob(f"{NAME}_*.step"):
        old.unlink()
    pts = [s.point + radius * s.opening for s in samples]
    paths = []
    for c, i in enumerate(range(0, len(pts) - 1, CHUNK)):
        pieces = []
        for a, b in pairwise(pts[i:i + CHUNK + 1]):
            axis = b - a
            if np.linalg.norm(axis) > 1e-6:
                pieces.append(BRepPrimAPI_MakeCylinder(
                    gp_Ax2(gp_Pnt(*a), gp_Dir(*axis)), radius, float(np.linalg.norm(axis))).Shape())
        paths.append(write_step(pieces, BEAD_DIR / f"{NAME}_{c:02d}.step"))
    return paths


# --- planowanie ---------------------------------------------------------------
job, positioner, checker, samples, plan = plan_job(NAME)
if plan is None:
    raise SystemExit("Nie znaleziono planu")
program = build_cell_program(ABB_IRB2400, plan, positioner, checker, HOME)
print("Program stanowiska (każdy odcinek bez kolizji):")
for seg in program:
    print(f"  {seg.label}")

# --- RoboDK -------------------------------------------------------------------
cell = RoboDKCell()
cell.set_base_pose(np.eye(4))
cell.remove("Rama", "Stół", "Rury", "Tarcza i uchwyt", "Hala", "Wózek toru", "Pozycjoner",
            "Detal", "Tarcza i przyrząd")
part_frame = cell.add_frame("Pozycjoner", positioner.T_base_part(0, 0))
part = cell.add_part(ROOT / "data" / job.step, np.eye(4), name="Detal", parent=part_frame)
fixture = cell.add_part(write_fixture_step(job), np.linalg.inv(job.T_table_part),
                        name="Tarcza i przyrząd", parent=part_frame)
torch_step = write_torch_step(TORCH_STEP, DEFAULT_TORCH.flange_to_tcp())
tool = cell.set_tool(DEFAULT_TORCH.flange_to_tcp(), name="Palnik MIG", geometry_step=torch_step)
try:
    part.setColor([0.75, 0.80, 0.90, 1.0])
    fixture.setColor([0.35, 0.35, 0.40, 1.0])
    tool.setColor([0.85, 0.55, 0.20, 1.0])
except Exception as e:  # noqa: BLE001
    print(f"(kolory pominięte: {e})")
bead_chunks = write_bead_chunks(samples, job.bead_radius)


def show(q: np.ndarray, tilt: float, rot: float) -> None:
    part_frame.setPose(to_mat(positioner.T_base_part(tilt, rot)))
    cell.set_joints(q)


def play(segment, steps_per_degree: float = 1.0) -> None:
    pts = list(zip(segment.robot, segment.positioner))
    if segment.kind == "weld":
        chunks = iter(bead_chunks)
        for i, (q, p) in enumerate(pts):
            show(q, *p)
            if i % CHUNK == CHUNK - 1 or i == len(pts) - 1:
                chunk = next(chunks, None)
                if chunk is not None:    # ścieg w układzie detalu – obraca się razem z nim
                    bead = cell.rdk.AddFile(str(chunk), part_frame)
                    try:
                        bead.setColor([0.80, 0.45, 0.15, 1.0])
                    except Exception:  # noqa: BLE001, S110
                        pass
        return
    for (qa, pa), (qb, pb) in pairwise(pts):
        span = max(np.degrees(np.max(np.abs(qb - qa))), np.max(np.abs(pb - pa)))
        for t in np.linspace(0.0, 1.0, max(2, int(span * steps_per_degree))):
            show(qa + (qb - qa) * t, *(pa + (pb - pa) * t))


show(HOME, 0.0, 0.0)
input("\nUstaw widok w RoboDK i naciśnij Enter, żeby zacząć... ")
for seg in program:
    print(f"  > {seg.label}")
    play(seg)
print("Koniec.")