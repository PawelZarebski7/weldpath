"""Demo w RoboDK: spawanie wielościegowe rur na obracającym się pozycjonerze.

W stacji nie ma modelu pozycjonera, więc go SYMULUJEMY: tworzymy układ odniesienia
"Pozycjoner", podpinamy pod niego rury, tarczę z uchwytem i kawałki ściegów,
a w każdym kroku ustawiamy ten układ tak, jak obróciłby go pozycjoner
(te same macierze co w planowaniu). Rosnący ścieg i szczęki uchwytu sprawiają,
że obrót rury jest widoczny – sam walec wyglądałby na nieruchomy.

Wymaga otwartego RoboDK z robotem "ABB IRB 2400/16" (np. data/weldcell.rdk).
Uruchom:  poetry run python examples/robodk_show_pipe.py
"""
from itertools import pairwise
from pathlib import Path

import fcl
import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.coordinated import build_cell_program, plan_coordinated, sample_seam
from weldpath.orientation import homogeneous
from weldpath.positioner import Positioner
from weldpath.robodk_bridge import RoboDKCell, to_mat, write_torch_step
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION
from weldpath.weld_job import DEFAULT_TORCH

ROOT = Path(__file__).resolve().parent.parent
PIPES = ROOT / "data" / "pipes.step"
FACEPLATE_STEP = ROOT / "data" / "faceplate.step"
BEAD_DIR = ROOT / "data" / "bead"           # kawałki ściegów do animacji
BEAD_RADIUS = {"grań": 2.5, "wypełnienie": 3.0, "lico": 4.0}
BEAD_COLOR = {"grań": [0.55, 0.25, 0.10, 1.0], "wypełnienie": [0.80, 0.45, 0.15, 1.0],
              "lico": [0.95, 0.85, 0.60, 1.0]}
BEAD_FILE = {"grań": "root", "wypełnienie": "fill", "lico": "cap"}   # nazwy plików bez polskich znaków
CHUNK = 5                                   # ile punktów spoiny na jeden kawałek ściegu
TORCH_STEP = ROOT / "data" / "torch.step"
POSITIONER = Positioner(tilt_axis_origin=np.array([1250.0, 0.0, 700.0]))
FACEPLATE = (fcl.Box(400.0, 400.0, 30.0), homogeneous(np.eye(3), np.array([0, 0, -15.0])))
# Pozycja parkowania (osie ABB): robot odchylony do tyłu, z dala od łuku, który zakreśla
# koniec rury przy uchylaniu pozycjonera. Z pozycji [0, 0, 0, 0, 30, 0] palnik wchodził w rurę.
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


def write_faceplate_step() -> None:
    """Tarcza pozycjonera Ø400 x 30 mm + uchwyt 3-szczękowy (układ detalu).

    Szczęki obracają się razem z rurą – bez nich obrotu walca nie widać.
    """
    shapes = [BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -30), gp_Dir(0, 0, 1)), 200, 30).Shape()]
    for angle in (0.0, 120.0, 240.0):
        a = np.radians(angle)
        radial, side = np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0])
        corner = radial * 85.0 - side * 15.0           # szczęka przylega do rury (r = 84.15)
        jaw = BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(*corner), gp_Dir(0, 0, 1), gp_Dir(*radial)),
                                  50.0, 30.0, 40.0).Shape()
        shapes.append(jaw)
    write_step(shapes, FACEPLATE_STEP)


def write_bead_chunks(samples, offset: float, radius: float, name: str) -> list[Path]:
    """Ścieg jako łańcuch cienkich walców wzdłuż spoiny, pocięty na kawałki do animacji."""
    BEAD_DIR.mkdir(exist_ok=True)
    # Ścieg leży w rowku: od jego dna przesunięty o offset + promień wzdłuż dwusiecznej.
    pts = [s.point + (offset + radius) * s.opening for s in samples]
    paths = []
    for c, i in enumerate(range(0, len(pts) - 1, CHUNK)):
        pieces = []
        for a, b in zip(pts[i:i + CHUNK], pts[i + 1:i + CHUNK + 1]):
            axis = b - a
            pieces.append(BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*a), gp_Dir(*axis)), radius, float(np.linalg.norm(axis))).Shape())
        paths.append(write_step(pieces, BEAD_DIR / f"{BEAD_FILE[name]}_{c:02d}.step"))
    return paths


# --- planowanie ---------------------------------------------------------------
bodies = solids(load_step(str(PIPES)))
seam = find_weld_seams(bodies)[0].seam
checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), POSITIONER.T_base_part(0, 0),
                     attachments=(FACEPLATE,))
plan = plan_coordinated(ABB_IRB2400, seam, POSITIONER, checker)
if plan is None:
    raise SystemExit("Nie znaleziono planu")
print(f"Plan gotowy: {len(plan.passes)} ściegi, robot rusza się o max {plan.robot_motion:.2f}°, "
      f"pozycjoner obraca rurę o {plan.positioner_rotation:.0f}° na ścieg")

# --- RoboDK -------------------------------------------------------------------
cell = RoboDKCell()
cell.remove("Rama", "Stół", "Rury", "Tarcza pozycjonera")  # sprzątamy po poprzednich demo
write_faceplate_step()
table_frame = cell.add_frame("Pozycjoner", POSITIONER.T_base_part(0, 0))
pipes = cell.add_part(PIPES, np.eye(4), name="Rury", parent=table_frame)
plate = cell.add_part(FACEPLATE_STEP, np.eye(4), name="Tarcza i uchwyt", parent=table_frame)
torch_step = write_torch_step(TORCH_STEP, DEFAULT_TORCH.flange_to_tcp())
tool = cell.set_tool(DEFAULT_TORCH.flange_to_tcp(), name="Palnik MIG", geometry_step=torch_step)
try:
    pipes.setColor([0.75, 0.80, 0.90, 1.0])
    plate.setColor([0.35, 0.35, 0.40, 1.0])
    tool.setColor([0.85, 0.55, 0.20, 1.0])
except Exception as e:  # noqa: BLE001
    print(f"(kolory pominięte: {e})")


def show(q: np.ndarray, tilt: float, rot: float) -> None:
    """Jeden krok animacji: pozycjoner (obracamy cały układ z rurami) + robot."""
    table_frame.setPose(to_mat(POSITIONER.T_base_part(tilt, rot)))
    cell.set_joints(q)


def play(segment, steps_per_degree: float = 1.0) -> None:
    """Odtwarza odcinek programu: liniowo między kolejnymi punktami robota i pozycjonera."""
    pts = list(zip(segment.robot, segment.positioner))
    if segment.kind == "weld":           # punkty spoiny są gęste – pokazujemy je wprost
        chunks = iter(bead_chunks[segment.weld_pass])
        for i, (q, p) in enumerate(pts):
            show(q, *p)
            if i % CHUNK == CHUNK - 1 or i == len(pts) - 1:
                chunk = next(chunks, None)
                if chunk is not None:    # kawałek ściegu w układzie pozycjonera – obraca się z rurą
                    bead = cell.rdk.AddFile(str(chunk), table_frame)
                    try:
                        bead.setColor(BEAD_COLOR[segment.weld_pass])
                    except Exception:  # noqa: BLE001, S110
                        pass
        return
    for (qa, pa), (qb, pb) in pairwise(pts):
        span = max(np.degrees(np.max(np.abs(qb - qa))), np.max(np.abs(pb - pa)))
        for t in np.linspace(0.0, 1.0, max(2, int(span * steps_per_degree))):
            show(qa + (qb - qa) * t, *(pa + (pb - pa) * t))


# Ściegi do animacji: te same punkty spoiny co w planie (co 10 mm, z zakładką 15 mm).
samples = sample_seam(seam, 10.0, 15.0)
bead_chunks = {p.name: write_bead_chunks(samples, p.offset, BEAD_RADIUS[p.name], p.name)
               for p in plan.passes}

# Pełny program: kolejność ruchów robota i pozycjonera, każdy odcinek sprawdzony na kolizje.
program = build_cell_program(ABB_IRB2400, plan, POSITIONER, checker, HOME)
print("Program stanowiska (każdy odcinek bez kolizji):")
for seg in program:
    print(f"  {seg.label}")

show(HOME, 0.0, 0.0)
input("\nUstaw widok w RoboDK i naciśnij Enter, żeby zacząć... ")
for seg in program:
    print(f"  > {seg.label}")
    play(seg)
print("Koniec.")