"""Demo w RoboDK: robot ABB IRB 2400 spawa wszystkie spoiny detalu leżącego na stole.

Domyślnie rama (data/frame.step); inny detal: podaj plik STEP jako argument, np.
    poetry run python examples/robodk_show_frame.py data/bracket.step

Wymaga otwartego RoboDK ze stacją, w której jest robot "ABB IRB 2400/16"
(np. data/weldcell.rdk). Planowanie liczy nasz kod (FCL), RoboDK tylko pokazuje.

Uruchom:  poetry run python examples/robodk_show_frame.py
"""
import sys
import time
from pathlib import Path

import fcl
import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker, mesh_from_shape
from weldpath.orientation import homogeneous
from weldpath.planner import RRTConnect, retract, shortcut
from weldpath.robodk_bridge import RoboDKCell, write_torch_step
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION
from weldpath.weld_job import DEFAULT_TORCH, IRB2400_LIMITS, plan_weld

ROOT = Path(__file__).resolve().parent.parent
STEP = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else ROOT / "data" / "frame.step"
# Wymiary detalu z jego siatki: środek w XY i spód – do ustawienia stołu i detalu.
_verts, _ = mesh_from_shape(load_step(str(STEP)))
PART_MIN, PART_MAX = _verts.min(axis=0), _verts.max(axis=0)
PART_CENTER = (PART_MIN + PART_MAX) / 2
# Środek detalu ląduje tam, gdzie środek ramy w plan_frame.py (dla ramy: bez zmian).
T_BASE_PART = homogeneous(np.eye(3), np.array([950.0 - PART_CENTER[0], -50.0 - PART_CENTER[1],
                                               300.0 - PART_MIN[2]]))
RETRACT_MM = 60.0
HOME = ABB_IRB2400_CONVENTION.to_opw([0, 0, 0, 0, 30, 0])  # ABB: przedramię poziomo, nadgarstek w dół

# Stół pod detalem (w układzie detalu): blat 900 x 700 x 50 mm, środek pod środkiem detalu.
TABLE_SIZE = np.array([900.0, 700.0, 50.0])
TABLE_CENTER = np.array([PART_CENTER[0], PART_CENTER[1], PART_MIN[2] - 25.0])
TABLE_STEP = ROOT / "data" / "table.step"
TORCH_STEP = ROOT / "data" / "torch.step"


def write_table_step() -> None:
    """Zapisuje blat jako STEP, żeby RoboDK mogło go wczytać jak każdy inny model."""
    corner = TABLE_CENTER - TABLE_SIZE / 2
    box = BRepPrimAPI_MakeBox(gp_Pnt(*corner), *TABLE_SIZE).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(box, STEPControl_AsIs)
    assert writer.Write(str(TABLE_STEP)) == IFSelect_RetDone


# Ten sam blat w modelu kolizji – jako element związany z detalem.
TABLE_COLLISION = (fcl.Box(*TABLE_SIZE), homogeneous(np.eye(3), TABLE_CENTER))


def log(msg: str) -> None:
    print(msg, flush=True)


# --- 1. planowanie (nasz kod) ------------------------------------------------
t0 = time.perf_counter()
bodies = solids(load_step(str(STEP)))
seams = find_weld_seams(bodies)
weld_checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), T_BASE_PART,
                          attachments=(TABLE_COLLISION,))
transit_checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), T_BASE_PART,
                             attachments=(TABLE_COLLISION,), margin=5.0)

log(f"Planowanie {len(seams)} spoin (to potrwa około minuty)...")
welds = []  # lista (nazwa, SeamPlan)
for i, ws in enumerate(seams):
    plans = plan_weld(ABB_IRB2400, ws.seam, T_BASE_PART, weld_checker)
    if not plans:
        log(f"  S{i:02d}: niewykonalna – pomijam")
        continue
    for k, p in enumerate(plans):
        welds.append((f"S{i:02d}" + ("ab"[k] if len(plans) > 1 else ""), p))
    log(f"  S{i:02d}: OK ({len(plans)} odcinek/odcinki)")

# --- 2. jedna długa ścieżka: dom -> spoiny z przejazdami -> dom -----------------
path: list[np.ndarray] = [HOME]
segments: list[tuple[str, int, int]] = []  # (opis, indeks początku, indeks końca) w path


def travel_to(q_goal: np.ndarray, label: str) -> None:
    """Bezkolizyjny przejazd z ostatniego punktu ścieżki do q_goal (RRT + wygładzanie)."""
    start = len(path) - 1
    result = RRTConnect(transit_checker, IRB2400_LIMITS, seed=len(path)).plan(path[-1], q_goal)
    if result.success:
        path.extend(shortcut(result.path, transit_checker)[1:])
    else:
        log(f"  UWAGA: brak bezkolizyjnego przejazdu ({label}) – przeskok")
        path.append(q_goal)
    segments.append((f"przejazd -> {label}", start, len(path) - 1))


for name, plan in welds:
    q_first, q_last = plan.joints[0], plan.joints[-1]
    q_approach = retract(ABB_IRB2400, q_first, plan.config, DEFAULT_TORCH, RETRACT_MM)
    q_leave = retract(ABB_IRB2400, q_last, plan.config, DEFAULT_TORCH, RETRACT_MM)
    if q_approach is None or transit_checker.in_collision(q_approach):
        q_approach = q_first  # nie da się odsunąć – podjazd bezpośrednio
    travel_to(q_approach, name)
    start = len(path) - 1
    path.append(q_first)                       # dojazd liniowy (MoveL)
    path.extend(plan.joints[1:])               # spawanie (MoveL wzdłuż spoiny)
    if q_leave is not None:
        path.append(q_leave)                   # odjazd liniowy (MoveL)
    segments.append((f"spawanie {name}", start, len(path) - 1))

travel_to(HOME, "dom")
log(f"Gotowe: {len(welds)} odcinków spoin, {len(path)} punktów ścieżki, "
    f"{time.perf_counter() - t0:.0f} s")

# --- 3. RoboDK: detal, palnik, animacja ----------------------------------------
cell = RoboDKCell()
cell.set_base_pose(np.eye(4))  # robot z powrotem na miejscu (po demo z torem)
cell.remove("Pozycjoner", "Rury", "Tarcza pozycjonera", "Hala", "Wózek toru",
            "Rama", "bracket")  # sprzątamy po demo rur
write_table_step()
table = cell.add_part(TABLE_STEP, T_BASE_PART, name="Stół")
frame = cell.add_part(STEP, T_BASE_PART, name="Rama" if STEP.stem == "frame" else STEP.stem)
try:  # kolory tylko dla czytelności – gdyby wersja RoboDK ich nie obsługiwała, pomijamy
    table.setColor([0.35, 0.35, 0.40, 1.0])
    frame.setColor([0.75, 0.80, 0.90, 1.0])
except Exception as e:  # noqa: BLE001
    log(f"(kolory pominięte: {e})")
torch_step = write_torch_step(TORCH_STEP, DEFAULT_TORCH.flange_to_tcp())
tool = cell.set_tool(DEFAULT_TORCH.flange_to_tcp(), name="Palnik MIG", geometry_step=torch_step)
try:
    tool.setColor([0.85, 0.55, 0.20, 1.0])  # miedziany palnik
except Exception as e:  # noqa: BLE001
    log(f"(kolor palnika pominięty: {e})")
cell.set_joints(HOME)

input("\nW RoboDK ustaw widok (np. przycisk 'Fit all' albo F2) i naciśnij Enter, "
      "żeby zacząć animację... ")
for label, a, b in segments:
    log(f"  {label}")
    cell.play(path[a:b + 1], steps_per_segment=4 if label.startswith("spawanie") else 12)
log("Koniec animacji.")
if "--loop" in sys.argv:
    while True:
        cell.play(path, steps_per_segment=6)