"""Detale spawane na pozycjonerze ruchem skoordynowanym: definicje i planowanie.

    rhs   – węzeł ramy przyczepy: profil 80x80x5 na płycie, spoina dookoła (4 proste + 4 łuki)
    tee   – stelaż z rurek: Ø42.4 na Ø60.3, spoina siodłowa (krzywa 3D)
    pipes – dwie rury Ø168.3 doczołowo, rowek V, trzy ściegi

Uruchom planowanie:  poetry run python examples/positioner_jobs.py rhs
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fcl
import numpy as np

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.coordinated import (
    DEFAULT_PASSES,
    chain_length,
    chain_seams,
    outer_chain,
    plan_coordinated_samples,
    sample_chain,
)
from weldpath.orientation import homogeneous
from weldpath.positioner import Positioner
from weldpath.robot import ABB_IRB2400

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Job:
    title: str
    step: str                       # plik STEP w data/
    T_table_part: np.ndarray        # jak detal leży na tarczy (oś obrotu = oś Z tarczy)
    fixtures: tuple                 # przyrząd: (lewy-dolny-róg, wymiar) prostopadłościanów, układ tarczy
    passes: tuple = (("ścieg", 0.0),)
    step_mm: float = 5.0            # gęstość punktów spoiny
    overlap: float = 15.0           # zakładka na zamknięciu spoiny
    bead_radius: float = 2.5


JOBS = {
    "rhs": Job(
        title="Węzeł ramy: profil 80x80x5 na płycie, spoina dookoła",
        step="rhs.step",
        T_table_part=homogeneous(np.eye(3), np.array([-150.0, -150.0, 0.0])),
        # cztery dociski na narożach płyty 300 x 300 (płyta: x,y ∈ [-150, 150] na tarczy)
        fixtures=tuple(((x, y, 15.0), (40.0, 40.0, 25.0))
                       for x in (-150.0, 110.0) for y in (-150.0, 110.0)),
    ),
    "tee": Job(
        title="Stelaż z rurek: Ø42.4 na Ø60.3, spoina siodłowa",
        step="tee.step",
        T_table_part=homogeneous(np.eye(3), np.array([0.0, 0.0, 30.15])),
        # dwie podpory (pryzmy) pod rurą główną, poza strefą spoiny
        fixtures=tuple(((x, -40.0, 0.0), (30.0, 80.0, 8.0)) for x in (-130.0, 100.0)),
    ),
    "pipes": Job(
        title="Rury Ø168.3 x 10 doczołowo, rowek V, trzy ściegi",
        step="pipes.step",
        T_table_part=np.eye(4),
        fixtures=(),
        passes=DEFAULT_PASSES,
        step_mm=10.0,
        bead_radius=3.0,
    ),
}

POSITIONER_ORIGIN = np.array([1250.0, 0.0, 700.0])
FACEPLATE_SIZE = (400.0, 400.0, 30.0)


def fixture_attachments(job: Job) -> tuple:
    """Tarcza i przyrząd jako przeszkody związane z detalem (obracają się razem z nim)."""
    T_part_table = np.linalg.inv(job.T_table_part)
    items = [(fcl.Box(*FACEPLATE_SIZE),
              T_part_table @ homogeneous(np.eye(3), np.array([0, 0, -FACEPLATE_SIZE[2] / 2])))]
    for corner, size in job.fixtures:
        center = np.array(corner) + np.array(size) / 2
        items.append((fcl.Box(*size), T_part_table @ homogeneous(np.eye(3), center)))
    return tuple(items)


def plan_job(name: str, log=print):
    """Zwraca (job, pozycjoner, checker, punkty spoiny, plan)."""
    job = JOBS[name]
    bodies = solids(load_step(str(ROOT / "data" / job.step)))
    seams = [w.seam for w in find_weld_seams(bodies, min_length=0)]
    chains = chain_seams(seams)
    chain = outer_chain(chains, job.T_table_part)
    samples = sample_chain(chain, job.step_mm, job.overlap)
    positioner = Positioner(tilt_axis_origin=POSITIONER_ORIGIN, T_table_part=job.T_table_part)
    checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), positioner.T_base_part(0, 0),
                         attachments=fixture_attachments(job))
    log(f"{job.title}")
    log(f"  krawędzie spoin: {len(seams)}, łańcuchy: {len(chains)} "
        f"(zewnętrzny: {len(chain)} krawędzi, {chain_length(chain):.1f} mm)")
    t0 = time.perf_counter()
    plan = plan_coordinated_samples(ABB_IRB2400, samples, chain[0][0], positioner, checker,
                                    passes=job.passes)
    log(f"  planowanie: {time.perf_counter() - t0:.1f} s, punktów spoiny: {len(samples)}")
    return job, positioner, checker, samples, plan


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "rhs"
    job, positioner, checker, samples, plan = plan_job(name)
    if plan is None:
        raise SystemExit("  NIEWYKONALNE")
    track = plan.passes[0].positioner
    print(f"  pozycjoner: uchylenie od {track[:, 0].min():+.0f}° do {track[:, 0].max():+.0f}°, "
          f"obrót o {abs(track[-1, 1] - track[0, 1]):.0f}°")
    print(f"  robot: największy ruch osi {plan.robot_motion:.1f}°, ściegi: "
          f"{', '.join(p.name for p in plan.passes)}")
    angles = [float(np.degrees(np.arccos(np.clip(s.normals[0] @ s.normals[1], -1, 1))))
              for s in samples]
    print(f"  kąt między ściankami wzdłuż spoiny: od {180 - max(angles):.0f}° "
          f"do {180 - min(angles):.0f}°")