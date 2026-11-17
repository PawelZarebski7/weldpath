"""Spoiny z wielu krawędzi na pozycjonerze: profil zamknięty dookoła i siodło rury na rurze."""
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weldpath.coordinated import build_cell_program, chain_length, chain_seams, is_closed
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION
from weldpath.seams import Seam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
from positioner_jobs import JOBS, plan_job

HOME = ABB_IRB2400_CONVENTION.to_opw([0, -50, 40, 0, 50, 0])
MAKE = {"rhs": "make_rhs.py", "tee": "make_tee.py"}


@pytest.fixture(scope="module", params=["rhs", "tee"])
def job_plan(request):
    name = request.param
    if not (ROOT / "data" / JOBS[name].step).exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / MAKE[name])], check=True)
    job, positioner, checker, samples, plan = plan_job(name, log=lambda *_: None)
    return SimpleNamespace(name=name, job=job, positioner=positioner, checker=checker,
                           samples=samples, plan=plan)


def _line(a, b) -> Seam:
    t = (b - a) / np.linalg.norm(b - a)
    up = np.array([0.0, 0.0, 1.0])
    return Seam(edge=None, faces=(None, None), start=a, end=b, tangent=t,
                normals=(up, np.cross(up, t)), opening_angle=90.0)


def test_chain_of_four_lines_closes():
    c = [np.array(p, float) for p in ((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0))]
    # krawędzie w przypadkowej kolejności i kierunku (jak z B-Rep) – łańcuch i tak się składa
    seams = [_line(c[1], c[0]), _line(c[2], c[3]), _line(c[1], c[2]), _line(c[3], c[0])]
    chains = chain_seams(seams)
    assert len(chains) == 1
    assert len(chains[0]) == 4
    assert chain_length(chains[0]) == pytest.approx(40.0)
    assert is_closed(chains[0])


def test_open_pieces_stay_separate():
    a, b, c, d = (np.array(p, float) for p in ((0, 0, 0), (10, 0, 0), (20, 5, 0), (30, 5, 0)))
    chains = chain_seams([_line(a, b), _line(c, d)])
    assert len(chains) == 2
    assert not any(is_closed(ch) for ch in chains)


def test_outer_chain_is_picked(job_plan):
    j = job_plan
    length = sum(np.linalg.norm(b.point - a.point) for a, b in pairwise(j.samples))
    if j.name == "rhs":
        # 4 proste po 60 mm + 4 ćwiartki łuku R10 → 240 + 62.8 = 302.8 mm (+ zakładka)
        assert 300.0 < length < 302.8 + j.job.overlap + 1.0
    else:
        # siodło Ø42.4 na Ø60.3 jest dłuższe niż obwód rury odgałęzienia (133.2 mm)
        assert np.pi * 42.4 < length < np.pi * 42.4 * 1.2 + j.job.overlap


def test_every_point_is_flat_pa(job_plan):
    j = job_plan
    assert j.plan is not None
    for s, (tilt, rot) in zip(j.samples, j.plan.passes[0].positioner):
        opening = j.positioner.T_base_part(tilt, rot)[:3, :3] @ s.opening
        assert opening[2] == pytest.approx(1.0, abs=1e-3)   # dwusieczna pionowo w górę


def test_positioner_carries_the_motion(job_plan):
    j = job_plan
    track = j.plan.passes[0].positioner
    assert abs(track[-1, 1] - track[0, 1]) > 360.0      # pełny obrót + zakładka
    if j.name == "rhs":
        assert np.ptp(track[:, 0]) < 1e-6                 # stałe uchylenie
        assert j.plan.robot_motion < 10.0                 # robot prawie stoi
    else:
        assert np.ptp(track[:, 0]) > 10.0                 # siodło: uchylenie zmienia się


def test_cell_program_is_collision_free(job_plan):
    j = job_plan
    program = build_cell_program(ABB_IRB2400, j.plan, j.positioner, j.checker, HOME)
    kinds = [seg.kind for seg in program]
    assert kinds[0] == "positioner" and kinds[-1] == "positioner"
    assert "weld" in kinds
    for q, p in zip(j.plan.passes[0].joints, j.plan.passes[0].positioner):
        j.checker.set_part_pose(j.positioner.T_base_part(*p))
        assert not j.checker.in_collision(q)