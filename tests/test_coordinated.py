import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import fcl
import numpy as np
import pytest

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.coordinated import (
    _combined_motion_ok,
    build_cell_program,
    plan_coordinated,
    positioner_track,
    sample_seam,
)
from weldpath.orientation import homogeneous
from weldpath.positioner import Positioner
from weldpath.robot import ABB_IRB2400, ABB_IRB2400_CONVENTION, forward_kinematics
from weldpath.weld_job import DEFAULT_TORCH

ROOT = Path(__file__).resolve().parent.parent
PIPES = ROOT / "data" / "pipes.step"
POSITIONER = Positioner(tilt_axis_origin=np.array([1250.0, 0.0, 700.0]))
FACEPLATE = (fcl.Box(400.0, 400.0, 30.0), homogeneous(np.eye(3), np.array([0, 0, -15.0])))
R_ROOT = 168.3 / 2 - 10.0 + 2.0   # promień dna rowka: wewnętrzny + próg


@pytest.fixture(scope="module")
def bodies():
    if not PIPES.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_pipes.py")], check=True)
    return solids(load_step(str(PIPES)))


@pytest.fixture(scope="module")
def seam(bodies):
    return find_weld_seams(bodies)[0].seam


@pytest.fixture(scope="module")
def plan(bodies, seam):
    checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), POSITIONER.T_base_part(0, 0),
                         attachments=(FACEPLATE,))
    return plan_coordinated(ABB_IRB2400, seam, POSITIONER, checker), checker


# --- wykrycie spoiny obwodowej --------------------------------------------

def test_one_circular_groove_seam(bodies):
    welds = find_weld_seams(bodies)
    assert len(welds) == 1
    seam = welds[0].seam
    assert set(welds[0].bodies) == {0, 1}
    assert not seam.is_straight
    assert seam.length == pytest.approx(2 * np.pi * R_ROOT, rel=1e-3)
    assert seam.opening_angle == pytest.approx(60.0, abs=0.1)   # rowek V 60°


def test_samples_follow_the_circle(seam):
    samples = sample_seam(seam, step=10.0, overlap=15.0)
    for s in samples:
        assert np.hypot(s.point[0], s.point[1]) == pytest.approx(R_ROOT)   # na okręgu dna
        assert s.point[2] == pytest.approx(250.0)                          # w płaszczyźnie złącza
        radial = np.array([s.point[0], s.point[1], 0.0]) / R_ROOT
        assert s.opening == pytest.approx(radial, abs=1e-9)    # rowek otwiera się na zewnątrz
        assert abs(s.tangent @ radial) < 1e-9                   # styczna prostopadła do promienia
    # zakładka: ostatni punkt 15 mm za startem, czyli znowu blisko początku
    assert len(samples) == int(np.ceil((seam.length + 15.0) / 10.0)) + 1


# --- ruch skoordynowany ---------------------------------------------------

def test_every_point_is_welded_flat(seam):
    samples = sample_seam(seam, 10.0, 15.0)
    track = positioner_track(POSITIONER, samples)
    assert np.allclose(track[:, 0], -90.0)                      # rura leży poziomo
    for s, (tilt, rot) in zip(samples, track):
        up = POSITIONER.T_base_part(tilt, rot)[:3, :3] @ s.opening
        assert up == pytest.approx([0, 0, 1], abs=1e-6)          # pozycja PA w każdym punkcie
    assert np.all(np.abs(np.diff(track[:, 1])) < 10.0)           # obrót płynny, bez przeskoków


def test_positioner_turns_full_circle_plus_overlap(plan):
    p, _ = plan
    overlap_deg = np.degrees(15.0 / R_ROOT)
    assert p.positioner_rotation == pytest.approx(360.0 + overlap_deg, abs=1.0)


def test_robot_holds_torch_still(plan):
    p, _ = plan
    assert p.robot_motion < 2.0          # cały obrót robi pozycjoner, robot prawie stoi
    for weld_pass in p.passes:
        tcp = np.array([(forward_kinematics(ABB_IRB2400, q) @ DEFAULT_TORCH.flange_to_tcp())[:3, 3]
                        for q in weld_pass.joints])
        assert np.ptp(tcp, axis=0).max() < 0.01   # TCP nieruchomo w przestrzeni


def test_passes_stack_up_in_the_groove(plan):
    p, _ = plan
    assert [w.name for w in p.passes] == ["grań", "wypełnienie", "lico"]
    heights = [(forward_kinematics(ABB_IRB2400, w.joints[0]) @ DEFAULT_TORCH.flange_to_tcp())[2, 3]
               for w in p.passes]
    assert np.diff(heights) == pytest.approx([4.0, 4.0], abs=1e-6)   # co 4 mm w górę rowka


def test_coordinated_plan_is_collision_free(plan):
    p, checker = plan
    for weld_pass in p.passes:
        for q, (tilt, rot) in zip(weld_pass.joints, weld_pass.positioner):
            checker.set_part_pose(POSITIONER.T_base_part(tilt, rot))
            assert not checker.in_collision(q)


# --- pełny program stanowiska ---------------------------------------------

PARK = ABB_IRB2400_CONVENTION.to_opw([0, -50, 40, 0, 50, 0])


def test_simultaneous_approach_would_collide(plan):
    # Dlaczego potrzebna jest kolejność ruchów: jazda robota RAZEM z kładzeniem rury
    # z pozycji [0, 0, 0, 0, 30, 0] wbija palnik w rurę.
    p, checker = plan
    old_home = ABB_IRB2400_CONVENTION.to_opw([0, 0, 0, 0, 30, 0])
    first = p.passes[0]
    assert not _combined_motion_ok(checker, POSITIONER, old_home, np.zeros(2),
                                   first.joints[0], first.positioner[0])


def test_cell_program_order_and_safety(plan):
    p, checker = plan
    program = build_cell_program(ABB_IRB2400, p, POSITIONER, checker, PARK)
    kinds = [s.kind for s in program]
    assert kinds[0] == "positioner" and kinds[-1] == "positioner"   # rura kładzie się / wstaje
    assert kinds.count("weld") == 3                                  # trzy ściegi
    assert program[0].positioner[0] == pytest.approx([0, 0])         # start: rura pionowo
    assert program[-1].positioner[-1][0] == pytest.approx(0.0)       # koniec: znowu pionowo
    for seg in (program[0], program[-1]):                            # robot czeka zaparkowany
        assert np.allclose(seg.robot[0], PARK) and np.allclose(seg.robot[-1], PARK)
    # Niezależne, gęste sprawdzenie każdego odcinka (krok 0.5° na osi).
    for seg in program:
        pts = list(zip(seg.robot, seg.positioner))
        for (qa, pa), (qb, pb) in pairwise(pts):
            assert _combined_motion_ok(checker, POSITIONER, qa, np.asarray(pa), qb,
                                       np.asarray(pb), resolution=np.radians(0.5)), seg.label