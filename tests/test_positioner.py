import subprocess
import sys
from pathlib import Path

import fcl
import numpy as np
import pytest

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.positioner import (
    Positioner,
    candidate_poses,
    gravity_angle,
    plan_weld_with_positioner,
    seam_opening,
    welding_position,
)
from weldpath.robot import ABB_IRB2400

ROOT = Path(__file__).resolve().parent.parent
FRAME = ROOT / "data" / "frame.step"
T_TABLE_PART = homogeneous(np.eye(3), np.array([-150.0, -100.0, 0.0]))
POSITIONER = Positioner(tilt_axis_origin=np.array([1000.0, 0.0, 500.0]),
                        T_table_part=T_TABLE_PART)


@pytest.fixture(scope="module")
def bodies():
    if not FRAME.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_frame.py")], check=True)
    return solids(load_step(str(FRAME)))


@pytest.fixture(scope="module")
def weld_seams(bodies):
    return [ws.seam for ws in find_weld_seams(bodies)]


# --- kinematyka pozycjonera -----------------------------------------------

def test_zero_pose_lays_part_flat():
    T = POSITIONER.T_base_part(0.0, 0.0)
    assert np.allclose(T[:3, :3], np.eye(3))
    # środek ramy (150, 100) leży na osi obrotu, tarcza 150 mm nad osią uchylania
    center = T @ np.array([150.0, 100.0, 0.0, 1.0])
    assert center[:3] == pytest.approx([1000.0, 0.0, 650.0])


def test_rotation_keeps_table_center_fixed():
    for rot in (-90.0, 45.0, 180.0):
        T = POSITIONER.T_base_part(30.0, rot)
        center = T @ np.array([150.0, 100.0, 0.0, 1.0])
        assert center[:3] == pytest.approx(POSITIONER.T_base_table(30.0, 0.0)[:3, 3])


def test_pointing_up_works_for_any_direction():
    rng = np.random.default_rng(0)
    for d in rng.normal(size=(300, 3)):
        solutions = POSITIONER.angles_pointing_up(d)
        # Uchylenie ±135° nie postawi pionowo kierunku celującego prawie w dół
        # (bliżej niż 45° od pionu w dół) – wtedy rozwiązań ma nie być.
        tilt_needed = np.degrees(np.arccos(d[2] / np.linalg.norm(d)))
        assert bool(solutions) == (tilt_needed <= 135.0)
        for tilt, rot in solutions:
            up = POSITIONER.T_base_part(tilt, rot)[:3, :3] @ (d / np.linalg.norm(d))
            assert up == pytest.approx([0, 0, 1], abs=1e-6)


# --- pozycje spawania -----------------------------------------------------

def test_flat_frame_has_pb_and_pf_seams(weld_seams):
    flat = POSITIONER.T_base_part(0.0, 0.0)
    positions = sorted(welding_position(s, flat) for s in weld_seams)
    assert positions.count("PB") == 8   # poziome spoiny pachwinowe na płasko = naboczne
    assert positions.count("PF") == 4   # pionowe narożniki (krawędź skierowana w górę)


def test_best_candidate_is_flat_position(weld_seams):
    for seam in weld_seams:
        _cost, tilt, rot = candidate_poses(POSITIONER, seam)[0]
        T = POSITIONER.T_base_part(tilt, rot)
        assert gravity_angle(seam, T) == pytest.approx(0.0, abs=1e-6)
        assert welding_position(seam, T) == "PA"


def test_opening_is_unit_bisector(weld_seams):
    for seam in weld_seams:
        v = seam_opening(seam)
        assert np.linalg.norm(v) == pytest.approx(1.0)
        assert v @ seam.normals[0] == pytest.approx(v @ seam.normals[1])  # dwusieczna


# --- planowanie (wolniejszy test integracyjny) ----------------------------

def test_vertical_corner_seam_becomes_flat(bodies, weld_seams):
    faceplate = (fcl.Box(500.0, 500.0, 30.0),
                 np.linalg.inv(T_TABLE_PART) @ homogeneous(np.eye(3), np.array([0, 0, -15.0])))
    checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), POSITIONER.T_base_part(0, 0),
                         attachments=(faceplate,))
    flat = POSITIONER.T_base_part(0.0, 0.0)
    vertical = next(s for s in weld_seams if welding_position(s, flat) in ("PF", "PG"))

    result = plan_weld_with_positioner(ABB_IRB2400, vertical, POSITIONER, checker)
    assert result is not None
    assert result.position == "PA"
    assert abs(result.tilt) == pytest.approx(90.0)  # pionową spoinę trzeba położyć
    for plan in result.plans:
        for q in plan.joints:
            assert not checker.in_collision(q)