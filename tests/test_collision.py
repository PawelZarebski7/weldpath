from pathlib import Path

import numpy as np
import pytest

from weldpath.cad import load_step
from weldpath.collision import FCLChecker, mesh_from_shape
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400, joint_positions
from weldpath.seams import find_seams
from weldpath.weld_job import plan_seam

DATA = Path(__file__).resolve().parent.parent / "data"
T_BASE_PART = homogeneous(np.eye(3), np.array([900.0, -100.0, 300.0]))
HOME = np.radians([0, 0, 90, 0, 30, 0])  # ramię w górę, przedramię poziomo


@pytest.fixture(scope="module")
def part():
    return load_step(str(DATA / "tjoint.step"))


@pytest.fixture(scope="module")
def checker(part):
    return FCLChecker(ABB_IRB2400, part, T_BASE_PART, table_top_z=300.0)


# --- geometria ------------------------------------------------------------

def test_mesh_covers_the_part(part):
    verts, tris = mesh_from_shape(part)
    assert verts.min(axis=0) == pytest.approx([0, 0, 0])
    assert verts.max(axis=0) == pytest.approx([200, 100, 90])
    assert tris.min() >= 0 and tris.max() < len(verts)  # poprawne indeksy


def test_joint_positions_match_link_lengths():
    rng = np.random.default_rng(0)
    k = np.hypot(ABB_IRB2400.a2, ABB_IRB2400.c3)
    for q in rng.uniform(-2, 2, size=(100, 6)):
        _, shoulder, elbow, wrist, flange = joint_positions(ABB_IRB2400, q)
        assert np.linalg.norm(elbow - shoulder) == pytest.approx(ABB_IRB2400.c2)
        assert np.linalg.norm(wrist - elbow) == pytest.approx(k)
        assert np.linalg.norm(flange - wrist) == pytest.approx(ABB_IRB2400.c4)


# --- kolizje --------------------------------------------------------------

def test_home_pose_is_free(checker):
    assert not checker.in_collision(HOME)
    assert checker.clearance(HOME) > 100


def test_arm_pushed_into_table_collides(checker):
    q = np.radians([-6, 80, 120, 0, 60, 0])  # ramię mocno w przód i w dół
    assert checker.in_collision(q)


def test_welding_path_is_collision_free(checker, part):
    # Podczas spawania palnik jest blisko blach, ale nie może w nie wchodzić.
    for seam in find_seams(part):
        plan = plan_seam(ABB_IRB2400, seam, T_BASE_PART)
        for q in plan.joints:
            assert not checker.in_collision(q)


def test_margin_makes_checker_stricter(part):
    # Ten sam plan spawania z marginesem 20 mm musi już kolidować (luz to ~3 mm).
    strict = FCLChecker(ABB_IRB2400, part, T_BASE_PART, margin=20.0)
    seam = find_seams(part)[0]
    plan = plan_seam(ABB_IRB2400, seam, T_BASE_PART)
    assert any(strict.in_collision(q) for q in plan.joints)


def test_check_counter(part):
    c = FCLChecker(ABB_IRB2400, part, T_BASE_PART)
    for _ in range(5):
        c.in_collision(HOME)
    assert c.checks == 5