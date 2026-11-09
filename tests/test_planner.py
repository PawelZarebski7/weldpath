from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.planner import RRTConnect, motion_valid, path_length, retract, shortcut
from weldpath.robot import ABB_IRB2400, forward_kinematics
from weldpath.seams import find_seams
from weldpath.weld_job import DEFAULT_TORCH, IRB2400_LIMITS, JointLimits, plan_seam

DATA = Path(__file__).resolve().parent.parent / "data"
T_BASE_PART = homogeneous(np.eye(3), np.array([900.0, -100.0, 300.0]))


# --- testy jednostkowe na sztucznej przestrzeni (bez robota i FCL) ----------

class BallChecker:
    """Przeszkoda = kula w przestrzeni przegubów. Szybka i przewidywalna."""

    def __init__(self, center, radius):
        self.center, self.radius = np.asarray(center, float), radius
        self.checks = 0

    def in_collision(self, q):
        self.checks += 1
        return np.linalg.norm(q - self.center) < self.radius


BOX = JointLimits(lower=np.full(6, -np.pi), upper=np.full(6, np.pi))
START = np.array([-2.0, 0, 0, 0, 0, 0])
GOAL = np.array([2.0, 0, 0, 0, 0, 0])


def assert_path_valid(path, checker, start, goal):
    assert np.allclose(path[0], start) and np.allclose(path[-1], goal)
    for a, b in pairwise(path):
        assert motion_valid(checker, a, b)


def test_free_space_gives_direct_path():
    checker = BallChecker(center=[0, 3, 0, 0, 0, 0], radius=0.5)  # przeszkoda z boku
    result = RRTConnect(checker, BOX).plan(START, GOAL)
    assert result.success and len(result.path) == 2 and result.iterations == 0


def test_plans_around_obstacle():
    checker = BallChecker(center=np.zeros(6), radius=1.0)  # kula dokładnie na prostej
    assert not motion_valid(checker, START, GOAL)
    result = RRTConnect(checker, BOX, seed=1).plan(START, GOAL)
    assert result.success
    assert_path_valid(result.path, checker, START, GOAL)


def test_start_in_collision_fails_fast():
    checker = BallChecker(center=START, radius=0.5)
    result = RRTConnect(checker, BOX).plan(START, GOAL)
    assert not result.success and result.iterations == 0


def test_same_seed_same_path():
    checker = BallChecker(center=np.zeros(6), radius=1.0)
    p1 = RRTConnect(checker, BOX, seed=7).plan(START, GOAL).path
    p2 = RRTConnect(checker, BOX, seed=7).plan(START, GOAL).path
    assert len(p1) == len(p2) and all(np.allclose(a, b) for a, b in zip(p1, p2))


def test_shortcut_keeps_path_valid_and_not_longer():
    checker = BallChecker(center=np.zeros(6), radius=1.0)
    path = RRTConnect(checker, BOX, seed=1).plan(START, GOAL).path
    smooth = shortcut(path, checker)
    assert_path_valid(smooth, checker, START, GOAL)
    assert len(smooth) <= len(path)
    assert path_length(smooth) <= path_length(path) + 1e-9


def test_motion_valid_detects_thin_obstacle_between_samples():
    # Cienka przeszkoda między dwoma wolnymi punktami: sprawdzenie samych końców by ją przeoczyło.
    checker = BallChecker(center=np.zeros(6), radius=0.05)
    a, b = np.array([-0.5, 0, 0, 0, 0, 0]), np.array([0.5, 0, 0, 0, 0, 0])
    assert not checker.in_collision(a) and not checker.in_collision(b)
    assert not motion_valid(checker, a, b)


# --- test integracyjny: przejazd nad żebrem prawdziwym robotem -------------

@pytest.fixture(scope="module")
def transit():
    part = load_step(str(DATA / "tjoint.step"))
    s0, s1 = find_seams(part)
    w0, w1 = plan_seam(ABB_IRB2400, s0, T_BASE_PART), plan_seam(ABB_IRB2400, s1, T_BASE_PART)
    checker = FCLChecker(ABB_IRB2400, part, T_BASE_PART, table_top_z=300.0, margin=5.0)
    qa = retract(ABB_IRB2400, w0.joints[-1], w0.config, DEFAULT_TORCH, 60.0)
    qb = retract(ABB_IRB2400, w1.joints[0], w1.config, DEFAULT_TORCH, 60.0)
    return checker, qa, qb


def test_retract_moves_tcp_along_torch_axis():
    q = np.radians([-20, 35, 110, 30, 70, 130])
    q_back = retract(ABB_IRB2400, q, config=0, torch=DEFAULT_TORCH, distance=60.0)
    T0 = forward_kinematics(ABB_IRB2400, q) @ DEFAULT_TORCH.flange_to_tcp()
    T1 = forward_kinematics(ABB_IRB2400, q_back) @ DEFAULT_TORCH.flange_to_tcp()
    assert np.allclose(T0[:3, :3], T1[:3, :3])                       # orientacja ta sama
    assert T1[:3, 3] == pytest.approx(T0[:3, 3] - 60.0 * T0[:3, 2])  # 60 mm wstecz po osi Z


def test_transit_over_rib(transit):
    checker, qa, qb = transit
    assert not motion_valid(checker, qa, qb)  # prosta droga przez żebro
    result = RRTConnect(checker, IRB2400_LIMITS, seed=0).plan(qa, qb)
    assert result.success
    smooth = shortcut(result.path, checker)
    assert_path_valid(smooth, checker, qa, qb)
    for q in smooth:
        assert IRB2400_LIMITS.ok(q)