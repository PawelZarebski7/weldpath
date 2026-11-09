import numpy as np
import pytest

from weldpath.robot import ABB_IRB2400, KUKA_KR6_R700, forward_kinematics, inverse_kinematics

ROBOTS = [KUKA_KR6_R700, ABB_IRB2400]


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Różnica kątów z uwzględnieniem zawinięcia (179° i -179° są blisko)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def random_configs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = np.radians([-170, -90, -120, -170, -120, -170])
    hi = np.radians([170, 90, 120, 170, 120, 170])
    return rng.uniform(lo, hi, size=(n, 6))


# --- FK -------------------------------------------------------------------

@pytest.mark.parametrize("robot", ROBOTS)
def test_zero_pose_points_straight_up(robot):
    T = forward_kinematics(robot, np.zeros(6))
    height = robot.c1 + robot.c2 + robot.c3 + robot.c4
    assert T[:3, 3] == pytest.approx([robot.a1 + robot.a2, robot.b, height])
    assert np.allclose(T[:3, :3], np.eye(3))


@pytest.mark.parametrize("robot", ROBOTS)
def test_axis1_rotates_around_vertical(robot):
    q = np.zeros(6)
    q[1] = np.radians(45)  # pochylamy ramię, żeby nadgarstek nie był nad osią 1
    p0 = forward_kinematics(robot, q)[:3, 3]
    q[0] = np.radians(90)
    p1 = forward_kinematics(robot, q)[:3, 3]
    assert p1[2] == pytest.approx(p0[2])                      # wysokość bez zmian
    assert np.hypot(*p1[:2]) == pytest.approx(np.hypot(*p0[:2]))  # ten sam promień


# --- IK -------------------------------------------------------------------

@pytest.mark.parametrize("robot", ROBOTS)
def test_every_ik_solution_reaches_the_pose(robot):
    for q in random_configs(500):
        T = forward_kinematics(robot, q)
        for sol in inverse_kinematics(robot, T):
            if np.isnan(sol).any():
                continue
            assert np.allclose(forward_kinematics(robot, sol), T, atol=1e-6)


@pytest.mark.parametrize("robot", ROBOTS)
def test_original_configuration_is_among_solutions(robot):
    for q in random_configs(500, seed=1):
        sols = inverse_kinematics(robot, forward_kinematics(robot, q))
        valid = sols[~np.isnan(sols).any(axis=1)]
        best = np.min(np.max(np.abs(angle_diff(valid, q)), axis=1))
        assert best < 1e-6


def test_wrist_flip_pairs():
    q = np.radians([20, 10, 30, 40, 50, 60])
    sols = inverse_kinematics(KUKA_KR6_R700, forward_kinematics(KUKA_KR6_R700, q))
    no_flip, flip = sols[0], sols[1]
    assert np.allclose(no_flip[:3], flip[:3])                         # ramię to samo
    # ±180° to ten sam obrót, więc porównujemy wartość bezwzględną różnicy.
    assert abs(angle_diff(flip[3], no_flip[3])) == pytest.approx(np.pi)  # q4 ± 180°
    assert flip[4] == pytest.approx(-no_flip[4])                         # q5 zmienia znak
    assert abs(angle_diff(flip[5], no_flip[5])) == pytest.approx(np.pi)  # q6 ± 180°


def test_unreachable_pose_gives_no_solutions():
    T = np.eye(4)
    T[:3, 3] = [5000, 0, 500]  # 5 m od robota o zasięgu ~0.7 m
    sols = inverse_kinematics(KUKA_KR6_R700, T)
    # q1 da się zawsze policzyć (to tylko kierunek), ale ramię nie sięga: q2..q6 = NaN.
    assert np.isnan(sols[:, 1:]).all()