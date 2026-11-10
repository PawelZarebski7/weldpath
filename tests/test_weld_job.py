from pathlib import Path

import numpy as np
import pytest

from weldpath.cad import load_step
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400, forward_kinematics
from weldpath.seams import find_seams
from weldpath.weld_job import (
    DEFAULT_TORCH,
    IRB2400_LIMITS,
    Torch,
    flange_targets,
    plan_seam,
    solve_seam,
)

DATA = Path(__file__).resolve().parent.parent / "data"
T_BASE_PART = homogeneous(np.eye(3), np.array([900.0, -100.0, 300.0]))


@pytest.fixture(scope="module")
def seams():
    return find_seams(load_step(str(DATA / "tjoint.step")))


@pytest.fixture(scope="module")
def plans(seams):
    return [plan_seam(ABB_IRB2400, s, T_BASE_PART) for s in seams]


# --- palnik ---------------------------------------------------------------

def test_torch_tcp_is_at_torch_length():
    T = Torch(length=350, neck_angle=22).flange_to_tcp()
    assert np.linalg.norm(T[:3, 3]) == pytest.approx(350)
    # oś Z palnika odchylona od osi kołnierza o kąt szyjki
    assert np.degrees(np.arccos(T[2, 2])) == pytest.approx(22)


def test_straight_torch_is_pure_offset():
    T = Torch(length=300, neck_angle=0).flange_to_tcp()
    assert np.allclose(T[:3, :3], np.eye(3))
    assert T[:3, 3] == pytest.approx([0, 0, 300])


# --- plan spoiny ----------------------------------------------------------

def test_both_seams_are_feasible(plans):
    assert all(p is not None and p.feasible for p in plans)


def test_plan_reaches_every_target(seams, plans):
    # Złota zasada: FK każdego punktu planu = żądana pozycja kołnierza.
    for seam, plan in zip(seams, plans):
        targets = flange_targets(seam, T_BASE_PART, DEFAULT_TORCH, plan.spin,
                                 10.0, None, 10.0)
        for q, T in zip(plan.joints, targets):
            assert np.allclose(forward_kinematics(ABB_IRB2400, q), T, atol=1e-6)


def test_plan_respects_joint_limits(plans):
    for plan in plans:
        for q in plan.joints:
            assert IRB2400_LIMITS.ok(q)


def test_no_wrist_flip_along_seam(plans):
    for plan in plans:
        q5 = plan.joints[:, 4]
        assert np.all(np.sign(q5) == np.sign(q5[0]))  # q5 nie zmienia znaku
        assert plan.max_joint_step < 5.0               # płynny ruch, brak skoków


def test_robot_faces_the_part(plans):
    # Detal leży przed robotem (+X), więc q1 musi być w przedniej półprzestrzeni.
    for plan in plans:
        assert np.all(np.abs(plan.joints[:, 0]) < np.radians(90))


def test_unreachable_part_gives_no_plan(seams):
    far = homogeneous(np.eye(3), np.array([1800.0, -100.0, 300.0]))
    assert plan_seam(ABB_IRB2400, seams[0], far) is None


def test_singularity_warning_is_reported(seams):
    # Nierealnie surowy próg osobliwości: każdy plan musi dostać ostrzeżenie.
    plan = solve_seam(ABB_IRB2400, seams[0], T_BASE_PART, singular_q5=179.0)
    assert any(w.startswith("UWAGA") for w in plan.warnings)
    assert plan.feasible  # ostrzeżenie nie blokuje planu


def test_spin_search_rescues_seam_at_edge_of_reach(seams):
    # Detal 1.4 m od robota: bez obrotu palnika poza zasięgiem (limity osi),
    # obrót wokół osi palnika przesuwa kołnierz bliżej i spoina staje się osiągalna.
    far = homogeneous(np.eye(3), np.array([1400.0, -100.0, 200.0]))
    assert solve_seam(ABB_IRB2400, seams[0], far, spin=0.0) is None
    best = plan_seam(ABB_IRB2400, seams[0], far)
    assert best is not None and best.feasible
    assert best.spin != 0.0