from pathlib import Path

import numpy as np
import pytest

from weldpath.cad import load_step
from weldpath.orientation import rotation_about_axis, torch_frame, torch_path
from weldpath.seams import find_seams

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def seams():
    return find_seams(load_step(str(DATA / "tjoint.step")))


def angle_to_face(z: np.ndarray, normal: np.ndarray) -> float:
    """Kąt między osią palnika a płaszczyzną ściany [°]."""
    return float(np.degrees(np.arcsin(abs(np.dot(z, normal)))))


# --- matematyka obrotów ---------------------------------------------------

def test_rodrigues_rotates_x_into_y():
    R = rotation_about_axis(np.array([0, 0, 1]), 90)
    assert R @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)


def test_rodrigues_zero_angle_is_identity():
    R = rotation_about_axis(np.array([1, 2, 3]), 0)
    assert np.allclose(R, np.eye(3))


# --- poprawność macierzy TCP ----------------------------------------------

@pytest.mark.parametrize("work_angle", [None, 30.0, 60.0])
@pytest.mark.parametrize("travel_angle", [-15.0, 0.0, 10.0])
def test_frame_is_valid_rotation(seams, work_angle, travel_angle):
    for seam in seams:
        R = torch_frame(seam, seam.start, work_angle, travel_angle)[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3))          # osie prostopadłe, jednostkowe
        assert np.linalg.det(R) == pytest.approx(1.0)   # układ prawoskrętny


def test_frame_position_is_the_seam_point(seams):
    seam = seams[0]
    T = torch_frame(seam, seam.end)
    assert T[:3, 3] == pytest.approx(seam.end)
    assert T[3] == pytest.approx([0, 0, 0, 1])


# --- parametry spawalnicze ------------------------------------------------

def test_default_is_bisector_45_degrees(seams):
    for seam in seams:
        z = torch_frame(seam, seam.start)[:3, 2]
        assert angle_to_face(z, seam.normals[0]) == pytest.approx(45.0)
        assert angle_to_face(z, seam.normals[1]) == pytest.approx(45.0)


def test_work_angle_is_measured_to_face_1(seams):
    for seam in seams:
        z = torch_frame(seam, seam.start, work_angle=30.0)[:3, 2]
        assert angle_to_face(z, seam.normals[0]) == pytest.approx(30.0)


def test_torch_points_into_material(seams):
    # Oś Z palnika musi mieć składową PRZECIWNĄ do normalnych obu ścian,
    # czyli celować w materiał, a nie od niego.
    for seam in seams:
        z = torch_frame(seam, seam.start)[:3, 2]
        assert np.dot(z, seam.normals[0]) < 0
        assert np.dot(z, seam.normals[1]) < 0


def test_push_tilts_forward_drag_tilts_backward(seams):
    seam = seams[0]
    z_push = torch_frame(seam, seam.start, travel_angle=10.0)[:3, 2]
    z_drag = torch_frame(seam, seam.start, travel_angle=-10.0)[:3, 2]
    assert np.dot(z_push, seam.tangent) == pytest.approx(np.sin(np.radians(10)))
    assert np.dot(z_drag, seam.tangent) == pytest.approx(-np.sin(np.radians(10)))


def test_x_axis_follows_travel_direction(seams):
    for seam in seams:
        x = torch_frame(seam, seam.start, travel_angle=10.0)[:3, 0]
        assert np.dot(x, seam.tangent) > 0.98


# --- trajektoria ----------------------------------------------------------

def test_path_covers_whole_seam(seams):
    seam = seams[0]
    path = torch_path(seam, step=10.0)
    assert len(path) == 21  # 200 mm / 10 mm + punkt końcowy
    assert path[0][:3, 3] == pytest.approx(seam.start)
    assert path[-1][:3, 3] == pytest.approx(seam.end)