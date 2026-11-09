from pathlib import Path

import numpy as np
import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from weldpath.cad import load_step
from weldpath.seams import find_seams

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def tjoint():
    return load_step(str(DATA / "tjoint.step"))


def test_tjoint_has_two_seams(tjoint):
    assert len(find_seams(tjoint)) == 2


def test_tjoint_seams_are_90_degrees(tjoint):
    for seam in find_seams(tjoint):
        assert seam.opening_angle == pytest.approx(90.0, abs=1e-6)
        assert seam.length == pytest.approx(200.0)


def test_tjoint_seams_lie_where_rib_meets_base(tjoint):
    ys = sorted(seam.start[1] for seam in find_seams(tjoint))
    assert ys == pytest.approx([45.0, 55.0])
    for seam in find_seams(tjoint):
        assert seam.start[2] == pytest.approx(10.0)


def test_normals_look_into_the_valley(tjoint):
    # Każda normalna musi mieć składową w stronę drugiej ściany:
    # suma normalnych wskazuje "wylot" doliny, czyli skąd wchodzi palnik.
    for seam in find_seams(tjoint):
        n1, n2 = seam.normals
        opening = n1 + n2
        assert opening[2] > 0  # dolina otwiera się do góry


def test_plain_box_has_no_seams():
    box = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
    assert find_seams(box) == []


def test_angle_filter_rejects_out_of_range(tjoint):
    assert find_seams(tjoint, min_angle=95.0) == []


def test_tangent_is_unit_vector(tjoint):
    for seam in find_seams(tjoint):
        assert np.linalg.norm(seam.tangent) == pytest.approx(1.0)