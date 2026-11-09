import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400
from weldpath.weld_job import plan_weld, reversed_seam, split_center_out, trimmed_seam

ROOT = Path(__file__).resolve().parent.parent
FRAME = ROOT / "data" / "frame.step"
T_BASE_PART = homogeneous(np.eye(3), np.array([800.0, -150.0, 300.0]))


@pytest.fixture(scope="module")
def bodies():
    if not FRAME.exists():  # model generujemy skryptem, gdyby go nie było
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_frame.py")], check=True)
    return solids(load_step(str(FRAME)))


@pytest.fixture(scope="module")
def weld_seams(bodies):
    return find_weld_seams(bodies)


@pytest.fixture(scope="module")
def checker(bodies):
    return FCLChecker(ABB_IRB2400, fuse_bodies(bodies), T_BASE_PART, table_top_z=300.0)


def find(weld_seams, bodies_pair, start):
    """Spoina łącząca dane bryły i zaczynająca się w danym punkcie."""
    for ws in weld_seams:
        if set(ws.bodies) == set(bodies_pair) and np.allclose(ws.seam.start, start):
            return ws.seam
    raise LookupError(start)


# --- wykrywanie spoin w złożeniu ------------------------------------------

def test_frame_has_four_bodies(bodies):
    assert len(bodies) == 4


def test_frame_has_twelve_seams(weld_seams):
    assert len(weld_seams) == 12


def test_seams_per_body_pair(weld_seams):
    pairs = Counter(tuple(sorted(ws.bodies)) for ws in weld_seams)
    # płyta-żebro A: zewnętrzna + 2 wewnętrzne (przerwane żebrem poprzecznym), tak samo B
    assert pairs == {(0, 1): 3, (0, 2): 3, (0, 3): 2, (1, 3): 2, (2, 3): 2}


def test_short_end_seams_are_filtered(bodies):
    # Dwie 10-milimetrowe krawędzie na czołach żebra poprzecznego.
    assert len(find_weld_seams(bodies, min_length=0)) == 14


def test_single_body_corners_are_not_welds():
    # Złącze teowe zapisane jako JEDNA bryła: krawędzie wklęsłe są, ale to nie spoiny.
    tjoint = load_step(str(ROOT / "data" / "tjoint.step"))
    assert find_weld_seams(solids(tjoint)) == []


# --- warianty spoiny ------------------------------------------------------

def test_reversed_and_trimmed_seam(weld_seams):
    seam = weld_seams[0].seam
    rev = reversed_seam(seam)
    assert rev.start == pytest.approx(seam.end) and rev.tangent == pytest.approx(-seam.tangent)
    cut = trimmed_seam(seam, 5.0, 10.0)
    assert cut.length == pytest.approx(seam.length - 15.0)


def test_split_center_out_has_no_gap(weld_seams):
    seam = weld_seams[0].seam
    a, b = split_center_out(seam)
    mid = (seam.start + seam.end) / 2
    assert a.start == pytest.approx(mid) and b.start == pytest.approx(mid)
    assert a.end == pytest.approx(seam.end) and b.end == pytest.approx(seam.start)


# --- planowanie z doborem parametrów (wolniejsze testy) -------------------

def test_open_outer_seam_uses_standard_settings(weld_seams, checker):
    seam = find(weld_seams, (0, 1), [0, 40, 10])
    plans = plan_weld(ABB_IRB2400, seam, T_BASE_PART, checker)
    assert len(plans) == 1
    p = plans[0]
    assert p.work_angle is None and p.travel_angle == 10.0 and not p.reversed
    assert p.trim == (0.0, 0.0)


def test_vertical_corner_seam_is_welded_from_above(weld_seams, checker):
    seam = find(weld_seams, (1, 3), [145, 50, 10])
    plans = plan_weld(ABB_IRB2400, seam, T_BASE_PART, checker)
    assert len(plans) == 1
    p = plans[0]
    assert p.feasible
    assert p.seam.start[2] > p.seam.end[2]  # spawanie z góry na dół


def test_closed_pocket_seam_is_split_center_out(weld_seams, checker):
    seam = find(weld_seams, (0, 3), [155, 50, 10])
    plans = plan_weld(ABB_IRB2400, seam, T_BASE_PART, checker)
    assert len(plans) == 2
    mid = (seam.start + seam.end) / 2
    for p in plans:
        assert p.feasible
        assert p.seam.start == pytest.approx(mid)  # obie połówki startują w środku
        for q in p.joints:
            assert not checker.in_collision(q)