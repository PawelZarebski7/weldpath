import subprocess
import sys
from collections import Counter
from pathlib import Path

import fcl
import numpy as np
import pytest

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400
from weldpath.track import (
    Track,
    accessible_from_track,
    plan_weld_on_track,
    setups_for_long_part,
    world_attachments,
)

ROOT = Path(__file__).resolve().parent.parent
BEAM = ROOT / "data" / "beam.step"
TRACK = Track()
T_WORLD_BEAM = homogeneous(np.eye(3), np.array([0.0, 950.0, 500.0]))
CENTER = np.array([1000.0, 0.0, 150.0])
SUPPORTS = [(fcl.Box(200.0, 300.0, 500.0), homogeneous(np.eye(3), np.array([x, 950.0, 250.0])))
            for x in (300.0, 1700.0)]


@pytest.fixture(scope="module")
def bodies():
    if not BEAM.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_beam.py")], check=True)
    return solids(load_step(str(BEAM)))


@pytest.fixture(scope="module")
def seams(bodies):
    return [w.seam for w in find_weld_seams(bodies)]


@pytest.fixture(scope="module")
def setups():
    return setups_for_long_part(T_WORLD_BEAM, CENTER)


def world_point(T, p):
    return (T @ np.append(p, 1.0))[:3]


# --- tor jezdny ---------------------------------------------------------------

def test_track_moves_robot_along_x():
    T = TRACK.T_base_part(500.0, np.eye(4))
    assert T[:3, 3] == pytest.approx([-500.0, 0.0, 0.0])   # robot w s=500 widzi świat przesunięty
    assert np.allclose(T[:3, :3], np.eye(3))
    assert TRACK.clamp(10_000.0) == TRACK.s_max


# --- belka i ustawienia -------------------------------------------------------

def test_beam_has_36_seams(bodies, seams):
    assert len(bodies) == 7                       # 3 blachy belki + 4 żebra
    assert len(seams) == 36


def test_setups_keep_beam_on_supports(setups):
    for _, T in setups:
        assert world_point(T, CENTER) == pytest.approx(world_point(T_WORLD_BEAM, CENTER))
    # ustawienie 3: pas górny (z = 300 w układzie belki) ląduje na dole (z świata = 500)
    flipped = setups[2][1]
    assert world_point(flipped, [1000.0, 0.0, 300.0])[2] == pytest.approx(500.0)


def test_supports_stay_on_the_floor(setups):
    # Stojaki przeliczone do układu belki muszą wrócić na swoje miejsce w każdym ustawieniu.
    for _, T in setups:
        for (_, T_world), (_, T_local) in zip(SUPPORTS, world_attachments(T, SUPPORTS)):
            assert np.allclose(T @ T_local, T_world)


def test_every_seam_is_accessible_in_some_setup(seams, setups):
    counts = Counter()
    for seam in seams:
        ok = [k for k, (_, T) in enumerate(setups, 1) if accessible_from_track(seam, T)]
        assert ok, "spoina niedostępna w żadnym ustawieniu"
        counts[ok[0]] += 1
    assert set(counts) == {1, 2, 3, 4}            # potrzebne są wszystkie cztery ustawienia


def test_overhead_seams_rejected_as_laid(seams, setups):
    T = setups[0][1]
    under_top_flange = [s for s in seams if s.start[2] == pytest.approx(288.0)
                        and s.end[2] == pytest.approx(288.0)]
    assert len(under_top_flange) == 14
    assert not any(accessible_from_track(s, T) for s in under_top_flange)


# --- planowanie z wyborem położenia wózka ------------------------------------

def test_track_position_follows_the_seam(bodies, seams, setups):
    T = setups[0][1]
    checker = FCLChecker(ABB_IRB2400, fuse_bodies(bodies), TRACK.T_base_part(0.0, T),
                         attachments=world_attachments(T, SUPPORTS))
    # Krótka spoina żebra z pasem dolnym, po stronie robota, na x = 1295.
    seam = next(s for s in seams if s.start[0] == pytest.approx(1295.0)
                and s.end[0] == pytest.approx(1295.0)
                and s.start[2] == pytest.approx(12.0) and s.end[2] == pytest.approx(12.0)
                and s.start[1] < 0)
    result = plan_weld_on_track(ABB_IRB2400, seam, TRACK, T, checker,
                                search={"spins": (0, 45, -45), "work_angles": (None,),
                                        "travel_angles": (10.0, 30.0), "trims": (0.0,)})
    assert result is not None and result.setup == 1
    assert result.s == pytest.approx(1295.0, abs=400.0)   # wózek naprzeciw spoiny
    checker.set_part_pose(TRACK.T_base_part(result.s, T))
    for plan in result.plans:
        for q in plan.joints:
            assert not checker.in_collision(q)