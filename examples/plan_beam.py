"""Belka dwuteowa 2 m z żebrami, robot na torze jezdnym, spawanie w kilku ustawieniach.

Uruchom:  poetry run python examples/make_beam.py
          poetry run python examples/plan_beam.py
Planowanie trwa ok. 3 minut (36 spoin x kilka ustawień i położeń wózka).
"""
import time
from collections import Counter
from pathlib import Path

import fcl
import numpy as np

from weldpath.assembly import find_weld_seams, fuse_bodies, solids
from weldpath.cad import load_step
from weldpath.collision import FCLChecker
from weldpath.orientation import homogeneous
from weldpath.robot import ABB_IRB2400
from weldpath.track import Track, plan_part_on_track, setups_for_long_part, world_attachments

ROOT = Path(__file__).resolve().parent.parent
TRACK = Track(s_min=-300.0, s_max=2300.0)
# Belka leży wzdłuż toru, 950 mm od jego osi, na stojakach wysokości 500 mm.
T_WORLD_BEAM = homogeneous(np.eye(3), np.array([0.0, 950.0, 500.0]))
BEAM_CENTER = np.array([1000.0, 0.0, 150.0])          # środek belki (układ belki)
SUPPORTS_WORLD = [(fcl.Box(200.0, 300.0, 500.0),
                   homogeneous(np.eye(3), np.array([x, 950.0, 250.0]))) for x in (300.0, 1700.0)]
# Węższe przeszukiwanie parametrów spawania niż domyślne – przy 36 spoinach liczy się czas.
SEARCH = {"spins": (0, 45, -45, 90, -90), "work_angles": (None, 35.0, 55.0),
          "travel_angles": (10.0, 0.0, -10.0, 30.0, -30.0), "trims": (0.0, 10.0)}


def make_checker(T_world_part, bodies):
    return FCLChecker(ABB_IRB2400, fuse_bodies(bodies), TRACK.T_base_part(0.0, T_world_part),
                      attachments=world_attachments(T_world_part, SUPPORTS_WORLD))


if __name__ == "__main__":
    bodies = solids(load_step(str(ROOT / "data" / "beam.step")))
    seams = [w.seam for w in find_weld_seams(bodies)]
    setups = setups_for_long_part(T_WORLD_BEAM, BEAM_CENTER)
    print(f"Belka: {len(bodies)} blach, {len(seams)} spoin, {len(setups)} możliwe ustawienia\n")

    t0 = time.perf_counter()
    done, failed = plan_part_on_track(ABB_IRB2400, seams, TRACK, setups,
                                      lambda T: make_checker(T, bodies),
                                      offsets=(0.0, 250.0, -250.0), search=SEARCH)
    print(f"\nWykonalne: {len(done)}/{len(seams)} spoin, {time.perf_counter() - t0:.0f} s")
    per_setup = Counter(w.setup for w in done)
    for k, (name, _) in enumerate(setups, 1):
        welds = sorted((w for w in done if w.setup == k), key=lambda w: w.s)
        if welds:
            print(f"  ustawienie {name}: {per_setup[k]} spoin, "
                  f"wózek od {welds[0].s:.0f} do {welds[-1].s:.0f} mm")
    if failed:
        print(f"  niewykonalne: {len(failed)}")