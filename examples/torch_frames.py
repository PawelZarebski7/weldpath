"""Wypisuje macierze TCP palnika dla każdej znalezionej spoiny.

Uruchom:  poetry run python examples/torch_frames.py data/tjoint.step
"""
import sys

import numpy as np

from weldpath.cad import load_step
from weldpath.orientation import torch_frame, torch_path
from weldpath.seams import find_seams

np.set_printoptions(precision=3, suppress=True)

WORK_ANGLE = None    # None = dwusieczna (45° dla złącza 90°)
TRAVEL_ANGLE = 10.0  # push 10°

shape = load_step(sys.argv[1] if len(sys.argv) > 1 else "data/tjoint.step")

for i, seam in enumerate(find_seams(shape)):
    T = torch_frame(seam, seam.start, WORK_ANGLE, TRAVEL_ANGLE)
    R = T[:3, :3]
    print(f"\n=== Spoina S{i}, punkt startowy {seam.start} ===")
    print(T)
    print(f"oś palnika Z     = {T[:3, 2]}")
    print(f"det(R)           = {np.linalg.det(R):.6f}   (musi być 1)")
    print(f"R^T·R == I       = {np.allclose(R.T @ R, np.eye(3))}  (osie prostopadłe)")

    path = torch_path(seam, step=50.0, work_angle=WORK_ANGLE, travel_angle=TRAVEL_ANGLE)
    print(f"trajektoria: {len(path)} punktów co 50 mm, "
          f"od {path[0][:3, 3]} do {path[-1][:3, 3]}")