"""Znajduje spoiny pachwinowe w modelu STEP.

Uruchom:  poetry run python examples/find_seams.py data/tjoint.step
"""
import sys

import numpy as np

from weldpath.cad import load_step
from weldpath.seams import find_seams

np.set_printoptions(precision=1, suppress=True)

shape = load_step(sys.argv[1] if len(sys.argv) > 1 else "data/tjoint.step")
seams = find_seams(shape)

print(f"Znalezione spoiny: {len(seams)}")
for i, s in enumerate(seams):
    print(f"\nS{i}: {s.start} -> {s.end}  ({s.length:.0f} mm)")
    print(f"    kierunek   t  = {s.tangent}")
    print(f"    normalne   n1 = {s.normals[0]}   n2 = {s.normals[1]}")
    print(f"    kąt otwarcia  = {s.opening_angle:.1f}°")