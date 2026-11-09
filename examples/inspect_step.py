"""Wypisuje ściany i krawędzie modelu STEP.

Uruchom:  poetry run python examples/inspect_step.py data/tjoint.step
"""
import sys

from weldpath.cad import curve_type, edge_length, edges, face_area, faces, load_step, surface_type

shape = load_step(sys.argv[1] if len(sys.argv) > 1 else "data/tjoint.step")

fs = list(faces(shape))
print(f"Ściany: {len(fs)}")
for i, f in enumerate(fs):
    print(f"  F{i:02d}  {surface_type(f):8s}  pole = {face_area(f):8.0f} mm²")

# TopExp_Explorer odwiedza każdą krawędź raz na każdą ścianę, do której należy,
# więc ta sama krawędź pojawia się 2x. IsSame() porównuje geometrię+topologię.
unique = []
for e in edges(shape):
    if not any(e.IsSame(u) for u in unique):
        unique.append(e)

print(f"\nKrawędzie (unikalne): {len(unique)}")
for i, e in enumerate(unique):
    print(f"  E{i:02d}  {curve_type(e):6s}  długość = {edge_length(e):6.1f} mm")