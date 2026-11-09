# weldpath

Automatic weld seam extraction and collision-free torch path planning for robotic welding cells.

## Pipeline

1. **Seam extraction**: read STEP (B-Rep), find weld seams and adjacent face normals
2. **Torch orientation**: seam geometry → TCP frames (work angle, push/drag angle)
3. **Motion planning**: collision-free transit moves between seams (RRT-Connect / RRT*)
4. **Kinematic constraints**: joint limits, singularity avoidance, no wrist flips

## Setup

Requires Python 3.11–3.14 and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run python examples/make_tjoint.py
poetry run python examples/inspect_step.py data/tjoint.step
```

## Project layout

```
weldpath/      library code
examples/      runnable scripts
tests/         pytest tests
data/          sample CAD models
```