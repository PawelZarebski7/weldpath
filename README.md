# weldpath

Automatic generation of collision-free welding programs for robotic cells:
**STEP model → weld seams → torch orientation → robot kinematics → collision-free motion**,
for a 6-axis arm, a tilt-rotate positioner and a linear track, visualised in **RoboDK**.

<p align="center">
  <img src="docs/pipe_welding.png" width="49%" alt="Pipe butt weld on a rotating positioner in RoboDK">
  <img src="docs/pipe_done.png" width="49%" alt="Finished three-pass circumferential weld">
</p>
<p align="center"><em>Pipe butt joint (Ø168.3 × 10, 60° V-groove), three passes. The positioner turns the pipe so every point is welded flat (PA); the robot barely moves (max 0.6° on any axis).</em></p>

## What it does

| Part | Cell | Result |
|---|---|---|
| T-joint | robot + table | 2 fillet seams, collision-free transit over the rib (RRT-Connect) |
| Frame, 4 separate plates | robot + table | **12/12** seams, incl. tight pockets: work/travel angle, direction, end trim and centre-out split chosen automatically |
| Same frame | robot + tilt-rotate positioner | every seam moved from PB/PF to **flat position PA** |
| Two pipes, V-groove | robot + rotating positioner | circumferential seam, **coordinated 8-axis motion**, root / fill / cap passes |
| Welded I-beam 2 m, 4 stiffeners | robot on **linear track** | **36/36** seams in 4 part set-ups (crane flips), track position chosen per seam |
| Bracket, arm at 60° + gusset | robot + table | seams at 60°, 90° and 120°, torch angle derived from geometry |
| RHS 80×80×5 on a base plate (trailer frame node) | robot + tilt-rotate positioner | weld-around: 4 lines + 4 corner arcs **chained into one continuous seam**, robot moves only 3° |
| Tube Ø42.4 on Ø60.3 (tubular frame) | robot + tilt-rotate positioner | **saddle weld** (3D curve, wall angle 90°→135°): tilt and rotation both vary, whole seam in PA |

Robot kinematics are **verified against RoboDK** for ABB IRB 2400/16: 0.000 mm / 0.000° on 200 random poses
(including the J2–J3 parallelogram coupling of the IRB 2400).

## Pipeline

```
STEP (B-Rep)                      weldpath.cad, weldpath.assembly
  └─ seams between bodies         concave edges of the fused model, owner bodies by point probing
      └─ torch frames             weldpath.orientation      bisector, work angle, push/drag, Gram-Schmidt
          └─ joint angles         weldpath.robot            analytic OPW IK (8 solutions), FK
              └─ seam plan        weldpath.weld_job         fixed configuration (no wrist flip),
                                                            limits in controller angles, singularity
                                                            margin, technology search
                  └─ transits     weldpath.planner          RRT-Connect + shortcutting, retract/approach
                      └─ cell     weldpath.positioner       tilt-rotate positioner, ISO 6947 positions
                                  weldpath.coordinated      curved seams, coordinated motion, multi-pass
                                  weldpath.track            linear track, part set-ups, accessibility
collision checks                  weldpath.collision        FCL: capsule robot, BVH part mesh, moving parts
RoboDK                            weldpath.robodk_bridge    joint conventions, FK check, station, animation
```

Every planner talks to collision checking only through the `CollisionChecker` protocol
(`in_collision(q) -> bool`), so the FCL backend can be swapped for RoboDK's collision engine
without touching the planners. A RoboDK-based checker is included (`RoboDKChecker`).

## Key decisions

- **Seam detection** — the assembly is fused; a seam is an edge where the two faces form a valley
  (`n1 · d2 > 0`). Each face is traced back to its original plate by probing a point just inside
  the material; edges inside a single plate are rejected.
- **Analytic IK (OPW model)** — 7 parameters describe most industrial arms; closed-form IK gives all
  8 configurations in microseconds, which matters when planners call it tens of thousands of times.
- **One configuration per seam** — the same IK branch is kept along the whole seam, so the wrist never
  flips mid-weld; joint jumps and `|q5|` (wrist singularity) are monitored.
- **Controller conventions** — joint limits are checked in ABB controller angles, after converting
  OPW angles with signs, offsets and the J2–J3 coupling found by an automatic comparison with RoboDK.
- **Technology search** — candidates are ordered by a "welding cost" (deviation from 45°/push 10°,
  torch spin, reversed direction, trimmed ends) and the search stops at the first good plan.
- **Positioner** — analytic tilt/rotate angles that point the seam bisector straight up (PA);
  for circular seams the table angle is unwrapped along the seam for smooth coordinated motion.
- **Seam chains** — B-Rep splits a weld-around into many edges (lines and corner fillets). Edges
  that meet end-to-end are chained, sampled as one curve with an overlap at the closing point, and
  welded in one coordinated motion; the chain farthest from the table axis is the outside weld.
- **Whole-cycle safety** — positioner moves, approach, weld passes, retract and return are
  sequenced and each segment is collision-checked (moving robot and positioner together was found
  to hit the pipe — there is a test for it).

## Run it

Requires Python 3.11–3.14 and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pytest -q                                   # 108 tests

poetry run python examples/make_frame.py               # generate example parts
poetry run python examples/plan_frame.py               # frame: 12 seams with parameter search
poetry run python examples/plan_positioner.py          # same frame on a positioner, all in PA
poetry run python examples/make_pipes.py && poetry run python examples/plan_pipe.py
poetry run python examples/make_beam.py && poetry run python examples/plan_beam.py     # ~3 min
poetry run python examples/make_bracket.py && poetry run python examples/plan_bracket.py
poetry run python examples/make_rhs.py && poetry run python examples/positioner_jobs.py rhs
poetry run python examples/make_tee.py && poetry run python examples/positioner_jobs.py tee
```

### RoboDK

Open RoboDK with an **ABB IRB 2400/16** in the station (free licence is enough), then:

```bash
poetry run python examples/robodk_check_fk.py          # compare our kinematics with RoboDK
poetry run python examples/robodk_show_frame.py        # frame on a table (or pass any STEP file)
poetry run python examples/robodk_show_pipe.py         # pipe on a rotating positioner, growing beads
poetry run python examples/robodk_show_beam.py         # I-beam from a linear track, 4 set-ups
poetry run python examples/robodk_show_job.py rhs      # weld-around an RHS profile on a positioner
poetry run python examples/robodk_show_job.py tee      # saddle weld of a tube branch
```

The positioner and the track are simulated by moving reference frames in the station
(there is no positioner/track model in the free station); with an OEM licence they would be
added as mechanisms synchronised with the robot.

## Known limitations / next steps

- Parts must touch — assembly gaps (0.5–1 mm) would need "nearly touching faces" detection.
- Sampling of curved seams is uniform in curve parameter (exact for lines and circles only).
- Axes 4 and 6 are limited to ±180° by IK wrapping (ABB allows ±200° / ±400°).
- Collision checks cost ~0.3 ms in Python (mostly object construction) — the next speed-up would be
  checking torch capsules first, caching robot links, and moving the hot loop to C++.
- RRT-Connect + shortcutting finds *a* path, not the shortest; RRT\* or trajectory optimisation
  (and time parameterisation) are next.
- Seam order is fixed; ordering/direction is a TSP-like problem, as is minimising part set-ups.
- Track position is chosen per seam; coordinated track motion during long seams is not used yet.
- Saddle welds are planned as fillets; groove preparation and weave along a varying angle are not modelled.

## Layout

```
weldpath/      library (≈2000 lines)
examples/      generators for example parts, planning demos, RoboDK demos
tests/         pytest (108 tests)
data/          generated STEP models and the RoboDK station
docs/          screenshots
```