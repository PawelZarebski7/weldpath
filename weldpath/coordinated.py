"""Krok 7: spoiny krzywoliniowe i ruch SKOORDYNOWANY robota z pozycjonerem.

Spoina obwodowa rury: zamiast wozić palnik dookoła rury (palnik raz na górze,
raz od spodu = pozycja okapowa), pozycjoner OBRACA rurę tak, żeby spawany punkt
był zawsze na górze (pozycja podolna PA). Robot prawie stoi w miejscu.

Dla każdego punktu spoiny (co `step` mm):
  1. liczymy lokalny kierunek spoiny i normalne obu ścian rowka (zmieniają się
     wzdłuż okręgu – na prostej były stałe),
  2. dobieramy obrót pozycjonera, przy którym dwusieczna rowka patrzy pionowo w górę,
  3. liczymy palnik w układzie detalu, przenosimy do świata przez pozycjoner
     i rozwiązujemy IK robota.

Grubą ściankę spawa się WIELOŚCIEGOWO: grań (dno rowka), wypełnienie, lico –
każdy ścieg to ta sama droga przesunięta wzdłuż dwusiecznej rowka na zewnątrz.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.gp import gp_Pnt, gp_Vec

from weldpath.orientation import homogeneous, rotation_about_axis, torch_frame
from weldpath.positioner import Positioner
from weldpath.robot import OPWParams, inverse_kinematics
from weldpath.seams import Seam, face_normal
from weldpath.weld_job import DEFAULT_TORCH, IRB2400_LIMITS, Torch


@dataclass
class SeamSample:
    point: np.ndarray       # punkt na spoinie (układ detalu)
    tangent: np.ndarray     # lokalny kierunek spoiny
    normals: tuple[np.ndarray, np.ndarray]

    @property
    def opening(self) -> np.ndarray:
        """Dwusieczna rowka/doliny – skąd przychodzi palnik."""
        v = self.normals[0] + self.normals[1]
        return v / np.linalg.norm(v)


def sample_seam(seam: Seam, step: float = 10.0, overlap: float = 0.0) -> list[SeamSample]:
    """Punkty co `step` mm wzdłuż krawędzi spoiny (także krzywej), z lokalnymi normalnymi.

    overlap: o ile mm przejechać za koniec (dla spoiny zamkniętej – zakładka na początku,
    żeby nie zostawić krateru w miejscu startu).
    Uwaga: krok w parametrze krzywej jest stały – dla prostej i okręgu to stały krok
    w milimetrach; dla dowolnej krzywej B-spline byłby tylko przybliżony.
    """
    curve = BRepAdaptor_Curve(seam.edge)
    u0, u1 = curve.FirstParameter(), curve.LastParameter()
    closed = curve.IsClosed() or curve.IsPeriodic()
    if not closed:
        overlap = 0.0
    n = max(1, int(np.ceil((seam.length + overlap) / step)))
    f1, f2 = seam.faces
    samples = []
    for s in np.linspace(0.0, seam.length + overlap, n + 1):
        u = u0 + (u1 - u0) * (s / seam.length)
        if closed:
            u = u0 + (u - u0) % (u1 - u0)
        p, v = gp_Pnt(), gp_Vec()
        curve.D1(u, p, v)
        point = np.array([p.X(), p.Y(), p.Z()])
        t = np.array([v.X(), v.Y(), v.Z()])
        samples.append(SeamSample(point, t / np.linalg.norm(t),
                                  (face_normal(f1, point), face_normal(f2, point))))
    return samples


@dataclass
class WeldPass:
    name: str
    offset: float            # przesunięcie od dna rowka wzdłuż dwusiecznej [mm]
    joints: np.ndarray       # (N, 6) kąty robota [rad]
    positioner: np.ndarray   # (N, 2) uchylenie i obrót pozycjonera [°]


@dataclass
class CoordinatedPlan:
    seam: Seam
    config: int
    spin: float
    passes: list[WeldPass] = field(default_factory=list)

    @property
    def robot_motion(self) -> float:
        """Największy zakres ruchu którejkolwiek osi robota w czasie spawania [°]."""
        q = np.vstack([p.joints for p in self.passes])
        return float(np.degrees(np.ptp(q, axis=0).max()))

    @property
    def positioner_rotation(self) -> float:
        """O ile obraca się tarcza pozycjonera w jednym ściegu [°]."""
        rot = self.passes[0].positioner[:, 1]
        return float(abs(rot[-1] - rot[0]))


DEFAULT_PASSES = (("grań", 0.0), ("wypełnienie", 4.0), ("lico", 8.0))


def positioner_track(positioner: Positioner, samples: list[SeamSample]) -> np.ndarray:
    """(N, 2) kąty pozycjonera stawiające każdy punkt spoiny w pozycji PA.

    Uchylenie trzymamy stałe (jak przy pierwszym punkcie), obrót tarczy "odwijamy",
    żeby rósł płynnie zamiast przeskakiwać z +180° na −180°.
    """
    tilt0 = positioner.angles_pointing_up(samples[0].opening)[0][0]
    track = []
    for smp in samples:
        options = positioner.angles_pointing_up(smp.opening)
        tilt, rot = min(options, key=lambda tr: abs(tr[0] - tilt0))
        track.append((tilt, rot))
    track = np.array(track)
    track[:, 1] = np.degrees(np.unwrap(np.radians(track[:, 1])))
    return track


def plan_coordinated(robot: OPWParams, seam: Seam, positioner: Positioner, checker,
                     torch: Torch = DEFAULT_TORCH, limits=IRB2400_LIMITS,
                     step: float = 10.0, overlap: float = 15.0, travel_angle: float = 10.0,
                     passes=DEFAULT_PASSES, spins=(0, 30, -30, 60, -60, 90, -90),
                     max_step: float = 15.0) -> CoordinatedPlan | None:
    """Plan spawania z obracającym się pozycjonerem: robot + 2 osie pozycjonera naraz."""
    samples = sample_seam(seam, step, overlap)
    track = positioner_track(positioner, samples)
    part_poses = [positioner.T_base_part(tilt, rot) for tilt, rot in track]
    T_tcp_flange = np.linalg.inv(torch.flange_to_tcp())

    for spin in spins:
        R_spin = homogeneous(rotation_about_axis(np.array([0.0, 0.0, 1.0]), spin), np.zeros(3))
        per_pass_solutions = []
        for _, offset in passes:
            targets = []
            for smp, T_part in zip(samples, part_poses):
                local = replace(seam, tangent=smp.tangent, normals=smp.normals)
                T_tcp = torch_frame(local, smp.point + offset * smp.opening, None, travel_angle)
                targets.append(T_part @ T_tcp @ R_spin @ T_tcp_flange)
            per_pass_solutions.append(np.array([inverse_kinematics(robot, T) for T in targets]))

        for config in range(8):
            qs = [sols[:, config, :] for sols in per_pass_solutions]
            if any(np.isnan(q).any() or not all(limits.ok(qi) for qi in q) for q in qs):
                continue
            steps = [np.abs((np.diff(q, axis=0) + np.pi) % (2 * np.pi) - np.pi) for q in qs]
            if max(np.degrees(s.max()) for s in steps) > max_step:
                continue
            collides = False
            for q in qs:
                for qi, T_part in zip(q, part_poses):
                    checker.set_part_pose(T_part)
                    if checker.in_collision(qi):
                        collides = True
                        break
                if collides:
                    break
            if collides:
                continue
            return CoordinatedPlan(
                seam=seam, config=config, spin=float(spin),
                passes=[WeldPass(name, off, q, track.copy())
                        for (name, off), q in zip(passes, qs)],
            )
    return None


# --- pełny program stanowiska: kolejność ruchów robota i pozycjonera ----------

@dataclass
class Segment:
    """Odcinek programu: punkty robota i pozycjonera (tyle samo), łączone liniowo."""
    label: str
    kind: str                 # "positioner" | "transit" | "linear" | "weld"
    robot: list[np.ndarray]
    positioner: list[np.ndarray]
    weld_pass: str | None = None


def _combined_motion_ok(checker, positioner: Positioner, qa, pa, qb, pb,
                        resolution: float = np.radians(1.0)) -> bool:
    """Ruch robota i pozycjonera naraz po prostej – sprawdzamy gęsto, krok ≤ 1° na osi."""
    span = max(np.max(np.abs(qb - qa)), np.radians(np.max(np.abs(pb - pa))))
    n = max(2, int(np.ceil(span / resolution)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        checker.set_part_pose(positioner.T_base_part(*(pa + (pb - pa) * t)))
        if checker.in_collision(qa + (qb - qa) * t):
            return False
    return True


def build_cell_program(robot: OPWParams, plan: CoordinatedPlan, positioner: Positioner,
                       checker, home: np.ndarray, torch: Torch = DEFAULT_TORCH,
                       limits=IRB2400_LIMITS, retract_mm: float = 60.0) -> list[Segment]:
    """Kolejność jak na prawdziwym stanowisku – robot i pozycjoner nie wchodzą sobie w drogę.

    1. robot w domu, pozycjoner kładzie rurę,
    2. robot podjeżdża (RRT) nad rowek, odsunięty o retract_mm wzdłuż osi palnika,
    3. dojazd liniowy, ściegi (ruch skoordynowany), przejścia między ściegami,
    4. odjazd liniowy, powrót robota do domu (RRT),
    5. pozycjoner podnosi rurę.
    Każdy odcinek jest sprawdzony na kolizje; gdy któregoś nie da się wykonać – wyjątek.
    """
    from weldpath.planner import RRTConnect, motion_valid, retract, shortcut

    flat = np.array([0.0, 0.0])
    passes = plan.passes
    tracks = []
    for k, weld_pass in enumerate(passes):
        track = weld_pass.positioner.copy()
        track[:, 1] -= 360.0 * k          # każdy ścieg zaczyna się w tym samym miejscu rury
        tracks.append(track)
    start, end = tracks[0][0], tracks[-1][-1]
    q_first, q_last = passes[0].joints[0], passes[-1].joints[-1]
    segments: list[Segment] = []

    def positioner_move(label, q, pa, pb):
        if not _combined_motion_ok(checker, positioner, q, pa, q, pb):
            raise RuntimeError(f"Kolizja: {label}")
        segments.append(Segment(label, "positioner", [q, q], [pa, pb]))

    def transit(label, qa, qb, pose):
        checker.set_part_pose(positioner.T_base_part(*pose))
        result = RRTConnect(checker, limits, seed=len(segments)).plan(qa, qb)
        if not result.success:
            raise RuntimeError(f"Brak bezkolizyjnej drogi: {label}")
        path = shortcut(result.path, checker)
        segments.append(Segment(label, "transit", path, [pose] * len(path)))

    def linear(label, qa, qb, pose):
        checker.set_part_pose(positioner.T_base_part(*pose))
        if not motion_valid(checker, qa, qb):
            raise RuntimeError(f"Kolizja: {label}")
        segments.append(Segment(label, "linear", [qa, qb], [pose, pose]))

    # 1. pozycjoner kładzie rurę, robot czeka w domu
    positioner_move("pozycjoner: kładzenie rury", home, flat, start)

    # 2-3. podjazd nad rowek i dojazd liniowy
    checker.set_part_pose(positioner.T_base_part(*start))
    q_approach = retract(robot, q_first, plan.config, torch, retract_mm)
    if q_approach is None or checker.in_collision(q_approach):
        raise RuntimeError("Punkt podjazdu nad rowkiem niedostępny")
    transit("robot: podjazd nad rowek", home, q_approach, start)
    linear("robot: dojazd do startu spoiny", q_approach, q_first, start)

    # ściegi i przejścia między nimi
    for k, (weld_pass, track) in enumerate(zip(passes, tracks)):
        if k > 0:
            qa, pa = passes[k - 1].joints[-1], tracks[k - 1][-1]
            qb, pb = weld_pass.joints[0], track[0]
            if not _combined_motion_ok(checker, positioner, qa, pa, qb, pb):
                raise RuntimeError(f"Kolizja: przejście do ściegu {weld_pass.name}")
            segments.append(Segment(f"przejście do ściegu: {weld_pass.name}", "linear",
                                    [qa, qb], [pa, pb]))
        segments.append(Segment(f"ścieg: {weld_pass.name}", "weld", list(weld_pass.joints),
                                list(track), weld_pass=weld_pass.name))

    # 4. odjazd liniowy i powrót do domu
    checker.set_part_pose(positioner.T_base_part(*end))
    q_depart = retract(robot, q_last, plan.config, torch, retract_mm)
    if q_depart is None or checker.in_collision(q_depart):
        raise RuntimeError("Punkt odjazdu nad rowkiem niedostępny")
    linear("robot: odjazd od spoiny", q_last, q_depart, end)
    transit("robot: powrót do domu", q_depart, home, end)

    # 5. pozycjoner podnosi rurę
    positioner_move("pozycjoner: podnoszenie rury", home, end, np.array([0.0, end[1]]))
    return segments