"""Krok 5b: planowanie bezkolizyjnego przejazdu w przestrzeni przegubów (RRT-Connect).

Przestrzeń konfiguracyjna (C-space): każdy punkt to 6 kątów robota. Przeszkody
w C-space to zbiory kątów, przy których robot w coś uderza – nie znamy ich kształtu,
możemy tylko zapytać checker "czy ten punkt koliduje?".

RRT-Connect (Kuffner & LaValle, 2000):
  - rosną DWA drzewa: od startu i od celu,
  - w każdej iteracji losujemy punkt, drzewo A robi krok w jego stronę (EXTEND),
  - drzewo B próbuje dociągnąć się do nowego węzła A tyle kroków, ile się da (CONNECT),
  - gdy się spotkają – mamy ścieżkę; potem drzewa zamieniają się rolami.
Na końcu ścieżkę wygładzamy (shortcutting), bo RRT daje zygzaki.
"""
import time
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from weldpath.collision import CollisionChecker
from weldpath.orientation import homogeneous
from weldpath.robot import OPWParams, forward_kinematics, inverse_kinematics
from weldpath.weld_job import JointLimits, Torch


def motion_valid(checker: CollisionChecker, qa: np.ndarray, qb: np.ndarray,
                 resolution: float = np.radians(2.0)) -> bool:
    """Czy ruch po prostej w przestrzeni przegubów qa -> qb jest bezkolizyjny?

    Sprawdzamy punkty co `resolution` (największa zmiana osi między próbkami).
    """
    n = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / resolution)))
    return all(not checker.in_collision(qa + (qb - qa) * i / n) for i in range(1, n + 1))


class _Tree:
    def __init__(self, root: np.ndarray):
        self.nodes = [root]
        self.parents = [-1]

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(np.array(self.nodes) - q, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, i: int) -> list[np.ndarray]:
        path = []
        while i != -1:
            path.append(self.nodes[i])
            i = self.parents[i]
        return path


@dataclass
class PlanResult:
    path: list[np.ndarray] | None   # lista konfiguracji od startu do celu
    iterations: int
    tree_nodes: int
    collision_checks: int
    seconds: float

    @property
    def success(self) -> bool:
        return self.path is not None


class RRTConnect:
    TRAPPED, ADVANCED, REACHED = range(3)

    def __init__(self, checker: CollisionChecker, limits: JointLimits,
                 step: float = np.radians(10.0), max_iterations: int = 5000,
                 seed: int | None = 0):
        self.checker = checker
        self.limits = limits
        self.step = step
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

    def _sample(self) -> np.ndarray:
        return self.rng.uniform(self.limits.lower, self.limits.upper)

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[int, int]:
        """Jeden krok drzewa w stronę celu. Zwraca (status, indeks nowego węzła)."""
        i = tree.nearest(target)
        q_near = tree.nodes[i]
        delta = target - q_near
        dist = np.linalg.norm(delta)
        q_new = target if dist <= self.step else q_near + delta * (self.step / dist)
        if not self.limits.ok(q_new) or not motion_valid(self.checker, q_near, q_new):
            return self.TRAPPED, -1
        j = tree.add(q_new, i)
        return (self.REACHED if dist <= self.step else self.ADVANCED), j

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[int, int]:
        """Rozciąga drzewo w stronę celu, dopóki się da."""
        while True:
            status, j = self._extend(tree, target)
            if status != self.ADVANCED:
                return status, j

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlanResult:
        t0 = time.perf_counter()
        checks0 = getattr(self.checker, "checks", 0)

        def result(path, it):
            return PlanResult(path, it, len(a.nodes) + len(b.nodes),
                              getattr(self.checker, "checks", 0) - checks0,
                              time.perf_counter() - t0)

        a, b = _Tree(np.asarray(q_start, float)), _Tree(np.asarray(q_goal, float))
        if self.checker.in_collision(a.nodes[0]) or self.checker.in_collision(b.nodes[0]):
            return result(None, 0)
        if motion_valid(self.checker, a.nodes[0], b.nodes[0]):  # prosta droga wolna?
            return result([a.nodes[0], b.nodes[0]], 0)

        a_is_start = True
        for it in range(1, self.max_iterations + 1):
            status, j = self._extend(a, self._sample())
            if status != self.TRAPPED:
                status_b, k = self._connect(b, a.nodes[j])
                if status_b == self.REACHED:
                    half_a = a.path_to_root(j)[::-1]   # korzeń A -> spotkanie
                    half_b = b.path_to_root(k)[1:]     # spotkanie -> korzeń B (bez dubla)
                    path = half_a + half_b
                    if not a_is_start:
                        path = path[::-1]
                    return result(path, it)
            a, b = b, a
            a_is_start = not a_is_start
        return result(None, self.max_iterations)


def shortcut(path: list[np.ndarray], checker: CollisionChecker, iterations: int = 200,
             seed: int | None = 0) -> list[np.ndarray]:
    """Wygładzanie: losujemy dwa punkty ścieżki i łączymy prostą, jeśli jest wolna."""
    rng = np.random.default_rng(seed)
    path = list(path)
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i > 1 and motion_valid(checker, path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def path_length(path: list[np.ndarray]) -> float:
    """Długość ścieżki w przestrzeni przegubów [rad]."""
    return float(sum(np.linalg.norm(b - a) for a, b in pairwise(path)))


def retract(robot: OPWParams, q: np.ndarray, config: int, torch: Torch,
            distance: float) -> np.ndarray | None:
    """Odsuwa palnik o `distance` mm wzdłuż jego osi (od detalu), w tej samej konfiguracji."""
    T_flange = forward_kinematics(robot, q)
    torch_axis = (T_flange @ torch.flange_to_tcp())[:3, 2]
    T_target = homogeneous(T_flange[:3, :3], T_flange[:3, 3] - distance * torch_axis)
    q_new = inverse_kinematics(robot, T_target)[config]
    return None if np.isnan(q_new).any() else q_new