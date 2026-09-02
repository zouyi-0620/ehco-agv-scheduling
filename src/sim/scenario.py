"""Scenario instances: tasks, AGV initial states, per-run RNG (SPEC.md sections 1, 8).

A Scenario is fully determined by (seed, layout params) and is shared across all
algorithms in a run (fairness requirement, SPEC section 8).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .warehouse import (Warehouse, build_grid, free_cells, storage_cells)


@dataclass
class Task:
    tid: int
    s_node: int          # start storage node id
    d_node: int          # destination storage node id
    urgent: bool
    omega: float         # urgency weight in (0, 1]
    m_load: float = C.M_LOAD


@dataclass
class AGVInit:
    agv_id: int
    start_node: int      # key-node id of initial position
    h0: float            # initial cumulative health


@dataclass
class Scenario:
    seed: int
    warehouse: Warehouse
    tasks: list[Task]
    agvs: list[AGVInit]
    # deadline reference distances: deadline uses allocation-moment position (D4)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)


def make_scenario(seed: int, n_agv: int = C.N_AGV, n_tasks: int = C.N_TASKS,
                  n_urgent: int = C.N_URGENT, wh: Warehouse | None = None,
                  h0_lo: float = C.H0_LO, h0_hi: float = C.H0_HI) -> Scenario:
    rng = np.random.default_rng(seed)

    # AGV initial positions: uniform over free cells that are not storage cells
    if wh is None:
        grid_free = free_cells(build_grid())
        storage_set = set(storage_cells())
    else:
        grid_free = free_cells(wh.grid)
        storage_set = set(wh.storage)
    candidates = [c for c in grid_free if c not in storage_set]
    idx = rng.choice(len(candidates), size=n_agv, replace=False)
    start_cells = [candidates[i] for i in idx]

    if wh is None or any(c not in wh.node_index for c in start_cells):
        wh = Warehouse.create(agv_start_cells=start_cells)

    h0 = rng.uniform(h0_lo, h0_hi, size=n_agv)
    agvs = [AGVInit(i, wh.node_index[c], float(h))
            for i, (c, h) in enumerate(zip(start_cells, h0))]

    # tasks: start/end drawn uniformly with replacement from K storage nodes;
    # zero-distance tasks rejected and re-drawn
    tasks = []
    n_reg = n_tasks - n_urgent
    urgent_ids = set(rng.choice(n_tasks, size=n_urgent, replace=False).tolist())
    for j in range(n_tasks):
        while True:
            s = int(rng.integers(C.N_STORAGE))
            d = int(rng.integers(C.N_STORAGE))
            if s != d:
                break
        urgent = j in urgent_ids
        omega = float(rng.uniform(C.OMEGA_URG_LO, C.OMEGA_URG_HI)) if urgent else 0.0
        tasks.append(Task(j, s, d, urgent, omega))
    return Scenario(seed=seed, warehouse=wh, tasks=tasks, agvs=agvs)
