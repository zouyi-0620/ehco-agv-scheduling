"""E4 dynamic-event simulation (SPEC.md sections 9, 10, 11; E13 timing).

Scenarios (SPEC section 10):
  A  sudden aisle congestion: at t_ev a channel loses capacity (aisle ~U{1..11},
     time ~U[100,600] s, reduction ~U[50,80]%)  -> LOCAL replanning (50 x 80,
     4,000 evals).  Congestion is modelled as a travel-time multiplier on every
     segment whose shortest-path cells cross the congested aisle; the local
     replanner sees the SAME multiplier (crossing node pairs cost D/(1-red)
     in the optimiser's distance matrix), so it re-sequences not-yet-started
     tasks away from the congested aisle.
  B  equipment fault warning: at t_inj an AGV's load rate jumps 0.4 -> 1.0
     (thermal fault) or the vibration channel jumps (vibration fault); the
     EWMA h_cum < H_SAFE = 0.6 then triggers the level-2 warning
     -> GLOBAL replanning (100 x 200, 20,000 evals) with the faulty AGV sent
     to the maintenance cell (49, 38).  Thermal faults derate the faulty AGV's
     speed protectively above 85 degC, so an open loop that keeps dispatching
     the faulty AGV pays a growing makespan penalty.  Randomised instances
     draw t_inj ~U[60,200] s: under the re-implemented thermal model
     (tau = 120 s + EWMA lag ~50 s) the warning trails the injection by
     ~110-180 s, so this window keeps most warnings inside the ~316 s mission
     (the SPEC-nominal instance keeps t = 200 s and mainly measures
     late-mission detection).
     Injection protocol: the load-rate jump must act on an operating vehicle,
     so the fault is injected into the AGV that is executing a loaded run at
     t_inj with the largest remaining workload (the SPEC-nominal AGV #3 is
     kept whenever it is still active at t_inj; per-seed AGV selection is part
     of the SPEC's "randomised AGV" requirement).  Health monitoring
     continues after the last task completes while a fault remains unwarned,
     so the detection statistics reflect the physical thermal response rather
     than an early termination of the run.
  C  3 urgent orders (omega = 0.9) inserted at t_ev ~U[300,500] s
     -> LOCAL replanning.  Open loop appends them to the least-loaded AGV's
     queue tail (no re-sequencing).

Protocol
--------
- Event randomisation (reviewer fix R3-MC4): >= 5 event instances per seed per
  scenario; instance 0 is the nominal SPEC configuration (used for the aligned
  time series of Fig 5c/d), instances 1..4 are drawn from the SPEC ranges.
- 30 seeds x 5 instances = 150 runs per scenario per strategy.
- Two strategies:
    * closed loop (EHCO):  initial AW-NSGA-II plan (run in-process) + event
                           + replanning when triggered
    * open loop (A*-GA):   initial A*-GA plan (run in-process) + event,
                           NO replanning (tasks stay fixed)
- Baseline f1 = same simulator with no event (no-event replay of the initial
  plan) -> performance degradation = (f1_event - f1_base)/f1_base * 100.
- Response time (E13): wall clock of the replanning call
  (time.perf_counter, host CPU, ms).
- RNG: event sampling uses default_rng(seed*1000 + inst*13 + 777); the
  replanning runs use default_rng(seed*100000 + inst*100 + k + 12345).  All
  numpy operations must run on the managed venv (numpy 2.5.1) so that the
  initial plans reproduce e1 exactly (anaconda numpy 2.3.5 draws a different
  PCG64 stream and is NOT usable).

Outputs (results/e4/):
  e4_raw.csv        per (scenario, strategy, seed, instance) rows
  e4_agg.csv        mean +- std per (scenario, strategy, metric)
  e4_table2.csv     Table-2-shaped aggregation (3 event rows)
  e4_ts_b.csv       scenario-B nominal-instance time series (Fig 5 c/d)
  e4_meta.json      configuration dump (instance lists, seeds, budgets)
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .. import constants as C
from ..algorithms import run_algorithm
from ..algorithms.common import EvalBudget, flc_weight_provider, nsga2_core
from ..dynamics import instant_health
from ..objectives import EvalConfig, evaluate_population
from ..scenario import AGVInit, Scenario, Task, make_scenario
from ..warehouse import cell_of

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e4")

SEEDS = list(range(1, 31))
N_INSTANCES = 5                    # >= 5 event instances (R3-MC4)
N_REPEATS = 30                     # 30 seeds as the repeat dimension
DT_SIM = 1.0                       # simulator step (s) -- also ts grid
TS_EVERY = 2.0                     # time-series sampling (s)
SIM_TMAX = 3000.0                  # safety cap on simulated time
DERATE_T0 = 85.0                   # protective derating onset (degC)
DERATE_T1 = 105.0                  # derating floor onset (degC)
T_FAULT_SS = 120.0                 # thermal-runaway steady state (degC) -- the
                                   # fault breaks cooling, so the motor
                                   # temperature rises above the safety limit
                                   # T_MAX=105 and the motor SoH goes negative,
                                   # which is what pushes EWMA h_cum below 0.6
                                   # (with AHP weights 0.4/0.35/0.25 a pure
                                   # 105 degC steady state bottoms out at ~0.62
                                   # and can NOT reach the warning line -- this
                                   # is documented in the manuscript).
VIB_FAULT_OFFSET = 45.0            # mm/s vibration-channel fault jump
                                   # (5 + 45 = 50 = V_CRIT -> mech SoH ~ 0)


# --------------------------------------------------------------------------
# path cells (scenario A congestion detection)
# --------------------------------------------------------------------------

def _neighbours(x: int, y: int, grid: np.ndarray):
    if x > 0 and grid[y, x - 1]:
        yield x - 1, y
    if x < C.GRID_W - 1 and grid[y, x + 1]:
        yield x + 1, y
    if y > 0 and grid[y - 1, x]:
        yield x, y - 1
    if y < C.GRID_H - 1 and grid[y + 1, x]:
        yield x, y + 1


def path_cells(src: tuple[int, int], dst: tuple[int, int],
               grid: np.ndarray) -> list[tuple[int, int]]:
    """Dijkstra with parent backtracking; cell sequence src .. dst inclusive."""
    dist = np.full((C.GRID_H, C.GRID_W), np.inf)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    dist[src[1], src[0]] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        if (x, y) == dst:
            break
        for nx, ny in _neighbours(x, y, grid):
            nd = d + 1.0
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                parent[(nx, ny)] = (x, y)
                heapq.heappush(pq, (nd, (nx, ny)))
    cells = [dst]
    cur = dst
    guard = 0
    while cur != src and guard < 10000:
        cur = parent[cur]
        cells.append(cur)
        guard += 1
    cells.reverse()
    return cells


def aisle_y_range(channel: int) -> tuple[int, int]:
    """Aisle `channel` (1..11) = the corridor between shelf rows, y in {3c,3c+1}."""
    return (3 * channel, 3 * channel + 1)


def _dijkstra_parents(src: tuple[int, int],
                      grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    """Single-source shortest-path Dijkstra (1 m cells) with parent pointers."""
    dist = np.full((C.GRID_H, C.GRID_W), np.inf)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    dist[src[1], src[0]] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for nx, ny in _neighbours(x, y, grid):
            nd = d + 1.0
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                parent[(nx, ny)] = (x, y)
                heapq.heappush(pq, (nd, (nx, ny)))
    return parent


_CROSS_CACHE: dict[int, np.ndarray] = {}


def crossing_matrix(wh, aisle: int) -> np.ndarray:
    """Boolean (n_nodes x n_nodes): does the shortest path src->dst pass
    through the y-band of `aisle` (endpoints included)?

    This mirrors EXACTLY the congestion penalty the simulator applies in
    `_speed` (path_cells uses the same pure-distance Dijkstra), so a planner
    fed the inflated matrix sees the same travel-time penalty the execution
    engine will impose.  Memoised per aisle -- the warehouse layout is
    deterministic across seeds."""
    if aisle in _CROSS_CACHE:
        return _CROSS_CACHE[aisle]
    y0, y1 = aisle_y_range(aisle)
    nodes = wh.nodes
    n = len(nodes)
    cross = np.zeros((n, n), dtype=bool)
    for si, src in enumerate(nodes):
        parent = _dijkstra_parents(src, wh.grid)
        src_hit = y0 <= src[1] <= y1
        for ti, dst in enumerate(nodes):
            if ti == si:
                continue
            hit = src_hit or (y0 <= dst[1] <= y1)
            cur = dst
            guard = 0
            while cur != src and not hit and guard < 10000:
                cur = parent[cur]
                if y0 <= cur[1] <= y1:
                    hit = True
                guard += 1
            cross[si, ti] = hit
    _CROSS_CACHE[aisle] = cross
    return cross


# --------------------------------------------------------------------------
# dynamic simulator
# --------------------------------------------------------------------------

@dataclass
class Seg:
    kind: str          # empty | loaded | idle
    dist: float        # m (0 for idle)
    dur: float         # nominal duration (s)
    src_node: int
    dst_node: int
    task_idx: int      # original task id (-1 for maintenance travel)
    remain_dist: float = 0.0   # m remaining (non-idle segments)
    remain_t: float = 0.0      # s remaining (idle segments)


class DynamicSimulator:
    """Discrete-time (1 s) execution engine over a task-assignment plan.

    AGVs serially execute their assigned tasks in queue order (FCFS), each task
    as empty run -> loaded run -> T_LOAD unloading.  Per 1 s step we advance
    positions, integrate the first-order thermal model, accumulate cycles/km,
    update the EWMA h_cum, and (optionally) fire scenario events.
    """

    def __init__(self, sc: Scenario, assign: np.ndarray, cfg: EvalConfig | None = None,
                 use_flc: bool = True):
        """use_flc=False disables the FLC weight provider inside replanning
        (fixed-weight 4-objective NSGA-II) -- the dynamic-ablation variant
        -FLC (e4_ablation.py).  Default True keeps the published EHCO path.

        Extension hooks (behaviour-preserving, defaults reproduce the
        published EHCO runs bit-exactly):
        * replan_obj_idx : objective subspace of the replanning optimiser
          (default the full 4 objectives; the health-agnostic C17 baseline
          overrides it with (0, 1, 2)).
        * fault_t_ss : steady-state temperature of the injected thermal
          fault (default T_FAULT_SS = 120 degC; the randomised-severity C18
          experiment overrides it per run)."""
        self.sc = sc
        self.assign = np.asarray(assign, dtype=int)
        self.use_flc = use_flc
        self.replan_obj_idx = (0, 1, 2, 3)
        self.fault_t_ss = T_FAULT_SS
        self.N = len(sc.agvs)
        self.D = sc.warehouse.dist_improved
        self.grid = sc.warehouse.grid
        self.cfg = cfg or EvalConfig()
        self.M_total = len(sc.tasks)
        self._cell_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._build_queues(sc.tasks, self.assign)
        self._replanned_global = False
        self.fault_agv: int | None = None
        self.fault_type: str | None = None
        # C20/C22 extension hook (behaviour-preserving default): when False the
        # local replanner keeps the STATIC distance tables (no congestion
        # inflation, no incumbent guard under the congested model) -- the
        # -costAdapt ablation (e_c22.py).
        self.congestion_aware_replan = True
        # C20: urgent tasks inserted mid-run are registered here at insertion
        # time so that LATER replanning calls (e.g. the compound fault-warning
        # global replan) see them in the remaining-task set and the frozen-map.
        self.extra_tasks_all: list[Task] = []

    # ---- queue construction -------------------------------------------
    def _build_queues(self, tasks, assign):
        D = self.D
        pos = [a.start_node for a in self.sc.agvs]
        self.seg_queues: list[list[Seg]] = [[] for _ in range(self.N)]
        for j, i in enumerate(assign):
            t = tasks[j]
            i = int(i)
            d_empty = D[pos[i], t.s_node]
            d_loaded = D[t.s_node, t.d_node]
            if d_empty > 1e-9:
                self.seg_queues[i].append(Seg("empty", float(d_empty), float(d_empty) / C.V_AVG,
                                              int(pos[i]), int(t.s_node), t.tid))
            self.seg_queues[i].append(Seg("loaded", float(d_loaded), float(d_loaded) / C.V_AVG,
                                          int(t.s_node), int(t.d_node), t.tid))
            self.seg_queues[i].append(Seg("idle", 0.0, C.T_LOAD, int(t.d_node), int(t.d_node), t.tid))
            pos[i] = t.d_node
        self.q: list[list[Seg]] = [list(q) for q in self.seg_queues]

    def _tail_node(self, i: int) -> int:
        if self.cur_seg[i] is not None:
            return int(self.cur_seg[i].dst_node)
        return int(self.pos[i])

    # ---- scenario-B fault targeting -------------------------------------
    def _rem_work(self, i: int) -> float:
        """Remaining workload of AGV i (m for travel, s for idle, mixed ok)."""
        w = 0.0
        s = self.cur_seg[i]
        if s is not None:
            w += s.remain_dist if s.kind != "idle" else s.remain_t
        for seg in self.q[i]:
            w += seg.remain_dist if seg.kind != "idle" else seg.remain_t
        return w

    def _pick_fault_target(self, rng: np.random.Generator,
                           preferred: int | None) -> int:
        """Choose the AGV the scenario-B fault is injected into.

        The load-rate jump 0.4 -> 1.0 is only physical for an operating
        vehicle, so the target is an AGV that is still working at t_inj; to
        keep the thermal response inside the simulated horizon we prefer one
        that is currently executing a LOADED run (hot, T near its steady
        state), breaking ties by the largest remaining workload.  The
        SPEC-nominal AGV (#3) is kept whenever it is still active.  If the
        whole fleet has already finished, we draw uniformly (the run then
        extends into post-shift monitoring and the warning still fires)."""
        if preferred is not None and self._rem_work(preferred) > 1e-9:
            return int(preferred)
        loaded = [i for i in range(self.N)
                  if self.cur_seg[i] is not None
                  and self.cur_seg[i].kind == "loaded"
                  and self._rem_work(i) > 1e-9]
        pool = loaded or [i for i in range(self.N) if self._rem_work(i) > 1e-9]
        if not pool:
            return int(rng.integers(self.N))
        return max(pool, key=self._rem_work)

    # ---- state reset ----------------------------------------------------
    def reset(self):
        self.pos = [a.start_node for a in self.sc.agvs]
        self.h = np.array([a.h0 for a in self.sc.agvs], dtype=float)
        self.T = np.full(self.N, C.T_AMB)
        self.cycles = np.zeros(self.N)
        self.km = np.zeros(self.N)
        self.t = 0.0
        self.done: set[int] = set()
        self.t_done: dict[int, float] = {}
        self.cur_seg: list[Seg | None] = [None] * self.N
        self.q = [list(q) for q in self.seg_queues]
        self.fault = [False] * self.N
        self.fault_vib = [0.0] * self.N
        self.congested_aisle: int | None = None
        self.congest_red = 0.0
        self.congest_active = False
        self.warned = [False] * self.N
        self.replan_walls_ms: list[float] = []
        self.warn_t: float | None = None
        self.warn_agv: int | None = None
        self.maint_enter: float | None = None
        self.hist_t: list[float] = []
        self.hist_frac: list[float] = []
        self.hist_h: list[np.ndarray] = []
        self._replanned_global = False
        self._urgent_done = False
        self.fault_agv = None
        self.fault_type = None
        self.fault_fallback = 0
        self.event_fired = False
        self.replan_info: list[dict] = []
        self.min_h = 1.0                 # C20 health-violation tracking
        self.extra_tasks_all = []        # C20: cleared on every reset

    # ---- per-step mechanics --------------------------------------------
    def _speed(self, i: int, seg: Seg) -> float:
        if seg.kind == "idle":
            return 0.0
        v = C.V_AVG
        if self.congest_active and seg.kind != "idle":
            key = (int(seg.src_node), int(seg.dst_node))
            if key not in self._cell_cache:
                a, b = cell_of(seg.src_node, self.sc.warehouse), cell_of(seg.dst_node, self.sc.warehouse)
                self._cell_cache[key] = path_cells(a, b, self.grid)
            y0, y1 = aisle_y_range(self.congested_aisle)
            if any(y0 <= y <= y1 for _, y in self._cell_cache[key]):
                v *= (1.0 - self.congest_red)
        if self.fault[i] and self.T[i] > DERATE_T0:
            # protective derating applies to both thermal and vibration faults
            # (both drive the motor temperature via the 0.4 -> 1.0 load-rate
            # injection; vibration additionally collapses the mech channel)
            v *= max(0.2, 1.0 - 0.8 * (self.T[i] - DERATE_T0) / (DERATE_T1 - DERATE_T0))
        return v

    def _health_step(self, dt: float):
        beta_sub = 1.0 - (1.0 - C.EWMA_BETA) ** (dt / C.DT)
        for i in range(self.N):
            seg = self.cur_seg[i]
            in_maint = seg is not None and seg.task_idx == -1
            if self.fault[i] and not in_maint:
                # fault load-rate 0.4 -> 1.0 (SPEC 10) plus thermal runaway:
                # cooling failure drives the motor past T_MAX towards the
                # fault steady state (default 120 degC)
                lr = C.LOAD_RATE_LOADED
                t_ss = self.fault_t_ss
            elif seg is None or seg.kind == "idle":
                lr = C.LOAD_RATE_IDLE
                t_ss = C.T_AMB + C.K_THERMAL * lr
            elif seg.kind == "empty":
                lr = C.LOAD_RATE_EMPTY
                t_ss = C.T_AMB + C.K_THERMAL * lr
            else:
                lr = C.LOAD_RATE_LOADED
                t_ss = C.T_AMB + C.K_THERMAL * lr
            tau = C.TAU_THERMAL if t_ss > self.T[i] else C.TAU_COOL
            self.T[i] = t_ss + (self.T[i] - t_ss) * np.exp(-dt / tau)
            v_rms = C.V0_VIB + C.K_WEAR * self.km[i] + self.fault_vib[i]
            h_inst = instant_health(self.cycles[i], self.T[i], v_rms)
            self.h[i] = beta_sub * h_inst + (1.0 - beta_sub) * self.h[i]
            if not self.warned[i] and self.h[i] < C.H_SAFE:
                self.warned[i] = True
                if self.warn_t is None:
                    self.warn_t = self.t
                    self.warn_agv = i
        self.min_h = min(self.min_h, float(self.h.min()))

    def step(self, dt: float):
        self.t += dt
        for i in range(self.N):
            if self.cur_seg[i] is None and self.q[i]:
                seg = self.q[i].pop(0)
                seg.remain_dist = seg.dist
                seg.remain_t = seg.dur
                self.cur_seg[i] = seg
            seg = self.cur_seg[i]
            if seg is None:
                continue
            if seg.kind == "idle":
                seg.remain_t -= dt
                finished = seg.remain_t <= 1e-9
            else:
                v = self._speed(i, seg)
                d = v * dt
                seg.remain_dist -= d
                self.cycles[i] += d / C.L_CYCLE
                self.km[i] += d / 1000.0
                finished = seg.remain_dist <= 1e-9
            if finished:
                self.cur_seg[i] = None
                if seg.kind != "idle":
                    self.pos[i] = seg.dst_node
                    if seg.task_idx == -1:
                        self.maint_enter = self.t
                if seg.kind == "idle":
                    self.done.add(seg.task_idx)
                    self.t_done[seg.task_idx] = self.t
        self._health_step(dt)

    # ---- replanning ------------------------------------------------------
    def _replan_scenario(self, extra_tasks: list[Task] | None = None,
                         exclude_agv: int | None = None) -> Scenario:
        frozen: dict[int, int] = {}
        for i in range(self.N):
            if self.cur_seg[i] is not None:
                frozen[i] = self.cur_seg[i].task_idx
        frozen_set = set(frozen.values())
        all_tasks = list(self.sc.tasks) + list(self.extra_tasks_all)
        tmap = {t.tid: t for t in all_tasks}
        rem = [t for t in all_tasks
               if t.tid not in self.done and t.tid not in frozen_set]
        if extra_tasks:
            rem = [t for t in rem if t.tid not in {x.tid for x in extra_tasks}]
            rem = rem + list(extra_tasks)
        agvs = []
        for i in range(self.N):
            if exclude_agv is not None and i == exclude_agv:
                continue
            if i in frozen:
                node = tmap[frozen[i]].d_node
            else:
                node = self.pos[i]
            agvs.append(AGVInit(i, node, float(self.h[i])))
        return Scenario(seed=self.sc.seed, warehouse=self.sc.warehouse,
                        tasks=rem, agvs=agvs)

    def _replan(self, sc_p: Scenario, rng: np.random.Generator,
                pop_size: int, budget: int, name: str,
                incumbent: np.ndarray | None = None):
        if len(sc_p.tasks) == 0 or len(sc_p.agvs) == 0:
            # nothing left to replan (e.g. the fault warning fired after the
            # last task completed) -- record and skip the optimisation call
            self.replan_info.append({"name": name, "budget": 0, "pop": pop_size,
                                     "n_tasks": len(sc_p.tasks),
                                     "n_agv": len(sc_p.agvs),
                                     "wall_ms": 0.0, "evals": 0,
                                     "skipped": True})
            return 0.0
        t0 = time.perf_counter()
        res = nsga2_core(
            sc_p, seed=int(rng.integers(1, 2 ** 31)),
            obj_idx=self.replan_obj_idx,
            cfg=self.cfg, budget=EvalBudget(budget), name=name,
            weight_fn=flc_weight_provider if self.use_flc else None,
            use_crowding=True, pop_size=pop_size)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self.replan_walls_ms.append(wall_ms)
        info = {"name": name, "budget": budget, "pop": pop_size,
                "n_tasks": len(sc_p.tasks), "n_agv": len(sc_p.agvs),
                "wall_ms": wall_ms, "evals": res.evals}
        applied = True
        if incumbent is not None:
            # incumbent guard (standard rolling-horizon practice): adopt the
            # re-optimised plan only if its estimated makespan under the SAME
            # (congestion-adjusted) model does not exceed the incumbent's.
            cur = evaluate_population(incumbent.reshape(1, -1), sc_p, self.cfg)
            f1_cur = float(cur["f1"][0])
            f1_new = float(res.best_F[0])
            applied = bool(f1_new <= f1_cur)
            info.update({"incumbent_f1": f1_cur, "new_f1": f1_new,
                         "applied": applied})
        self.replan_info.append(info)
        if applied:
            self._apply_new_assign(res.best_assign, sc_p)
        return wall_ms

    def _apply_new_assign(self, assign: np.ndarray, sc_p: Scenario):
        D = self.D
        tmap = {t.tid: t for t in list(self.sc.tasks)
                + list(self.extra_tasks_all)}
        new_q: list[list[Seg]] = [[] for _ in range(self.N)]
        # keep the remaining segments of every frozen (in-progress) task at the
        # head of its AGV queue; the replanning only re-assigns not-started tasks
        for i in range(self.N):
            seg = self.cur_seg[i]
            if seg is not None and seg.task_idx >= 0:
                j = seg.task_idx
                t = tmap[j]
                if seg.kind == "empty":
                    d_loaded = D[t.s_node, t.d_node]
                    new_q[i].append(Seg("loaded", float(d_loaded),
                                        float(d_loaded) / C.V_AVG,
                                        int(t.s_node), int(t.d_node), j))
                    new_q[i].append(Seg("idle", 0.0, C.T_LOAD,
                                        int(t.d_node), int(t.d_node), j))
                elif seg.kind == "loaded":
                    new_q[i].append(Seg("idle", 0.0, C.T_LOAD,
                                        int(t.d_node), int(t.d_node), j))
        # the new empty runs start from the true tail of each AGV: the last
        # queued segment's destination (frozen task tail) > current segment's
        # destination > current position
        tail = {}
        for i in range(self.N):
            if new_q[i]:
                tail[i] = int(new_q[i][-1].dst_node)
            elif self.cur_seg[i] is not None:
                tail[i] = int(self.cur_seg[i].dst_node)
            else:
                tail[i] = int(self.pos[i])
        for k, a in enumerate(assign):
            i = sc_p.agvs[int(a)].agv_id
            t = sc_p.tasks[k]
            d_empty = D[tail[i], t.s_node]
            d_loaded = D[t.s_node, t.d_node]
            if d_empty > 1e-9:
                new_q[i].append(Seg("empty", float(d_empty), float(d_empty) / C.V_AVG,
                                    int(tail[i]), int(t.s_node), t.tid))
            new_q[i].append(Seg("loaded", float(d_loaded), float(d_loaded) / C.V_AVG,
                                int(t.s_node), int(t.d_node), t.tid))
            new_q[i].append(Seg("idle", 0.0, C.T_LOAD, int(t.d_node), int(t.d_node), t.tid))
            tail[i] = t.d_node
        self.q = new_q

    def _congested_D(self) -> np.ndarray:
        """Planner-side distance matrix under the active congestion: node pairs
        whose shortest path crosses the congested aisle band cost
        D / (1 - reduction) -- exactly the travel-time multiplier the
        execution engine applies in `_speed`.  Used ONLY inside the
        replanning optimisation; executed segments keep physical distances
        and the simulator applies the derate itself."""
        cross = crossing_matrix(self.sc.warehouse, int(self.congested_aisle))
        D = np.array(self.D, dtype=float, copy=True)
        D[cross] *= 1.0 / (1.0 - self.congest_red)
        return D

    def _local_replan(self, rng: np.random.Generator, extra_tasks: list[Task] | None = None):
        sc_p = self._replan_scenario(extra_tasks=extra_tasks)
        if not sc_p.tasks:
            return 0.0
        incumbent = None
        if (self.congestion_aware_replan and self.congest_active
                and self.congested_aisle is not None):
            # congestion-aware replanning: hand the optimiser the inflated
            # matrix so it re-sequences tasks away from the congested aisle;
            # guard the adoption with the incumbent plan's estimate
            wh2 = copy.copy(self.sc.warehouse)
            wh2.dist_improved = self._congested_D()
            wh2.dist_standard = wh2.dist_improved
            sc_p.warehouse = wh2
            if not extra_tasks and all(t.tid < len(self.assign)
                                       for t in sc_p.tasks):
                # current assignment of the not-yet-started tasks (values are
                # sc_p-local AGV indices; local replan keeps the full fleet).
                # Skipped when mid-run-inserted urgent tasks (tid >= M) are
                # present: they have no initial-plan assignment.
                incumbent = np.array([int(self.assign[t.tid])
                                      for t in sc_p.tasks], dtype=int)
        return self._replan(sc_p, rng, C.NP_LOCAL, C.NP_LOCAL * C.GMAX_LOCAL,
                            "AW-local", incumbent=incumbent)

    def _global_replan(self, rng: np.random.Generator, exclude_agv: int):
        sc_p = self._replan_scenario(exclude_agv=exclude_agv)
        wall = self._replan(sc_p, rng, C.NP_GLOBAL, C.NP_GLOBAL * C.GMAX_GLOBAL,
                            "AW-global")
        # faulty AGV: after the frozen in-progress segment finishes, travel to
        # the maintenance cell and stop (idle cooling recovers h_cum).
        # The frozen task's remaining segments are already at the head of
        # self.q[exclude_agv] (kept by _apply_new_assign); append the
        # maintenance travel after them.
        m_node = self.sc.warehouse.node_index[C.MAINTENANCE_CELL]
        tail = self._tail_node(exclude_agv)
        for s in self.q[exclude_agv]:
            tail = s.dst_node
        d = self.D[tail, m_node]
        if d > 1e-9:
            self.q[exclude_agv].append(Seg("empty", float(d), float(d) / C.V_AVG,
                                           int(tail), int(m_node), -1))
        return wall

    # ---- open-loop (no replanning) helpers -------------------------------
    def _append_urgent(self, tasks: list[Task]):
        """Open loop: append urgent orders to the least-loaded AGV's tail."""
        for t in tasks:
            loads = []
            for i in range(self.N):
                rem = sum(s.remain_dist if s.kind != "idle" else s.remain_t
                          for s in self.q[i])
                if self.cur_seg[i] is not None:
                    cs = self.cur_seg[i]
                    rem += cs.remain_dist if cs.kind != "idle" else cs.remain_t
                loads.append(rem)
            i = int(np.argmin(loads))
            tail = self._tail_node(i)
            # if this AGV has a tail position, chain from the last queued dst
            for s in self.q[i]:
                tail = s.dst_node
            d_empty = self.D[tail, t.s_node]
            d_loaded = self.D[t.s_node, t.d_node]
            if d_empty > 1e-9:
                self.q[i].append(Seg("empty", float(d_empty), float(d_empty) / C.V_AVG,
                                     int(tail), int(t.s_node), t.tid))
            self.q[i].append(Seg("loaded", float(d_loaded), float(d_loaded) / C.V_AVG,
                                 int(t.s_node), int(t.d_node), t.tid))
            self.q[i].append(Seg("idle", 0.0, C.T_LOAD, int(t.d_node), int(t.d_node), t.tid))

    # ---- master loop ------------------------------------------------------
    def run(self, event: dict | None = None, closed_loop: bool = True,
            ts_out: bool = False) -> dict:
        self.reset()
        ev = event or {}
        kind = ev.get("kind")
        rng = ev.get("rng", np.random.default_rng(0))
        extra: list[Task] = []
        if kind in ("urgent", "compound"):
            for k in range(ev.get("n_orders", 3)):
                r2 = np.random.default_rng(int(rng.integers(1, 2 ** 31)))
                while True:
                    s = int(r2.integers(C.N_STORAGE))
                    d = int(r2.integers(C.N_STORAGE))
                    if s != d:
                        break
                extra.append(Task(tid=100 + k, s_node=s, d_node=d,
                                  urgent=True, omega=0.9, m_load=C.M_LOAD))
            self.M_total = len(self.sc.tasks) + len(extra)

        last_ts = -1.0

        def _still_monitoring() -> bool:
            """Run until all tasks finish; keep monitoring while an injected
            fault has not yet tripped the warning (post-shift cooling-failure
            response), capped by SIM_TMAX."""
            if len(self.done) < self.M_total:
                return True
            return (self.fault_agv is not None
                    and not self.warned[self.fault_agv])

        while self.t < SIM_TMAX and _still_monitoring():
            t = self.t
            # ---- event dispatch ----------------------------------------
            if kind == "congestion" and not self.congest_active and t >= ev["t_ev"]:
                self.congested_aisle = int(ev["aisle"])
                self.congest_red = float(ev["reduction"])
                self.congest_active = True
                self.event_fired = True
                if closed_loop:
                    self._local_replan(rng)
            if kind == "compound" and not self.congest_active and t >= ev["t_c"]:
                # C20 compound disturbance: cascading congestion -> thermal
                # fault -> urgent orders, each firing its own trigger
                self.congested_aisle = int(ev["aisle"])
                self.congest_red = float(ev["reduction"])
                self.congest_active = True
                self.event_fired = True
                if closed_loop:
                    self._local_replan(rng)
            if kind == "fault" and self.fault_agv is None and t >= ev["t_inj"]:
                target = self._pick_fault_target(rng, ev.get("agv"))
                self.fault_fallback = int(ev.get("agv") is not None
                                          and target != int(ev["agv"]))
                self.fault_agv = target
                self.fault_type = ev["fault_type"]
                self.fault[target] = True
                self.event_fired = True
                if self.fault_type == "vibration":
                    self.fault_vib[target] = VIB_FAULT_OFFSET
            if kind == "compound" and self.fault_agv is None and t >= ev["t_f"]:
                # adaptive targeting: the most-loaded AGV in a loaded run
                target = self._pick_fault_target(rng, None)
                self.fault_fallback = 0
                self.fault_agv = target
                self.fault_type = ev.get("fault_type", "thermal")
                self.fault[target] = True
                self.event_fired = True
                if self.fault_type == "vibration":
                    self.fault_vib[target] = VIB_FAULT_OFFSET
            if (kind in ("fault", "compound") and self.fault_agv is not None
                    and self.warned[self.fault_agv] and not self._replanned_global):
                self._replanned_global = True
                if closed_loop:
                    self._global_replan(rng, exclude_agv=self.fault_agv)
            if kind == "urgent" and not self._urgent_done and t >= ev["t_ev"]:
                self._urgent_done = True
                self.event_fired = True
                if closed_loop:
                    # register the inserted orders so later replanning calls
                    # (none in the single-event scenario) would also see them
                    self.extra_tasks_all = list(extra)
                    self._local_replan(rng)
                else:
                    self._append_urgent(extra)
            if kind == "compound" and not self._urgent_done and t >= ev["t_u"]:
                self._urgent_done = True
                self.event_fired = True
                if closed_loop:
                    self.extra_tasks_all = list(extra)
                    self._local_replan(rng)
                else:
                    self._append_urgent(extra)
            # ---- advance ------------------------------------------------
            self.step(DT_SIM)
            if ts_out and self.t >= last_ts + TS_EVERY:
                last_ts = self.t
                self.hist_t.append(self.t)
                self.hist_frac.append(len(self.done) / self.M_total * 100.0)
                self.hist_h.append(self.h.copy())

        makespan = max(self.t_done.values()) if self.t_done else self.t
        timeout = len(self.done) < self.M_total
        return {
            "makespan": float(makespan),
            "timeout": timeout,
            "n_done": len(self.done),
            "n_total": self.M_total,
            "replan_walls_ms": list(self.replan_walls_ms),
            "warn_t": self.warn_t,
            "warn_agv": self.warn_agv,
            "warned_any": any(self.warned),
            "n_warned": int(sum(self.warned)),
            "warned_indices": [int(i) for i, w in enumerate(self.warned) if w],
            "maint_enter": self.maint_enter,
            "fault_agv": self.fault_agv,
            "fault_fallback": self.fault_fallback,
            "event_fired": self.event_fired,
            "min_h": float(self.min_h),
            "n_replans": int(sum(1 for r in self.replan_info
                                 if not r.get("skipped", False))),
            "hist": (self.hist_t, self.hist_frac, self.hist_h),
            "replan_info": list(self.replan_info),
        }


# --------------------------------------------------------------------------
# event instances
# --------------------------------------------------------------------------

def make_instances(kind: str) -> list[dict]:
    """5 instances per scenario; inst 0 = nominal SPEC configuration."""
    out = []
    for k in range(N_INSTANCES):
        rng = np.random.default_rng(777 + k * 131)
        if kind == "congestion":
            if k == 0:
                out.append({"kind": "congestion", "t_ev": 150.0, "aisle": 6,
                            "reduction": 0.70})
            else:
                # t_ev ~U[100,280] s (narrowed from the SPEC's U[100,600]
                # drawing range: the re-implemented mission completes at
                # ~316 s, so later onsets never fire inside the run)
                out.append({"kind": "congestion",
                            "t_ev": float(rng.uniform(100.0, 280.0)),
                            "aisle": int(rng.integers(1, 12)),
                            "reduction": float(rng.uniform(0.50, 0.80))})
        elif kind == "fault":
            if k == 0:
                # SPEC nominal: t=200 s, AGV #3, thermal
                out.append({"kind": "fault", "t_inj": 200.0, "agv": 3,
                            "fault_type": "thermal", "mode": "nominal"})
            else:
                # agv=None: the fault hits the most-loaded AGV currently in a
                # loaded run at t_inj (per-seed; SPEC randomises the AGV and a
                # load-rate jump on an idle vehicle is vacuous).
                # Injection window U[60,200] s (narrowed from the SPEC's
                # U[100,600] drawing range): under the re-implemented thermal
                # model (tau = 120 s + EWMA lag) the warning trails the
                # injection by ~110-180 s, so later injections fire after the
                # ~316 s mission and only measure detection, not response.
                out.append({"kind": "fault",
                            "t_inj": float(rng.uniform(60.0, 200.0)),
                            "agv": None,
                            "fault_type": "thermal" if rng.random() < 0.5
                            else "vibration",
                            "mode": "adaptive"})
        elif kind == "compound":
            # C20: cascading compound disturbance -- congestion, then an
            # equipment fault, then urgent orders.  Instance 0 is the nominal
            # timeline (t_c = 150 s, t_f = 180 s, t_u = 220 s, aisle 6, 70%
            # reduction, thermal fault); instances 1..4 randomise the onset
            # times (strict cascade ordering t_c < t_f < t_u, gaps 20-60 s),
            # the aisle, the severity and the fault type.
            if k == 0:
                out.append({"kind": "compound", "t_c": 150.0, "aisle": 6,
                            "reduction": 0.70, "t_f": 180.0,
                            "fault_type": "thermal", "t_u": 220.0,
                            "n_orders": 3})
            else:
                t_c = float(rng.uniform(100.0, 180.0))
                t_f = float(t_c + rng.uniform(20.0, 60.0))
                t_u = float(t_f + rng.uniform(20.0, 60.0))
                out.append({"kind": "compound", "t_c": t_c,
                            "aisle": int(rng.integers(1, 12)),
                            "reduction": float(rng.uniform(0.50, 0.80)),
                            "t_f": t_f,
                            "fault_type": "thermal" if rng.random() < 0.5
                            else "vibration",
                            "t_u": t_u, "n_orders": 3})
        else:  # urgent
            if k == 0:
                out.append({"kind": "urgent", "t_ev": 400.0, "n_orders": 3})
            else:
                out.append({"kind": "urgent",
                            "t_ev": float(rng.uniform(300.0, 500.0)),
                            "n_orders": 3})
    return out


# --------------------------------------------------------------------------
# per-scenario runner
# --------------------------------------------------------------------------

def run_scenario(kind: str, seed: int, inst: dict, closed_loop: bool,
                 plans: dict) -> dict:
    """Run one (seed, instance, strategy) configuration."""
    sc = plans["scenarios"][seed]
    assign = plans["aw"][seed] if closed_loop else plans["astar"][seed]
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)

    sim = DynamicSimulator(sc, assign)
    # baseline: no-event replay of the SAME initial plan
    base = sim.run(event=None, closed_loop=False, ts_out=False)
    # event replay (nominal instance also emits the aligned time series)
    ts_out = (kind == "fault" and closed_loop and inst.get("_nominal", False)) \
        or (kind == "fault" and not closed_loop and inst.get("_nominal", False))
    out = sim.run(event=ev, closed_loop=closed_loop, ts_out=ts_out)

    f1_base = base["makespan"]
    f1_ev = out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    row = {
        "scenario": kind, "strategy": "EHCO" if closed_loop else "A*-GA",
        "seed": seed, "instance": inst.get("_idx", 0),
        "t_ev": ev.get("t_ev") if ev.get("kind") != "fault" else ev.get("t_inj"),
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "timeout": int(out["timeout"]),
        "response_ms": out["replan_walls_ms"][0] if out["replan_walls_ms"] else "",
        "replan_evals": out["replan_info"][0]["evals"] if out["replan_info"] else "",
        "warn_t": out["warn_t"], "warn_agv": out["warn_agv"],
        "n_warned": out["n_warned"], "n_agv": C.N_AGV,
        "event_fired": int(out["event_fired"]),
        "replan_applied": int(bool(out["replan_info"])
                              and not out["replan_info"][0].get("skipped", False)
                              and out["replan_info"][0].get("applied", True)),
        "maint_enter": out["maint_enter"] if kind == "fault" else "",
    }
    if kind == "fault":
        row["fault_agv"] = out["fault_agv"]
        row["fault_type"] = ev["fault_type"]
        row["fault_fallback"] = out["fault_fallback"]
        row["detected"] = int(out["warn_agv"] == out["fault_agv"])
        row["false_pos"] = int(sum(1 for i in out["warned_indices"]
                                   if i != out["fault_agv"]))
    return row, (sim if ts_out else None)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _mean_std(vals) -> tuple[float, float]:
    vals = np.asarray([v for v in vals if v is not None and v != "" and v == v],
                      dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def aggregate(raw: list[dict]) -> list[dict]:
    agg: list[dict] = []
    by = defaultdict(list)
    for r in raw:
        by[(r["scenario"], r["strategy"])].append(r)
    for (scenario, strategy), rows in by.items():
        degs = [r["deg_pct"] for r in rows]
        m, s = _mean_std(degs)
        agg.append({"scenario": scenario, "strategy": strategy,
                    "metric": "deg_pct", "mean": m, "std": s, "n": len(rows)})
        resp = [r["response_ms"] for r in rows]
        if any(v not in ("", None) for v in resp):
            m2, s2 = _mean_std(resp)
            agg.append({"scenario": scenario, "strategy": strategy,
                        "metric": "response_ms", "mean": m2, "std": s2, "n": len(rows)})
        fired = [r.get("event_fired", 1) for r in rows]
        agg.append({"scenario": scenario, "strategy": strategy,
                    "metric": "event_fired_rate", "mean": float(np.mean(fired)),
                    "std": 0.0, "n": len(rows)})
        appl = [r.get("replan_applied", 0) for r in rows]
        agg.append({"scenario": scenario, "strategy": strategy,
                    "metric": "replan_applied_rate", "mean": float(np.mean(appl)),
                    "std": 0.0, "n": len(rows)})
        if scenario == "fault":
            # rows reused from e4_raw.csv (published) carry no per-run
            # detection columns; detection metrics are platform-level
            # (identical across strategies) and are only aggregated over
            # rows that carry them.
            det_rows = [r for r in rows if r.get("detected") is not None]
            if det_rows:
                det = np.array([r["detected"] for r in det_rows])
                fp = np.array([r["false_pos"] for r in det_rows])
                det_rate = float(det.mean())
                far = float(fp.mean()) / (C.N_AGV - 1)
                tp = int(det.sum()); fn = int((1 - det).sum()); fpp = int(fp.sum())
                prec = tp / (tp + fpp) if (tp + fpp) else float("nan")
                rec = tp / (tp + fn) if (tp + fn) else float("nan")
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
                agg.append({"scenario": scenario, "strategy": strategy,
                            "metric": "detection_rate", "mean": det_rate, "std": 0.0,
                            "n": len(det_rows)})
                agg.append({"scenario": scenario, "strategy": strategy,
                            "metric": "false_alarm_rate", "mean": far, "std": 0.0,
                            "n": len(det_rows)})
                agg.append({"scenario": scenario, "strategy": strategy,
                            "metric": "f1_score", "mean": f1, "std": 0.0,
                            "n": len(det_rows)})
                wt = [r["warn_t"] for r in det_rows if r.get("warn_t") is not None]
                if wt:
                    agg.append({"scenario": scenario, "strategy": strategy,
                                "metric": "warn_t_mean",
                                "mean": _mean_std(wt)[0],
                                "std": _mean_std(wt)[1], "n": len(wt)})
    return agg


def table2_rows(agg: list[dict]) -> list[dict]:
    """Table-2-shaped rows: one per event scenario."""
    def get(sc, strategy, metric):
        for r in agg:
            if r["scenario"] == sc and r["strategy"] == strategy and r["metric"] == metric:
                return r["mean"], r["std"]
        return float("nan"), float("nan")

    rows = []
    for sc, label, rtype in (("congestion", "A: sudden aisle congestion",
                              "local"), ("fault", "B: equipment fault warning",
                                          "global"), ("urgent", "C: urgent order insertion",
                                                      "local")):
        d_e, s_e = get(sc, "EHCO", "deg_pct")
        d_a, s_a = get(sc, "A*-GA", "deg_pct")
        r_m, r_s = get(sc, "EHCO", "response_ms")
        rows.append({
            "event": label, "replan_type": rtype,
            "triggers_per_run": 1,
            "trigger_rate": get(sc, "EHCO", "event_fired_rate")[0],
            "replan_applied_rate": get(sc, "EHCO", "replan_applied_rate")[0],
            "response_ms_mean": r_m, "response_ms_std": r_s,
            "deg_ehco_mean": d_e, "deg_ehco_std": s_e,
            "deg_astar_mean": d_a, "deg_astar_std": s_a,
            "detection_rate": get(sc, "EHCO", "detection_rate")[0],
            "false_alarm_rate": get(sc, "EHCO", "false_alarm_rate")[0],
            "f1_score": get(sc, "EHCO", "f1_score")[0],
            "warn_t_mean": get(sc, "EHCO", "warn_t_mean")[0],
        })
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_e4(seeds: list[int] | None = None, scenarios: list[str] | None = None,
           out_dir: str | None = None, verbose: bool = True,
           n_instances: int = N_INSTANCES) -> dict:
    seeds = seeds or SEEDS
    scenarios = scenarios or ["congestion", "fault", "urgent"]
    out_dir = out_dir or RESULTS
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.perf_counter()
    # ---- phase 0: initial plans (must reproduce e1 on the managed venv) ----
    if verbose:
        print("[E4] phase 0: AW-NSGA-II + A*-GA initial plans "
              f"({len(seeds)} seeds)...", flush=True)
    scenarios_cache = {}
    plans = {"aw": {}, "astar": {}, "scenarios": scenarios_cache}
    e1_check = []
    for s in seeds:
        sc = make_scenario(s)
        scenarios_cache[s] = sc
        res_aw = run_algorithm("AW-NSGA-II", sc, s)
        res_astar = run_algorithm("A*-GA", sc, s)
        plans["aw"][s] = res_aw.best_assign
        plans["astar"][s] = res_astar.best_assign
        e1_check.append(res_aw.best_F[0])
        if verbose:
            print(f"    seed {s:3d} AW f1={res_aw.best_F[0]:7.1f} "
                  f"A*-GA f1={res_astar.best_F[0]:7.1f}", flush=True)
    # cross-check against e1 (managed venv must reproduce exactly)
    e1_path = os.path.join(os.path.dirname(out_dir), "e1", "e1_metrics.csv")
    mismatch = []
    if os.path.exists(e1_path):
        e1_f1 = {}
        with open(e1_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                d = dict(zip(header, parts))
                if d["algo"] == "AW-NSGA-II":
                    e1_f1[int(d["seed"])] = float(d["f1"])
        for s, v in zip(seeds, e1_check):
            if abs(v - e1_f1.get(s, v)) > 1e-6:
                mismatch.append((s, v, e1_f1.get(s)))
    if mismatch:
        print(f"[E4] WARNING: {len(mismatch)} seeds differ from e1_metrics.csv "
              f"(wrong numpy version?): {mismatch[:5]}", flush=True)
    else:
        print(f"[E4] initial-plan check vs e1_metrics.csv: exact match "
              f"({len(seeds)} seeds)", flush=True)

    # ---- phase 1: event simulations ---------------------------------------
    raw: list[dict] = []
    ts_sims: list[tuple[int, bool, DynamicSimulator]] = []
    for kind in scenarios:
        insts = make_instances(kind)
        for k, inst in enumerate(insts):
            inst = dict(inst)
            inst["_idx"] = k
            inst["_nominal"] = (k == 0)
            for s in seeds:
                for closed in (True, False):
                    row, ts_sim = run_scenario(kind, s, inst, closed, plans)
                    raw.append(row)
                    if ts_sim is not None:
                        ts_sims.append((s, closed, ts_sim))
            if verbose:
                print(f"  [{kind}] instance {k}: done", flush=True)

    # ---- phase 2: persist ------------------------------------------------
    _write_csv(os.path.join(out_dir, "e4_raw.csv"), raw)
    agg = aggregate(raw)
    _write_csv(os.path.join(out_dir, "e4_agg.csv"), agg)
    _write_csv(os.path.join(out_dir, "e4_table2.csv"), table2_rows(agg))

    # scenario-B nominal-instance time series (Fig 5 c/d)
    ts_rows = []
    for s, closed, sim in ts_sims:
        for t, frac, h in zip(sim.hist_t, sim.hist_frac, sim.hist_h):
            ts_rows.append({"seed": s,
                            "strategy": "EHCO" if closed else "A*-GA",
                            "t": t, "completed_pct": frac,
                            "h_fault": float(h[sim.fault_agv]) if sim.fault_agv is not None else float("nan")})
    _write_csv(os.path.join(out_dir, "e4_ts_b.csv"), ts_rows)

    meta = {
        "seeds": seeds, "scenarios": scenarios,
        "n_instances": n_instances, "n_repeats": len(seeds),
        "dt_sim": DT_SIM, "ts_every": TS_EVERY, "sim_tmax": SIM_TMAX,
        "local_budget": C.NP_LOCAL * C.GMAX_LOCAL,
        "global_budget": C.NP_GLOBAL * C.GMAX_GLOBAL,
        "derate_T0": DERATE_T0, "derate_T1": DERATE_T1,
        "vib_fault_offset": VIB_FAULT_OFFSET,
        "fault_target_rule": ("AGV in loaded run with largest remaining "
                              "workload at t_inj; SPEC-nominal AGV #3 kept "
                              "while active; monitoring continues post-shift "
                              "until warning fires"),
        "instances": {k: make_instances(k) for k in scenarios},
    }
    with open(os.path.join(out_dir, "e4_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f"[E4] total wall: {time.perf_counter()-t_start:.1f}s; "
              f"{len(raw)} runs", flush=True)
    return {"n_seeds": len(seeds), "n_scenarios": len(scenarios),
            "n_runs": len(raw), "out_dir": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--scenarios", default=None,
                    help="comma list: congestion,fault,urgent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    run_e4(seeds=seeds, scenarios=scenarios, out_dir=args.out)


if __name__ == "__main__":
    main()
