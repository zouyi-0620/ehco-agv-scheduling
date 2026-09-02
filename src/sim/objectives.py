"""Vectorized four-objective fitness evaluation (SPEC.md section 5).

Design: the layout is fixed, so all path queries are lookups in the precomputed
node-to-node distance matrices. Population evaluation loops over tasks (M=50
vectorized steps across the population), giving sub-millisecond per-individual
cost.

Model notes (SPEC decision points):
- D1: with initial h_cum in [0.7, 1.0] the piecewise health penalty f4 is 0 in
  static scenarios; SoH-aware allocation pressure comes from the maintenance
  cost proxy c_m*(1-h) inside f2. Implemented exactly as specified.
- D4: deadline_ij = 1.5*((d(pos_i, s_j) + d(s_j, d_j))/v + T_load) with the
  allocation-moment planned position; queueing delays cause tardiness.
- Collision waiting is not modelled in static fitness (manuscript: all
  collisions resolved by replanning / minor departure adjustments, none
  infeasible); dynamic-scenario congestion is handled by the event simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .dynamics import instant_health
from .scenario import Scenario, Task


@dataclass
class EvalConfig:
    """Switches for algorithm/ablation variants (SPEC section 9, E2)."""
    use_health_objective: bool = True    # f4 + hard exclusion (h < h_crit)
    use_maintenance_cost: bool = True    # c_m * (1-h) inside f2
    use_improved_astar: bool = True      # improved vs standard A* tables
    hard_exclusion_only: bool = False    # exclusion at h < h_safe, no f4/c_m


def _distance_matrix(sc: Scenario, cfg: EvalConfig) -> np.ndarray:
    if cfg.use_improved_astar and sc.warehouse.dist_improved is not None:
        return sc.warehouse.dist_improved
    return sc.warehouse.dist_standard


def evaluate_population(assign: np.ndarray, sc: Scenario,
                        cfg: EvalConfig | None = None,
                        h_cum: np.ndarray | None = None) -> dict:
    """Evaluate a population of task-assignment chromosomes.

    Parameters
    ----------
    assign : (P, M) int array, values in [0, N) - AGV of each task
    sc     : Scenario (tasks sorted by index = FCFS queue order)
    cfg    : variant switches
    h_cum  : (N,) allocation-moment cumulative health (default: initial h0)

    Returns dict with arrays (P,): f1..f4, makespan, cost, energy_kJ,
    health_pen, tardiness, n_tasks_per_agv (P, N).
    """
    cfg = cfg or EvalConfig()
    P, M = assign.shape
    N = len(sc.agvs)
    D = _distance_matrix(sc, cfg)
    assert (assign >= 0).all() and (assign < N).all()

    if h_cum is None:
        h_cum = np.array([a.h0 for a in sc.agvs])

    tasks: list[Task] = sc.tasks
    s = np.array([t.s_node for t in tasks])
    d = np.array([t.d_node for t in tasks])
    omega = np.array([t.omega for t in tasks])

    rows = np.arange(P)
    start_nodes = np.array([a.start_node for a in sc.agvs])
    cur_node = np.tile(start_nodes, (P, 1))            # (P, N)
    cur_time = np.zeros((P, N))                        # s
    cum_energy_wh = np.zeros((P, N))                   # Wh
    cum_tardy_cost = np.zeros((P, N))                  # CNY
    cum_health_pen = np.zeros((P, N))
    n_tasks = np.zeros((P, N), dtype=int)

    h_of_assigned = h_cum[assign]                      # (P, M) allocation h
    # piecewise health penalty rate per unit loaded-distance (D1)
    hp_rate = np.where(h_of_assigned >= C.H_SAFE, 0.0,
                       np.where(h_of_assigned < C.H_CRIT, np.inf,
                                C.ALPHA_PENALTY * (1.0 - h_of_assigned / C.H_SAFE)))
    infeasible_agv = h_cum < C.H_CRIT                  # hard constraint (N,)
    if cfg.hard_exclusion_only:
        # "hard-exclusion-only" ablation: exclude at h_safe, no f4/c_maint
        infeasible_agv = h_cum < C.H_SAFE
    elif not cfg.use_health_objective:
        infeasible_agv = np.zeros(N, dtype=bool)
    if not cfg.use_health_objective:
        hp_rate = np.zeros_like(hp_rate)

    for j in range(M):
        agv = assign[:, j]                             # (P,)
        pos = cur_node[rows, agv]                      # (P,)
        d_empty = D[pos, s[j]]                         # (P,)
        d_loaded = D[s[j], d[j]]
        t_seg = (d_empty + d_loaded) / C.V_AVG + C.T_LOAD
        e_wh = (C.P_MOTOR * 1000.0 * (C.LAMBDA_EMPTY * d_empty
                                      + C.LAMBDA_LOADED * d_loaded)
                / C.V_AVG)                             # W * s = J / 3600? no: Wh below
        e_wh = e_wh / 3600.0
        new_time = cur_time[rows, agv] + t_seg
        deadline = C.DEADLINE_MULT * t_seg             # D4 (same-segment form)
        slack = deadline * (1.0 + C.BETA_SLACK * omega[j])
        tardy = np.maximum(0.0, new_time - slack)
        cur_time[rows, agv] = new_time
        cum_energy_wh[rows, agv] += e_wh
        cum_tardy_cost[rows, agv] += C.C_P * tardy
        cum_health_pen[rows, agv] += hp_rate[:, j] * d_loaded
        cur_node[rows, agv] = d[j]
        n_tasks[rows, agv] += 1

    makespan = cur_time.max(axis=1)
    energy_wh = cum_energy_wh.sum(axis=1)
    c_energy = C.C_E * energy_wh / 1000.0              # kWh -> CNY
    if cfg.use_maintenance_cost:
        active = n_tasks > 0
        c_maint = (C.C_M * np.maximum(0.0, 1.0 - h_cum)[None, :] * active).sum(axis=1)
    else:
        c_maint = 0.0
    c_tardy = cum_tardy_cost.sum(axis=1)
    cost = c_energy + c_maint + c_tardy
    health_pen = cum_health_pen.sum(axis=1)

    # infeasible individuals: any task on an infeasible AGV
    bad = infeasible_agv[assign].any(axis=1)
    f4 = np.where(bad, np.inf, health_pen)

    return {
        "f1": makespan,
        "f2": cost,
        "f3": energy_wh * 3.6,                          # Wh -> kJ
        "f4": f4,
        "makespan": makespan,
        "cost": cost,
        "tardiness_cost": c_tardy,
        "n_tasks_per_agv": n_tasks,
        "infeasible": bad,
    }


# --------------------------------------------------------------------------
# Plan-level true-degradation simulation (for delta-h, replay correction E8)
# --------------------------------------------------------------------------

@dataclass
class PlanSegment:
    kind: str          # "empty" | "loaded" | "idle"
    duration: float    # s
    dist: float        # m


def build_plan(assign: np.ndarray, sc: Scenario,
               cfg: EvalConfig | None = None) -> list[list[PlanSegment]]:
    """Per-AGV ordered segment list for one chromosome (FCFS by task index)."""
    cfg = cfg or EvalConfig()
    D = _distance_matrix(sc, cfg)
    plans: list[list[PlanSegment]] = [[] for _ in sc.agvs]
    pos = [a.start_node for a in sc.agvs]
    for j, t in enumerate(sc.tasks):
        i = int(assign[j])
        d_empty = D[pos[i], t.s_node]
        d_loaded = D[t.s_node, t.d_node]
        if d_empty > 0:
            plans[i].append(PlanSegment("empty", d_empty / C.V_AVG, d_empty))
        plans[i].append(PlanSegment("loaded", d_loaded / C.V_AVG, d_loaded))
        plans[i].append(PlanSegment("idle", C.T_LOAD, 0.0))
        pos[i] = t.d_node
    return plans


def simulate_plan_health(plans: list[list[PlanSegment]], sc: Scenario,
                         substep: float = 1.0) -> dict:
    """Simulate true degradation + EWMA SoH along executed plans.

    Thermal state uses the closed-form first-order response per substep;
    EWMA is integrated at `substep` seconds (beta rescaled exactly, so the
    1 s discretisation only approximates the within-step h_inst trajectory).
    Returns per-AGV h trajectory summary and fleet metrics."""
    N = len(sc.agvs)
    T = np.full(N, C.T_AMB)
    cycles = np.zeros(N)
    km = np.zeros(N)
    h_cum = np.array([a.h0 for a in sc.agvs])
    beta_sub = 1.0 - (1.0 - C.EWMA_BETA) ** (substep / C.DT)
    h_start_task = np.full((N, 64), np.nan)   # h_cum at each task start (E8)
    n_task_idx = np.zeros(N, dtype=int)

    for i in range(N):
        for seg in plans[i]:
            lr = {"empty": C.LOAD_RATE_EMPTY, "loaded": C.LOAD_RATE_LOADED,
                  "idle": C.LOAD_RATE_IDLE}[seg.kind]
            t_ss = C.T_AMB + C.K_THERMAL * lr
            tau = C.TAU_THERMAL if t_ss > T[i] else C.TAU_COOL
            dist_rate = seg.dist / seg.duration if seg.duration > 0 else 0.0
            t_remaining = seg.duration
            while t_remaining > 1e-9:
                dt = min(substep, t_remaining)
                # closed-form thermal update
                T[i] = t_ss + (T[i] - t_ss) * np.exp(-dt / tau)
                cycles[i] += dist_rate * dt / C.L_CYCLE
                km[i] += dist_rate * dt / 1000.0
                v_rms = C.V0_VIB + C.K_WEAR * km[i]
                h_inst = instant_health(cycles[i], T[i], v_rms)
                h_cum[i] = beta_sub * h_inst + (1 - beta_sub) * h_cum[i]
                t_remaining -= dt
                if seg.kind == "idle" and t_remaining < 1e-9:
                    pass
            if seg.kind == "loaded":
                k = n_task_idx[i]
                if k < 64:
                    h_start_task[i, k] = h_cum[i]
                n_task_idx[i] += 1

    h0 = np.array([a.h0 for a in sc.agvs])
    delta_h = h0 - h_cum
    return {
        "h0": h0, "h_end": h_cum, "delta_h": delta_h,
        "delta_h_mean": float(delta_h.mean()),
        "delta_h_max": float(delta_h.max()),
        "gini": _gini(delta_h),
        "T_end": T, "cycles": cycles, "km": km,
        "h_start_task": h_start_task,
    }


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


# --------------------------------------------------------------------------
# Decision selection from a Pareto front (SPEC D11: knee point)
# --------------------------------------------------------------------------

def knee_point_selection(F: np.ndarray) -> int:
    """Index of the knee point on a (normalised) front: the point with maximum
    distance to the line joining the two extreme points of the front."""
    F = np.asarray(F, dtype=float)
    if len(F) < 2:
        return 0
    # normalise columns to [0, 1]
    lo, hi = F.min(axis=0), F.max(axis=0)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Fn = (F - lo) / rng
    # extremes: min distance to ideal (0-vector)
    dist_ideal = np.sqrt((Fn ** 2).sum(axis=1))
    e1, e2 = np.argsort(dist_ideal)[:2]
    p1, p2 = Fn[e1], Fn[e2]
    line = p2 - p1
    norm = np.linalg.norm(line)
    if norm < 1e-12:
        return int(e1)
    u = line / norm
    rel = Fn - p1
    d = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    return int(np.argmax(d))
