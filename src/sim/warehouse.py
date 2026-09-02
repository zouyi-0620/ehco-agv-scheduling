"""Warehouse layout, grid, storage locations, and shortest-path precomputation.

Layout (SPEC.md section 1, decision D10):
  - 50 x 40 grid, 1 m cells
  - 12 shelf rows (1 cell thick obstacles) at y in SHELF_ROW_Y, spanning x=1..48
  - 2-cell aisles between rows; cross aisles top (y=36..39) and bottom (y=0..1);
    perimeter corridors x=0 and x=49
  - 96 storage locations: pick cells adjacent to shelf rows at PICK_X columns,
    alternating above/below each row
  - Maintenance area at (49, 38)

Key nodes for path precomputation: 96 storage cells + 10 AGV start cells +
maintenance cell. Because AGV positions after each task are always storage
cells, all fitness-evaluation path queries are table lookups.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from . import constants as C


def build_grid() -> np.ndarray:
    """Return (GRID_H, GRID_W) bool array; True = traversable."""
    grid = np.ones((C.GRID_H, C.GRID_W), dtype=bool)
    for y in C.SHELF_ROW_Y:
        x0, x1 = C.SHELF_X_RANGE
        grid[y, x0:x1 + 1] = False
    return grid


def storage_cells() -> list[tuple[int, int]]:
    """96 pick cells. Pick face alternates above (y-1) / below (y+1) each row."""
    cells = []
    for r, y in enumerate(C.SHELF_ROW_Y):
        for c, x in enumerate(C.PICK_X):
            # alternate side per (row, column) so both faces are used
            above = (r + c) % 2 == 0
            py = y - 1 if above else y + 1
            cells.append((x, py))
    assert len(cells) == C.N_STORAGE
    return cells


def free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(grid)
    return list(zip(xs.tolist(), ys.tolist()))


def _neighbours(x: int, y: int, grid: np.ndarray):
    if x > 0 and grid[y, x - 1]:
        yield x - 1, y
    if x < C.GRID_W - 1 and grid[y, x + 1]:
        yield x + 1, y
    if y > 0 and grid[y - 1, x]:
        yield x, y - 1
    if y < C.GRID_H - 1 and grid[y + 1, x]:
        yield x, y + 1


def dijkstra_from(src: tuple[int, int], grid: np.ndarray,
                  cost: np.ndarray | None = None):
    """Dijkstra over the grid. If cost is given, edge cost = cost[cell] (fused
    multi-cost, improved A*); the geometric length of each resulting path is
    tracked in parallel. Returns (cost_dist, geo_dist) arrays (GRID_H, GRID_W);
    with cost=None both equal pure grid distance in metres."""
    dist = np.full((C.GRID_H, C.GRID_W), np.inf)
    geo = np.full((C.GRID_H, C.GRID_W), np.inf)
    dist[src[1], src[0]] = 0.0
    geo[src[1], src[0]] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for nx, ny in _neighbours(x, y, grid):
            step_cost = 1.0 if cost is None else float(cost[ny, nx])
            nd = d + step_cost
            ng = geo[y, x] + 1.0
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                geo[ny, nx] = ng
                heapq.heappush(pq, (nd, (nx, ny)))
    return dist, geo


@dataclass
class Warehouse:
    grid: np.ndarray
    storage: list[tuple[int, int]]
    nodes: list[tuple[int, int]] = field(default_factory=list)   # key nodes
    node_index: dict[tuple[int, int], int] = field(default_factory=dict)
    # pairwise distance matrices among key nodes (m)
    dist_standard: np.ndarray | None = None      # pure-distance A* (metres)
    dist_improved: np.ndarray | None = None      # improved A* path lengths (m)

    @classmethod
    def create(cls, agv_start_cells: list[tuple[int, int]] | None = None,
               improved: bool = True) -> "Warehouse":
        grid = build_grid()
        storage = storage_cells()
        nodes = list(storage)
        if agv_start_cells:
            for cell in agv_start_cells:
                if cell not in nodes:
                    nodes.append(cell)
        if C.MAINTENANCE_CELL not in nodes:
            nodes.append(C.MAINTENANCE_CELL)
        wh = cls(grid=grid, storage=storage, nodes=nodes)
        wh.node_index = {cell: i for i, cell in enumerate(nodes)}
        n = len(nodes)
        wh.dist_standard = np.zeros((n, n))
        for i, cell in enumerate(nodes):
            d, _ = dijkstra_from(cell, grid)
            for j, (x, y) in enumerate(nodes):
                wh.dist_standard[i, j] = d[y, x]
        if improved:
            wh.dist_improved = wh._improved_lengths()
        return wh

    def _fused_cell_costs(self) -> np.ndarray:
        """Fused multi-cost per-cell weight (improved A*, D2): 1 + w_e*(energy
        factor). The health term is 0 for h >= h_safe (static-typical case);
        AGVs in the penalty band are handled by callers via the distance table."""
        e_loaded = C.P_MOTOR * C.LAMBDA_LOADED / C.V_AVG / C.E0_PER_M
        e_avg = 0.5 * (e_loaded + 1.0)
        return np.full((C.GRID_H, C.GRID_W), 1.0 + C.W_E * e_avg)

    def _improved_lengths(self) -> np.ndarray:
        """Geometric lengths (m) of min-fused-cost paths (improved A*)."""
        cost = self._fused_cell_costs()
        n = len(self.nodes)
        out = np.zeros((n, n))
        for i, cell in enumerate(self.nodes):
            _, geo = dijkstra_from(cell, self.grid, cost)
            for j, (x, y) in enumerate(self.nodes):
                out[i, j] = geo[y, x]
        return out

    # convenience lookups -----------------------------------------------
    def d(self, a: int, b: int, improved: bool = False) -> float:
        m = self.dist_improved if improved else self.dist_standard
        return float(m[a, b])


def cell_of(node_id: int, wh: Warehouse) -> tuple[int, int]:
    return wh.nodes[node_id]
