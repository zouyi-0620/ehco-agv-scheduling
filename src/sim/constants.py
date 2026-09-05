"""EHCO simulation core - single source of truth for all parameters.

All values transcribed from SPEC.md (D:/WorkBuddy/2026-08-05-10-06-28/SPEC.md),
which was extracted from the manuscript (review/manuscript_full.txt).
Decision points D1-D10 are documented in SPEC.md section 0.
"""
import numpy as np

# ---------------------------------------------------------------- warehouse
GRID_W = 50          # x cells (1 m each)
GRID_H = 40          # y cells
SHELF_ROWS = 12      # shelf rows (obstacles, 1 cell thick)
SHELF_ROW_Y = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]  # D10
SHELF_X_RANGE = (1, 48)          # shelf row spans x = 1..48
PICK_X = [4, 10, 16, 22, 28, 34, 40, 46]   # 8 pick columns per row
N_STORAGE = 96                   # K = 12 * 8
MAINTENANCE_CELL = (49, 38)      # maintenance area (dynamic scenario B)
C_AISLE = 2                      # max parallel AGVs per aisle
AISLE_CAPACITY_NOMINAL = 2.5     # m (nominal; grid-rounded to 2 cells, D10)

# ---------------------------------------------------------------- fleet / tasks
N_AGV = 10
N_TASKS = 50
N_REGULAR = 40
N_URGENT = 10
H0_LO, H0_HI = 0.7, 1.0          # initial h_cum ~ U[0.7, 1.0]
OMEGA_URG_LO, OMEGA_URG_HI = 0.8, 1.0

# ---------------------------------------------------------------- AGV spec (Table S1)
P_MOTOR = 0.75                   # kW
V_AVG = 1.5                      # m/s
C_MAX = 2000                     # battery cycles
L_CYCLE = 500.0                  # m per cycle
T_MAX = 105.0                    # deg C
T_AMB = 25.0                     # deg C
TAU_THERMAL = 120.0              # s (heating)
TAU_COOL = 180.0                 # s (cooling)
K_THERMAL = 80.0                 # deg C per load-rate
V0_VIB = 5.0                     # mm/s baseline
K_WEAR = 0.5                     # mm/(s*km)
V_CRIT = 50.0                    # mm/s
M_BASE, M_RATED = 50.0, 100.0    # kg
M_LOAD = 100.0                   # every task carries rated load (D6)
LAMBDA_LOADED = 1.0              # (50+100)/(50+100)
LAMBDA_EMPTY = M_BASE / (M_BASE + M_RATED)   # 0.3333
LOAD_RATE_EMPTY = 0.4            # D7
LOAD_RATE_LOADED = 1.0
LOAD_RATE_IDLE = 0.2
T_LOAD = 5.0                     # s loading/unloading per task

# ---------------------------------------------------------------- true dynamics
DT = 0.05                        # 50 ms data cycle

# ---------------------------------------------------------------- SoH model
AHP_W = np.array([0.40, 0.35, 0.25])   # battery, motor-thermal, mechanical
EWMA_BETA = 0.001                # per 50 ms step (half-life ~35 s)
DEGRAD_WINDOW = 300.0            # s observation window
DEGRAD_THRESHOLD = -1e-4         # s^-1 warning threshold

# ---------------------------------------------------------------- EKF
Q_DIAG = np.array([0.1**2, 0.1**2, 0.05**2, 0.05**2, 0.5**2])
R_DIAG = np.array([0.5**2, 0.5**2, 0.3**2])
H_OBS = np.zeros((3, 5))
H_OBS[0, 0] = 1.0
H_OBS[1, 1] = 1.0
H_OBS[2, 4] = 1.0
SIGMA_RFID = 0.5                 # m
SIGMA_TEMP = 0.3                 # deg C
SIGMA_VIB = 0.03                 # mm/s (independent channel, not in EKF)
WAREHOUSE_DIAG = 64.0            # m, NRMSE normalisation

# ---------------------------------------------------------------- objectives
C_E = 0.8                        # CNY/kWh
C_M = 200.0                      # CNY
C_P = 0.5                        # CNY/s
H_SAFE = 0.6
H_CRIT = 0.2
ALPHA_PENALTY = 0.5
BETA_SLACK = 0.3                 # deadline tightness factor for urgent tasks:
                                 # slack = deadline*(1 - BETA_SLACK*omega), so
                                 # more urgent (higher omega) -> tighter deadline
DEADLINE_MULT = 1.5              # deadline multiplier (D4)
RHO_MAX = 0.8                    # congestion constraint

# ---------------------------------------------------------------- A* (D2)
W_E = 0.2                        # energy weight (normalised fusion)
W_H = 0.5                        # health weight
E0_PER_M = P_MOTOR * LAMBDA_EMPTY / V_AVG   # kWh per m, empty (normalisation)

# ---------------------------------------------------------------- AW-NSGA-II
NP = 100
GMAX = 200
PC_MIN, PC_MAX = 0.6, 0.9
# pm ramps 0.01 -> 0.02 over generations. Calibrated after the E1 diagnostic:
# the original 0.05 -> 0.2 per-gene reassignment was destructively strong
# (each generation randomly reassigns 2.5-10 genes per individual, destroying
# promising partial solutions); 0.01 -> 0.02 preserves the paper's "mutation
# rate ramps up with generation" schedule while keeping search effective.
PM_MIN, PM_MAX = 0.01, 0.02
NP_LOCAL, GMAX_LOCAL = 50, 80    # local replanning budget
NP_GLOBAL, GMAX_GLOBAL = 100, 200

# baselines
NSGA3_H = 5                      # Das-Dennis division -> 56 ref points (4-obj)
MOEAD_N_WV = 100                 # uniform weight vectors
MOEAD_T = 20                     # neighbourhood size
LWC_A, LWC_D = 0.34, 0.40        # linear weight controller

# ---------------------------------------------------------------- experiments
N_RUNS = 30
SEEDS = list(range(1, N_RUNS + 1))
