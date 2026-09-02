"""True degradation dynamics, EWMA SoH estimation, and EKF (SPEC.md sections 2-4).

Ground-truth engine steps at 50 ms; EKF fuses noisy observations; SoH uses
EKF-estimated temperature and median-filtered vibration (independent channel).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C


@dataclass
class AGVTruth:
    """Ground-truth physical state of one AGV."""
    x: float
    y: float
    theta: float = 0.0
    v: float = C.V_AVG
    T: float = C.T_AMB
    cycles: float = 0.0
    km: float = 0.0
    load_rate: float = C.LOAD_RATE_IDLE
    fault_load_rate: float | None = None   # dynamic scenario B injection

    def step(self, dt: float = C.DT) -> None:
        """Advance thermal state by dt seconds (called every 50 ms)."""
        lr = self.load_rate if self.fault_load_rate is None else self.fault_load_rate
        t_ss = C.T_AMB + C.K_THERMAL * lr
        tau = C.TAU_THERMAL if t_ss > self.T else C.TAU_COOL
        self.T = t_ss + (self.T - t_ss) * np.exp(-dt / tau)

    def travel(self, dist_m: float) -> None:
        """Accumulate distance-driven degradation."""
        self.cycles += dist_m / C.L_CYCLE
        self.km += dist_m / 1000.0

    # true instantaneous sub-indicator health (ground truth, not SoH estimate)
    def true_instant_health(self) -> float:
        soh_batt = 1.0 - self.cycles / C.C_MAX
        soh_motor = 1.0 - max(0.0, self.T - C.T_AMB) / (C.T_MAX - C.T_AMB)
        soh_mech = 1.0 - (C.V0_VIB + C.K_WEAR * self.km) / C.V_CRIT
        return float(C.AHP_W @ np.array([soh_batt, soh_motor, soh_mech]))


def soh_battery(cycles: float) -> float:
    return 1.0 - cycles / C.C_MAX


def soh_motor(T: float) -> float:
    return 1.0 - max(0.0, T - C.T_AMB) / (C.T_MAX - C.T_AMB)


def soh_mech(v_rms: float) -> float:
    return 1.0 - v_rms / C.V_CRIT


def instant_health(cycles: float, T_est: float, v_rms_filt: float) -> float:
    """SoH estimate from (battery cycles, EKF-estimated T, filtered vibration)."""
    return float(C.AHP_W @ np.array([
        soh_battery(cycles), soh_motor(T_est), soh_mech(v_rms_filt)]))


def ewma_step(h_cum: float, h_inst: float, beta: float = C.EWMA_BETA) -> float:
    return beta * h_inst + (1.0 - beta) * h_cum


@dataclass
class EKF:
    """Extended Kalman filter for [x, y, theta, v, T] (SPEC section 4).

    The transition is weakly nonlinear through cos/sin of theta; we use the
    standard EKF linearisation. Observation: x, y (RFID) and T (motor)."""
    x: np.ndarray                                   # state (5,)
    P: np.ndarray = field(default_factory=lambda: np.eye(5) * 1.0)
    Q: np.ndarray = field(default_factory=lambda: np.diag(C.Q_DIAG))
    R: np.ndarray = field(default_factory=lambda: np.diag(C.R_DIAG))

    def predict(self, dt: float = C.DT) -> None:
        x, y, th, v, T = self.x
        self.x = np.array([
            x + v * dt * np.cos(th),
            y + v * dt * np.sin(th),
            th, v, T])
        F = np.eye(5)
        F[0, 2] = -v * dt * np.sin(th)
        F[0, 3] = dt * np.cos(th)
        F[1, 2] = v * dt * np.cos(th)
        F[1, 3] = dt * np.sin(th)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        H = C.H_OBS
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    @property
    def est(self) -> np.ndarray:
        return self.x


def run_ekf_validation(truth_traj: list[tuple[float, float, float, float, float]],
                       rng: np.random.Generator) -> dict:
    """Run EKF over a truth trajectory with sensor noise; return consistency
    metrics: NRMSE (normalised by warehouse diagonal), residual mean/std."""
    ekf = EKF(x=np.array(truth_traj[0], dtype=float))
    errs, residuals = [], []
    for truth in truth_traj:
        ekf.predict()
        tx, ty, _, _, tT = truth
        z = np.array([
            tx + rng.normal(0.0, C.SIGMA_RFID),
            ty + rng.normal(0.0, C.SIGMA_RFID),
            tT + rng.normal(0.0, C.SIGMA_TEMP)])
        ekf.update(z)
        errs.append(np.hypot(ekf.x[0] - tx, ekf.x[1] - ty))
        residuals.append(z[0] - (C.H_OBS @ ekf.x)[0])
    errs = np.array(errs)
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    return {
        "rmse_m": rmse,
        "nrmse": rmse / C.WAREHOUSE_DIAG,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
    }
