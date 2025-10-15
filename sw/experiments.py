from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .SW_model import SWModel

__all__ = [
    "hall_effects",
    "hall_from_angle_scan",
    "switching_fields",
    "astroid_theta_scan",
    "astroid_phi_scan",
    "astroid_3d",
    # plotting
    "plot_hysteresis",
    "plot_angle_scan",
    "plot_halls",
]

# ---------- Transport (AMR / PHE / AHE) ---------- #

def hall_effects(
    model: SWModel,
    theta_H: float,
    phi_H: float,
    H_values: np.ndarray,
    *,
    DeltaR: float = 0.1,
    R0: float = 1.0,
    A_phe: float = 19.36,
    R_ahe: float = 0.0108,
    J: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute AMR, PHE, AHE and total vs magnetic field.

    Returns arrays (N,2) with columns [H, value].
    """
    if J is None:
        J = model.J_dir
    M = model.hysteresis(theta_H, phi_H, H_values)[:, 1:4]
    Hcol = H_values.reshape(-1, 1)

    JM = (J @ M.T)
    AMR = np.hstack([Hcol, (DeltaR * JM * JM / R0).reshape(-1, 1)])
    PHE = np.hstack([Hcol, (J[0] * DeltaR * A_phe * M[:, 0] * M[:, 1]).reshape(-1, 1)])
    AHE = np.hstack([Hcol, (J[0] * R_ahe * M[:, 2]).reshape(-1, 1)])
    TOT = np.hstack([Hcol, (PHE[:, 1] + AHE[:, 1]).reshape(-1, 1)])
    return AMR, PHE, AHE, TOT


def hall_from_angle_scan(
    scan: np.ndarray,
    *,
    DeltaR: float = 0.1,
    R0: float = 1.0,
    A_phe: float = 19.36,
    R_ahe: float = 0.0108,
    J: np.ndarray = np.array([1.0, 0.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert an angle scan array [angle, Mx, My, Mz, E] to AMR/PHE/AHE vs angle."""
    angle = scan[:, 0:1]
    M = scan[:, 1:4]
    JM = (J @ M.T)
    AMR = np.hstack([angle, (DeltaR * JM * JM / R0).reshape(-1, 1)])
    PHE = np.hstack([angle, (J[0] * DeltaR * A_phe * M[:, 0] * M[:, 1]).reshape(-1, 1)])
    AHE = np.hstack([angle, (J[0] * R_ahe * M[:, 2]).reshape(-1, 1)])
    TOT = np.hstack([angle, (PHE[:, 1] + AHE[:, 1]).reshape(-1, 1)])
    return AMR, PHE, AHE, TOT


# ---------- Switching / Astroid helpers ---------- #

def switching_fields(model: SWModel, theta_H: float, phi_H: float, H_values: np.ndarray, *, threshold: float = 0.1) -> np.ndarray:
    M = model.hysteresis(theta_H, phi_H, H_values)
    dMz = np.diff(M[:, 3])
    idx = np.where(np.abs(dMz) > float(threshold))[0]
    return M[idx]


def astroid_theta_scan(model: SWModel, phi_fixed: float, H_sweep: np.ndarray, *, resolution: int = 45) -> tuple[np.ndarray, np.ndarray]:
    thetas = np.linspace(0.0, np.pi, int(resolution))
    Hx_list: List[np.ndarray] = []
    Hy_list: List[np.ndarray] = []
    for th in thetas:
        sw = switching_fields(model, th, phi_fixed, H_sweep)
        if sw.size:
            H_switch = sw[:, 0]
            Hx_list.append(H_switch * np.cos(th))
            Hy_list.append(H_switch * np.sin(th))
    if not Hx_list:
        return np.array([]), np.array([])
    return np.concatenate(Hx_list), np.concatenate(Hy_list)


def astroid_phi_scan(model: SWModel, theta_fixed: float, H_sweep: np.ndarray, *, resolution: int = 90) -> tuple[np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 2 * np.pi, int(resolution), endpoint=False)
    Hx_list: List[np.ndarray] = []
    Hy_list: List[np.ndarray] = []
    for ph in phis:
        sw = switching_fields(model, theta_fixed, ph, H_sweep)
        if sw.size:
            H_switch = sw[:, 0]
            Hx_list.append(H_switch * np.cos(ph))
            Hy_list.append(H_switch * np.sin(ph))
    if not Hx_list:
        return np.array([]), np.array([])
    return np.concatenate(Hx_list), np.concatenate(Hy_list)


def astroid_3d(model: SWModel, H_sweep: np.ndarray, *, theta_res: int = 24, phi_res: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thetas = np.linspace(0.0, np.pi, int(theta_res))
    phis = np.linspace(0.0, 2 * np.pi, int(phi_res), endpoint=False)
    Hx_list: List[np.ndarray] = []
    Hy_list: List[np.ndarray] = []
    Hz_list: List[np.ndarray] = []
    for th in thetas:
        cth, sth = np.cos(th), np.sin(th)
        for ph in phis:
            cph, sph = np.cos(ph), np.sin(ph)
            sw = switching_fields(model, th, ph, H_sweep)
            if sw.size:
                H_switch = sw[:, 0]
                Hx_list.append(H_switch * sth * cph)
                Hy_list.append(H_switch * sth * sph)
                Hz_list.append(H_switch * cth)
    if not Hx_list:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(Hx_list), np.concatenate(Hy_list), np.concatenate(Hz_list)


# ---------- Plotting ---------- #

def plot_hysteresis(data: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for ax, col, label in [(ax1, 1, "$M_x$"), (ax2, 2, "$M_y$"), (ax3, 3, "$M_z$")]:
        ax.plot(data[:, 0], data[:, col], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="-")
        ax.set_xlabel("H")
        ax.set_ylabel(label)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Magnetization vs Field (Stoner-Wohlfarth)")
    fig.tight_layout()
    return fig


def plot_angle_scan(data: np.ndarray, *, angle_label: str) -> Figure:
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for ax, col, label in [(ax1, 1, "$M_x$"), (ax2, 2, "$M_y$"), (ax3, 3, "$M_z$")]:
        ax.plot(data[:, 0], data[:, col], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="-")
        ax.set_xlabel(angle_label)
        ax.set_ylabel(label)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Magnetization vs {angle_label} at fixed |H|")
    fig.tight_layout()
    return fig


def plot_halls(AMR: np.ndarray, PHE: np.ndarray, AHE: np.ndarray, TOT: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(AMR[:, 0], AMR[:, 1], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="-", label="AMR")
    ax2.plot(PHE[:, 0], PHE[:, 1], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="-", label="PHE")
    ax3.plot(AHE[:, 0], AHE[:, 1], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="-", label="AHE")
    for ax in (ax1, ax2, ax3):
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xlabel("H")
    fig.suptitle("Hall effects vs Field")
    fig.tight_layout()
    return fig