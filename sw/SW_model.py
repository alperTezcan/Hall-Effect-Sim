from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
from scipy.optimize import minimize

from .anisotropies import Anisotropy

__all__ = [
    "SWModel",
    "m_from_angles",
    "angles_from_vec",
]

def m_from_angles(theta: float, phi: float) -> np.ndarray:
    """Convert spherical angles to Cartesian unit vector."""
    s, c = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([s * cp, s * sp, c], dtype=float)

def angles_from_vec(v: np.ndarray) -> Tuple[float, float]:
    """Convert Cartesian vector to spherical angles."""
    v = v / np.linalg.norm(v)
    theta = float(np.arccos(v[2]))
    phi = float(np.arctan2(v[1], v[0]))
    return theta, phi

@dataclass
class SWModel:
    """Stoner-Wohlfarth macrospin core: energy + minimization.

    Parameters
    ----------
    anisotropies : Sequence[Anisotropy]
        Collection of anisotropy energy terms.
    Hd : float
        Demagnetizing coefficient in 1/2 Hd cos^2(theta_M).
    J_dir : np.ndarray
        Current direction (for transport projections), unit vector.
    """

    anisotropies: Sequence[Anisotropy]
    Hd: float = 0.0
    J_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))

    # ---------------- Energies ---------------- #
    @staticmethod
    def zeeman(H: float, theta_H: float, phi_H: float, theta_M: float, phi_M: float) -> float:
        cosgamma = (
            np.sin(theta_H) * np.cos(phi_H) * np.sin(theta_M) * np.cos(phi_M)
            + np.sin(theta_H) * np.sin(phi_H) * np.sin(theta_M) * np.sin(phi_M)
            + np.cos(theta_H) * np.cos(theta_M)
        )
        return -H * cosgamma

    def demag(self, theta_M: float) -> float:
        return 0.5 * self.Hd * (np.cos(theta_M) ** 2)

    def anisotropy_sum(self, theta_M: float, phi_M: float) -> float:
        M = m_from_angles(theta_M, phi_M)
        return float(sum(ani.energy(M) for ani in self.anisotropies))

    def energy(self, H: float, theta_H: float, phi_H: float, theta_M: float, phi_M: float) -> float:
        return (
            self.zeeman(H, theta_H, phi_H, theta_M, phi_M)
            + self.demag(theta_M)
            + self.anisotropy_sum(theta_M, phi_M)
        )

    # -------------- Minimization -------------- #
    def _angle_energy(self, params: Sequence[float], H: float, theta_H: float, phi_H: float) -> float:
        th, ph = params
        return self.energy(H, theta_H, phi_H, th, ph)

    def minimize_energy(
        self,
        H: float,
        theta_H: float,
        phi_H: float,
        initial_guess: Tuple[float, float],
        *,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        method: str = None,
        options: Optional[Dict] = None,
    ) -> Tuple[float, float]:
        # TODO: bounds, method, options cause minimzer to fail. Needs to be investigated.
        #if options is None:
        #    options = {"maxiter": 500}
        #if bounds is None:
        #    bounds = ((0.0, np.pi), (0, 2 * np.pi))
        res = minimize(
            self._angle_energy,
            x0=np.array(initial_guess, dtype=float),
            args=(H, theta_H, phi_H),
            bounds=bounds,
            method=method,
            options=options,
        )
        th, ph = map(float, res.x)
        return th, ph

    # -------------- Sweeps & Scans -------------- #
    def hysteresis(
        self,
        theta_H: float,
        phi_H: float,
        H_values: np.ndarray,
        *,
        initial_theta_phi: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        if initial_theta_phi is None:
            initial_theta_phi = (np.sign(H_values[0]) * theta_H, np.sign(H_values[0]) * phi_H)
        out: List[List[float]] = []
        guess = initial_theta_phi
        for H in H_values:
            tM, pM = self.minimize_energy(H, theta_H, phi_H, guess)
            M = m_from_angles(tM, pM)
            out.append([float(H), float(M[0]), float(M[1]), float(M[2])])
            guess = (tM, pM)
        return np.asarray(out, dtype=float)

    def angle_scan_theta(
        self,
        H: float,
        phi_H: float,
        thetas: np.ndarray,
        *,
        initial_theta_phi: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        out: List[List[float]] = []
        guess = initial_theta_phi if initial_theta_phi is not None else (thetas[0], phi_H)
        for th in thetas:
            tM, pM = self.minimize_energy(H, th, phi_H, guess)
            M = m_from_angles(tM, pM)
            E = self.energy(H, th, phi_H, tM, pM)
            out.append([float(th), float(M[0]), float(M[1]), float(M[2]), float(E)])
            guess = (tM, pM)
        return np.asarray(out, dtype=float)

    def angle_scan_phi(
        self,
        H: float,
        theta_H: float,
        phis: np.ndarray,
        *,
        initial_theta_phi: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        out: List[List[float]] = []
        guess = initial_theta_phi if initial_theta_phi is not None else (theta_H, phis[0])
        for ph in phis:
            tM, pM = self.minimize_energy(H, theta_H, ph, guess)
            M = m_from_angles(tM, pM)
            E = self.energy(H, theta_H, ph, tM, pM)
            out.append([float(ph), float(M[0]), float(M[1]), float(M[2]), float(E)])
            guess = (tM, pM)
        return np.asarray(out, dtype=float)