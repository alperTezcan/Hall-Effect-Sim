from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Sequence
import numpy as np

__all__ = [
    "Anisotropy",
    "Uniaxial",
    "ManyUniaxial",
    "Cubic",
    "Hexagonal",
    "axis_from_degrees",
    "euler_zyx",
]

# ---------------- Base interface ---------------- #

class Anisotropy(ABC):
    """Abstract anisotropy contribution (energy per unit volume)."""
    @abstractmethod
    def energy(self, M: np.ndarray) -> float: ...

# ---------------- Helpers ---------------- #

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n

def euler_zyx(z_deg: float, y_deg: float, x_deg: float) -> np.ndarray:
    """R = Rz(z) @ Ry(y) @ Rx(x) (degrees). Use for LAB --> CRYSTAL mapping."""
    z, y, x = np.deg2rad([z_deg, y_deg, x_deg])
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    return Rz @ Ry @ Rx

# ---------------- Uniaxial ---------------- #

@dataclass(frozen=True)
class Uniaxial(Anisotropy):
    """Uniaxial anisotropy. E = 1/2 Han (1 - (M . u)^2)."""
    Han: float
    u: np.ndarray  # unit vector in LAB frame
    def __post_init__(self) -> None:
        object.__setattr__(self, "u", _unit(self.u))
    def energy(self, M: np.ndarray) -> float:
        M = _unit(M)
        dot = float(np.dot(M, self.u))
        return 0.5 * self.Han * (1.0 - dot * dot)

def axis_from_degrees(Han: float, theta_deg: float, phi_deg: float) -> Uniaxial:
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    st, ct = np.sin(th), np.cos(th)
    cp, sp = np.cos(ph), np.sin(ph)
    u = np.array([st * cp, st * sp, ct], dtype=float)
    return Uniaxial(Han=Han, u=u)

# ---------------- Many-Uniaxial (sum) ---------------- #

@dataclass
class ManyUniaxial(Anisotropy):
    """Sum of multiple uniaxial contributions."""
    Han_list: Sequence[float] | None = None
    u_list: Sequence[np.ndarray] | None = None
    terms: List[Uniaxial] | None = None
    def __post_init__(self) -> None:
        if self.terms is None:
            if self.Han_list is None or self.u_list is None:
                raise ValueError("Provide either (Han_list & u_list) or terms")
            if len(self.Han_list) != len(self.u_list):
                raise ValueError("Han_list and u_list must have same length")
            self.terms = [Uniaxial(float(h), _unit(u)) for h, u in zip(self.Han_list, self.u_list)]
        self.terms = [Uniaxial(t.Han, _unit(t.u)) for t in self.terms]
    def energy(self, M: np.ndarray) -> float:
        return float(sum(t.energy(M) for t in self.terms))

# ---------------- Cubic crystal ---------------- #

@dataclass(frozen=True)
class Cubic(Anisotropy):
    """Cubic crystal anisotropy in crystal frame.
    E = K1 Σ a_i^2 a_j^2 + K2 (a1^2 a2^2 a3^2), a = R @ M
    """
    K1: float
    K2: float = 0.0
    R: np.ndarray | None = None
    def __post_init__(self) -> None:
        if self.R is None:
            object.__setattr__(self, "R", np.eye(3))
    def energy(self, M: np.ndarray) -> float:
        a = self.R @ _unit(M)
        a1, a2, a3 = float(a[0]), float(a[1]), float(a[2])
        s2 = a1*a1*a2*a2 + a2*a2*a3*a3 + a3*a3*a1*a1
        s3 = (a1*a1)*(a2*a2)*(a3*a3)
        return self.K1 * s2 + self.K2 * s3

# ---------------- Hexagonal crystal ---------------- #

@dataclass(frozen=True)
class Hexagonal(Anisotropy):
    """Hexagonal (hcp) anisotropy in crystal frame.
    E = K1 sin^2(theta) + K2 sin^4(theta) + K6 sin^6(theta) cos(6phi), angles from Mc = R @ M
    """
    K1: float
    K2: float = 0.0
    K6: float = 0.0
    R: np.ndarray | None = None
    def __post_init__(self) -> None:
        if self.R is None:
            object.__setattr__(self, "R", np.eye(3))
    def energy(self, M: np.ndarray) -> float:
        mc = self.R @ _unit(M)
        mx, my, mz = float(mc[0]), float(mc[1]), float(mc[2])
        sin2 = max(0.0, 1.0 - mz*mz)
        phi = float(np.arctan2(my, mx))
        sin4 = sin2 * sin2
        sin6 = sin4 * sin2
        return self.K1 * sin2 + self.K2 * sin4 + self.K6 * sin6 * np.cos(6.0 * phi)