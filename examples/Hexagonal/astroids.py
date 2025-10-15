from __future__ import print_function

import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from sw import (
    SWModel, Uniaxial, Hexagonal, axis_from_degrees, euler_zyx,
    hall_effects, hall_from_angle_scan,
    plot_hysteresis, plot_angle_scan, plot_halls,
    astroid_theta_scan, astroid_phi_scan
)

def main():
    # Shape anisotropy (uniaxial)
    shape = axis_from_degrees(Han=0.5, theta_deg=0.0, phi_deg=0.0)
    
    # Hexagonal anisotropy (crystal)
    hex_cryst = Hexagonal(K1=0.08, K2=0.0, K6=0.04)

    # Build model with both terms
    model = SWModel([shape, hex_cryst], Hd=0.0)

    # Field sweep
    H_forward  = np.linspace(+6.0, -6.0, 2401)
    H_backward = np.linspace(-6.0, +6.0, 2401)
    H = np.concatenate([H_forward, H_backward])

    # Astroid slice theta
    Hx, Hy = astroid_theta_scan(model, phi_fixed=0.0, H_sweep=H, resolution=50)
    plt.figure(); plt.scatter(Hx, Hy, s=6); plt.gca().set_aspect("equal"); plt.title("Astroid theta-scan")

    # Astroid slice phi
    Hx, Hy = astroid_phi_scan(model, theta_fixed=np.pi/2, H_sweep=H, resolution=90)
    plt.figure(); plt.scatter(Hx, Hy, s=6); plt.gca().set_aspect("equal"); plt.title("Astroid phi-scan")

    plt.show()

if __name__ == "__main__":
    main()
