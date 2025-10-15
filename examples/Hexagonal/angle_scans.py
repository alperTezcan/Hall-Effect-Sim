from __future__ import print_function

import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from sw import (
    SWModel, Uniaxial, Hexagonal, axis_from_degrees, euler_zyx,
    hall_effects, hall_from_angle_scan,
    plot_hysteresis, plot_angle_scan, plot_halls,
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

    # ------------------------------
    # Angle scan at fixed |H|
    # ------------------------------
    H_abs = 6  # choose a field where both anisotropies compete

    # θ-scan with φ fixed
    phi_fixed = np.deg2rad(0.0)
    theta_grid = np.deg2rad(
        np.concatenate(
            [np.linspace(-180.0, 180.0, 181), 
             np.linspace(180.0, -180.0, 181)]
            )
        )
    scan_theta = model.angle_scan_theta(H_abs, phi_fixed, theta_grid)
    fig3 = plot_angle_scan(scan_theta, angle_label=r"$\theta_H$ (rad)")
    fig3.suptitle("Angle scan in θ at |H| = {:.2f}".format(H_abs))

    # Convert θ-scan to transport vs angle
    AMR_t, PHE_t, AHE_t, TOT_t = hall_from_angle_scan(scan_theta)
    plt.figure(figsize=(24,8))
    plt.plot(theta_grid, AMR_t[:, 1], label="AMR")
    plt.plot(theta_grid, TOT_t[:, 1], label="PHE + AHE")
    plt.xlabel(r"$\theta_H$ (rad)"); plt.ylabel("signal (arb)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("Transport vs θ at |H| = {:.2f}".format(H_abs))

    # φ-scan with θ fixed
    theta_fixed = np.deg2rad(90.0)
    phi_grid = np.deg2rad(
        np.concatenate(
            [np.linspace(0.0, 360.0, 181),
             np.linspace(360.0, 0.0, 181)]
            )
        )
    scan_phi = model.angle_scan_phi(H_abs, theta_fixed, phi_grid)
    fig4 = plot_angle_scan(scan_phi, angle_label=r"$\phi_H$ (rad)")
    fig4.suptitle("Angle scan in φ at |H| = {:.2f}".format(H_abs))

    # Convert φ-scan to transport vs angle
    AMR_p, PHE_p, AHE_p, TOT_p = hall_from_angle_scan(scan_phi)
    plt.figure(figsize=(6,4))
    plt.plot(phi_grid, AMR_p[:, 1], label="AMR")
    plt.plot(phi_grid, TOT_p[:, 1], label="PHE + AHE")
    plt.xlabel(r"$\phi_H$ (rad)"); plt.ylabel("signal (arb)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("Transport vs φ at |H| = {:.2f}".format(H_abs))

    plt.show()

if __name__ == "__main__":
    main()
