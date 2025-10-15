from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from sw import (
    SWModel, axis_from_degrees, hall_from_angle_scan, plot_angle_scan
)

def main():
    # Define anisotropy: easy axis along +z with Han=0.5
    axes = [axis_from_degrees(Han=.5, theta_deg=0., phi_deg=0.)]
    model = SWModel(axes, Hd=0.0)

    # Angle scans at fixed |H|
    H_abs = 2.0
    theta_grid = np.deg2rad(np.linspace(-180.0, 180.0, 181))
    phi_H = np.deg2rad(120)
    scan_theta = model.angle_scan_theta(H_abs, phi_H, theta_grid)
    plot_angle_scan(scan_theta, angle_label=r"$\theta_H$ (rad)")

    # Hall vs angle
    AMR_t, PHE_t, AHE_t, TOT_t = hall_from_angle_scan(scan_theta)
    plt.figure(); plt.plot(theta_grid, AMR_t[:,1]); plt.title("AMR vs theta at |H|=const")

    plt.show()

if __name__ == "__main__":
    main()