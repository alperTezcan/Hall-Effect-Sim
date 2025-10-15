from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from sw import (
    SWModel, Uniaxial, axis_from_degrees,
    hall_effects, hall_from_angle_scan,
    plot_hysteresis, plot_angle_scan, plot_halls,
    astroid_theta_scan,
)

def main():
    # Define anisotropy: easy axis along +z with Han=0.5
    axes = [axis_from_degrees(Han=.5, theta_deg=0., phi_deg=0.)]
    model = SWModel(axes, Hd=0.0)

    # Field sweep: from +H to −H and back
    H_forward = np.linspace(+6.0, -6.0, 2401)
    H_backward = np.linspace(-6.0, +6.0, 2401)
    H = np.concatenate([H_forward, H_backward])  

    # Apply field at theta_H = 5 deg, phi_H = 0 deg
    #theta_H = np.deg2rad(30)
    #phi_H = np.deg2rad(120)
    #data = model.hysteresis(theta_H=theta_H, phi_H=phi_H, H_values=H)

    # Plot hysteresis
    #plot_hysteresis(data)

    # Transport curves
    #AMR, PHE, AHE, TOT = hall_effects(model, theta_H, phi_H, H)
    #plot_halls(AMR, PHE, AHE, TOT)

    # Angle scans at fixed |H|
    #H_abs = 2.0
    #theta_grid = np.deg2rad(np.linspace(0.0, 180.0, 181))
    #scan_theta = model.angle_scan_theta(H_abs, phi_H, theta_grid)
    #plot_angle_scan(scan_theta, angle_label=r"$\theta_H$ (rad)")

    # Hall vs angle
    #AMR_t, PHE_t, AHE_t, TOT_t = hall_from_angle_scan(scan_theta)
    #plt.figure(); plt.plot(theta_grid, AMR_t[:,1]); plt.title("AMR vs theta at |H|=const")

    # Astroid slice
    Hx, Hy = astroid_theta_scan(model, phi_fixed=0.0, H_sweep=H, resolution=30)
    plt.figure(); plt.scatter(Hx, Hy, s=4); plt.gca().set_aspect("equal"); plt.title("Astroid theta-scan")

    plt.show()

if __name__ == "__main__":
    main()