from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from sw import (
    SWModel, axis_from_degrees, hall_effects, plot_hysteresis, plot_halls
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
    theta_H = np.deg2rad(30)
    phi_H = np.deg2rad(120)
    data = model.hysteresis(theta_H=theta_H, phi_H=phi_H, H_values=H)

    # Plot hysteresis
    plot_hysteresis(data)

    # Transport curves
    AMR, PHE, AHE, TOT = hall_effects(model, theta_H, phi_H, H)
    plot_halls(AMR, PHE, AHE, TOT)

    plt.show()

if __name__ == "__main__":
    main()