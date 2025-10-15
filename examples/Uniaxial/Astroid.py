from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from sw import (
    SWModel, axis_from_degrees, astroid_theta_scan
)

def main():
    # Define anisotropy: easy axis along +z with Han=0.5
    axes = [axis_from_degrees(Han=.5, theta_deg=0., phi_deg=0.)]
    model = SWModel(axes, Hd=0.0)

    # Field sweep: from +H to −H and back
    H_forward = np.linspace(+6.0, -6.0, 2401)
    H_backward = np.linspace(-6.0, +6.0, 2401)
    H = np.concatenate([H_forward, H_backward])  

    # Astroid slice
    Hx, Hy = astroid_theta_scan(model, phi_fixed=0.0, H_sweep=H, resolution=45)
    plt.figure(); plt.scatter(Hx, Hy, s=6); plt.gca().set_aspect("equal"); plt.title("Astroid theta-scan")

    plt.show()

if __name__ == "__main__":
    main()