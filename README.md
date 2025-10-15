# Stoner-Wohlfarth (SW) Macrospin Simulations

**Author:** Alper TEZCAN • PhD Track, Institut Polytechnique de Paris  
**Advisors:** Prof. Jean‑Eric Wegrowe (supervisor), Valentin Desbuis (mentorship)  
**License:** MIT (see `LICENSE`)

This library implements fast, modular simulations of the Stoner-Wohlfarth model with multiple anisotropy terms and experiment helpers.

---

## Features

- Core SW energy + numerical minimization (L‑BFGS‑B)
- Anisotropy hierarchy:
  - `Uniaxial` (single axis)
  - `ManyUniaxial` (sum of multiple uniaxials)
  - `Cubic` crystal anisotropy
  - `Hexagonal` crystal anisotropy
- Angle scans ((theta/phi)) and field sweeps (hysteresis)
- Transport helpers: AMR / PHE / AHE, plus basic astroid utilities
- Simple plotting utilities for quick inspection

**Requirements:** Python >= 3.9, NumPy, SciPy, Matplotlib (see `pyproject.toml`).

---

## Package Layout

```
sw/
  ├─ anisotropies.py   # Anisotropy hierarchy (Uniaxial, ManyUniaxial, Cubic, Hexagonal)
  ├─ SW_model.py       # SWModel: energies, minimization, sweeps & scans
  ├─ experiments.py    # Transport, switching/astroid, plotting helpers
  └─ __init__.py       # Public API
examples/
  └─ run_demo.py
pyproject.toml
README.md
LICENSE
```

---

## API Highlights

- **Anisotropies**
  - `Uniaxial(Han, u)` — with `axis_from_degrees(Han, theta_deg, phi_deg)` helper
  - `ManyUniaxial(Han_list=[...], u_list=[...])` or `ManyUniaxial(terms=[Uniaxial(...), ...])`
  - `Cubic(K1, K2=0.0, R=None)` — energy in the **crystal frame** (`a = R @ M`)
  - `Hexagonal(K1, K2=0.0, K6=0.0, R=None)` — uses (theta, phi) in the crystal frame
  - `euler_zyx(z, y, x)` — build rotation matrix mapping **LAB → CRYSTAL**

- **SW_model**
  - `SWModel(axes, Hd=0.0)`
  - `hysteresis(theta_H, phi_H, H_values)`
  - `angle_scan_theta(H, phi_H, thetas)` / `angle_scan_phi(H, theta_H, phis)`

- **Experiments**
  - `hall_effects(...)` / `hall_from_angle_scan(...)`
  - `switching_fields(...)`, `astroid_*` helpers
  - `plot_hysteresis(...)`, `plot_angle_scan(...)`, `plot_halls(...)`

---

## Citing

If this software assists your research, please cite it. Example BibTeX:

```bibtex
@software{tezcan_stoner_wohlfarth_2025,
  author  = {Alper Tezcan},
  title   = {Stoner-Wohlfarth Macrospin Simulations},
  year    = {2025},
  version = {2.0.0},
  url     = {https://github.com/your-org-or-user/stoner-wohlfarth}
}
```

---

## Contributing

Pull requests and issues are welcome. Please include a minimal failing example if you report a bug.