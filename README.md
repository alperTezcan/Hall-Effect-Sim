---

Developed by **Alper TEZCAN**, January 2025
PhD Track, Institut Polytechnique de Paris

---

# Magnetic Anisotropy Simulations

This repository contains simulation and data analysis tools developed to study and generalize the **Stoner-Wohlfarth model** for magnetic anisotropy, under the supervision of Prof. Jean-Eric Wegrowe and with the mentorship of Valentin Desbuis.

## 📁 Project Structure

- **Codes/**  
  Contains the main simulation and analysis codebase.
  - Newer versions are modular and object-oriented, structured as a Python library compatible with Jupyter Notebooks.
  - Older versions are stored in `Codes/Old/`, and follow a single-script, linear style for prototyping purposes.

- **Data/**  
  Contains experimental measurement data provided by Valentin. These are used for comparison and validation against simulation results.

## ⚙️ Features

- Simulation of magnetic field and angle sweep experiments
- Support for multiple uniaxial and crystalline anisotropies (e.g., cubic, tetragonal, hexagonal)
- Visualization of magnetization curves and Stoner-Wohlfarth asteroids
- Energy minimization and comparison with experimental Hall effect data
- Basic data analysis tools using `numpy`, `pandas`, and Jupyter Notebooks

## 🧠 Future Plans

- Rewriting core simulation components in C++ for performance (ROOT framework)
- Advanced anisotropy extraction using machine learning
- Integration of transport phenomena and extended material systems