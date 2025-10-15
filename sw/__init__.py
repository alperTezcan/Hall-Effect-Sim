from .anisotropies import (
    Anisotropy, Uniaxial, ManyUniaxial, Cubic, Hexagonal,
    axis_from_degrees, euler_zyx,
)
from .SW_model import SWModel, m_from_angles, angles_from_vec
from .experiments import (
    hall_effects, hall_from_angle_scan, switching_fields,
    astroid_theta_scan, astroid_phi_scan, astroid_3d,
    plot_hysteresis, plot_angle_scan, plot_halls,
)

__all__ = [
    "Anisotropy", "Uniaxial", "ManyUniaxial", "Cubic", "Hexagonal",
    "axis_from_degrees", "euler_zyx",
    "SWModel", "m_from_angles", "angles_from_vec",
    "hall_effects", "hall_from_angle_scan", "switching_fields",
    "astroid_theta_scan", "astroid_phi_scan", "astroid_3d",
    "plot_hysteresis", "plot_angle_scan", "plot_halls",
]

__version__ = "2.0.0"