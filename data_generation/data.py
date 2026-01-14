"""
Centralized physical parameters, mesh tags, and simulation settings.
All constants are defined with clear units and documentation.
"""

from pathlib import Path
from typing import Final, List

# Mesh Physical Group Tags
# Must match tags in generate_mesh.py
FLUID_TAG: Final[int] = 1
PCB_TAG: Final[int] = 2
COMPONENT_BASE_TAG: Final[int] = 3
COMPONENT_TAGS: Final[List[int]] = [3, 4, 5, 6, 7]

INLET_TAG: Final[int] = 11
OUTLET_TAG: Final[int] = 12
WALLS_TAG: Final[int] = 13
SOLID_TAG: Final[int] = 14  # Fluid-solid interface (no-slip)

# Fluid Properties (Air at 20°C, 1 atm)
NU: Final[float] = 1e-5
K_FLUID: Final[float] = 0.0263  # Thermal conductivity [W/m·K]
RHO_CP_FLUID: Final[float] = 1213.435  # Volumetric heat cap. [J/m³·K]

# Solid Properties
K_PCB: Final[float] = 0.3  # PCB thermal conductivity [W/m·K]

K_COMPONENTS: Final[List[float]] = [
    381.0,  # Copper-like
    253.0,  # Aluminum-like
    187.0,
    324.0,
    293.0,
]  # Component thermal conductivities [W/m·K]

Q_DOT_COMP: Final[List[float]] = [
    1.0e1,  # 10 W
    2.0e1,  # 20 W
    1.4e1,
    2.1e1,
    1.8e1,
]  # Heat generation per component [W]

# Boundary Conditions
TEMP_INLET: Final[float] = 293.15  # Inlet temperature [K] (20°C)
VEL_INLET: Final[float] = 0.025  # Inlet velocity (x-direction) [m/s]

# Simulation and I/O Settings
MESH_FILE: Final[str] = "circuit_mesh_2d.msh"
MESH_SIZE: Final[float] = 0.25  # Characteristic mesh size [m]
OUTPUT_FOLDER: Final[str] = "results"

# Simulation parameters
MAX_ITERS: Final[int] = 25
TOL: Final[float] = 5e-3
K: Final[int] = 1  # Polynomial degree

DATASET_FOLDER = Path("dataset")
N_SAMPLES = 500
MAX_CPUS = 24

# Parameter Ranges
PARAM_RANGES = {
    "k_comps": (200.0, 500.0),
    "k_pcb": (0.01, 1.0),
    "vel_inlet": (0.001, 0.01),
    "h_pcb": (0.05, 0.2),
    "w_pcb": (0.5, 0.85),
    "n_up": (0, 5),
    "w_comps": (0.1, 0.3),
    "h_comps": (1.0, 1.5),
    "q_comps": (10.0, 30.0),
}

DIMENSIONS = 25


def log(msg: str) -> None:
    """
    Print message to console.

    Args:
        msg: Message to print.
    """
    print(msg, flush=True)
