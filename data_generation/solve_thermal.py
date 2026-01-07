"""
solve_thermal.py - Direct Topological Mapping (Fixed Interpolation)
"""

import warnings
from typing import Tuple

import numpy as np

# Local imports
from data import (
    COMPONENT_TAGS,
    K_COMPONENTS,
    K_FLUID,
    K_PCB,
    MESH_SIZE,
    Q_DOT_COMP,
    RHO_CP_FLUID,
    TEMP_INLET,
)
from skfem import (
    Basis,
    BilinearForm,
    ElementQuad0,
    ElementQuad1,
    ElementVectorH1,
    LinearForm,
    Mesh,
    asm,
    condense,
    solve,
)
from skfem.helpers import dot, grad

# Suppress skfem warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. Weak Forms
# =============================================================================


@BilinearForm
def advection_diffusion(u, v, w):
    """Bilinear form for advection-diffusion equation."""
    # Diffusion
    diffusion = w.k * dot(grad(u), grad(v))

    # Advection
    vel = np.array([w.u_vel, w.v_vel])
    conv_term = dot(vel, grad(u))
    advection = w.rho_cp * conv_term * v

    # SUPG Stabilization
    v_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + 1e-12)
    h = MESH_SIZE
    tau = 1.0 / (4.0 * w.k / (h**2 * w.rho_cp + 1e-12) + 2.0 * v_mag / h)

    residual = w.rho_cp * dot(vel, grad(u))
    v_stream = dot(vel, grad(v))

    supg = tau * residual * v_stream

    return diffusion + advection + supg


@LinearForm
def heat_source_load(v, w):
    """Linear form for heat source load."""
    return w.q_val * v


# =============================================================================
# 2. Main Thermal Solver
# =============================================================================


def _setup_material_properties(mesh: Mesh):
    """Initialize material properties (k, rho_cp, heat source) on elements."""
    k_elem = np.zeros(mesh.nelements)
    rho_cp_elem = np.ones(mesh.nelements) * 1e-3
    q_elem = np.zeros(mesh.nelements)

    if "Fluid" in mesh.subdomains:
        idx = mesh.subdomains["Fluid"]
        k_elem[idx] = K_FLUID
        rho_cp_elem[idx] = RHO_CP_FLUID

    if "PCB" in mesh.subdomains:
        k_elem[mesh.subdomains["PCB"]] = K_PCB

    for i, _ in enumerate(COMPONENT_TAGS):
        comp_name = f"Component_{i+1}"
        if comp_name in mesh.subdomains:
            idx = mesh.subdomains[comp_name]
            k_elem[idx] = K_COMPONENTS[i]
            # Heat Source
            _basis_tmp = Basis(mesh, ElementQuad1(), elements=idx)
            area_comp = np.sum(_basis_tmp.dx)
            if area_comp > 1e-12:
                q_elem[idx] = Q_DOT_COMP[i] / area_comp
    return k_elem, rho_cp_elem, q_elem


def _map_velocity_to_thermal(mesh: Mesh, fluid_sol: np.ndarray, fluid_basis: Basis):
    """Maps velocity from fluid basis to thermal basis (P1)."""
    thermal_basis = Basis(mesh, ElementQuad1())
    fluid_nodes = np.unique(mesh.t[:, mesh.subdomains["Fluid"]])

    u_global = np.zeros(thermal_basis.N)
    v_global = np.zeros(thermal_basis.N)

    u_global[thermal_basis.nodal_dofs[0, fluid_nodes]] = fluid_sol[
        fluid_basis.nodal_dofs[0, fluid_nodes]
    ]
    v_global[thermal_basis.nodal_dofs[0, fluid_nodes]] = fluid_sol[
        fluid_basis.nodal_dofs[1, fluid_nodes]
    ]

    vec_p1_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
    vel_p1_flat = vec_p1_basis.zeros()
    idxs = vec_p1_basis.split_indices()
    vel_p1_flat[idxs[0]], vel_p1_flat[idxs[1]] = u_global, v_global

    return u_global, v_global, thermal_basis, vel_p1_flat


def _setup_thermal_bcs(thermal_basis, mesh):
    """Setup thermal boundary conditions."""
    thermal_dofs = thermal_basis.get_dofs(mesh.boundaries)
    d_dofs = np.array([], dtype=int)
    x_init = thermal_basis.zeros()

    if "Inlet" in thermal_dofs:
        inlet_nodes = thermal_dofs["Inlet"].all()
        x_init[inlet_nodes] = TEMP_INLET
        d_dofs = np.union1d(d_dofs, inlet_nodes)
    return x_init, d_dofs


def solve_thermal(
    mesh: Mesh, fluid_sol: np.ndarray, fluid_basis: Basis
) -> Tuple[np.ndarray, Basis, np.ndarray]:
    """
    Solves thermal equation using TOPOLOGICAL mapping of vertex DOFs.
    Directly maps indices from Fluid Basis to Thermal Basis.
    """
    print("Initializing Thermal Finite Elements (P1)...")
    k_elem, rho_cp_elem, q_elem = _setup_material_properties(mesh)

    print("Mapping Velocity (Topological DOF Match)...")
    u_global, v_global, thermal_basis, vel_p1_flat = _map_velocity_to_thermal(
        mesh, fluid_sol, fluid_basis
    )

    print("Assembling Thermal System...")
    basis0 = Basis(mesh, ElementQuad0(), quadrature=thermal_basis.quadrature)

    x_init, d_dofs = _setup_thermal_bcs(thermal_basis, mesh)

    print("Solving Thermal Linear System...")
    t_sol = solve(
        *condense(
            asm(
                advection_diffusion,
                thermal_basis,
                k=basis0.interpolate(k_elem),
                rho_cp=basis0.interpolate(rho_cp_elem),
                u_vel=u_global,
                v_vel=v_global,
            ),
            asm(heat_source_load, thermal_basis, q_val=basis0.interpolate(q_elem)),
            x=x_init,
            D=d_dofs,
        )
    )

    print(f"  -> Solved. Range: [{t_sol.min():.2f}, {t_sol.max():.2f}] K")
    return t_sol, thermal_basis, vel_p1_flat
