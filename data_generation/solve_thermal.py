"""
solve_thermal.py - Direct Topological Mapping (Numba Accelerated + PyPardiso)
"""

import warnings
from typing import Tuple, List

import numpy as np
import numba  # <--- NEW: Numba import

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
)
# Nota: solve è sostituito da pypardiso
from pypardiso import spsolve 
from skfem.helpers import dot, grad

# Suppress skfem warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 0. Numba Kernels (High Performance Math)
# =============================================================================

@numba.jit(nopython=True, fastmath=True, cache=True)
def _advection_diffusion_kernel(du, dv, u_val, v_val, u_vel, v_vel, k, rho_cp, h):
    """
    Kernel JIT compilato per calcolare: Diffusion + Advection + SUPG
    Operazioni eseguite in codice macchina puro.
    """
    # 1. Diffusion: k * (grad(u) . grad(v))
    # du e dv sono shape (2, N_points). du[0] = dx, du[1] = dy
    dot_grad = du[0] * dv[0] + du[1] * dv[1]
    diffusion = k * dot_grad

    # 2. Advection: rho_cp * (vel . grad(u)) * v
    vel_dot_grad_u = u_vel * du[0] + v_vel * du[1]
    advection = rho_cp * vel_dot_grad_u * v_val

    # 3. SUPG Stabilization
    # Magnitude velocity
    v_mag_sq = u_vel * u_vel + v_vel * v_vel
    # Evitiamo sqrt se possibile, ma qui serve per la formula tau standard
    v_mag = np.sqrt(v_mag_sq + 1e-12)

    # Tau parameter calculation
    # tau = 1 / ( 4k/(h^2*rho_cp) + 2|u|/h )
    # Denominatore sicuro
    denom_diff = (4.0 * k) / (h * h * rho_cp + 1e-12)
    denom_adv = (2.0 * v_mag) / h
    tau = 1.0 / (denom_diff + denom_adv + 1e-15)

    # Residuals
    residual = rho_cp * vel_dot_grad_u
    
    # Streamline derivative of test function: vel . grad(v)
    v_stream = u_vel * dv[0] + v_vel * dv[1]

    supg = tau * residual * v_stream

    return diffusion + advection + supg


# =============================================================================
# 1. Weak Forms
# =============================================================================

@BilinearForm
def advection_diffusion(u, v, w):
    """Bilinear form wrapper that calls Numba kernel."""
    # Pass raw numpy arrays to the JIT kernel
    return _advection_diffusion_kernel(
        u.grad,          # (2, N) array
        v.grad,          # (2, N) array
        u.value,         # (N) array (test function value)
        v.value,         # (N) array (test function value)
        w.u_vel,         # (N) from w
        w.v_vel,         # (N) from w
        w.k,             # (N) material prop
        w.rho_cp,        # (N) material prop
        MESH_SIZE        # constant
    )


@LinearForm
def heat_source_load(v, w):
    """Linear form for heat source load (Simple enough for pure NumPy)."""
    return w.q_val * v


# =============================================================================
# 2. Main Thermal Solver
# =============================================================================


def _setup_material_properties(
    mesh: Mesh,
    k_pcb: float = K_PCB,
    k_comps: List[float] = K_COMPONENTS,
    q_dot_comp: List[float] = Q_DOT_COMP,
):
    """Initialize material properties (k, rho_cp, heat source) on elements."""
    k_elem = np.zeros(mesh.nelements)
    rho_cp_elem = np.ones(mesh.nelements) * 1e-3
    q_elem = np.zeros(mesh.nelements)

    if "Fluid" in mesh.subdomains:
        idx = mesh.subdomains["Fluid"]
        k_elem[idx] = K_FLUID
        rho_cp_elem[idx] = RHO_CP_FLUID

    if "PCB" in mesh.subdomains:
        k_elem[mesh.subdomains["PCB"]] = k_pcb

    for i, _ in enumerate(COMPONENT_TAGS):
        comp_name = f"Component_{i+1}"
        if comp_name in mesh.subdomains:
            idx = mesh.subdomains[comp_name]
            k_elem[idx] = k_comps[i] if isinstance(k_comps, (list, np.ndarray)) else k_comps
            # Heat Source
            _basis_tmp = Basis(mesh, ElementQuad1(), elements=idx)
            area_comp = np.sum(_basis_tmp.dx)
            if area_comp > 1e-12:
                q_val = q_dot_comp[i] if isinstance(q_dot_comp, (list, np.ndarray)) else q_dot_comp
                q_elem[idx] = q_val / area_comp
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
    mesh: Mesh,
    fluid_sol: np.ndarray,
    fluid_basis: Basis,
    k_pcb: float = K_PCB,
    k_comps: List[float] = K_COMPONENTS,
    q_dot_comp: List[float] = Q_DOT_COMP,
) -> Tuple[np.ndarray, Basis, np.ndarray]:
    """
    Solves thermal equation using TOPOLOGICAL mapping of vertex DOFs.
    """
    print("Initializing Thermal Finite Elements (P1)...")
    k_elem, rho_cp_elem, q_elem = _setup_material_properties(mesh, k_pcb, k_comps, q_dot_comp)

    print("Mapping Velocity (Topological DOF Match)...")
    u_global, v_global, thermal_basis, vel_p1_flat = _map_velocity_to_thermal(
        mesh, fluid_sol, fluid_basis
    )

    print("Assembling Thermal System (Numba Optimized)...")
    basis0 = Basis(mesh, ElementQuad0(), quadrature=thermal_basis.quadrature)

    x_init, d_dofs = _setup_thermal_bcs(thermal_basis, mesh)
    
    # Assembly matrix A
    A = asm(
        advection_diffusion,
        thermal_basis,
        k=basis0.interpolate(k_elem),
        rho_cp=basis0.interpolate(rho_cp_elem),
        u_vel=u_global,
        v_vel=v_global,
    )
    
    # Assembly RHS b
    b = asm(heat_source_load, thermal_basis, q_val=basis0.interpolate(q_elem))

    print("Solving Thermal Linear System (PyPardiso)...")
    
    # Use PyPardiso instead of standard scipy/skfem solve
    A_cond, b_cond = condense(A, b, x=x_init, D=d_dofs, expand=False)
    
    # Solve condensed system
    x_sol_c = spsolve(A_cond, b_cond)
    
    # Expand solution
    t_sol = x_init.copy()
    i_dof = thermal_basis.complement_dofs(d_dofs)
    t_sol[i_dof] = x_sol_c

    print(f"  -> Solved. Range: [{t_sol.min():.2f}, {t_sol.max():.2f}] K")
    return t_sol, thermal_basis, vel_p1_flat