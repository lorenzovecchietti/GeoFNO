"""
Thermal solver for GeoFNO using topological vertex mapping and Numba acceleration.
"""

import warnings
from typing import List, Tuple

import numba
import numpy as np
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
from pypardiso import spsolve
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

warnings.filterwarnings("ignore", category=RuntimeWarning)


@numba.jit(nopython=True, fastmath=True, cache=True)
def _advection_diffusion_kernel(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    du, dv, _u_val, v_val, u_vel, v_vel, k, rho_cp, h
):
    """
    JIT-compiled kernel for Diffusion + Advection + SUPG calculation.
    """
    # 1. Diffusion: k * (grad(u) . grad(v))
    dot_grad = du[0] * dv[0] + du[1] * dv[1]
    diffusion = k * dot_grad

    # 2. Advection: rho_cp * (vel . grad(u)) * v
    vel_dot_grad_u = u_vel * du[0] + v_vel * du[1]
    advection = rho_cp * vel_dot_grad_u * v_val

    # 3. SUPG Stabilization
    v_mag_sq = u_vel * u_vel + v_vel * v_vel
    v_mag = np.sqrt(v_mag_sq + 1e-12)

    # Tau calculation
    denom_diff = (4.0 * k) / (h * h * rho_cp + 1e-12)
    denom_adv = (2.0 * v_mag) / h
    tau = 1.0 / (denom_diff + denom_adv + 1e-15)

    residual = rho_cp * vel_dot_grad_u
    v_stream = u_vel * dv[0] + v_vel * dv[1]

    supg = tau * residual * v_stream

    return diffusion + advection + supg


@BilinearForm
def advection_diffusion(u, v, w):
    """Bilinear form wrapper calling Numba kernel."""
    return _advection_diffusion_kernel(
        u.grad,
        v.grad,
        u.value,
        v.value,
        w.u_vel,
        w.v_vel,
        w.k,
        w.rho_cp,
        MESH_SIZE,
    )


@LinearForm
def heat_source_load(v, w):
    """Linear form for heat source load."""
    return w.q_val * v


def _setup_material_properties(
    mesh: Mesh,
    k_pcb: float = K_PCB,
    k_comps: List[float] | None = None,
    q_dot_comp: List[float] | None = None,
):
    """Initialize material properties (k, rho_cp, heat source) on mesh elements."""
    if k_comps is None:
        k_comps = K_COMPONENTS
    if q_dot_comp is None:
        q_dot_comp = Q_DOT_COMP

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
            k_elem[idx] = (
                k_comps[i] if isinstance(k_comps, (list, np.ndarray)) else k_comps
            )
            _basis_tmp = Basis(mesh, ElementQuad1(), elements=idx)
            area_comp = np.sum(_basis_tmp.dx)
            if area_comp > 1e-12:
                q_val = (
                    q_dot_comp[i]
                    if isinstance(q_dot_comp, (list, np.ndarray))
                    else q_dot_comp
                )
                q_elem[idx] = q_val / area_comp
    return k_elem, rho_cp_elem, q_elem


def _map_velocity_to_thermal(mesh: Mesh, fluid_sol: np.ndarray, fluid_basis: Basis):
    """Map velocity from fluid basis (Quad2) to thermal basis (Quad1)."""
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
    """Setup Dirichlet boundary conditions for temperature."""
    thermal_dofs = thermal_basis.get_dofs(mesh.boundaries)
    d_dofs = np.array([], dtype=int)
    x_init = thermal_basis.zeros()

    if "Inlet" in thermal_dofs:
        inlet_nodes = thermal_dofs["Inlet"].all()
        x_init[inlet_nodes] = TEMP_INLET
        d_dofs = np.union1d(d_dofs, inlet_nodes)
    return x_init, d_dofs


def solve_thermal(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    mesh: Mesh,
    fluid_sol: np.ndarray,
    fluid_basis: Basis,
    k_pcb: float = K_PCB,
    k_comps: List[float] | None = None,
    q_dot_comp: List[float] | None = None,
) -> Tuple[np.ndarray, Basis, np.ndarray]:
    """Solve the heat equation with advection on the mesh."""
    if k_comps is None:
        k_comps = K_COMPONENTS
    if q_dot_comp is None:
        q_dot_comp = Q_DOT_COMP

    print("Initializing Thermal Finite Elements (P1)...")
    k_elem, rho_cp_elem, q_elem = _setup_material_properties(
        mesh, k_pcb, k_comps, q_dot_comp
    )

    print("Mapping Velocity...")
    u_global, v_global, thermal_basis, vel_p1_flat = _map_velocity_to_thermal(
        mesh, fluid_sol, fluid_basis
    )

    print("Assembling Thermal System...")
    basis0 = Basis(mesh, ElementQuad0(), quadrature=thermal_basis.quadrature)
    x_init, d_dofs = _setup_thermal_bcs(thermal_basis, mesh)

    mat_a = asm(
        advection_diffusion,
        thermal_basis,
        k=basis0.interpolate(k_elem),
        rho_cp=basis0.interpolate(rho_cp_elem),
        u_vel=u_global,
        v_vel=v_global,
    )

    vec_b = asm(heat_source_load, thermal_basis, q_val=basis0.interpolate(q_elem))

    print("Solving Thermal Linear System...")
    mat_a_cond, vec_b_cond = condense(mat_a, vec_b, x=x_init, D=d_dofs, expand=False)
    x_sol_c = spsolve(mat_a_cond, vec_b_cond)

    t_sol = x_init.copy()
    i_dof = thermal_basis.complement_dofs(d_dofs)
    t_sol[i_dof] = x_sol_c

    print(f"  -> Solved. Range: [{t_sol.min():.2f}, {t_sol.max():.2f}] K")
    return t_sol, thermal_basis, vel_p1_flat
