"""
solve_thermal.py - Direct Topological Mapping (Fixed Interpolation)
"""

import warnings
from typing import Tuple

import numpy as np
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

# Suppress skfem warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. Weak Forms
# =============================================================================

@BilinearForm
def advection_diffusion(u, v, w):
    # Diffusion
    diffusion = w.k * dot(grad(u), grad(v))
    
    # Advection
    vel = np.array([w.u_vel, w.v_vel])
    conv_term = dot(vel, grad(u))
    advection = w.rho_cp * conv_term * v
    
    # SUPG Stabilization
    v_mag = np.sqrt(vel[0]**2 + vel[1]**2 + 1e-12)
    h = MESH_SIZE
    tau = 1.0 / (4.0 * w.k / (h**2 * w.rho_cp + 1e-12) + 2.0 * v_mag / h)

    residual = w.rho_cp * dot(vel, grad(u)) 
    v_stream = dot(vel, grad(v))
    
    supg = tau * residual * v_stream
    
    return diffusion + advection + supg


@LinearForm
def heat_source_load(v, w):
    return w.q_val * v


# =============================================================================
# 2. Main Thermal Solver
# =============================================================================

def solve_thermal(mesh: Mesh, fluid_sol: np.ndarray, fluid_basis: Basis) -> Tuple[np.ndarray, Basis, np.ndarray]:
    """
    Solves thermal equation using TOPOLOGICAL mapping of vertex DOFs.
    Directly maps indices from Fluid Basis to Thermal Basis.
    """
    print("Initializing Thermal Finite Elements (P1)...")
    thermal_basis = Basis(mesh, ElementQuad1())
    print(f"  -> Thermal DOFs: {thermal_basis.N}")

    # --- A. Material Properties (P0 Elemental) ---
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

    # =========================================================================
    # B. Direct Topological Velocity Transfer
    # =========================================================================
    print("Mapping Velocity (Topological DOF Match)...")
    
    # 1. Identificare i vertici fluido
    fluid_elem_indices = mesh.subdomains["Fluid"]
    fluid_vertex_indices = np.unique(mesh.t[:, fluid_elem_indices])
    
    # 2. Estrarre i valori P2 Fluid (usando nodal_dofs)
    # Assumiamo che fluid_basis sia mista (Vel, Pressure). 
    # nodal_dofs[0] -> u, nodal_dofs[1] -> v
    u_dof_indices = fluid_basis.nodal_dofs[0, fluid_vertex_indices]
    v_dof_indices = fluid_basis.nodal_dofs[1, fluid_vertex_indices]
    
    u_vals = fluid_sol[u_dof_indices]
    v_vals = fluid_sol[v_dof_indices]
    
    # 3. Assegnare i valori P1 Thermal
    u_global = np.zeros(thermal_basis.N)
    v_global = np.zeros(thermal_basis.N)
    
    target_dofs = thermal_basis.nodal_dofs[0, fluid_vertex_indices]
    
    u_global[target_dofs] = u_vals
    v_global[target_dofs] = v_vals
    
    # 4. Creazione vettore piatto per ritorno (opzionale, mantenuto per compatibilità)
    vec_p1_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
    vel_p1_flat = vec_p1_basis.zeros()
    vp1_idxs = vec_p1_basis.split_indices()
    vel_p1_flat[vp1_idxs[0]] = u_global
    vel_p1_flat[vp1_idxs[1]] = v_global
    
    print(f"  -> Transferred velocity on {len(fluid_vertex_indices)} shared vertices.")

    # =========================================================================
    # 3. Assembly & Solve
    # =========================================================================
    print("Assembling Thermal System...")
    
    # Correctly interpolate elemental properties to quadrature points using P0 basis
    # This avoids "ValueError: Input array has wrong size" and "NotImplementedError"
    basis0 = Basis(mesh, ElementQuad0(), quadrature=thermal_basis.quadrature)
    k_df = basis0.interpolate(k_elem)
    rho_cp_df = basis0.interpolate(rho_cp_elem)
    q_df = basis0.interpolate(q_elem)

    A = asm(advection_diffusion, thermal_basis, 
            k=k_df, rho_cp=rho_cp_df, u_vel=u_global, v_vel=v_global)
            
    b = asm(heat_source_load, thermal_basis, q_val=q_df)
    
    # BCs
    thermal_dofs = thermal_basis.get_dofs(mesh.boundaries)
    D_dofs = np.array([], dtype=int)
    x_init = thermal_basis.zeros()
    
    if "Inlet" in thermal_dofs:
        inlet_nodes = thermal_dofs["Inlet"].all()
        x_init[inlet_nodes] = TEMP_INLET
        D_dofs = np.union1d(D_dofs, inlet_nodes)
        
    print("Solving Thermal Linear System...")
    t_sol = solve(*condense(A, b, x=x_init, D=D_dofs))
    
    print(f"  -> Solved. Range: [{t_sol.min():.2f}, {t_sol.max():.2f}] K")
    return t_sol, thermal_basis, vel_p1_flat