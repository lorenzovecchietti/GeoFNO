"""
solve_fluid.py - Optimized Version (Split Assembly + Fast Solver + Numba)
"""

import time
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import numba  # <--- NEW: Numba import

# Local imports
from data import MAX_ITERS, MESH_FILE, MESH_SIZE, NU, OUTPUT_FOLDER, TOL, VEL_INLET
from generate_mesh import CircuitBoard, generate_gmsh_mesh_2d
from pypardiso import spsolve

# Explicit imports
from skfem import (
    Basis,
    BilinearForm,
    ElementQuad1,
    ElementQuad2,
    ElementVectorH1,
    LinearForm,
    Mesh,
    asm,
    condense,
)
from skfem.helpers import div, dot, grad, inner

# Suppress skfem warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. Numba Kernels
# =============================================================================

@numba.jit(nopython=True, fastmath=True, cache=True)
def _navier_dynamic_kernel(u_grad, p_grad, v_val, v_grad, u_prev, nu, h, delta_supg, delta_graddiv):
    """
    Kernel JIT per Navier-Stokes iterativo:
    Calcola Convection + SUPG + GradDiv in un singolo passaggio ottimizzato.
    """
    # u_grad shape: (2, 2, N_points). indices: [component_u, deriv_dir, point]
    # v_val shape: (2, N_points)
    # v_grad shape: (2, 2, N_points)
    # u_prev shape: (2, N_points)
    
    # 1. Convection Picard: (u_prev . grad(u)) . v
    # (u_prev[0]*du/dx + u_prev[1]*du/dy) * v[0] ...
    
    # Calcolo (u_prev . grad) applicato a u_x e u_y
    # du_x / dt_streamline
    u_adv_x = u_prev[0] * u_grad[0, 0] + u_prev[1] * u_grad[0, 1] 
    # du_y / dt_streamline
    u_adv_y = u_prev[0] * u_grad[1, 0] + u_prev[1] * u_grad[1, 1]
    
    convection = u_adv_x * v_val[0] + u_adv_y * v_val[1]

    # 2. SUPG Parameter Calculation
    u_mag_sq = u_prev[0]**2 + u_prev[1]**2
    u_mag = np.sqrt(u_mag_sq + 1e-12)
    
    # Tau formula
    # (2|u|/h)^2
    term1 = (2.0 * u_mag / h)**2
    # (9 * 4nu / h^2)^2
    term2 = (36.0 * nu / (h*h))**2
    
    tau = delta_supg / np.sqrt(term1 + term2 + 1e-15)

    # 3. SUPG Residuals & Test Function
    # Res_x = (u_prev . grad(u))_x + dp/dx
    res_x = u_adv_x + p_grad[0]
    res_y = u_adv_y + p_grad[1]
    
    # Streamline derivative of test function v: (u_prev . grad(v))
    v_stream_x = u_prev[0] * v_grad[0, 0] + u_prev[1] * v_grad[0, 1]
    v_stream_y = u_prev[0] * v_grad[1, 0] + u_prev[1] * v_grad[1, 1]
    
    supg_term = tau * (res_x * v_stream_x + res_y * v_stream_y)
    
    # 4. Grad-Div Stabilization
    # div(u) = du_x/dx + du_y/dy
    div_u = u_grad[0, 0] + u_grad[1, 1]
    div_v = v_grad[0, 0] + v_grad[1, 1]
    
    graddiv_term = delta_graddiv * u_mag * h * div_u * div_v
    
    return convection + supg_term + graddiv_term

# =============================================================================
# 1. Weak Forms - SPLIT FOR SPEED
# =============================================================================
# Stabilization params
DELTA_SUPG = 0.5
DELTA_GRADDIV = 0.1
EPS_PENALTY = 1e-6


@BilinearForm
def stokes_static(u, p, v, q, w):
    """
    PART A: STATIC TERMS
    Depends only on 'w.nu' and mesh geometry.
    This is linear, skfem's default numpy broadcasting is fast enough here
    since it runs only once per viscosity step.
    """
    return (
        w.nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u) - EPS_PENALTY * p * q
    )


@BilinearForm
def navier_dynamic(u, p, v, _q, w):
    """
    PART B: DYNAMIC TERMS (Non-Linear)
    Depends on 'w.u_prev'. Calls Numba Kernel.
    """
    return _navier_dynamic_kernel(
        u.grad,             # (2, 2, N)
        p.grad,             # (2, N)
        v.value,            # (2, N)
        v.grad,             # (2, 2, N)
        w.u_prev,           # (2, N)
        w.nu,               # float/scalar broadcasted
        MESH_SIZE,          # constant
        DELTA_SUPG,         # constant
        DELTA_GRADDIV       # constant
    )


@LinearForm
def navier_stokes_rhs(_v, _q, _w):
    """Zero RHS for Navier-Stokes."""
    return 0.0


# =============================================================================
# 2. Helper Functions (Unchanged)
# =============================================================================


def get_mesh(cb: CircuitBoard = None, mesh_path: Path = None) -> Mesh:
    """Load or generate the mesh."""
    if mesh_path is None:
        out_dir = Path(OUTPUT_FOLDER)
        out_dir.mkdir(exist_ok=True, parents=True)
        mesh_path = out_dir / MESH_FILE
    else:
        mesh_path = Path(mesh_path)
        mesh_path.parent.mkdir(exist_ok=True, parents=True)

    if not mesh_path.exists():
        print(f"Generating mesh at {mesh_path}...")
        if cb is None:
            cb = CircuitBoard(
                h_pcb=0.05,
                w_pcb=0.7,
                n_up=3,
                w_comps=[0.1, 0.2, 0.05, 0.15, 0.2],
                h_comps=[1.1, 1.5, 1.0, 1.2, 1.5],
            )
        generate_gmsh_mesh_2d(cb, mesh_size=MESH_SIZE, output_file=str(mesh_path))
    return Mesh.load(str(mesh_path))


def _get_inlet_dofs(basis, dofs, idx_u, idx_v, vel_inlet=VEL_INLET):
    u_inlet_vec = np.zeros(basis.N)
    u_inlet_dofs = np.array([], dtype=int)
    v_inlet_dofs = np.array([], dtype=int)

    if "Inlet" in dofs:
        inlet_nodes = dofs["Inlet"].all()
        u_inlet_dofs = np.intersect1d(inlet_nodes, idx_u)
        v_inlet_dofs = np.intersect1d(inlet_nodes, idx_v)

        y_coords = basis.doflocs[1, u_inlet_dofs]
        if len(y_coords) > 0:
            y_min, y_max = y_coords.min(), y_coords.max()
            h_h = y_max - y_min
            y_mid = (y_max + y_min) / 2.0
            u_profile = vel_inlet * (1.0 - (2.0 * (y_coords - y_mid) / h_h) ** 2)
            u_inlet_vec[u_inlet_dofs] = u_profile
    return u_inlet_dofs, v_inlet_dofs, u_inlet_vec


def get_boundary_dofs(basis: Basis, mesh: Mesh, vel_inlet=VEL_INLET):
    """Compute Dirichlet DOFs for fluid solver."""
    dofs = basis.get_dofs(mesh.boundaries)
    idxs = basis.split_indices()[0]
    idx_u, idx_v = idxs[basis.split_bases()[0].split_indices()]

    u_inlet_dofs, v_inlet_dofs, u_inlet_vec = _get_inlet_dofs(basis, dofs, idx_u, idx_v, vel_inlet)

    # Wall & Ghost Setup
    wall_dofs = np.array([], dtype=int)
    for b_name in ["Walls", "SolidInterfaces"]:
        if b_name in dofs:
            bd_dofs = dofs[b_name].all()
            wall_dofs = np.concatenate(
                [
                    wall_dofs,
                    np.intersect1d(bd_dofs, idx_u),
                    np.intersect1d(bd_dofs, idx_v),
                ]
            )

    fluid_dofs_indices = np.unique(basis.element_dofs)
    solid_ghost_dofs = np.setdiff1d(np.arange(basis.N), fluid_dofs_indices)

    d_dofs = np.unique(
        np.concatenate([u_inlet_dofs, v_inlet_dofs, wall_dofs, solid_ghost_dofs])
    ).astype(int)

    return d_dofs, u_inlet_vec


def _picard_iteration(basis, x_sol, a_static, d_dofs, nu_curr):
    """Perform a single Picard iteration."""
    try:
        # Numba-optimized assembly
        a_dynamic = asm(
            navier_dynamic,
            basis,
            u_prev=basis.interpolate(x_sol)[0],
            nu=nu_curr,
            h=MESH_SIZE,
        )
        a_condensed, b_c = condense(
            a_static + a_dynamic,
            asm(navier_stokes_rhs, basis),
            x=x_sol,
            D=d_dofs,
            expand=False,
        )
        i_dof = basis.complement_dofs(d_dofs)
        x_new = x_sol.copy()
        
        # PyPardiso Solver
        x_new[i_dof] = x_sol[i_dof] + spsolve(
            a_condensed, b_c - a_condensed @ x_sol[i_dof]
        )
        return 0.7 * x_new + 0.3 * x_sol, x_new
    except RuntimeError as err:
        print(f"  Solver failed (Singular?): {err}")
        return None, None


def run_solver_loop(basis, x_sol, d_dofs, viscosities):
    """
    Run the iteration loop for Navier-Stokes solver.
    """
    for step_idx, nu_curr in enumerate(viscosities):
        print(f"\n--- STEP {step_idx + 1}/{len(viscosities)}: Nu = {nu_curr:.2e} ---")

        t0 = time.time()
        a_static = asm(stokes_static, basis, nu=nu_curr)
        print(f"  Static Assembly: {time.time() - t0:.4f}s")

        for i in range(MAX_ITERS):
            t_iter_start = time.time()
            x_sol_new, x_new = _picard_iteration(
                basis, x_sol, a_static, d_dofs, nu_curr
            )
            if x_sol_new is None:
                break

            # Check Convergence
            rel_error = np.linalg.norm(x_new - x_sol) / (np.linalg.norm(x_new) + 1e-12)
            x_sol = x_sol_new

            print(
                f"  Iter {i + 1:2d}: err={rel_error:.2e}"
                + f" ({time.time() - t_iter_start:.3f}s)"
            )

            if rel_error < TOL:
                print("  Convergence reached.")
                break

    return x_sol


def solve_fluid(mesh: Mesh, vel_inlet=VEL_INLET) -> Tuple[np.ndarray, Basis]:
    """
    Solves Navier-Stokes on the Fluid subdomain.
    """
    print("Initializing Finite Elements...")
    # Quad2 (Velocity) + Quad1 (Pressure) = Taylor-Hood elements
    element = ElementVectorH1(ElementQuad2()) * ElementQuad1()

    if "Fluid" not in mesh.subdomains:
        raise ValueError("Mesh missing 'Fluid' subdomain.")

    basis = Basis(mesh, element, elements=mesh.subdomains["Fluid"])
    print(f"  -> Total DOFs: {basis.N}")

    print("Setting up Boundary Conditions...")
    d_dofs, x_sol = get_boundary_dofs(basis, mesh, vel_inlet)

    # Viscosity stepping (Continuation Method)
    viscosities = np.geomspace(5e-2, NU, 3)

    print("Initializing with Stokes (Static)...")
    a_init = asm(stokes_static, basis, nu=viscosities[0])
    b_init = asm(navier_stokes_rhs, basis)

    # Solve initial Stokes with PyPardiso
    a_condensed, b_c = condense(a_init, b_init, x=x_sol, D=d_dofs, expand=False)
    x_sol_c = spsolve(a_condensed, b_c)
    i_dof = basis.complement_dofs(d_dofs)
    x_sol[i_dof] = x_sol_c

    print("  -> Stokes initialization successful.")

    print(f"\nStarting Solver Loop (Numba Accelerated). Target Nu={NU:.2e}")
    x_sol = run_solver_loop(basis, x_sol, d_dofs, [viscosities[-1]])

    return x_sol, basis


# =============================================================================
# 3. Main Execution
# =============================================================================


def main():
    """Main function for standalone testing of fluid solver."""
    mesh = get_mesh()
    solve_fluid(mesh)
    print("\nGeneration Complete.")


if __name__ == "__main__":
    main()