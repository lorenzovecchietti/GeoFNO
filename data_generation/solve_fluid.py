"""
solve_fluid.py - Optimized Version (Split Assembly + Fast Solver)
"""

import warnings
from pathlib import Path
from typing import Tuple
import time

import numpy as np

# Local imports
from data import MAX_ITERS, MESH_FILE, MESH_SIZE, NU, OUTPUT_FOLDER, TOL, VEL_INLET
from generate_mesh import CircuitBoard, generate_gmsh_mesh_2d

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

from pypardiso import spsolve

# Suppress skfem warnings about singular matrix if we handle them
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    Re-assembled ONLY when viscosity 'nu' changes.
    """
    return (
        w.nu * inner(grad(u), grad(v))
        - p * div(v)
        - q * div(u)
        - EPS_PENALTY * p * q
    )


@BilinearForm
def navier_dynamic(u, p, v, q, w):
    """
    PART B: DYNAMIC TERMS (Non-Linear)
    Depends on 'w.u_prev'.
    Re-assembled EVERY iteration.
    Includes: Convection, SUPG, Grad-Div
    """
    # 1. Convection Picard: (u_prev · ∇) u
    convection = dot(w.u_prev, u.grad[0]) * v[0] + dot(w.u_prev, u.grad[1]) * v[1]

    # Calculate magnitude for stabilization parameters
    u_mag = np.sqrt(w.u_prev[0] ** 2 + w.u_prev[1] ** 2 + 1e-12)

    # 2. SUPG Stabilization
    # Tau calculation
    tau = DELTA_SUPG / np.sqrt(
        (2.0 * u_mag / w.h) ** 2 + (9.0 * 4 * w.nu / w.h**2) ** 2
    )

    # Residuals for SUPG (Simplified for Picard: convection + grad(p))
    # Note: We apply SUPG logic to the test functions
    res_x = dot(w.u_prev, u.grad[0]) + p.grad[0]
    res_y = dot(w.u_prev, u.grad[1]) + p.grad[1]

    supg = tau * (res_x * dot(w.u_prev, v.grad[0]) + res_y * dot(w.u_prev, v.grad[1]))

    # 3. Grad-Div Stabilization
    graddiv = DELTA_GRADDIV * u_mag * w.h * div(u) * div(v)

    return convection + supg + graddiv


@LinearForm
def navier_stokes_rhs(_v, _q, _w):
    return 0.0


# =============================================================================
# 2. Helper Functions (Unchanged mostly)
# =============================================================================

def get_mesh() -> Mesh:
    """Load or generate the mesh."""
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(exist_ok=True, parents=True)
    mesh_path = out_dir / MESH_FILE

    if not mesh_path.exists():
        print(f"Generating mesh at {mesh_path}...")
        cb = CircuitBoard(
            h_pcb=0.05, w_pcb=0.7, n_up=3,
            w_comps=[0.1, 0.2, 0.05, 0.15, 0.2],
            h_comps=[1.1, 1.5, 1.0, 1.2, 1.5],
        )
        generate_gmsh_mesh_2d(cb, mesh_size=MESH_SIZE, output_file=str(mesh_path))
    return Mesh.load(str(mesh_path))


def get_boundary_dofs(basis: Basis, mesh: Mesh):
    """Compute Dirichlet DOFs for fluid solver."""
    dofs = basis.get_dofs(mesh.boundaries)
    idxs = basis.split_indices()[0]
    idx_u, idx_v = idxs[basis.split_bases()[0].split_indices()]

    # Inlet Setup
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
            # Parabolic profile
            u_profile = VEL_INLET * (1.0 - (2.0 * (y_coords - y_mid) / h_h) ** 2)
            u_inlet_vec[u_inlet_dofs] = u_profile

    # Wall & Ghost Setup
    wall_dofs = np.array([], dtype=int)
    for b_name in ["Walls", "SolidInterfaces"]:
        if b_name in dofs:
            bd_dofs = dofs[b_name].all()
            wall_dofs = np.concatenate([
                wall_dofs,
                np.intersect1d(bd_dofs, idx_u),
                np.intersect1d(bd_dofs, idx_v),
            ])

    fluid_dofs_indices = np.unique(basis.element_dofs)
    solid_ghost_dofs = np.setdiff1d(np.arange(basis.N), fluid_dofs_indices)

    d_dofs = np.unique(
        np.concatenate([u_inlet_dofs, v_inlet_dofs, wall_dofs, solid_ghost_dofs])
    ).astype(int)

    return d_dofs, u_inlet_vec


def run_solver_loop(basis, x_sol, d_dofs, viscosities):
    """
    Run the iteration loop for Navier-Stokes solver.
    """
    # Pre-assemble RHS (always zero in this formulation)
    b_zero = asm(navier_stokes_rhs, basis)

    # --- FIX: Calculate Free Indices (I) manually ---
    I = basis.complement_dofs(d_dofs)

    for step_idx, nu_curr in enumerate(viscosities):
        print(f"\n--- STEP {step_idx + 1}/{len(viscosities)}: Nu = {nu_curr:.2e} ---")

        t0 = time.time()
        A_static = asm(stokes_static, basis, nu=nu_curr)
        print(f"  Static Assembly: {time.time() - t0:.4f}s")

        for i in range(MAX_ITERS):
            t_iter_start = time.time()

            args_dyn = {
                "u_prev": basis.interpolate(x_sol)[0],
                "nu": nu_curr,
                "h": MESH_SIZE,
            }

            try:
                A_dynamic = asm(navier_dynamic, basis, **args_dyn)
                A_total = A_static + A_dynamic
                A_c, b_c = condense(A_total, b_zero, x=x_sol, D=d_dofs, expand=False)
                x_c = x_sol[I]
                dx = spsolve(A_c, b_c - A_c @ x_c)
                x_new_c = x_c + dx
                x_new = x_sol.copy()
                x_new[I] = x_new_c
                x_sol = 0.7 * x_new + 0.3 * x_sol

            except RuntimeError as err:
                print(f"  Solver failed (Singular?): {err}")
                break

            # Check Convergence
            rel_error = np.linalg.norm(x_new - x_sol) / (np.linalg.norm(x_new) + 1e-12)

            t_iter_end = time.time()
            print(f"  Iter {i + 1:2d}: err={rel_error:.2e} ({t_iter_end - t_iter_start:.3f}s)")

            if rel_error < TOL:
                print("  Convergence reached.")
                break

    return x_sol


def solve_fluid(mesh: Mesh) -> Tuple[np.ndarray, Basis]:
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
    d_dofs, x_sol = get_boundary_dofs(basis, mesh)

    # Viscosity stepping (Continuation Method)
    viscosities = np.geomspace(5e-2, NU, 3)

    print("Initializing with Stokes (Static)...")
    # We can reuse the static form here!
    A_init = asm(stokes_static, basis, nu=viscosities[0])
    b_init = asm(navier_stokes_rhs, basis)

    # Solve initial Stokes
    A_c, b_c = condense(A_init, b_init, x=x_sol, D=d_dofs, expand=False)
    x_sol_c = spsolve(A_c, b_c)
    I = basis.complement_dofs(d_dofs)
    x_sol[I] = x_sol_c

    print("  -> Stokes initialization successful.")

    print(f"\nStarting Solver Loop. Target Nu={NU:.2e}")
    x_sol = run_solver_loop(basis, x_sol, d_dofs, [viscosities[-1]])

    return x_sol, basis


# =============================================================================
# 3. Main Execution
# =============================================================================

def main():
    """Main function for standalone testing of fluid solver."""
    mesh = get_mesh()
    x_sol, basis = solve_fluid(mesh)
    print("\nGeneration Complete.")

if __name__ == "__main__":
    main()