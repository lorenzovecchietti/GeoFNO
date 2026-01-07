"""
solve_fluid.py - Modular & Robust Penalty Method Version
"""

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np

# Local imports
from data import MAX_ITERS, MESH_FILE, MESH_SIZE, NU, OUTPUT_FOLDER, TOL, VEL_INLET
from generate_mesh import CircuitBoard, generate_gmsh_mesh_2d
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

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
    solve,
)
from skfem.helpers import div, dot, grad, inner
from skfem.visuals.matplotlib import draw, plot

# Suppress skfem warnings about singular matrix if we handle them
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. Weak Forms (Penalty Method) - PURE & STATELESS
# =============================================================================
# Stabilization params
DELTA_SUPG = 0.5
DELTA_GRADDIV = 0.1
EPS_PENALTY = 1e-6


@BilinearForm
def stokes_flow(u, p, v, q, w):
    """
    Stokes flow bilinear form using parameters passed via 'w'.
    """
    return (
        w.nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u) - EPS_PENALTY * p * q
    )


@BilinearForm
def navier_stokes_lhs(u, p, v, q, w):
    """
    Navier-Stokes Linearized LHS (Picard).
    Reads 'w.u_prev', 'w.nu', 'w.h' from assembly context.
    """
    # Convection Picard: (u_prev · ∇) u
    convection = dot(w.u_prev, u.grad[0]) * v[0] + dot(w.u_prev, u.grad[1]) * v[1]

    u_mag = np.sqrt(w.u_prev[0] ** 2 + w.u_prev[1] ** 2 + 1e-12)

    # Tau SUPG
    tau = DELTA_SUPG / np.sqrt(
        (2.0 * u_mag / w.h) ** 2 + (9.0 * 4 * w.nu / w.h**2) ** 2
    )

    # 1. Galerkin Terms
    galerkin = (
        w.nu * inner(grad(u), grad(v))
        - p * div(v)
        - q * div(u)
        + convection
        - EPS_PENALTY * p * q
    )

    # 2. SUPG Terms
    res_x = dot(w.u_prev, u.grad[0]) + p.grad[0]
    res_y = dot(w.u_prev, u.grad[1]) + p.grad[1]
    supg = tau * (res_x * dot(w.u_prev, v.grad[0]) + res_y * dot(w.u_prev, v.grad[1]))

    # 3. Grad-Div Stabilization
    graddiv = DELTA_GRADDIV * u_mag * w.h * div(u) * div(v)

    return galerkin + supg + graddiv


@LinearForm
def navier_stokes_rhs(_v, _q, _w):
    """
    Navier-Stokes Linearized RHS (Picard).
    Reads 'w.u_prev', 'w.nu', 'w.h' from assembly context.
    """
    return 0.0


@LinearForm
def rhs_zero(_v, _q, _w):
    """
    Zero RHS.
    """
    return 0.0


# =============================================================================
# 2. Helper Functions
# =============================================================================


def get_mesh() -> Mesh:
    """Load or generate the mesh."""
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(exist_ok=True, parents=True)
    mesh_path = out_dir / MESH_FILE

    if not mesh_path.exists():
        print(f"Generating mesh at {mesh_path}...")
        cb = CircuitBoard(
            h_pcb=0.05,
            w_pcb=0.7,
            n_up=3,
            w_comps=[0.1, 0.2, 0.05, 0.15, 0.2],
            h_comps=[1.1, 1.5, 1.0, 1.2, 1.5],
        )
        generate_gmsh_mesh_2d(cb, mesh_size=MESH_SIZE, output_file=str(mesh_path))

    print(f"Loading mesh from {mesh_path}...")
    return Mesh.load(str(mesh_path))


def _get_inlet_dofs(basis, dofs, idx_u, idx_v):
    """Setup inlet velocity profile and return indices/values."""
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
            h_height = y_max - y_min
            y_mid = (y_max + y_min) / 2.0
            u_profile = VEL_INLET * (1.0 - (2.0 * (y_coords - y_mid) / h_height) ** 2)
            u_inlet_vec[u_inlet_dofs] = u_profile
    return u_inlet_vec, u_inlet_dofs, v_inlet_dofs


def _get_wall_and_ghost_dofs(basis, dofs, idx_u, idx_v):
    """Setup wall and ghost DOFs indices."""
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
    return wall_dofs, solid_ghost_dofs


def get_boundary_dofs(basis: Basis, mesh: Mesh):
    """Compute Dirichlet DOFs for fluid solver."""
    dofs = basis.get_dofs(mesh.boundaries)
    idxs = basis.split_indices()[0]
    idx_u, idx_v = idxs[basis.split_bases()[0].split_indices()]

    u_inlet_vec, u_inlet_dofs, v_inlet_dofs = _get_inlet_dofs(basis, dofs, idx_u, idx_v)
    wall_dofs, ghost_dofs = _get_wall_and_ghost_dofs(basis, dofs, idx_u, idx_v)

    d_dofs = np.unique(
        np.concatenate([u_inlet_dofs, v_inlet_dofs, wall_dofs, ghost_dofs])
    ).astype(int)

    return d_dofs, u_inlet_vec


def run_solver_loop(basis, x_sol, d_dofs, viscosities):
    """Run the iteration loop for Navier-Stokes solver."""
    for step_idx, nu_curr in enumerate(viscosities):
        print(f"\n--- STEP {step_idx+1}/{len(viscosities)}: Nu = {nu_curr:.2e} ---")

        for i in range(MAX_ITERS):
            args = {
                "u_prev": basis.interpolate(x_sol)[0],
                "nu": nu_curr,
                "h": MESH_SIZE,
            }
            try:
                # Assemble and Solve with Relaxation
                x_new = (
                    0.7
                    * solve(
                        *condense(
                            asm(navier_stokes_lhs, basis, **args),
                            asm(navier_stokes_rhs, basis, **args),
                            x=x_sol,
                            D=d_dofs,
                        )
                    )
                    + 0.3 * x_sol
                )
            except RuntimeError as err:
                print(f"  Solver failed (Singular?): {err}")
                break

            rel_error = np.linalg.norm(x_new - x_sol) / (np.linalg.norm(x_new) + 1e-12)
            x_sol = x_new
            u_i, v_i, _ = basis.nodal_dofs
            max_vel = np.max(np.linalg.norm(np.stack([x_sol[u_i], x_sol[v_i]]), axis=0))
            print(f"  Iter {i+1:2d}: err={rel_error:.2e}, max_u={max_vel:.3f}")

            if rel_error < TOL:
                print("  Convergence reached.")
                break
    return x_sol


def solve_fluid(mesh: Mesh) -> Tuple[np.ndarray, Basis]:
    """
    Solves Navier-Stokes on the Fluid subdomain.
    """
    print("Initializing Finite Elements...")
    element = ElementVectorH1(ElementQuad2()) * ElementQuad1()

    if "Fluid" not in mesh.subdomains:
        raise ValueError("Mesh missing 'Fluid' subdomain.")

    basis = Basis(mesh, element, elements=mesh.subdomains["Fluid"])
    print(f"  -> Total DOFs: {basis.N}")

    print("Setting up Boundary Conditions...")
    d_dofs, x_sol = get_boundary_dofs(basis, mesh)

    # Viscosity stepping (Continuation Method)
    viscosities = np.geomspace(5e-2, NU, 8)

    print("Initializing with Penalized Stokes...")
    a_stokes = asm(stokes_flow, basis, nu=viscosities[0])
    b_stokes = asm(rhs_zero, basis)
    x_sol = solve(*condense(a_stokes, b_stokes, x=x_sol, D=d_dofs))
    print("  -> Stokes initialization successful.")

    print(f"\nStarting Solver Loop. Target Nu={NU:.2e}")
    x_sol = run_solver_loop(basis, x_sol, d_dofs, viscosities)

    return x_sol, basis


# =============================================================================
# 3. Main Execution
# =============================================================================


def _plot_fluid_results(mesh, basis, x_sol, out_dir):
    """Internal helper to plot fluid simulation results."""
    u_idx, v_idx, p_idx = basis.nodal_dofs
    u_sol = np.stack([x_sol[u_idx], x_sol[v_idx]]).T

    mesh.save(
        str(out_dir / "fluid_solution.vtk"),
        {"velocity": u_sol, "pressure": x_sol[p_idx]},
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    val_mag = np.linalg.norm(u_sol, axis=1)

    plot(mesh, val_mag, ax=ax, cmap="turbo", shading="gouraud")
    draw(mesh, ax=ax, color="gray", alpha=0.1, linewidth=0.05)
    draw(mesh, ax=ax, boundaries_only=True, color="black", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    norm = mcolors.Normalize(vmin=val_mag.min(), vmax=val_mag.max())
    s_map = cm.ScalarMappable(norm=norm, cmap="turbo")
    s_map.set_array([])
    fig.colorbar(s_map, ax=ax).set_label("Velocity Magnitude [m/s]")

    plt.savefig(out_dir / "fluid_velocity.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    """Main function for standalone testing of fluid solver."""
    mesh = get_mesh()
    x_sol, basis = solve_fluid(mesh)
    print("\nGenerating Output...")
    _plot_fluid_results(mesh, basis, x_sol, Path(OUTPUT_FOLDER))
    print("Simulation completed.")


if __name__ == "__main__":
    main()
