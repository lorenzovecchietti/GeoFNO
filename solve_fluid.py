"""
solve fluid skfem
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# Local imports
from data import MAX_ITERS, MESH_FILE, MESH_SIZE, NU, OUTPUT_FOLDER, TOL, VEL_INLET
from generate_mesh import CircuitBoard, generate_gmsh_mesh_2d

# Configuration
BACKEND = "numba"  # Forces JIT compilation for speed

# =============================================================================
# 1. Mesh Generation and Loading
# =============================================================================
mesh_path = Path(MESH_FILE)

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
mesh = Mesh.load(str(mesh_path))

# =============================================================================
# 2. Element Definition (Taylor-Hood P2-P1)
# =============================================================================
element = ElementVectorH1(ElementQuad2()) * ElementQuad1()
basis = Basis(mesh, element, elements=mesh.subdomains["Fluid"])

# =============================================================================
# 3. Weak Forms (Optimized for Numba & Matrix Splitting)
# =============================================================================


@BilinearForm
def stokes_flow(u, p, v, q, w):
    """
    Standard Stokes part (Viscosity + Pressure).
    Linear and independent of velocity u. Assembled once per viscosity step.
    """
    return w.nu_val * inner(grad(u), grad(v)) - p * div(v) - q * div(u)


@BilinearForm
def supg_convection_linearized(u, _p, v, _q, w):
    """
    Combined Convection (Newton Linearization) + SUPG Stabilization (LHS).
    """
    u_prev = w.u_prev
    nu = w.nu_val
    h = w.h

    # SUPG Parameter (Tau) calculation
    u_mag = np.sqrt(u_prev[0] ** 2 + u_prev[1] ** 2 + 1e-12)
    tau = 1.0 / np.sqrt((2.0 * u_mag / h) ** 2 + (4.0 * nu / h**2) ** 2)

    # 1. Standard Galerkin Convection (Newton: u_k * grad(u) + u * grad(u_k))
    # Term: (u_prev . grad) u
    conv1 = dot(grad(u), u_prev)
    # Term: (u . grad) u_prev
    conv2 = dot(grad(u_prev), u)

    galerkin = dot(conv1 + conv2, v)

    # 2. SUPG Stabilization
    # Residual of linearized momentum
    res_lin = conv1 + conv2
    # Streamline test function
    v_stream = dot(grad(v), u_prev)

    supg = tau * dot(res_lin, v_stream)

    return galerkin + supg


@LinearForm
def supg_convection_rhs(v, _q, w):
    """
    RHS for Newton Method: Residuals of Convection + SUPG.
    """
    u_prev = w.u_prev
    nu = w.nu_val
    h = w.h

    u_mag = np.sqrt(u_prev[0] ** 2 + u_prev[1] ** 2 + 1e-12)
    tau = 1.0 / np.sqrt((2.0 * u_mag / h) ** 2 + (4.0 * nu / h**2) ** 2)

    # Convection at previous step: (u_prev . grad) u_prev
    conv_prev = dot(grad(u_prev), u_prev)

    # Galerkin RHS
    galerkin = dot(conv_prev, v)

    # SUPG RHS
    v_stream = dot(grad(v), u_prev)
    supg = tau * dot(conv_prev, v_stream)

    return galerkin + supg


# =============================================================================
# 4. Boundary Conditions
# =============================================================================
# Retrieve DOFs
dofs = basis.get_dofs(mesh.boundaries)
u_inlet_vec = np.zeros(basis.N)

# Helper indices for vector components
idx_u = basis.split_indices()[0][basis.split_bases()[0].split_indices()[0]]
idx_v = basis.split_indices()[0][basis.split_bases()[0].split_indices()[1]]
idx_p = basis.split_indices()[1]

# Set Inlet Velocity
if "Inlet" in dofs:
    inlet_dofs_all = dofs["Inlet"].all()
    # Intersect to find u and v DOFs on the inlet
    u_inlet_dofs = np.intersect1d(inlet_dofs_all, idx_u)
    v_inlet_dofs = np.intersect1d(inlet_dofs_all, idx_v)
    
    u_inlet_vec[u_inlet_dofs] = VEL_INLET
    u_inlet_vec[v_inlet_dofs] = 0.0

# Wall DOFs (No-slip)
wall_dofs = np.array([], dtype=int)
for b_name in ["Walls", "SolidInterfaces"]:
    if b_name in dofs:
        bd_dofs = dofs[b_name].all()
        wall_dofs = np.concatenate(
            [wall_dofs, np.intersect1d(bd_dofs, idx_u), np.intersect1d(bd_dofs, idx_v)]
        )

# Inlet DOFs
inlet_dofs = np.array([], dtype=int)
if "Inlet" in dofs:
    inlet_dofs_all = dofs["Inlet"].all()
    inlet_dofs = np.concatenate(
        [np.intersect1d(inlet_dofs_all, idx_u), np.intersect1d(inlet_dofs_all, idx_v)]
    )

# Outlet Pressure Reference
outlet_dofs = np.array([], dtype=int)
if "Outlet" in dofs:
    outlet_dofs = np.intersect1d(dofs["Outlet"].all(), idx_p)

# Identify Solid DOFs (inactive fluid nodes)
fluid_dofs_indices = np.unique(basis.element_dofs)
all_dofs = np.arange(basis.N)
solid_dofs = np.setdiff1d(all_dofs, fluid_dofs_indices)

# Compile Dirichlet DOFs
D_dofs = np.unique(
    np.concatenate([inlet_dofs, wall_dofs, outlet_dofs, solid_dofs])
).astype(int)

x_sol = u_inlet_vec.copy()

# =============================================================================
# 5. Solver Loop
# =============================================================================
out_dir = Path(OUTPUT_FOLDER)
out_dir.mkdir(exist_ok=True, parents=True)

print(f"Starting Solver. Target Nu={NU}. Backend={BACKEND}")


viscosities = np.geomspace(1e-2, NU, 10)
ux_idx, uy_idx, p_idx = basis.nodal_dofs

for step_idx, nu_curr in enumerate(viscosities):
    print(f"\n--- STEP {step_idx+1}/{len(viscosities)}: Nu = {nu_curr:.2e} ---")

    # Optimization: Assemble Stokes matrix (Linear) ONLY ONCE per viscosity step
    t0 = time.time()
    A_stokes = asm(stokes_flow, basis, nu_val=nu_curr)
    print(f"  Stokes matrix assembled in {time.time()-t0:.2f}s")

    for i in range(MAX_ITERS):
        t_iter = time.time()

        # Update interpolation of previous solution
        x_prev_func = basis.interpolate(x_sol)

        # Assemble Nonlinear parts (Convection + SUPG)
        A_nonlinear = asm(
            supg_convection_linearized,
            basis,
            u_prev=x_prev_func[0],
            nu_val=nu_curr,
        )

        f_nonlinear = asm(
            supg_convection_rhs,
            basis,
            u_prev=x_prev_func[0],
            nu_val=nu_curr,
        )

        # Combine matrices
        A_mat = A_stokes + A_nonlinear

        # Solve
        x_new = solve(*condense(A_mat, f_nonlinear, x=x_sol, D=D_dofs))

        # Error check
        diff = np.linalg.norm(x_new - x_sol) / (np.linalg.norm(x_new) + 1e-12)
        x_sol = x_new

        print(f"  Iter {i+1}: error = {diff:.2e} [{time.time()-t_iter:.2f}s]")

        if diff < TOL:
            print("  Convergence reached.")
            break
    else:
        print("  Warning: Max iterations reached without convergence.")

    # Save output
    u_solution = np.stack([x_sol[ux_idx], x_sol[uy_idx]]).T
    p_solution = x_sol[p_idx]

    vtk_path = out_dir / f"fluid_step_{step_idx:02d}.vtk"
    png_filename = f"velocity_step_{step_idx:02d}_nu_{nu_curr:.1e}.png"

    mesh.save(
        str(vtk_path),
        {
            "velocity": np.stack([x_sol[ux_idx], x_sol[uy_idx]]).T,
            "pressure": x_sol[p_idx],
        },
    )

    fig, ax_plot = plt.subplots(1, 1, figsize=(10, 5))
    display_u_mag = np.linalg.norm(u_solution, axis=1)
    plot(mesh, display_u_mag, ax=ax_plot, cmap="turbo")
    draw(mesh, ax=ax_plot, boundaries_only=True)
    ax_plot.set_title(f"Velocity Magnitude [m/s] (nu={nu_curr:.2e})")
    ax_plot.set_axis_off()
    plt.savefig(out_dir / png_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

print("\nSimulation completed.")
