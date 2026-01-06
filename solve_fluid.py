"""
solve_fluid.py - Robust Penalty Method Version
"""

import time
import warnings
from pathlib import Path

import numpy as np
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

# Local imports
from data import MAX_ITERS, MESH_FILE, MESH_SIZE, NU, OUTPUT_FOLDER, TOL, VEL_INLET
from generate_mesh import CircuitBoard, generate_gmsh_mesh_2d

# Suppress skfem warnings about singular matrix if we handle them
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. Mesh Generation and Loading
# =============================================================================
out_dir = Path(OUTPUT_FOLDER)
out_dir.mkdir(exist_ok=True, parents=True)
mesh_path = Path(OUTPUT_FOLDER) / Path(MESH_FILE)

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
print("Initializing Finite Elements...")
# P2 (Velocity) * P1 (Pressure)
element = ElementVectorH1(ElementQuad2()) * ElementQuad1()

# Basis defined ONLY on Fluid subdomain
basis = Basis(mesh, element, elements=mesh.subdomains["Fluid"])
print(f"  -> Total DOFs: {basis.N}")

# =============================================================================
# 3. Weak Forms (Penalty Method)
# =============================================================================
# Stabilization params
DELTA_SUPG = 0.5  # 1.0
DELTA_GRADDIV = 0.1  # 1.0
EPS_PENALTY = 1e-6


@BilinearForm
def stokes_flow(u, p, v, q, w):
    """
    LHS = (u_prev · ∇) u + ∇p - div(u)q
    """
    return (
        w.nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u) - EPS_PENALTY * p * q
    )


@BilinearForm
def navier_stokes_lhs(u, p, v, q, w):
    """
    LHS = (u_prev · ∇) u + ∇p - div(u)q
    """

    # Convezione Picard (Trasporto): (u_prev · ∇) u
    # In skfem per ElementVectorH1:
    # u.grad[0] è grad(ux), u.grad[1] è grad(uy)
    convection = dot(w.u_prev, u.grad[0]) * v[0] + dot(w.u_prev, u.grad[1]) * v[1]

    u_mag = np.sqrt(w.u_prev[0] ** 2 + w.u_prev[1] ** 2 + 1e-12)
    # Tau SUPG standard
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

    # 2. SUPG Terms (Coerente: stabilizziamo l'operatore LHS)
    # Residuo LHS ≈ (u_prev · ∇)u + ∇p
    res_x = dot(w.u_prev, u.grad[0]) + p.grad[0]
    res_y = dot(w.u_prev, u.grad[1]) + p.grad[1]
    supg = tau * (
        res_x * dot(w.u_prev, v.grad[0]) + res_y * dot(w.u_prev, v.grad[1])
    )  # Streamline vector: (u_prev · ∇) v

    # 3. Grad-Div Stabilization
    graddiv = DELTA_GRADDIV * u_mag * w.h * div(u) * div(v)

    return galerkin + supg + graddiv


@LinearForm
def navier_stokes_rhs(_v, _q, _w):
    """
    CORREZIONE: In iterazione di Picard (Total Solution), il RHS
    contiene solo le forze esterne (o 0). NON deve contenere la convezione
    precedente se questa è già nel LHS, altrimenti si cancellano.
    """
    return 0.0


@LinearForm
def rhs_zero(_v, _q, _w):
    """
    RHS = 0
    """
    return 0.0


# =============================================================================
# 4. Boundary Conditions
# =============================================================================
print("Setting up Boundary Conditions...")
dofs = basis.get_dofs(mesh.boundaries)
u_inlet_vec = np.zeros(basis.N)

# Indices helper
idx_u = basis.split_indices()[0][basis.split_bases()[0].split_indices()[0]]
idx_v = basis.split_indices()[0][basis.split_bases()[0].split_indices()[1]]

# --- 1. Inlet (Parabolic) ---
u_inlet_dofs = np.array([], dtype=int)
v_inlet_dofs = np.array([], dtype=int)

if "Inlet" in dofs:
    inlet_nodes = dofs["Inlet"].all()
    u_inlet_dofs = np.intersect1d(inlet_nodes, idx_u)
    v_inlet_dofs = np.intersect1d(inlet_nodes, idx_v)

    y_coords = basis.doflocs[1, u_inlet_dofs]
    if len(y_coords) > 0:
        y_min, y_max = y_coords.min(), y_coords.max()
        H = y_max - y_min
        y_mid = (y_max + y_min) / 2.0
        # Profilo
        u_profile = VEL_INLET * (1.0 - (2.0 * (y_coords - y_mid) / H) ** 2)
        u_inlet_vec[u_inlet_dofs] = u_profile
        print(f"  -> Inlet configured with {len(u_inlet_dofs)} velocity nodes.")

# --- 2. Walls ---
wall_dofs = np.array([], dtype=int)
for b_name in ["Walls", "SolidInterfaces"]:
    if b_name in dofs:
        bd_dofs = dofs[b_name].all()
        wall_dofs = np.concatenate(
            [wall_dofs, np.intersect1d(bd_dofs, idx_u), np.intersect1d(bd_dofs, idx_v)]
        )
print(f"  -> Walls configured with {len(wall_dofs)} nodes.")

# --- 3. Solid Ghost Nodes (CRITICO) ---
# Dobbiamo vincolare a 0 tutti i nodi che appartengono agli elementi Solid
# ma che la base ha inizializzato comunque. Altrimenti generano righe di zeri.
fluid_dofs_indices = np.unique(basis.element_dofs)
all_dofs = np.arange(basis.N)
solid_ghost_dofs = np.setdiff1d(all_dofs, fluid_dofs_indices)
print(f"  -> Constraining {len(solid_ghost_dofs)} solid/ghost DOFs.")

# Compile Dirichlet DOFs (Velocità Inlet + Walls + Ghost)
# NOTA: Non fissiamo la pressione esplicitamente, usiamo EPS_PENALTY
D_dofs = np.unique(
    np.concatenate([u_inlet_dofs, v_inlet_dofs, wall_dofs, solid_ghost_dofs])
).astype(int)

print(f"  -> Total Dirichlet DOFs: {len(D_dofs)}")

# Init Soluzione
x_sol = u_inlet_vec.copy()

# =============================================================================
# 5. Solver Loop
# =============================================================================

# Viscosity stepping
viscosities = np.geomspace(5e-2, NU, 8)
ux_idx, uy_idx, p_idx = basis.nodal_dofs

print("Initializing with Penalized Stokes...")

A_s = asm(stokes_flow, basis, nu=viscosities[0])
b_s = asm(rhs_zero, basis)
# Usa 'x' come guess e 'D' per condensare
x_sol = solve(*condense(A_s, b_s, x=x_sol, D=D_dofs))
print("  -> Stokes initialization successful.")


print(f"\nStarting Solver Loop. Target Nu={NU:.2e}")


for step_idx, nu_curr in enumerate(viscosities):
    print(f"\n--- STEP {step_idx+1}/{len(viscosities)}: Nu = {nu_curr:.2e} ---")

    relax = 0.7

    for i in range(MAX_ITERS):
        t_iter = time.time()

        # Interpolate for prev terms
        x_prev_func = basis.interpolate(x_sol)

        # Assemble Newton
        # Passiamo h e nu esplicitamente
        A_mat = asm(
            navier_stokes_lhs, basis, u_prev=x_prev_func[0], nu=nu_curr, h=MESH_SIZE
        )
        b_vec = asm(
            navier_stokes_rhs, basis, u_prev=x_prev_func[0], nu=nu_curr, h=MESH_SIZE
        )

        # Solve
        try:
            x_new_cand = solve(*condense(A_mat, b_vec, x=x_sol, D=D_dofs))
        except RuntimeError as e:
            print(f"  Solver failed (Singular?): {e}")
            break

        # Update
        x_new = relax * x_new_cand + (1.0 - relax) * x_sol

        # Error
        diff = np.linalg.norm(x_new - x_sol)
        sol_norm = np.linalg.norm(x_new)
        rel_error = diff / (sol_norm + 1e-12)

        x_sol = x_new

        max_vel = np.max(
            np.linalg.norm(np.stack([x_sol[ux_idx], x_sol[uy_idx]]), axis=0)
        )
        print(f"  Iter {i+1:2d}: err={rel_error:.2e}, max_u={max_vel:.3f}")

        if rel_error < TOL:
            print("  Convergence reached.")
            break

    # Save VTK
    u_sol = np.stack([x_sol[ux_idx], x_sol[uy_idx]]).T
    vtk_path = out_dir / f"fluid_step_{step_idx:02d}.vtk"
    mesh.save(str(vtk_path), {"velocity": u_sol, "pressure": x_sol[p_idx]})

    # --- PLOTTING (MODIFICATO) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    val_mag = np.linalg.norm(u_sol, axis=1)
    vmin, vmax = val_mag.min(), val_mag.max()

    # 1. Campo Velocità
    plot(mesh, val_mag, ax=ax, cmap="turbo", shading="gouraud")

    # 2. MESH WIREFRAME (Grigio trasparente alpha 0.4)
    # linewidth=0.5 rende la griglia sottile ed elegante
    draw(mesh, ax=ax, color="gray", alpha=0.1, linewidth=0.2)

    # 3. Muri/Bordi (Nero pieno per definire la geometria)
    draw(mesh, ax=ax, boundaries_only=True, color="black", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")

    # Colorbar Fix
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    s_map = cm.ScalarMappable(norm=norm, cmap="turbo")
    s_map.set_array([])

    cbar = fig.colorbar(s_map, ax=ax)
    cbar.set_label("Velocity Magnitude [m/s]")

    ax.set_title(f"Velocity Magnitude [Nu={nu_curr:.2e}]")
    ax.set_axis_off()
    plt.savefig(
        out_dir / f"velocity_step_{step_idx:02d}.png", dpi=350, bbox_inches="tight"
    )
    plt.close(fig)

print("\nSimulation completed.")
