"""
main_data_generation.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Local imports
from data import OUTPUT_FOLDER
from skfem import Basis, ElementQuad1, ElementVectorH1
from skfem.visuals.matplotlib import draw, plot
from solve_fluid import get_mesh, solve_fluid
from solve_thermal import solve_thermal

# Suppress warnings
warnings.filterwarnings("ignore")


def _plot_simulation(mesh, thermal_basis, temp_sol, vel_p1_flat, out_path):
    """Internal helper to generate the final simulation image."""

    # --- MODIFICA 1: Layout e Dimensioni ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    # --- Prepare Velocity Magnitude ---
    vec_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
    ux_idxs, uy_idxs = vec_basis.split_indices()
    u_vals, v_vals = vel_p1_flat[ux_idxs], vel_p1_flat[uy_idxs]
    vel_mag = np.sqrt(u_vals ** 2 + v_vals ** 2)

    p1_scalar_basis = Basis(mesh, ElementQuad1())

    # --- 1. Velocity Plot ---
    ax = axs[0]

    # ERROR FIX: 'plot' returns the axis, not the image.
    # We plot first, then grab the mappable from ax.collections[0]
    plot(p1_scalar_basis, vel_mag, ax=ax, cmap="turbo", shading="gouraud")
    img1 = ax.collections[0]

    # --- MODIFICA 2: Visibilità Mesh ---
    draw(mesh, ax=ax, color="gray", linewidth=0.1, alpha=0.5)
    draw(mesh, ax=ax, boundaries_only=True, color="black", linewidth=1.5)

    ax.set_title("Velocity Magnitude [m/s]", fontsize=14)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- MODIFICA 3: Colorbar ---
    # Now 'img1' is a QuadMesh/TriMesh object, which has the 'cmap' attribute
    fig.colorbar(img1, ax=ax, fraction=0.046, pad=0.04, label="Magnitude [m/s]")

    # --- 2. Temperature Plot ---
    ax = axs[1]

    # ERROR FIX: Same here for temperature
    plot(thermal_basis, temp_sol, ax=ax, cmap="inferno", shading="gouraud")
    img2 = ax.collections[0]

    draw(mesh, ax=ax, color="gray", linewidth=0.1, alpha=0.5)
    draw(mesh, ax=ax, boundaries_only=True, color="black", linewidth=1.5)

    ax.set_title("Temperature [K]", fontsize=14)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colorbar per la temperatura
    fig.colorbar(img2, ax=ax, fraction=0.046, pad=0.04, label="Temp [K]")

    plt.savefig(out_path, dpi=350)
    # It is good practice to close the figure to free memory if this runs in a loop
    plt.close(fig)

    return u_vals, v_vals


def main():
    """
    Main execution pipeline for fluid and thermal data generation.
    """
    print("=== Starting Data Generation Pipeline ===")

    # 1. Mesh
    mesh = get_mesh()

    # 2. Fluid Solve
    print("\n--- Solving Fluid Dynamics ---")
    fluid_sol, fluid_basis = solve_fluid(mesh)

    # 3. Thermal Solve
    print("\n--- Solving Thermal Transfer ---")
    # Nota: solve_thermal ora restituisce vel_p1_flat (coefficienti P1 globali)
    temp_sol, thermal_basis, vel_p1_flat = solve_thermal(mesh, fluid_sol, fluid_basis)

    # 4. Visualization & Output
    print("\n--- Generating Visualization & Output ---")

    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(exist_ok=True, parents=True)

    out_path = out_dir / "final_simulation_result.png"
    u_vals, v_vals = _plot_simulation(
        mesh, thermal_basis, temp_sol, vel_p1_flat, out_path
    )
    print(f"Final image saved to {out_path}")

    # Save VTK (Opzionale, utile per ParaView)
    vtk_path = out_dir / "solution.vtk"
    mesh.save(
        str(vtk_path),
        point_data={
            "Velocity": np.stack([u_vals, v_vals, np.zeros_like(u_vals)], axis=1),
            "Temperature": temp_sol,
        },
    )

    print("\n=== Pipeline Completed ===")


if __name__ == "__main__":
    main()
