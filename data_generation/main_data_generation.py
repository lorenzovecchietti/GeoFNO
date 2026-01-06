"""
main_data_generation.py
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from skfem import Basis, ElementVectorH1, ElementQuad1, Mesh
from skfem.visuals.matplotlib import plot, draw

# Local imports
from data import OUTPUT_FOLDER
from solve_fluid import solve_fluid, get_mesh
from solve_thermal import solve_thermal

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
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
    
    # Setup Figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # --- Prepare Velocity Magnitude for Plotting ---
    # vel_p1_flat contiene [ux_all, uy_all]. Dobbiamo calcolare la magnitudo sui nodi P1.
    vec_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
    ux_idxs = vec_basis.split_indices()[0]
    uy_idxs = vec_basis.split_indices()[1]
    
    u_vals = vel_p1_flat[ux_idxs]
    v_vals = vel_p1_flat[uy_idxs]
    vel_mag = np.sqrt(u_vals**2 + v_vals**2)
    
    # Base scalare P1 per il plot
    p1_scalar_basis = Basis(mesh, ElementQuad1())

    # 1. Velocity Plot
    ax = axs[0]
    # Scalar Plot (Heatmap)
    plot(p1_scalar_basis, vel_mag, ax=ax, cmap="turbo", shading="gouraud")
    # Mesh Overlay (Gray, thin)
    draw(mesh, ax=ax, boundaries_only=False, color="gray", linewidth=0.05)
    # Boundaries (Black, thick)
    draw(mesh, ax=ax, boundaries_only=True, color="black", linewidth=1.5)
    
    ax.set_title("Velocity Magnitude [m/s]")
    ax.set_aspect("equal")
    ax.axis("off")
    
    # 2. Temperature Plot
    ax = axs[1]
    # Scalar Plot
    plot(thermal_basis, temp_sol, ax=ax, cmap="inferno", shading="gouraud")
    # Mesh Overlay
    draw(mesh, ax=ax, boundaries_only=False, color="gray", linewidth=0.05)
    # Boundaries
    draw(mesh, ax=ax, boundaries_only=True, color="black", linewidth=1.5)
    
    ax.set_title("Temperature [K]")
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Save
    out_path = out_dir / "final_simulation_result.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Final image saved to {out_path}")
    
    # Save VTK (Opzionale, utile per ParaView)
    vtk_path = out_dir / "solution.vtk"
    mesh.save(str(vtk_path), point_data={
        "Velocity": np.stack([u_vals, v_vals, np.zeros_like(u_vals)], axis=1),
        "Temperature": temp_sol
    })
    
    print("\n=== Pipeline Completed ===")

if __name__ == "__main__":
    main()