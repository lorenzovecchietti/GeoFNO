"""
Parallel Dataset Generation with Sobol Sampling for GeoFNO.
"""

import gc
import multiprocessing as mp
import os
import pickle
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Local imports
from data import (
    DATASET_FOLDER,
    DIMENSIONS,
    MAX_CPUS,
    N_SAMPLES,
    PARAM_RANGES,
)
from generate_mesh import CircuitBoard
from scipy.spatial import cKDTree
from scipy.stats import qmc
from skfem import Basis, ElementQuad1, ElementQuad2, ElementVectorH1
from skfem.visuals.matplotlib import draw, plot
from solve_fluid import get_mesh, solve_fluid
from solve_thermal import _setup_material_properties, solve_thermal

# Set environment variables for single-thread operations in subprocesses
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

def generate_samples(n_samples: int) -> np.ndarray:
    """Generate Sobol samples in [0, 1]^DIMENSIONS."""
    sampler = qmc.Sobol(d=DIMENSIONS, scramble=True)
    return sampler.random(n=n_samples)


def map_sample_to_params(sample: np.ndarray) -> Dict[str, Any]:
    """Map a Sobol unit sample to the physical parameter space."""
    idx = 0
    params = {}

    # K_COMPONENTS (5)
    params["k_comps"] = (
        sample[idx : idx + 5]
        * (PARAM_RANGES["k_comps"][1] - PARAM_RANGES["k_comps"][0])
        + PARAM_RANGES["k_comps"][0]
    )
    idx += 5

    # K_PCB (1)
    params["k_pcb"] = (
        sample[idx] * (PARAM_RANGES["k_pcb"][1] - PARAM_RANGES["k_pcb"][0])
        + PARAM_RANGES["k_pcb"][0]
    )
    idx += 1

    # vel_inlet (1)
    params["vel_inlet"] = (
        sample[idx] * (PARAM_RANGES["vel_inlet"][1] - PARAM_RANGES["vel_inlet"][0])
        + PARAM_RANGES["vel_inlet"][0]
    )
    idx += 1

    # h_pcb (1)
    params["h_pcb"] = (
        sample[idx] * (PARAM_RANGES["h_pcb"][1] - PARAM_RANGES["h_pcb"][0])
        + PARAM_RANGES["h_pcb"][0]
    )
    idx += 1

    # w_pcb (1)
    params["w_pcb"] = (
        sample[idx] * (PARAM_RANGES["w_pcb"][1] - PARAM_RANGES["w_pcb"][0])
        + PARAM_RANGES["w_pcb"][0]
    )
    idx += 1

    # n_up (1) - Round to nearest integer
    params["n_up"] = int(
        np.round(
            sample[idx] * (PARAM_RANGES["n_up"][1] - PARAM_RANGES["n_up"][0])
            + PARAM_RANGES["n_up"][0]
        )
    )
    idx += 1

    # w_comps (5)
    params["w_comps"] = (
        sample[idx : idx + 5]
        * (PARAM_RANGES["w_comps"][1] - PARAM_RANGES["w_comps"][0])
        + PARAM_RANGES["w_comps"][0]
    )
    idx += 5

    # h_comps (5)
    params["h_comps"] = (
        sample[idx : idx + 5]
        * (PARAM_RANGES["h_comps"][1] - PARAM_RANGES["h_comps"][0])
        + PARAM_RANGES["h_comps"][0]
    )
    idx += 5

    # q_comps (5)
    params["q_comps"] = (
        sample[idx : idx + 5]
        * (PARAM_RANGES["q_comps"][1] - PARAM_RANGES["q_comps"][0])
        + PARAM_RANGES["q_comps"][0]
    )
    idx += 5

    return params


def _plot_simulation(
    mesh, thermal_basis, temp_sol, vel_p1_flat, out_path
):  # pylint: disable=too-many-locals
    """Generate simulation results plot."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    vec_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
    ux_idxs, uy_idxs = vec_basis.split_indices()
    u_vals, v_vals = vel_p1_flat[ux_idxs], vel_p1_flat[uy_idxs]
    vel_mag = np.sqrt(u_vals**2 + v_vals**2)
    p1_scalar_basis = Basis(mesh, ElementQuad1())

    ax_vel = axs[0]
    plot(p1_scalar_basis, vel_mag, ax=ax_vel, cmap="turbo", shading="gouraud")
    img1 = ax_vel.collections[0]
    draw(mesh, ax=ax_vel, color="gray", linewidth=0.1, alpha=0.5)
    draw(mesh, ax=ax_vel, boundaries_only=True, color="black", linewidth=1.5)
    ax_vel.set_title("Velocity Magnitude [m/s]", fontsize=14)
    ax_vel.set_aspect("equal")
    ax_vel.axis("off")
    fig.colorbar(img1, ax=ax_vel, fraction=0.046, pad=0.04)

    ax_temp = axs[1]
    plot(thermal_basis, temp_sol, ax=ax_temp, cmap="inferno", shading="gouraud")
    img2 = ax_temp.collections[0]
    draw(mesh, ax=ax_temp, color="gray", linewidth=0.1, alpha=0.5)
    draw(mesh, ax=ax_temp, boundaries_only=True, color="black", linewidth=1.5)
    ax_temp.set_title("Temperature [K]", fontsize=14)
    ax_temp.set_aspect("equal")
    ax_temp.axis("off")
    fig.colorbar(img2, ax=ax_temp, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def element_to_nodal(mesh, elem_vals):
    """Average element values to nodes."""
    nodal_vals = np.zeros(mesh.nnodes)
    counts = np.zeros(mesh.nnodes)
    for e in range(mesh.nelements):
        for n in mesh.t[:, e]:
            nodal_vals[n] += elem_vals[e]
            counts[n] += 1
    return nodal_vals / (counts + 1e-12)


def worker(args):  # pylint: disable=too-many-locals,too-many-statements
    """Worker function for parallel data generation."""
    sample_idx, sample = args
    case_dir = DATASET_FOLDER / f"case_{sample_idx:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = case_dir / "generation.log"

    with open(log_file_path, "w", buffering=1, encoding="utf-8") as f_log:
        with redirect_stdout(f_log), redirect_stderr(f_log):
            try:
                print(f"--- Starting Case {sample_idx} ---")

                params = map_sample_to_params(sample)

                # Save parameters
                with open(case_dir / "params.pkl", "wb") as f:
                    pickle.dump(params, f)

                # 1. Mesh Generation
                cb = CircuitBoard(
                    h_pcb=params["h_pcb"],
                    w_pcb=params["w_pcb"],
                    n_up=params["n_up"],
                    w_comps=params["w_comps"].tolist(),
                    h_comps=params["h_comps"].tolist(),
                )
                mesh_path = case_dir / "mesh.msh"
                mesh = get_mesh(cb, mesh_path)

                # 2. Fluid Solve
                fluid_sol, fluid_basis = solve_fluid(
                    mesh, vel_inlet=params["vel_inlet"]
                )

                # 3. Thermal Solve
                temp_sol, thermal_basis, vel_p1_flat = solve_thermal(
                    mesh,
                    fluid_sol,
                    fluid_basis,
                    k_pcb=params["k_pcb"],
                    k_comps=params["k_comps"],
                    q_dot_comp=params["q_comps"],
                )

                # 4. Visualization & Output
                img_path = case_dir / "simulation.png"
                _plot_simulation(mesh, thermal_basis, temp_sol, vel_p1_flat, img_path)
                total_nodes = mesh.p.shape[1]

                # --- VELOCITY ---
                vec_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
                ux_idxs, uy_idxs = vec_basis.split_indices()
                vx = vel_p1_flat[ux_idxs]
                vy = vel_p1_flat[uy_idxs]

                # --- PRESSURE (Geometric Mapping) ---
                # 1. Mixed Basis Setup
                fluid_element = ElementVectorH1(ElementQuad2()) * ElementQuad1()
                full_fluid_basis = Basis(
                    mesh, fluid_element, elements=mesh.subdomains["Fluid"]
                )
                _, p_idxs = full_fluid_basis.split_indices()

                # 2. Pressure and Coordinates
                pressure_values = fluid_sol[p_idxs]
                pressure_dof_locs = full_fluid_basis.doflocs[:, p_idxs]

                # 3. Mesh Physical Nodes
                fluid_nodes_indices = np.unique(mesh.t[:, mesh.subdomains["Fluid"]])
                mesh_node_locs = mesh.p[:, fluid_nodes_indices]

                # 4. KDTree for mapping
                tree = cKDTree(pressure_dof_locs.T)
                _, closest_p_idx = tree.query(mesh_node_locs.T, k=1)

                # 5. Pressure Assignment
                p_full = np.zeros(total_nodes)

                if fluid_nodes_indices.max() >= total_nodes:
                    raise RuntimeError(
                        f"Mesh index mismatch: max index {fluid_nodes_indices.max()} "
                        f">= total nodes {total_nodes}"
                    )

                p_full[fluid_nodes_indices] = pressure_values[closest_p_idx]

                solutions = {
                    "temperature": temp_sol,
                    "pressure": p_full,
                    "vx": vx,
                    "vy": vy,
                }
                np.save(case_dir / "solutions.npy", solutions)

                # 6. Extract Inputs (K, Q)
                real_nnodes = mesh.t.max() + 1

                # Calculate physical properties per element
                k_elem, _, q_elem = _setup_material_properties(
                    mesh,
                    k_pcb=params["k_pcb"],
                    k_comps=params["k_comps"],
                    q_dot_comp=params["q_comps"],
                )

                # Robust element to nodal mapping
                def robust_element_to_nodal(indices, values, num_nodes):
                    nodal_sum = np.zeros(num_nodes)
                    nodal_count = np.zeros(num_nodes)

                    # Vectorized accumulation
                    for i in range(indices.shape[0]):
                        np.add.at(nodal_sum, indices[i], values)
                        np.add.at(nodal_count, indices[i], 1)

                    mask = nodal_count > 0
                    nodal_sum[mask] /= nodal_count[mask]
                    return nodal_sum

                k_nodal = robust_element_to_nodal(mesh.t, k_elem, real_nnodes)
                q_nodal = robust_element_to_nodal(mesh.t, q_elem, real_nnodes)

                inputs = {"conductivity": k_nodal, "power": q_nodal}
                np.save(case_dir / "inputs.npy", inputs)

                # Cleanup
                del mesh, fluid_sol, temp_sol, vec_basis
                if "fluid_basis" in locals():
                    del fluid_basis
                if "full_fluid_basis" in locals():
                    del full_fluid_basis

                gc.collect()
                print(f"Case {sample_idx} completed successfully.")

            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Case {sample_idx} failed: {e}")
                traceback.print_exc()


def main():
    """Main entry point for dataset generation."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"=== Starting Dataset Generation ({N_SAMPLES} samples) ===")

    DATASET_FOLDER.mkdir(exist_ok=True)

    samples = generate_samples(N_SAMPLES)
    args_list = [(i, samples[i]) for i in range(N_SAMPLES)]

    max_workers = min(mp.cpu_count(), MAX_CPUS, N_SAMPLES)
    print(f"Using {max_workers} workers for parallel generation.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(worker, args_list)

    print(f"\n=== Dataset Generation Completed. Saved to {DATASET_FOLDER} ===")


if __name__ == "__main__":
    main()
