"""
main_data_generation.py - Parallel Dataset Generation with Sobol Sampling
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import pickle
import warnings
import gc
import sys
from pathlib import Path
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Local imports
from data import OUTPUT_FOLDER, MESH_SIZE, COMPONENT_TAGS
from generate_mesh import CircuitBoard
from solve_fluid import get_mesh, solve_fluid
from solve_thermal import solve_thermal, _setup_material_properties
from skfem import Basis, ElementQuad1, ElementVectorH1, ElementQuad2

# Suppress warnings
warnings.filterwarnings("ignore")

DATASET_FOLDER = Path("dataset_100")
N_SAMPLES = 100  # Set to 100 for dataset generation
MAX_CPUS = 1

# Parameter Ranges
PARAM_RANGES = {
    "k_comps": (200.0, 500.0),    # 5 values
    "k_pcb": (0.01, 1.0),         # 1 value
    "vel_inlet": (0.001, 0.01),  # 1 value
    "h_pcb": (0.01, 0.1),         # 1 value
    "w_pcb": (0.085, 0.5),        # 1 value
    "n_up": (0, 5),               # 1 value (integer)
    "w_comps": (0.05, 0.3),       # 5 values
    "h_comps": (1.0, 1.5),        # 5 values
    "q_comps": (100.0, 300.0),       # 5 values
}

# Total dimensions for Sobol: 5+1+1+1+1+1+5+5+5 = 25
DIMENSIONS = 25

def generate_samples(n_samples: int) -> np.ndarray:
    """Generate Sobol samples in [0, 1]^DIMENSIONS."""
    sampler = qmc.Sobol(d=DIMENSIONS, scramble=True)
    return sampler.random(n=n_samples)

def map_sample_to_params(sample: np.ndarray) -> Dict[str, Any]:
    """Map a Sobol unit sample to the physical parameter space."""
    idx = 0
    params = {}
    
    # K_COMPONENTS (5)
    params["k_comps"] = sample[idx:idx+5] * (PARAM_RANGES["k_comps"][1] - PARAM_RANGES["k_comps"][0]) + PARAM_RANGES["k_comps"][0]
    idx += 5
    
    # K_PCB (1)
    params["k_pcb"] = sample[idx] * (PARAM_RANGES["k_pcb"][1] - PARAM_RANGES["k_pcb"][0]) + PARAM_RANGES["k_pcb"][0]
    idx += 1
    
    # vel_inlet (1)
    params["vel_inlet"] = sample[idx] * (PARAM_RANGES["vel_inlet"][1] - PARAM_RANGES["vel_inlet"][0]) + PARAM_RANGES["vel_inlet"][0]
    idx += 1
    
    # h_pcb (1)
    params["h_pcb"] = sample[idx] * (PARAM_RANGES["h_pcb"][1] - PARAM_RANGES["h_pcb"][0]) + PARAM_RANGES["h_pcb"][0]
    idx += 1
    
    # w_pcb (1)
    params["w_pcb"] = sample[idx] * (PARAM_RANGES["w_pcb"][1] - PARAM_RANGES["w_pcb"][0]) + PARAM_RANGES["w_pcb"][0]
    idx += 1
    
    # n_up (1) - Round to nearest integer
    params["n_up"] = int(np.round(sample[idx] * (PARAM_RANGES["n_up"][1] - PARAM_RANGES["n_up"][0]) + PARAM_RANGES["n_up"][0]))
    idx += 1
    
    # w_comps (5)
    params["w_comps"] = sample[idx:idx+5] * (PARAM_RANGES["w_comps"][1] - PARAM_RANGES["w_comps"][0]) + PARAM_RANGES["w_comps"][0]
    idx += 5
    
    # h_comps (5)
    params["h_comps"] = sample[idx:idx+5] * (PARAM_RANGES["h_comps"][1] - PARAM_RANGES["h_comps"][0]) + PARAM_RANGES["h_comps"][0]
    idx += 5

    # q_comps (5)
    params["q_comps"] = sample[idx:idx+5] * (PARAM_RANGES["q_comps"][1] - PARAM_RANGES["q_comps"][0]) + PARAM_RANGES["q_comps"][0]
    idx += 5
    
    return params

def _plot_simulation(mesh, thermal_basis, temp_sol, vel_p1_flat, out_path):
    """Internal helper to generate the final simulation image."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    
    vec_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
    ux_idxs, uy_idxs = vec_basis.split_indices()
    u_vals, v_vals = vel_p1_flat[ux_idxs], vel_p1_flat[uy_idxs]
    vel_mag = np.sqrt(u_vals**2 + v_vals**2)
    p1_scalar_basis = Basis(mesh, ElementQuad1())

    from skfem.visuals.matplotlib import draw, plot
    
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
    """Averages element values to nodes."""
    nodal_vals = np.zeros(mesh.nnodes)
    counts = np.zeros(mesh.nnodes)
    for e in range(mesh.nelements):
        for n in mesh.t[:, e]:
            nodal_vals[n] += elem_vals[e]
            counts[n] += 1
    return nodal_vals / (counts + 1e-12)

def worker(args):
    """Worker function for parallel data generation."""
    sample_idx, sample = args
    case_dir = DATASET_FOLDER / f"case_{sample_idx:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = case_dir / "generation.log"
    
    with open(log_file_path, "w", buffering=1) as f_log:
        with redirect_stdout(f_log), redirect_stderr(f_log):
            try:
                print(f"--- Starting Case {sample_idx} ---")
                
                params = map_sample_to_params(sample)
                
                # Save parameters
                with open(case_dir / "params.pkl", "wb") as f:
                    pickle.dump(params, f)
                    
                # 1. Mesh
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
                fluid_sol, fluid_basis = solve_fluid(mesh, vel_inlet=params["vel_inlet"])
                
                # 3. Thermal Solve
                temp_sol, thermal_basis, vel_p1_flat = solve_thermal(
                    mesh, fluid_sol, fluid_basis,
                    k_pcb=params["k_pcb"],
                    k_comps=params["k_comps"],
                    q_dot_comp=params["q_comps"]
                )
                
                # 4. Visualization & Output
                img_path = case_dir / "simulation.png"
                _plot_simulation(mesh, thermal_basis, temp_sol, vel_p1_flat, img_path)
                
                # 5. Extract Solutions (T, P, VX, VY)
                vec_basis = Basis(mesh, ElementVectorH1(ElementQuad1()))
                ux_idxs, uy_idxs = vec_basis.split_indices()
                vx = vel_p1_flat[ux_idxs]
                vy = vel_p1_flat[uy_idxs]
                
                # Pressure extraction
                fluid_element = ElementVectorH1(ElementQuad2()) * ElementQuad1()
                full_fluid_basis = Basis(mesh, fluid_element, elements=mesh.subdomains["Fluid"])
                _, p_idxs = full_fluid_basis.split_indices()
                pressure_on_fluid = fluid_sol[p_idxs]
                
                p_full = np.zeros(mesh.nnodes)
                fluid_nodes = np.unique(mesh.t[:, mesh.subdomains["Fluid"]])
                p_full[fluid_nodes] = pressure_on_fluid
                
                solutions = {
                    "temperature": temp_sol,
                    "pressure": p_full,
                    "vx": vx,
                    "vy": vy
                }
                np.save(case_dir / "solutions.npy", solutions)
                
                # 6. Extract Inputs (K, Q)
                k_elem, _, q_elem = _setup_material_properties(
                    mesh, k_pcb=params["k_pcb"], k_comps=params["k_comps"], q_dot_comp=params["q_comps"]
                )
                k_nodal = element_to_nodal(mesh, k_elem)
                q_nodal = element_to_nodal(mesh, q_elem)
                
                inputs = {
                    "conductivity": k_nodal,
                    "power": q_nodal
                }
                np.save(case_dir / "inputs.npy", inputs)
                
                del mesh, fluid_sol, fluid_basis, temp_sol, thermal_basis, vec_basis
                gc.collect() # Forza il rilascio della RAM
                print(f"Case {sample_idx} completed.")
                
            except Exception as e:
                print(f"Case {sample_idx} failed: {e}")
                import traceback
                traceback.print_exc()
    

def main():
    # Use 'spawn' to avoid issues with MKL/Pypardiso after forking
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"=== Starting Dataset Generation ({N_SAMPLES} samples) ===")
    
    DATASET_FOLDER.mkdir(exist_ok=True)
    
    samples = generate_samples(N_SAMPLES)
    
    args_list = [(i, samples[i]) for i in range(N_SAMPLES)]
    
    # Use max_workers = number of physical cores or less to avoid oversubscription
    max_workers = min(min(mp.cpu_count(), MAX_CPUS), N_SAMPLES)
    print(f"Using {max_workers} workers for parallel generation.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(worker, args_list)
        
    print(f"\n=== Dataset Generation Completed. Saved to {DATASET_FOLDER} ===")

if __name__ == "__main__":
    main()
