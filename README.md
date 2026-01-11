# GeoFNO: Geometric Fourier Neural Operator for Conjugate Heat Transfer

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Generation: FEM Simulations](#data-generation-fem-simulations)
3. [Neural Network: GeoFNO Architecture](#neural-network-geofno-architecture)
4. [Practical Usage](#practical-usage)
5. [References](#references)

---

## Project Overview

This project combines **Finite Element Method (FEM)** simulations with **Deep Learning** to create a fast surrogate model for conjugate heat transfer problems. The workflow consists of two main components:

1. **Data Generation**: High-fidelity FEM simulations of laminar fluid flow and heat transfer over electronic components
2. **Neural Network**: GeoFNO learns to predict temperature, pressure, and velocity fields from geometry and boundary conditions

**Goal**: Replace expensive FEM simulations (minutes/hours) with instant neural network predictions (milliseconds) while maintaining accuracy.

---

# Data Generation: FEM Simulations

## 1. Simulation Overview

The data generation pipeline simulates **laminar, steady-state airflow** over a Printed Circuit Board (PCB) with mounted electronic components inside a cooling channel. The simulation solves coupled physics:
- **Incompressible Navier-Stokes equations** (fluid flow)
- **Advection-diffusion equation** (heat transfer)

Both are solved using the **Finite Element Method (FEM)** with the `scikit-fem` library.

### Simulated Variables

**Fluid Flow:**
- **Velocity field** ($\mathbf{u}$): Vector field $[u_x, u_y]$ representing fluid speed and direction
- **Pressure field** ($p$): Scalar field representing static pressure distribution

**Heat Transfer:**
- **Temperature field** ($T$): Scalar field representing temperature distribution in both fluid and solid regions

### Fixed Data & Parameters

The physical setup uses properties of **Air at 20°C** and standard PCB geometry:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Domain Size** | $20.0 \times 5.0$ m | Rectangular channel |
| **PCB Geometry** | Width: 14.0, Height: 0.25 m | Centered in the domain |
| **Kinematic Viscosity** ($\nu$) | $1.0 \times 10^{-5}$ $m^2/s$ | Air viscosity |
| **Inlet Velocity**| $U_{max} = 0.05$ m/s | Parabolic profile |
| **Solid Components** | 5 Components | Modeled as obstacles with heat sources |
| **Thermal Conductivity (Air)** | Variable | Low conductivity |
| **Thermal Conductivity (Solid)** | Variable | High conductivity |
| **Power Dissipation** | Variable per component | Heat sources in solids |

**Boundary Conditions:**

*Fluid Flow:*
- **Inlet**: Prescribed parabolic velocity profile
- **Outlet**: Natural boundary condition (traction-free)
- **Walls & Solids**: No-slip condition ($\mathbf{u} = 0$)

*Heat Transfer:*
- **Inlet**: Fixed temperature (ambient)
- **Outlet**: Natural boundary condition
- **Walls**: Adiabatic (no heat flux)
- **Solid Components**: Internal heat generation (power dissipation)

---

## 2. Mesh Generation

The mesh is generated using the **GMSH** Python API with the OpenCASCADE kernel to ensure robust boolean operations. The generation process is handled in `data_generation/generate_mesh.py`.

### Process:

1. **Geometry Definition**: Rectangles are defined for the channel, the PCB, and the 5 electronic components
2. **Fragmentation**: The `gmsh.model.occ.fragment` operation cuts the solid shapes out of the fluid domain. This ensures the mesh is **conformal**—nodes match perfectly at the fluid-solid interfaces
3. **Physical Tagging**: Since IDs change after fragmentation, the script automatically identifies surfaces using their **Center of Mass** to assign physical tags (Fluid, PCB, Inlet, Outlet, etc.)
4. **Discretization**: The domain is meshed with **Quadrilaterals** (via `RecombineAll=1`) to support high-quality tensor product elements

**Why conformal meshes?**
- Ensures continuity of temperature at fluid-solid interfaces
- Enables accurate heat flux calculations
- Simplifies coupling between fluid and thermal solvers

---

## 3. Mathematical Formulation

### 3.1 Fluid Flow Solver (`solve_fluid.py`)

The solver uses **scikit-fem (skfem)** to discretize the weak form of the Navier-Stokes equations.

#### Finite Elements

We use **Taylor-Hood (P2-P1)** elements, which satisfy the LBB (Ladyzhenskaya-Babuska-Brezzi) condition for stability:
- **Velocity**: Quadratic (P2) Quadrilateral elements (`ElementVectorH1(ElementQuad2)`)
- **Pressure**: Linear (P1) Quadrilateral elements (`ElementQuad1`)

**Why Taylor-Hood?**
- Prevents spurious pressure oscillations
- Satisfies inf-sup condition for mixed formulations
- Standard choice for incompressible flow

#### Linearization (Picard Iteration)

The convective term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ is non-linear. We linearize it using **Picard iterations** (Fixed Point iteration):

$$
(\mathbf{u}^{k+1} \cdot \nabla) \mathbf{u}^{k+1} \approx (\mathbf{u}^{k} \cdot \nabla) \mathbf{u}^{k+1}
$$

Where $\mathbf{u}^k$ is the solution from the previous iteration.

#### Continuation Method (Viscosity Stepping)

Directly solving for low viscosity ($\nu = 10^{-5}$) is difficult due to non-linearity. The solver employs a **Continuation Method**:

1. Start with high viscosity ($\nu = 5 \times 10^{-2}$), where the flow is Stokes-like (diffusion dominated) and easy to solve
2. Gradually reduce $\nu$ in steps: $[5\cdot 10^{-2}, \dots, 10^{-5}]$
3. Use the solution from the previous step as the initial guess for the next

**Why continuation?**
- Avoids convergence issues at high Reynolds numbers
- Provides good initial guesses for non-linear solver
- More robust than direct solution

---

### 3.2 Stabilization Techniques

To handle the convection-dominated nature of the flow at lower viscosities (higher Reynolds numbers), standard Galerkin FEM becomes unstable (generating spurious oscillations). We employ a **Variational Multiscale (VMS)** approach with stabilization terms:

#### A. SUPG (Streamline-Upwind Petrov-Galerkin)

Added to stabilize the convective term. It adds numerical diffusion **only in the direction of the flow streamlines**, preventing cross-wind diffusion.

$$
\text{Term}_{\text{SUPG}} = \sum_{K} \tau_K (\mathbf{u}^k \cdot \nabla \mathbf{v}, \mathcal{R}(\mathbf{u}, p))_K
$$

- **Purpose**: Prevents node-to-node oscillations in velocity
- $\tau_K$: Stabilization parameter calculated based on local element size ($h$) and velocity magnitude

#### B. Grad-Div Stabilization

Adds a penalty on the divergence of the velocity test functions.

$$
\text{Term}_{\text{GradDiv}} = \gamma \| \mathbf{u} \| h (\nabla \cdot \mathbf{u}, \nabla \cdot \mathbf{v})
$$

- **Purpose**: Enforces mass conservation more strictly and improves the coupling between velocity and pressure solver blocks

#### C. Penalty Method

A small penalty term $\epsilon p q$ is subtracted from the continuity equation ($-\epsilon p q$).

- **Purpose**: Regularizes the pressure matrix, ensuring solvability even if pressure boundary conditions are not strictly defined (removes the "hydrostatic pressure mode" singularity)

---

### 3.3 Thermal Solver (`solve_thermal.py`)

After solving the fluid flow, the velocity field is used to solve the **advection-diffusion equation** for temperature:

$$
\rho c_p (\mathbf{u} \cdot \nabla T) - \nabla \cdot (k \nabla T) = Q
$$

Where:
- $\mathbf{u}$: Velocity field from fluid solver (interpolated onto thermal mesh)
- $k$: Thermal conductivity (different in fluid vs. solid regions)
- $Q$: Volumetric heat source (non-zero only in electronic components)
- $\rho c_p$: Heat capacity (assumed constant)

**Coupling Strategy:**
1. Solve fluid flow to get velocity field
2. Interpolate velocity onto thermal mesh nodes
3. Solve thermal problem with advection term
4. One-way coupling (fluid affects temperature, but not vice versa)

**Why one-way coupling?**
- Simplifies the problem (no need for iterative coupling)
- Valid for small temperature differences (Boussinesq approximation not needed)
- Faster computation

---

### 3.4 Data Storage

For each simulation case, the following files are saved:

```
dataset/case_XXXX/
├── mesh.msh              # GMSH mesh file
├── inputs.npy            # Input fields: {conductivity, power}
├── solutions.npy         # Output fields: {temperature, pressure, vx, vy}
├── fluid_solution.png    # Visualization of velocity field
└── thermal_solution.png  # Visualization of temperature field
```

**Data Format:**
- `inputs.npy`: Dictionary with node-wise values
- `solutions.npy`: Dictionary with node-wise values
- All arrays have shape `(N_nodes,)` where nodes correspond to mesh vertices

---

# Neural Network: GeoFNO Architecture

## Overview

GeoFNO (Geometric Fourier Neural Operator) is a neural network architecture designed to learn mappings between function spaces for solving partial differential equations (PDEs) on irregular geometries. This implementation specifically targets conjugate heat transfer problems involving fluid flow and thermal transport in complex geometries with solid obstacles.

## Problem Statement

The network learns to predict:
- **Temperature field** (T)
- **Pressure field** (P)
- **Velocity components** (vx, vy)

Given:
- **Thermal conductivity** distribution (k)
- **Power/heat source** distribution (Q)
- **Geometry** (solid/fluid regions)

**Key Challenge**: FEM data lives on irregular meshes, but neural networks prefer regular grids. GeoFNO bridges this gap.

---

## Architecture Design

### 1. Encoder: Grid-Based Input Processing

**Input Channels (5 total):**
```
[conductivity, power, grid_x, grid_y, solid_mask]
```

**Why a regular grid?**
- Fourier Neural Operators require structured data for efficient FFT operations
- Grid representation enables spectral convolutions in frequency domain
- Allows for translation-equivariant feature learning

**Design Choice:**
- `nn.Conv2d(5, width, 1)`: Projects multi-channel input to latent space
- Uses 1×1 convolution for channel-wise feature extraction
- Grid resolution: 128×128 (balances accuracy and computational cost)

**Mesh-to-Grid Conversion:**
- Irregular FEM mesh data is interpolated onto a regular 128×128 grid
- Hybrid interpolation: linear (smooth) + nearest-neighbor (robust)
- Preserves geometry information through explicit solid/fluid mask channel

---

### 2. Spectral Processing: FNO Blocks

**Core Component: Spectral Convolution**

The `SpectralConv2d` layer operates in Fourier space:

```python
x_ft = FFT(x)
out_ft = ComplexMultiply(x_ft, learnable_weights)
out = IFFT(out_ft)
```

**Why Fourier space?**
- **Global receptive field**: Each point sees the entire domain instantly
- **Multi-scale learning**: Different Fourier modes capture different scales
- **Parameter efficiency**: Fewer parameters than equivalent spatial convolutions
- **Physical intuition**: Many PDEs have natural frequency-domain representations

**Fourier Modes Configuration:**
- `modes1 = modes2 = 16`
- Captures low to mid-frequency patterns
- Higher modes = more detail, but risk overfitting
- Chosen empirically to balance expressiveness and generalization

**FNO Block Structure:**
```
FNOBlock = SpectralConv + Conv1x1 + InstanceNorm + GELU + Residual
```

**Why this combination?**
- **Spectral path**: Learns global, frequency-based patterns
- **Spatial path (Conv1x1)**: Learns local, pointwise transformations
- **Residual connection**: Enables deep networks, stabilizes training
- **Instance Normalization**: Normalizes per-sample, crucial for varying geometries
- **GELU activation**: Smooth, non-linear, performs well in transformers/FNOs

**Network Depth:**
- 4 FNO blocks
- Each block refines features at different abstraction levels
- Depth chosen to balance capacity and training stability

**Latent Width:**
- `width = 128` channels
- Sufficient capacity for complex multi-physics interactions
- Larger than typical FNO implementations due to coupled fluid-thermal problem

---

### 3. Geometric Querying: Grid-to-Mesh Interpolation

**Challenge:** 
FNO operates on regular grids, but FEM solutions live on irregular meshes.

**Solution:**
```python
x_sampled = F.grid_sample(x_latent, query_coords, mode='bilinear')
```

**Why grid_sample?**
- Differentiable interpolation from grid to arbitrary points
- Bilinear interpolation provides smooth gradients
- Enables querying at exact mesh node locations
- Handles irregular geometries without mesh-specific operations

**Coordinate Normalization:**
- Input coordinates normalized to [-1, 1] for `grid_sample`
- Ensures consistent interpolation across different domain sizes

**This is the key innovation**: The network learns on a regular grid but can predict at arbitrary mesh points!

---

### 4. Decoder: Coordinate Injection MLP

**Architecture:**
```
Input: [latent_features (128), physical_coords (2)] → 130 dimensions
FC1: 130 → 128 (GELU + Dropout)
FC2: 128 → 128 (GELU + Dropout)
FC3: 128 → 4 (T, P, vx, vy)
```

**Why Coordinate Injection?**
- **Inductive bias**: Explicitly provides spatial information
- **Geometry awareness**: Network knows exact query location
- **Continuous representation**: Can query at any point, not just grid nodes
- **Inspired by**: Neural implicit representations (NeRF, DeepSDF)

**Why MLP instead of convolution?**
- Operates on irregular mesh points (no spatial structure)
- Pointwise prediction at each query location
- Flexible output at arbitrary resolutions

**Dropout (0.1):**
- Regularization to prevent overfitting
- Particularly important in decoder where overfitting is common

---

## Training Strategy

### Loss Function: Relative L2 Norm

```
Loss = ||Prediction - Target||₂ / ||Target||₂
```

**Why relative instead of absolute?**
- **Scale invariance**: Works across different problem scales
- **Physical relevance**: Measures relative error, not absolute
- **Standard in FNO literature**: Enables fair comparison
- **Handles multi-scale outputs**: Temperature and velocity have different magnitudes

**Masking:**
- Padded mesh nodes are masked out during loss computation
- Ensures loss only computed on valid data points

---

### Optimization

**Optimizer: AdamW**
- `lr = 1e-3`: Standard learning rate for FNO
- `weight_decay = 1e-1`: Strong regularization to prevent overfitting

**Scheduler: StepLR**
- Reduces learning rate by 0.5 every 20 epochs
- Helps convergence in later training stages

---

### Data Augmentation

**Geometric Augmentations:**

1. **Random horizontal flip** (50% probability)
   - Flips grid and coordinates
   - Negates vy velocity component

2. **Random rotation** (0°, 90°, 180°, 270°)
   - Rotates grid using `torch.rot90`
   - Applies rotation matrix to coordinates and velocity vectors
   - Preserves physics (velocity is a vector field)

**Why augmentation?**
- Limited training data (expensive FEM simulations)
- Improves generalization to unseen geometries
- Exploits symmetries in physics (rotation/reflection invariance)

**Input Noise Injection:**
```python
noise = torch.randn_like(x_grid[:, :2, ...]) * 0.001
```
- Small noise on conductivity and power channels
- Regularization technique (similar to dropout)
- Improves robustness to input perturbations

---

## Data Pipeline

### Mesh-to-Grid Interpolation

**Challenge:** FEM data is on irregular meshes, FNO needs regular grids.

**Hybrid Interpolation Strategy:**
1. **Linear interpolation** (primary method)
   - Smooth, accurate within convex hull
2. **Nearest-neighbor fallback** (for NaN regions)
   - Fills extrapolation regions outside mesh
   - Ensures no missing data

**Solid/Fluid Mask Generation:**
- Automatically detects fluid as most frequent conductivity value
- Creates binary mask: 1 = solid, 0 = fluid
- Uses nearest-neighbor interpolation to preserve sharp boundaries
- Critical for geometry-aware learning

---

### Normalization

**Z-score normalization:**
```
x_norm = (x - mean) / std
```

**Why normalize?**
- Different physical quantities have vastly different scales
- Improves gradient flow and training stability
- Computed globally across entire dataset for consistency

**Cached Statistics:**
- Mean and std saved to `stats.pt`
- Ensures consistent normalization between train/test
- Avoids recomputation on every run

---

### Data Caching

**Pre-loading Strategy:**
- All data loaded into RAM at initialization
- Grid interpolation cached to disk (`grid_hybrid_*.npy`)
- Eliminates I/O bottleneck during training

**Why cache?**
- Mesh-to-grid interpolation is expensive
- Training becomes I/O bound without caching
- Enables fast iteration and experimentation

---

## Key Design Decisions Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Grid Resolution** | 128×128 | Balance between detail and memory |
| **Fourier Modes** | 16×16 | Captures relevant scales without overfitting |
| **Latent Width** | 128 | Sufficient for multi-physics coupling |
| **FNO Depth** | 4 blocks | Deep enough for abstraction, shallow enough to train |
| **Normalization** | InstanceNorm | Handles varying geometries better than BatchNorm |
| **Activation** | GELU | Smooth, performs well in transformer-like architectures |
| **Decoder** | Coordinate Injection MLP | Enables continuous, geometry-aware predictions |
| **Loss** | Relative L2 | Scale-invariant, standard for operator learning |
| **Augmentation** | Flip + Rotation | Exploits physical symmetries, improves generalization |
| **Interpolation** | Hybrid (Linear + Nearest) | Smooth inside mesh, robust outside |

---

## Advantages Over Alternatives

**vs. Standard CNNs:**
- Global receptive field from first layer (Fourier transform)
- More parameter-efficient for PDE learning
- Better at capturing long-range dependencies

**vs. Graph Neural Networks:**
- No need for explicit graph construction
- Faster training (structured grid operations)
- Easier to implement and debug

**vs. Standard FNO:**
- Coordinate injection enables irregular mesh queries
- Better geometry handling through explicit masking
- More robust to varying domain shapes

**vs. Traditional FEM:**
- **Speed**: Milliseconds vs. minutes/hours
- **Scalability**: Amortized cost over many queries
- **Differentiability**: Can be integrated into optimization loops

---

## Limitations and Future Work

**Current Limitations:**
1. Fixed grid resolution (cannot easily change at inference)
2. 2D only (extension to 3D requires significant memory)
3. Assumes rectangular bounding box
4. Limited to single-phase flow
5. One-way coupling (no thermal feedback on flow)

**Potential Improvements:**
1. Multi-resolution training (pyramid approach)
2. Adaptive Fourier modes based on geometry complexity
3. Physics-informed loss terms (conservation laws)
4. Uncertainty quantification (Bayesian extension)
5. Transfer learning across different geometries
6. Two-way coupling for natural convection

---

# Practical Usage

## Project Structure

```
GeoFNO/
├── data_generation/
│   ├── generate_mesh.py      # Mesh generation with GMSH
│   ├── solve_fluid.py         # Navier-Stokes solver
│   ├── solve_thermal.py       # Heat transfer solver
│   ├── data.py                # Material properties and geometry
│   ├── main_data_generation.py # Orchestrates full simulation pipeline
│   └── dataset/               # Generated FEM data
│       └── case_XXXX/
│           ├── mesh.msh
│           ├── inputs.npy
│           └── solutions.npy
├── NN/
│   ├── model.py               # GeoFNO architecture
│   ├── loader.py              # Dataset class (mesh-to-grid conversion)
│   ├── train.py               # Training loop
│   ├── utils.py               # Loss, visualization, augmentation
│   └── results/               # Training outputs
│       ├── best_model.pth
│       ├── prediction_epoch_*.png
│       └── training_history.png
└── README.md                  # This file
```

---

## Installation

### Requirements

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision  # Or use conda for CUDA support
pip install scikit-fem meshio gmsh numpy scipy matplotlib pypardiso
```

**Key Dependencies:**
- `torch`: Neural network framework
- `scikit-fem`: FEM solver
- `gmsh`: Mesh generation
- `meshio`: Mesh I/O
- `pypardiso`: Fast sparse linear solver (optional but recommended)

---

## Workflow

### Step 1: Generate Training Data

```bash
cd data_generation
python main_data_generation.py
```

**What it does:**
1. Generates random geometries (component positions, sizes)
2. Creates conformal meshes with GMSH
3. Solves fluid flow (Navier-Stokes)
4. Solves heat transfer (advection-diffusion)
5. Saves inputs and solutions to `dataset/case_XXXX/`

**Configuration:**
- Edit `data.py` to change material properties, domain size, etc.
- Edit `main_data_generation.py` to change number of cases

**Expected time:** 
- ~1-5 minutes per case (depends on mesh size and convergence)
- Generate 100-1000 cases for good training data

---

### Step 2: Train Neural Network

```bash
cd NN
python train.py
```

**What it does:**
1. Loads all FEM data from `../data_generation/dataset/`
2. Converts irregular meshes to regular 128×128 grids
3. Trains GeoFNO for 1000 epochs (configurable)
4. Saves best model to `results/best_model.pth`
5. Generates visualizations every 10 epochs

**Configuration:**
Edit `train.py` to change:
- `BATCH_SIZE`: Default 16 (reduce if out of memory)
- `EPOCHS`: Default 1000
- `LR`: Learning rate (default 1e-3)
- Grid size, Fourier modes, network width, etc.

**Expected time:**
- ~1-2 hours for 1000 epochs on GPU (dataset dependent)
- CPU training possible but 10-100× slower

**Monitoring:**
- Watch terminal for loss values
- Check `results/prediction_epoch_*.png` for visual quality
- Final loss curves saved to `results/training_history.png`

---

### Step 3: Inference (Using Trained Model)

```python
import torch
from model import GeoFNO
from loader import MeshToGridDataset

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeoFNO(modes1=16, modes2=16, width=128, dropout_rate=0.1).to(device)
model.load_state_dict(torch.load("results/best_model.pth"))
model.eval()

# Load test data
dataset = MeshToGridDataset(
    root_dir="../data_generation/dataset",
    grid_size=(128, 128),
    input_keys=["conductivity", "power"],
    output_keys=["temperature", "pressure", "vx", "vy"],
)

# Predict on a single case
sample = dataset[0]
x_grid = sample["x_grid"].unsqueeze(0).to(device)
coords = sample["query_coords"].unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(x_grid, coords)  # Shape: (1, N_nodes, 4)

# Extract fields
T_pred = prediction[0, :, 0].cpu().numpy()   # Temperature
P_pred = prediction[0, :, 1].cpu().numpy()   # Pressure
vx_pred = prediction[0, :, 2].cpu().numpy()  # Velocity X
vy_pred = prediction[0, :, 3].cpu().numpy()  # Velocity Y
```

**Speed:**
- Inference: ~10-50ms per case on GPU
- FEM simulation: ~1-5 minutes per case
- **Speedup: 1000-10000×**

---

## Hyperparameter Tuning

### Key Hyperparameters to Tune:

1. **Fourier Modes** (`modes1`, `modes2`):
   - Start with 12-16 for simple geometries
   - Increase to 20-32 for complex geometries
   - Higher = more capacity but slower and risk overfitting

2. **Latent Width** (`width`):
   - 64: Fast, less capacity
   - 128: Balanced (default)
   - 256: High capacity, slower

3. **Learning Rate** (`LR`):
   - Too high: Unstable training
   - Too low: Slow convergence
   - Start with 1e-3, reduce if loss oscillates

4. **Weight Decay** (`weight_decay`):
   - Controls regularization
   - 1e-1 (default): Strong regularization
   - Reduce if underfitting, increase if overfitting

5. **Batch Size**:
   - Larger = more stable gradients, more memory
   - Smaller = noisier gradients, less memory
   - 16 is a good default

---

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
- Reduce batch size
- Reduce grid resolution (128 → 64)
- Reduce latent width (128 → 64)
- Use gradient checkpointing

### Issue: Training Loss Not Decreasing

**Solutions:**
- Check data normalization (should have mean~0, std~1)
- Reduce learning rate
- Check for NaN values in data
- Visualize predictions to see if network is learning anything

### Issue: Good Training Loss, Poor Test Loss

**Solutions:**
- Overfitting: Increase weight decay, dropout
- Increase data augmentation probability
- Generate more training data
- Reduce model capacity (fewer modes, smaller width)

### Issue: Slow Data Loading

**Solutions:**
- Ensure caching is enabled (`force_recompute=False`)
- Pre-load all data to RAM (already done in `loader.py`)
- Use SSD instead of HDD for dataset storage

---

## References

### FEM and Stabilization:
- **Taylor-Hood Elements**: Taylor & Hood (1973), "A numerical solution of the Navier-Stokes equations"
- **SUPG Stabilization**: Brooks & Hughes (1982), "Streamline upwind/Petrov-Galerkin formulations"
- **Grad-Div Stabilization**: Olshanskii (2002), "A low order Galerkin finite element method"

### Neural Operators:
- **Fourier Neural Operator**: Li et al. (2020), "Fourier Neural Operator for Parametric Partial Differential Equations"
- **Geometry-Informed Neural Operator**: Li et al. (2022), "Geometry-Informed Neural Operator for Large-Scale 3D PDEs"

### Implicit Representations:
- **NeRF**: Mildenhall et al. (2020), "NeRF: Representing Scenes as Neural Radiance Fields"
- **DeepSDF**: Park et al. (2019), "DeepSDF: Learning Continuous Signed Distance Functions"

---

## License

This project is provided as-is for research and educational purposes.

---

## Contact

For questions or issues, please open an issue on the project repository.
