# GeoFNO: Geometric Fourier Neural Operator for Conjugate Heat Transfer

![Simulation](data_generation/dataset/case_040/simulation.png)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Generation: FEM Simulations](#data-generation-fem-simulations)
3. [Neural Network: GeoFNO Architecture](#neural-network-geofno-architecture)
4. [Practical Usage](#practical-usage)

---

## Project Overview

The workflow consists of two main components:

1. **Data Generation**: High-fidelity FEM simulations of laminar fluid flow and heat transfer over electronic components
2. **Neural Network**: GeoFNO learns to predict temperature, pressure, and velocity fields from geometry and boundary conditions

---

# Data Generation: FEM Simulations

## 1. Simulation Overview

The data generation pipeline simulates **laminar, steady-state airflow** over a Printed Circuit Board (PCB) with mounted electronic components inside a cooling channel. The simulation solves coupled physics:
- **Incompressible Navier-Stokes equations** (fluid flow)
- **Advection-diffusion equation** (heat transfer)

Both are solved using the **Finite Element Method (FEM)** with the `scikit-fem` library.

### Fixed Data & Parameters

The physical setup uses properties of **Air at 20°C** and standard PCB geometry:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Domain Size** | $20.0 \times 5.0$ m | Rectangular channel |
| **PCB Geometry** | Width: 14.0, Height: 0.25 m | Centered in the domain |
| **Kinematic Viscosity** ($\nu$) | $1.0 \times 10^{-5}$ $m^2/s$ | Air viscosity |
| **Inlet Velocity**| Variable | Parabolic profile |
| **Solid Components** | 5 Components | Modeled as obstacles with heat sources |
| **Thermal Conductivity (Air)** | 0.0263 $W/m\cdot K$ | Air conductivity |
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

The mesh is generated using the **GMSH** Python API. The generation process is handled in `data_generation/generate_mesh.py`.
**Strategy**: Conformal quadrilateral meshing. Solid components are fragmented from the fluid domain to ensure continuous nodes at interfaces.

---

## 3. Mathematical Formulation

This project solves the coupled system of fluid dynamics and heat transfer using the **Finite Element Method (FEM)**. The solver is built on `scikit-fem` and utilizes custom Numba-accelerated kernels for the efficient assembly of stabilization terms.

### 3.1 Fluid Dynamics: Incompressible Navier-Stokes

The fluid flow is governed by the steady-state incompressible Navier-Stokes equations. We solve for the velocity field $\mathbf{u} = (u_x, u_y)$ and the kinematic pressure field $p$:

$$
\begin{aligned}
(\mathbf{u} \cdot \nabla) \mathbf{u} - \nu \Delta \mathbf{u} + \nabla p &= 0 \quad \text{in } \Omega_f \\
\nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega_f
\end{aligned}
$$

Where $\nu$ is the kinematic viscosity ($1.0 \times 10^{-5} \, m^2/s$).

#### Discretization (Taylor-Hood Elements)
To satisfy the **LBB (Ladyzhenskaya-Babuska-Brezzi)** inf-sup condition and ensure numerical stability, we employ **Taylor-Hood ($P_2-P_1$)** mixed elements:
- **Velocity Space ($\mathbf{V}_h$):** Continuous piecewise Quadratic ($P_2$) elements (`ElementVectorH1(ElementQuad2)`).
- **Pressure Space ($Q_h$):** Continuous piecewise Linear ($P_1$) elements (`ElementQuad1`).

#### Linearization (Picard Iteration)
The convective term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ is non-linear. We resolve this using **Picard Iterations** (Fixed Point method). At each iteration $k$, the convective velocity is frozen from the previous step:

$$
(\mathbf{u}^{k} \cdot \nabla) \mathbf{u}^{k+1} - \nu \Delta \mathbf{u}^{k+1} + \nabla p^{k+1} = 0
$$

The loop continues until the relative error drops below the tolerance $\epsilon = 5 \times 10^{-3}$.

### 3.2 Stabilization Techniques (VMS)

Standard Galerkin FEM is inherently unstable for convection-dominated flows (high Reynolds numbers). We implement a **Variational Multiscale (VMS)** formulation with three specific stabilization terms added to the weak form.

#### A. SUPG (Streamline-Upwind Petrov-Galerkin)
To prevent spurious node-to-node oscillations in the velocity field, we add diffusion strictly along the flow streamlines. The stabilization term is:

$$
S_{SUPG} = \sum_{K} \tau_K \left( (\mathbf{u}^k \cdot \nabla \mathbf{u}^{k+1} + \nabla p) \cdot (\mathbf{u}^k \cdot \nabla \mathbf{v}) \right)_K
$$

The stabilization parameter $\tau_K$ is calculated dynamically per element based on the local element size $h$ and velocity magnitude $|\mathbf{u}|$:

$$
\tau_K = \frac{\delta_{supg}}{\sqrt{ \left(\frac{2 |\mathbf{u}|}{h}\right)^2 + \left(\frac{36 \nu}{h^2}\right)^2 }}
$$

Where $\delta_{supg} = 0.5$.

#### B. Grad-Div Stabilization
This term penalizes the divergence of the velocity field to improve mass conservation and the coupling between velocity and pressure blocks.

$$
S_{GradDiv} = \gamma |\mathbf{u}| h (\nabla \cdot \mathbf{u}) (\nabla \cdot \mathbf{v})
$$

Where $\gamma = 0.1$ is the scaling factor (`DELTA_GRADDIV`).

#### C. Pressure Penalty Regularization
To ensure the solvability of the linear system and remove hydrostatic pressure modes, a small penalty term is subtracted from the continuity equation:

$$
S_{Penalty} = - \epsilon p q
$$

Where $\epsilon = 10^{-6}$ (`EPS_PENALTY`). This relaxes the incompressibility constraint slightly to $\nabla \cdot \mathbf{u} + \epsilon p = 0$.

#### Final Stabilized Weak Form (Navier-Stokes)

Find the solution $(\mathbf{u}^{k+1}, p^{k+1}) \in \mathbf{V}_h \times Q_h$ such that for all test functions $(\mathbf{v}, q) \in \mathbf{V}_h \times Q_h$, the following residual equation is satisfied:

$$
\begin{aligned}
\mathcal{R}(\mathbf{u}^{k+1}, p^{k+1}; \mathbf{v}, q) &= \\
&\quad \underbrace{ \int_\Omega \nu \nabla \mathbf{u}^{k+1} : \nabla \mathbf{v} \, d\Omega - \int_\Omega p^{k+1} (\nabla \cdot \mathbf{v}) \, d\Omega - \int_\Omega q (\nabla \cdot \mathbf{u}^{k+1}) \, d\Omega }_{\text{Standard Galerkin (Viscous + Pressure + Continuity)}} \\
&\quad + \underbrace{ \int_\Omega [(\mathbf{u}^k \cdot \nabla) \mathbf{u}^{k+1}] \cdot \mathbf{v} \, d\Omega }_{\text{Convection (Picard Linearized)}} \\
&\quad - \underbrace{ \int_\Omega \epsilon \, p^{k+1} q \, d\Omega }_{\text{Penalty Regularization}} \\
&\quad + \underbrace{ \sum_{K \in \mathcal{T}_h} \int_K \tau_K \left( (\mathbf{u}^k \cdot \nabla) \mathbf{u}^{k+1} + \nabla p^{k+1} \right) \cdot \left( (\mathbf{u}^k \cdot \nabla) \mathbf{v} \right) \, d\Omega }_{\text{SUPG Stabilization}} \\
&\quad + \underbrace{ \sum_{K \in \mathcal{T}_h} \int_K \gamma \|\mathbf{u}^k\| h_K (\nabla \cdot \mathbf{u}^{k+1}) (\nabla \cdot \mathbf{v}) \, d\Omega }_{\text{Grad-Div Stabilization}} = 0
\end{aligned}
$$


**Where:**
* $\mathbf{u}^k$: Velocity field from the previous iteration (Frozen).
* $\nu$: Kinematic viscosity (`1e-5`).
* $\epsilon$: Penalty parameter (`1e-6`).
* $\tau_K$: SUPG stabilization parameter (dynamic per element).
* $\gamma$: Grad-Div scaling factor (`0.1`).
* $h_K$: Local element size.

### 3.3 Conjugate Heat Transfer

After obtaining the converged velocity field $\mathbf{u}$, we solve the linear Advection-Diffusion equation for temperature $T$ over the entire domain (Fluid + Solid):

$$
\rho c_p (\mathbf{u} \cdot \nabla T) - \nabla \cdot (k(\mathbf{x}) \nabla T) = Q(\mathbf{x})
$$

- **Coupling:** One-way coupling. The velocity $\mathbf{u}$ is interpolated from the fluid solution onto the thermal mesh.
- **Conductivity ($k$):** Spatially varying field defined in `data.py`:
  - Air: $k \approx 0.0263 \, W/m\cdot K$
  - PCB: $k = 0.3 \, W/m\cdot K$
  - Components: $k \in [200, 500] \, W/m\cdot K$
- **Heat Source ($Q$):** Non-zero volumetric heat generation only within the electronic component subdomains ($10W$ to $30W$ per component).

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

Given:
- **Thermal conductivity** distribution (k)
- **Power/heat source** distribution (Q)
- **Inlet velocity** (scalar value)
- **Geometry** (solid/fluid regions)

**Key Challenge**: FEM data lives on irregular meshes, but neural networks prefer regular grids. GeoFNO bridges this gap.

---

## Architecture Design

### 1. Encoder: Grid-Based Input Processing

**Input Channels (6 total):**
```
[conductivity, power, grid_x, grid_y, solid_mask, vel_inlet]
```

- **conductivity**: Thermal conductivity field (varies between fluid and solid regions)
- **power**: Heat generation field (non-zero in electronic components)
- **grid_x, grid_y**: Spatial coordinates normalized to [-1, 1]
- **solid_mask**: Binary mask distinguishing solid (1) from fluid (0) regions
- **vel_inlet**: Scalar inlet velocity value broadcast across the entire grid

**Why a regular grid?**
- Fourier Neural Operators require structured data for efficient FFT operations
- Grid representation enables spectral convolutions in frequency domain
- Allows for translation-equivariant feature learning

**Design Choice:**
- `nn.Conv2d(6, width, 1)`: Projects multi-channel input to latent space
- Uses 1×1 convolution for channel-wise feature extraction
- Grid resolution: 128×128 (balances accuracy and computational cost)

**Mesh-to-Grid Conversion:**
- Irregular FEM mesh data is interpolated onto a regular 128×128 grid
- Hybrid interpolation: linear (smooth) + nearest-neighbor (robust)
- Preserves geometry information through explicit solid/fluid mask channel
- Inlet velocity is added as a constant scalar channel to provide global boundary condition information

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
- 6 FNO blocks
- Each block refines features at different abstraction levels
- Increased depth for better capacity in temperature prediction

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
FC1: 130 → 130 (GELU + Dropout)
FC2: 130 → 1 (Temperature output)
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

**Simplified Output:**
- Single output channel for temperature prediction
- Focused architecture eliminates multi-task interference
- Better performance for the primary thermal prediction objective

**Dropout (0.1):**
- Regularization to prevent overfitting
- Particularly important in decoder where overfitting is common

---

## Training Strategy

### Loss Function: Mean Squared Error (MSE)

```python
Loss = mse_loss(Prediction, Target, reduction="mean")
```

**Why MSE?**
- **Standard regression loss**: Directly penalizes the difference between predicted and actual temperature values.
- **Smooth gradients**: Provides stable gradients for backpropagation.
- **Physical relevance**: Minimizing the squared difference is equivalent to finding the mean temperature distribution.

**Relative Error Monitoring:**
For evaluation, the model computes a **Relative Error (%)** defined as:
$$
\text{Error}_{\text{rel}} = \frac{|T_{\text{gt}} - T_{\text{pred}}|}{\max(T_{\text{gt}}) - \min(T_{\text{gt}})} \times 100
$$
This provides a more intuitive measure of accuracy independent of the absolute temperature scale.

---

### Optimization

**Optimizer: AdamW**
- `lr = 5e-2`: A high learning rate is used to speed up initial convergence, compensated by the robust weight decay.
- `weight_decay = 1e-4`: Regularization to prevent overfitting by penalizing large weights.
- **Gradient Clipping**: `max_norm=1.0` is applied to prevent gradient explosion and stabilize training.

**Scheduler: StepLR**
- Reduces learning rate by 0.5 every 20 epochs.
- Enables fine-tuning of weights in later training stages.

**Early Stopping:**
- `patience = 50`: Training stops if the test loss does not improve for 50 consecutive epochs.
- Saves computational resources and prevents overfitting to the training set.

---

### Data Augmentation

**Geometric Augmentations:**

1. **Random horizontal flip** (50% probability)
   - Flips the grid and negates the horizontal coordinate.
   - Preserves the thermal physics as the domain is symmetric.

2. **Random rotation** (0°, 90°, 180°, 270°)
   - Rotates the grid using `torch.rot90`.
   - Applies a corresponding rotation matrix to the mesh coordinates.
   - Significantly increases the effective dataset size and improves generalization to different component orientations.

**Why augmentation?**
- Exploits physical symmetries (rotation/reflection invariance).
- Reduces the need for thousands of expensive FEM simulations.
- Encourages the network to learn position-independent features.

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
| **Input Channels** | 6 (k, Q, x, y, mask, vel_inlet) | Includes all relevant physics and boundary conditions |
| **Grid Resolution** | 128×128 | Balance between detail and memory |
| **Fourier Modes** | 16×16 | Captures relevant scales without overfitting |
| **Latent Width** | 128 | Sufficient for thermal physics representation |
| **FNO Depth** | 6 blocks | Deeper network for better feature abstraction |
| **Normalization** | InstanceNorm | Handles varying geometries better than BatchNorm |
| **Activation** | GELU | Smooth, performs well in transformer-like architectures |
| **Decoder** | Coordinate Injection MLP | Enables continuous, geometry-aware predictions |
| **Output** | Temperature only | Focused prediction eliminates multi-task interference |
| **Loss** | Relative L2 (MSE) | Scale-invariant, standard for operator learning |
| **Augmentation** | Flip + Rotation | Exploits physical symmetries, improves generalization |
| **Interpolation** | Hybrid (Linear + Nearest) | Smooth inside mesh, robust outside |

---

## Results

### Training Performance

The model was trained on 500 simulated cases with varying geometries, thermal conductivities, and heat sources. The training converged successfully with both training and test errors decreasing consistently.

![Training History](NN/results/training_history.png)

**Training Statistics:**
- **Training Loss (MSE)**: Converged to ~0.03
- **Test Error (Rel L2)**: Converged to ~0.025 (2.5% relative error)
- **Epochs**: 180 epochs shown
- **Convergence**: Smooth convergence without overfitting

### Example Predictions

Below are example predictions from the test set, showing the model's ability to accurately predict temperature fields for unseen geometries:

#### Test Example
![Test Prediction](NN/results/test_examples/test_sample_001.png)


**Observations:**
- Ground truth and predictions show excellent visual agreement
- Relative error typically below 10% across the domain
- Model correctly captures temperature gradients around heat sources
- Solid-fluid interfaces are accurately resolved
- Cool inlet air and heated components produce expected thermal patterns

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

# Practical Usage

## Project Structure

```
GeoFNO/
├── data_generation/
│   ├── mesh.py                # Mesh generation utility
│   ├── solve_fluid.py         # Navier-Stokes solver (FEM)
│   ├── solve_thermal.py       # Heat transfer solver (FEM)
│   ├── data.py                # Material properties and domain settings
│   ├── generate_data.py       # Main script to run full simulations
│   └── dataset/               # Generated FEM data
│       └── case_XXXX/
│           ├── mesh.msh       # Conformal mesh file
│           ├── inputs.npy     # Field inputs (k, power)
│           ├── solutions.npy  # Solution fields (T, P, u, v)
│           ├── params.pkl     # Case parameters (e.g., vel_inlet)
│           └── simulation.png # Visualization of FEM result
├── NN/
│   ├── model.py               # GeoFNO network architecture
│   ├── loader.py              # Parallel dataset loading and grid interpolation
│   ├── train.py               # Training and evaluation loop
│   ├── utils.py               # Collation, augmentation, and visualization
│   └── results/               # Training outputs
│       ├── best_model.pth     # Trained weights
│       ├── training_history.png
│       └── test_examples/     # Prediction visualizations
└── README.md
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
python generate_data.py
```

**What it does:**
1. Generates random geometries (component positions and sizes).
2. Sets random boundary conditions (variable inlet velocity).
3. Solves the coupled fluid-thermal problem using scikit-fem.
4. Saves high-quality data and visualizations for training.

**Configuration:**
- Edit `data.py` to change domain dimensions or physical constants.
- The script uses parallel processing to speed up generation.

---

### Step 2: Train Neural Network

```bash
cd NN
python train.py
```

**What it does:**
1. Loads all cases from the dataset directory in parallel.
2. Interpolates mesh data onto a 128x128 grid for spectral processing.
3. Pre-pads all samples for efficient batch processing.
4. Trains the GeoFNO model with MSE loss and AdamW optimizer.
5. Employs early stopping and learning rate decay.
6. Automatically saves visualizations of predictions in `results/test_examples/`.

**Configuration:**
Edit `train.py` to change:
- `BATCH_SIZE`: Default 8.
- `EPOCHS`: Default 2000.
- `LR`: Learning rate (default 5e-2).
- `PATIENCE`: Early stopping patience (default 50).
