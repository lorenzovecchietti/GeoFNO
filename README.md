# CFD Simulation: Fluid Flow Over a Circuit Board

## 1. Simulation Overview
This project simulates the **laminar, steady-state airflow** over a Printed Circuit Board (PCB) with mounted electronic components inside a cooling channel. The simulation solves the **incompressible Navier-Stokes equations** using the Finite Element Method (FEM).

### Simulated Variables
- **Velocity field ($\mathbf{u}$)**: Vector field $[u_x, u_y]$ representing fluid speed and direction.
- **Pressure field ($p$)**: Scalar field representing static pressure distribution.

### Fixed Data & Parameters
The physical setup uses properties of **Air at 20°C** and standard PCB geometry:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Domain Size** | $20.0 \times 5.0$ Units | Rectangular channel. |
| **PCB Geometry** | Width: 14.0, Height: 0.25 | Centered in the domain. |
| **Fluid Density** | (Implicit in $\nu$) | Incompressible assumption. |
| **Kinematic Viscosity ($\nu$)** | $1.0 \times 10^{-5}$ $m^2/s$ | Target viscosity (Air). |
| **Inlet Velocity**| $U_{max} = 0.05$ m/s | Parabolic profile. |
| **Solid Solids** | 5 Components | Modeled as obstacles (holes in fluid mesh). |

**Boundary Conditions:**
- **Inlet**: Prescribed parabolic velocity profile.
- **Outlet**: Natural boundary condition (do-nothing/traction free).
- **Walls & Solids**: No-slip condition ($\mathbf{u} = 0$).

---

## 2. Mesh Generation
The mesh is generated using the **GMSH** Python API with the OpenCASCADE kernel to ensure robust boolean operations. The generation process is handled in `data_generation/generate_mesh.py`.

### Process:
1.  **Geometry Definition**: Rectangles are defined for the channel, the PCB, and the 5 electronic components.
2.  **Fragmentation**: The `gmsh.model.occ.fragment` operation cuts the solid shapes out of the fluid domain. This ensures the mesh is **conformal**—nodes match perfectly at the fluid-solid interfaces.
3.  **Physical Tagging**: Since IDs change after fragmentation, the script automatically identifies surfaces using their **Center of Mass** to assign physical tags (Fluid, PCB, Inlet, Outlet, etc.).
4.  **Discretization**: The domain is meshed with **Quadrilaterals** (via `RecombineAll=1`) to support high-quality tensor product elements.

---

## 3. Mathematical Formulation (`solve_fluid.py`)
The solver uses **scikit-fem (skfem)** to discretize the weak form of the Navier-Stokes equations.

### Finite Elements
We use **Taylor-Hood (P2-P1)** elements, which satisfy the LBB (Ladyzhenskaya-Babuska-Brezzi) condition for stability:
- **Velocity**: Quadratic (P2) Quadrilateral elements (`ElementVectorH1(ElementQuad2)`).
- **Pressure**: Linear (P1) Quadrilateral elements (`ElementQuad1`).

### Linearization (Picard Estimator)
The convective term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ is non-linear. We linearize it using **Picard iterations** (Fixed Point iteration):
$$
(\mathbf{u}^{k+1} \cdot \nabla) \mathbf{u}^{k+1} \approx (\mathbf{u}^{k} \cdot \nabla) \mathbf{u}^{k+1}
$$
Where $\mathbf{u}^k$ is the solution from the previous iteration.

### Continuation Method (Viscosity Stepping)
Directly solving for low viscosity ($\nu = 10^{-5}$) is difficult due to non-linearity. The solver employs a **Continuation Method**:
1.  Start with high viscosity ($\nu = 5 \times 10^{-2}$), where the flow is Stokes-like (diffusion dominated) and easy to solve.
2.  Gradually reduce $\nu$ in steps: $[5\cdot 10^{-2}, \dots, 10^{-5}]$.
3.  Use the solution from the previous step as the initial guess for the next.

---

## 4. Stabilization Techniques
To handle the convection-dominated nature of the flow at lower viscosities (higher Reynolds numbers), standard Galerkin FEM becomes unstable (generating spurious oscillations). We employ a **Variational Multiscale (VMS)** approach with two key stabilization terms added to the weak form:

### A. SUPG (Streamline-Upwind Petrov-Galerkin)
Added to stabilize the convective term. It adds numerical diffusion **only in the direction of the flow streamlines**, preventing cross-wind diffusion.
$$
\text{Term}_{\text{SUPG}} = \sum_{K} \tau_K (\mathbf{u}^k \cdot \nabla \mathbf{v}, \mathcal{R}(\mathbf{u}, p))_K
$$
- **Purpose**: Prevents node-to-node oscillations in velocity.
- $\tau_K$: Stabilization parameter calculated based on local element size ($h$) and velocity magnitude.

### B. Grad-Div Stabilization
Adds a penalty on the divergence of the velocity test functions.
$$
\text{Term}_{\text{GradDiv}} = \gamma \| \mathbf{u} \| h (\nabla \cdot \mathbf{u}, \nabla \cdot \mathbf{v})
$$
- **Purpose**: Enforces mass conservation more strictly and improves the coupling between velocity and pressure solver blocks.

### C. Penalty Method
A small penalty term $\epsilon p q$ is subtracted from the continuity equation ($-\epsilon p q$).
- **Purpose**: Regularizes the pressure matrix, ensuring solvability even if pressure boundary conditions are not strictly defined (removes the "hydrostatic pressure mode" singularity).
