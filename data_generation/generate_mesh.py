# pylint: disable=too-many-locals
"""
generate_mesh.py

Modularized 2D GMSH mesh generator for a circuit board with fluid domain,
PCB, and electronic components. Uses OpenCASCADE kernel and physical groups.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gmsh

from data import (
    COMPONENT_BASE_TAG,
    FLUID_TAG,
    INLET_TAG,
    OUTLET_TAG,
    PCB_TAG,
    SOLID_TAG,
    WALLS_TAG,
)

# -------------------------------
# Data Structures
# -------------------------------


@dataclass
class CircuitBoard:
    # pylint: disable=too-many-instance-attributes
    """Parameters defining the circuit board geometry."""

    h_pcb: float  # PCB height fraction of domain
    w_pcb: float  # PCB width fraction of domain
    n_up: int  # Number of components on top side
    w_comps: List[float]  # Width fractions of components
    h_comps: List[float]  # Height fractions of components
    h: float = 5.0  # Total domain height
    w: float = 20.0  # Total domain width

    def __post_init__(self) -> None:
        """Compute number of components on bottom side."""
        self.n_down = 5 - self.n_up


# -------------------------------
# Geometry Definition
# -------------------------------


def define_geometry(
    params: CircuitBoard,
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[List[Tuple[float, float]]],
]:
    """
    Define rectangular regions for domain, PCB, and components.

    Returns:
        domain: [(x0,y0), (x1,y1)]
        pcb:    [(x0,y0), (x1,y1)]
        components: list of [(x0,y0), (x1,y1)]
    """
    domain = [(0.0, 0.0), (params.w, params.h)]

    pcb_width = params.w * params.w_pcb
    pcb_height = params.h * params.h_pcb
    pcb_x0 = 0.5 * params.w * (1 - params.w_pcb)
    pcb_y0 = 0.5 * params.h * (1 - params.h_pcb)
    pcb = [(pcb_x0, pcb_y0), (pcb_x0 + pcb_width, pcb_y0 + pcb_height)]

    components = []
    total_comps = params.n_up + params.n_down

    for i in range(total_comps):
        if i < params.n_up:
            sub_idx = i
            sign = 1
            n_slots = params.n_up
            y_start = pcb[1][1]
        else:
            sub_idx = i - params.n_up
            sign = -1
            n_slots = params.n_down
            y_start = pcb[0][1]

        slot_width = pcb_width / n_slots
        comp_width = params.w_comps[i] * slot_width
        x_start = pcb[0][0] + slot_width * sub_idx + 0.5 * (slot_width - comp_width)
        x_end = x_start + comp_width
        y_end = y_start + sign * params.h_comps[i] * pcb_height

        components.append([(x_start, y_start), (x_end, y_end)])

    return domain, pcb, components


# -------------------------------
# Geometry Creation in GMSH
# -------------------------------


def create_geometry_entities(
    domain_coords: List[Tuple[float, float]],
    pcb_coords: List[Tuple[float, float]],
    components: List[List[Tuple[float, float]]],
) -> Tuple[int, int, List[int]]:
    """
    Create 2D rectangles in GMSH using OpenCASCADE.

    Returns:
        domain_tag, pcb_tag, list of component_tags
    """
    domain_tag = gmsh.model.occ.addRectangle(
        domain_coords[0][0],
        domain_coords[0][1],
        0,
        domain_coords[1][0] - domain_coords[0][0],
        domain_coords[1][1] - domain_coords[0][1],
    )

    pcb_tag = gmsh.model.occ.addRectangle(
        pcb_coords[0][0],
        pcb_coords[0][1],
        0,
        pcb_coords[1][0] - pcb_coords[0][0],
        pcb_coords[1][1] - pcb_coords[0][1],
    )

    component_tags = []
    for comp in components:
        tag = gmsh.model.occ.addRectangle(
            comp[0][0], comp[0][1], 0, comp[1][0] - comp[0][0], comp[1][1] - comp[0][1]
        )
        component_tags.append(tag)

    return domain_tag, pcb_tag, component_tags


# -------------------------------
# Domain Fragmentation
# -------------------------------


def fragment_domain(domain_tag: int, solid_tags: List[int]) -> None:
    """Cut solids out of the fluid domain using boolean fragment."""
    gmsh.model.occ.fragment([(2, domain_tag)], [(2, tag) for tag in solid_tags])
    gmsh.model.occ.synchronize()


# -------------------------------
# Physical Group Assignment (Surfaces)
# -------------------------------


def assign_surface_physical_groups(
    pcb_tag: int, component_tags_2d: List[int], tol: float = 1e-6
) -> Tuple[List[int], int, List[int]]:
    """
    Identify fluid, PCB, and component surfaces after fragmentation.
    Assign physical groups.

    Returns:
        fluid_surfaces, pcb_surface_tag, component_surface_tags
    """
    surfaces = gmsh.model.getEntities(2)

    # Pre-compute centers of mass
    com_pcb = gmsh.model.occ.getCenterOfMass(2, pcb_tag)
    com_comps = {
        tag: gmsh.model.occ.getCenterOfMass(2, tag) for tag in component_tags_2d
    }

    pcb_surf: int = -1
    comp_surf_map: Dict[int, int] = {}
    solid_surfs = set()

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)

        # Match PCB
        if abs(com[0] - com_pcb[0]) < tol and abs(com[1] - com_pcb[1]) < tol:
            pcb_surf = tag
            solid_surfs.add(tag)
            continue

        # Match components
        for orig_tag, orig_com in com_comps.items():
            if abs(com[0] - orig_com[0]) < tol and abs(com[1] - orig_com[1]) < tol:
                comp_surf_map[orig_tag] = tag
                solid_surfs.add(tag)
                break

    all_surfs = {tag for _, tag in surfaces}
    fluid_surfs: list[int] = list(all_surfs - solid_surfs)
    comp_surfs = [comp_surf_map[t] for t in component_tags_2d if t in comp_surf_map]

    # Assign physical groups
    gmsh.model.addPhysicalGroup(2, fluid_surfs, FLUID_TAG)
    gmsh.model.setPhysicalName(2, FLUID_TAG, "Fluid")

    if pcb_surf:
        gmsh.model.addPhysicalGroup(2, [pcb_surf], PCB_TAG)
        gmsh.model.setPhysicalName(2, PCB_TAG, "PCB")

    for i, surf_tag in enumerate(comp_surfs):
        tag_val = COMPONENT_BASE_TAG + i
        gmsh.model.addPhysicalGroup(2, [surf_tag], tag_val)
        gmsh.model.setPhysicalName(2, tag_val, f"Component_{i+1}")

    return fluid_surfs, pcb_surf, comp_surfs


# -------------------------------
# Physical Group Assignment (Boundaries)
# -------------------------------


def assign_boundary_physical_groups(
    domain_coords: List[Tuple[float, float]], pcb_surf: int, comp_surfs: List[int]
) -> None:
    """
    Identify and tag inlet, outlet, walls, and fluid-solid interfaces.
    """
    lines = gmsh.model.getEntities(1)
    inlet, outlet, walls = [], [], []

    for dim, tag in lines:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x, y = com[0], com[1]

        if abs(x - domain_coords[0][0]) < 1e-4:
            inlet.append(tag)
        elif abs(x - domain_coords[1][0]) < 1e-4:
            outlet.append(tag)
        elif abs(y - domain_coords[0][1]) < 1e-4 or abs(y - domain_coords[1][1]) < 1e-4:
            if tag not in inlet and tag not in outlet:
                walls.append(tag)

    # Fluid-solid interface lines
    pcb_lines = [t for d, t in gmsh.model.getBoundary([(2, pcb_surf)], oriented=False)]
    comp_lines = []
    for surf in comp_surfs:
        comp_lines += [
            t for d, t in gmsh.model.getBoundary([(2, surf)], oriented=False)
        ]

    interface_lines_set = set(pcb_lines + comp_lines)
    interface_lines = [
        t for t in interface_lines_set if t not in inlet + outlet + walls
    ]

    # Assign physical groups
    if inlet:
        gmsh.model.addPhysicalGroup(1, inlet, INLET_TAG)
        gmsh.model.setPhysicalName(1, INLET_TAG, "Inlet")
    if outlet:
        gmsh.model.addPhysicalGroup(1, outlet, OUTLET_TAG)
        gmsh.model.setPhysicalName(1, OUTLET_TAG, "Outlet")
    if walls:
        gmsh.model.addPhysicalGroup(1, walls, WALLS_TAG)
        gmsh.model.setPhysicalName(1, WALLS_TAG, "Walls")
    if interface_lines:
        gmsh.model.addPhysicalGroup(1, interface_lines, SOLID_TAG)
        gmsh.model.setPhysicalName(1, SOLID_TAG, "SolidInterfaces")


# -------------------------------
# Mesh Generation
# -------------------------------


def generate_mesh(mesh_size: float) -> None:
    """Set mesh options and generate 2D mesh."""
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size / 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 32)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)


# -------------------------------
# Main Function
# -------------------------------


def generate_gmsh_mesh_2d(
    params: CircuitBoard,
    mesh_size: float = 0.1,
    output_file: str = "circuit_mesh_2d.msh",
) -> None:
    """
    Generate a 2D structured-like mesh of a circuit board in fluid.

    Creates:
      - 2D domains: fluid, PCB, components
      - 1D boundaries: inlet, outlet, walls, fluid-solid interfaces
    """
    gmsh.initialize()
    gmsh.model.add("circuit_board_2d")

    try:
        # 1. Define geometry
        domain_coords, pcb_coords, components = define_geometry(params)

        # 2. Create entities
        domain_tag, pcb_tag, comp_tags = create_geometry_entities(
            domain_coords, pcb_coords, components
        )

        # 3. Fragment domain
        fragment_domain(domain_tag, [pcb_tag] + comp_tags)

        # 4. Assign surface groups
        _, pcb_surf, comp_surfs = assign_surface_physical_groups(pcb_tag, comp_tags)

        # 5. Assign boundary groups
        assign_boundary_physical_groups(domain_coords, pcb_surf, comp_surfs)

        # 6. Generate and save mesh
        generate_mesh(mesh_size)
        gmsh.write(output_file)

    finally:
        gmsh.finalize()
