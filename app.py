import math
from typing import List, Dict, Tuple

import numpy as np
import openseespy.opensees as ops
from munch import Munch

from viktor.parametrization import (ViktorParametrization, NumberField, Text, GeometrySelectField, DynamicArray,
                                    IntegerField, Step, OptionField)
from viktor import ViktorController, UserError
from viktor.geometry import Material, Color, Group, Point, RectangularExtrusion, Line, Sphere, Cone, Vector
from viktor.views import GeometryView, GeometryResult, Label

FLOOR_HEIGHT = 4
DEFAULT_NUMBER_FLOORS = 10
NODE_RADIUS = 0.5
b = 0.3  # Width columns and beams

# Set structural properties
area = 50  # cross-sectional area of the elements
E = 29500.0  # Young's modulus of the elements
mass_x_node = 0.49  # node mass per unit length
mass_x_element = 0.  # element mass per unit length
G = 1000.  # Shear modulus
Jxx = 1000.  # Torsional moment of inertia of cross section
Iy = 2150.  # Second moment of area about the local y-axis
Iz = 2150.  # Second moment of area about the local z-axis
coord_transf = "Linear"  # Linear, PDelta, Corotational
mass_type = "-lMass"  # -lMass, -cMass

material_basic_nodes = Material("Node", color=Color.viktor_blue())
material_basic = Material("Building", color=Color(165, 165, 165))
material_deformed = Material("Node", color=Color.viktor_blue())
material_deformed_arrow = Material("Arrow", color=Color(255, 0, 0))
material_undeformed = Material("Node", color=Color(220, 220, 220), opacity=0.5)
material_undeformed_arrow = Material("Arrow", color=Color(255, 158, 145), opacity=0.5)

OFFSET_LABEL_SCALE = 40


def create_load_arrow(point_node: Point, magnitude: float, direction: str, material=None) -> Group:
    """Function to create a load arrow from a selected node"""
    size_arrow = magnitude / 20
    scale_point = 1.5
    scale_arrow_line = 7

    # Create points for the origin of the arrow point and line, based on the coordinate of the node with the load
    origin_of_arrow_point = Point(point_node.x - size_arrow - NODE_RADIUS, point_node.y,
                                  point_node.z)
    origin_of_arrow_line = Point(origin_of_arrow_point.x - size_arrow, origin_of_arrow_point.y,
                                 origin_of_arrow_point.z)

    # Creating the arrow with Viktor Cone and RectangularExtrusion
    arrow_point = Cone(size_arrow / scale_point, size_arrow, origin=origin_of_arrow_point, orientation=Vector(1, 0, 0),
                       material=material)
    arrow_line = RectangularExtrusion(size_arrow / scale_arrow_line, size_arrow / scale_arrow_line,
                                      Line(origin_of_arrow_line, origin_of_arrow_point),
                                      material=material)

    arrow = Group([arrow_point, arrow_line])

    # Rotate the arrow if the direction is not 'x'
    if direction == "y":
        arrow.rotate(0.5 * math.pi, Vector(0, 0, 1), point=point_node)
    if direction == "z":
        arrow.rotate(0.5 * math.pi, Vector(0, 1, 0), point=point_node)

    return arrow


def run_opensees_model(nodes_with_load: List[Dict]) -> None:
    """Function to run the opensees model.
    First, the analysis is defined and the loads are added before the analysis can be run.
    """
    # Define Static Analysis
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.analysis("Static")

    # Adding loads
    for node in nodes_with_load:
        if node["direction"] == "x":
            ops.load(node["node_tag"], node["magnitude"], 0, 0, 0, 0, 0)
        elif node["direction"] == "y":
            ops.load(node["node_tag"], 0, node["magnitude"], 0, 0, 0, 0)
        elif node["direction"] == "z":
            ops.load(node["node_tag"], 0, 0, -1 * node["magnitude"], 0, 0, 0)

    # Run Analysis
    ops.analyze(10)

    return


def add_base_floor(params: Munch) -> RectangularExtrusion:
    """Function to add a base floor to the model for visualization.
    This has the width of the building plus some extra space. The extra space is defined by which direction is the
    largest: the width or the length.
    """
    extra_width_floor = max(params.step_1.width, params.step_1.length) / 2
    base_floor = RectangularExtrusion(
        width=params.step_1.width + 2 * extra_width_floor,
        height=3 * b,
        line=Line(Point(0.5 * params.step_1.width, -extra_width_floor, 0),
                  Point(0.5 * params.step_1.width, params.step_1.length + extra_width_floor, 0))
    )
    return base_floor


def add_nodes(params: Munch, mode_of_deformation: str, nodes_with_load: List[Dict] | List, material_nodes: Material) \
        -> Tuple[List[Point], List[Sphere], List[Label]]:
    """Function to add nodes to the OpenSees model and for visualisation.
    First, an offset is defined for the positioning of the labels next to the nodes. Then, the nodes will be added by
    looping over the building.
    If the mode is 'deformed', the displacement will be considered. If the mode is 'undeformed', nodes will be added to
    the OpenSees model and the node with a load need to be found.
    """
    # Defining the offset for the label of the node
    offset_label_x = params.step_1.width / OFFSET_LABEL_SCALE
    offset_label_y = params.step_1.length / OFFSET_LABEL_SCALE
    offset_label_z = (params.step_1.number_floors * FLOOR_HEIGHT) / OFFSET_LABEL_SCALE

    # Adding the nodes by looping through the levels, width and length of the building for the number of nodes.
    nodes = []
    points = []
    node_labels = []
    node_tag = 1
    for z in range(0, (params.step_1.number_floors + 1) * FLOOR_HEIGHT, FLOOR_HEIGHT):
        for x in np.linspace(0, params.step_1.width, params.step_1.no_nodes):
            for y in np.linspace(0, params.step_1.length, params.step_1.no_nodes):
                if mode_of_deformation == 'deformed':
                    ux = ops.nodeDisp(node_tag, 1) * params.step_2.deformation_scale
                    uy = ops.nodeDisp(node_tag, 2) * params.step_2.deformation_scale
                    uz = ops.nodeDisp(node_tag, 3) * params.step_2.deformation_scale
                else:
                    ux, uy, uz = 0, 0, 0
                point = Point(x + ux, y + uy, z + uz)
                points.append(point)
                # Create Viktor node and label to visualize
                nodes.append(Sphere(centre_point=point,
                                    radius=NODE_RADIUS,
                                    material=material_nodes,
                                    identifier=f"{x + ux}-{y + uy}-{z + uz}"))
                node_labels.append(
                    Label(Point(x + ux + offset_label_x, y + uy + offset_label_y, z + uz + offset_label_z),
                          str(node_tag), size_factor=0.6)
                )

                if mode_of_deformation == 'undeformed':
                    # Create the OpenSees structural node. The node is identified with a node tag.
                    ops.node(node_tag, x, y, z)
                    ops.mass(node_tag, mass_x_node, mass_x_node, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)

                    # If the node is on the ground floor, it is fixed so this should be added the the OpenSees node.
                    if z == 0:
                        ops.fix(node_tag, 1, 1, 1, 1, 1, 1)

                    # Check if this node is the selected node with a load to find the node tag of the node with a load.
                    if [x, y, z] in [load["coords"] for load in nodes_with_load]:
                        index = [load["coords"] for load in nodes_with_load].index([x, y, z])
                        nodes_with_load[index]["node_tag"] = node_tag

                node_tag += 1

    return points, nodes, node_labels


def add_arrows(nodes_with_load: List[Dict] | List, points: List[Point], material_arrow: Material) -> List[Group] | List:
    """Function to add an arrow to the building for visualisation.
    To create the load arrows, loop through the nodes with a load.
    """
    arrows = []
    for node in nodes_with_load:
        arrow = create_load_arrow(points[node["node_tag"] - 1], node["magnitude"], node["direction"],
                                  material=material_arrow)
        arrows.append(arrow)

    return arrows


def add_columns(params: Munch, mode_of_deformation: str, points: List[Point], material: Material) \
        -> Tuple[List[RectangularExtrusion], int]:
    """Function to add columns to the model.
    Adding columns by looping over the floors, and the nodes in the width and length of the building.
    If the mode is 'undeformed', the columns will be added to the OpenSees model with as an element.
    """
    element_tag = 1
    node_tag1 = 1
    columns = []
    for k in range(0, params.step_1.number_floors):
        for i in range(0, params.step_1.no_nodes):
            for j in range(0, params.step_1.no_nodes):
                # Find the node and its coordinates that is at the same x,y location but on the next floor to create
                # the vertical column.
                node_tag2 = node_tag1 + params.step_1.no_nodes * params.step_1.no_nodes
                i_node = points[node_tag1 - 1]
                j_node = points[node_tag2 - 1]

                if mode_of_deformation == "undeformed":
                    # Create the OpenSees element
                    # Definition of element in OpenSees docs: ops.element('elasticBeamColumn', eleTag, *eleNodes, Area,
                    # E_mod, G_mod, Jxx, Iy, Iz, transfTag, <'-mass', mass>, <'-cMass'>)
                    ops.element("elasticBeamColumn", element_tag, node_tag1, node_tag2, area, E, G, Jxx, Iy, Iz, 1,
                                "-mass", mass_x_element, mass_type)

                # Create the column to visualize
                col = RectangularExtrusion(width=b, height=b, line=Line(i_node, j_node), material=material)
                columns.append(col)

                element_tag += 1
                node_tag1 += 1

    return columns, element_tag


def add_beams(params: Munch, mode_of_deformation: str, points: List[Point], element_tag: int, material: Material) \
        -> List[RectangularExtrusion]:
    """"Function to add beams to the model.
    First in the x-direction, then in the y-direction by looping over the levels and the width and length.
    If the mode is 'undeformed', add the beams as an element to the OpenSees model."""
    beams = []

    # Add beam elements in x-direction. Start on the first floor. Loop in the width over the number of nodes but skip
    # the last node because otherwise the beam would extend outwards of the building. And loop over the length.
    node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes
    for j in range(1, params.step_1.number_floors + 1):
        for i in range(0, (params.step_1.no_nodes - 1)):
            for k in range(0, params.step_1.no_nodes):
                # To get the other node (one to the right) to which the beam is connected, add the number of nodes. Add
                # the structural element of the beam.
                node_tag2 = node_tag1 + params.step_1.no_nodes
                if mode_of_deformation == "undeformed":
                    ops.element("elasticBeamColumn", element_tag, node_tag1, node_tag2, 50., E, 1000., 1000., 2150.,
                                2150., 2, "-mass", mass_x_element, mass_type)

                # Find the coordinates of the nodes and add the Viktor element for the visualization of the beam
                i_node = points[node_tag1 - 1]
                j_node = points[node_tag2 - 1]
                beam = RectangularExtrusion(width=b, height=b, line=Line(i_node, j_node), material=material)
                beams.append(beam)

                element_tag += 1
                node_tag1 += 1

        node_tag1 += params.step_1.no_nodes  # To go the next column of nodes (in the x,y-plane)
    node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes  # To go to the next floo

    # Add beam elements in y-direction. Start on the first floor. Loop in the width. And loop over the length over the
    # number of nodes but skip the last node because otherwise the beam would extend outwards of the building.
    for j in range(1, params.step_1.number_floors + 1):
        for i in range(0, params.step_1.no_nodes):
            for k in range(0, (params.step_1.no_nodes - 1)):
                # To get the other node (one forward) to which the beam is connected, add 1. And add the structural
                # element of the beam.
                node_tag2 = node_tag1 + 1
                if mode_of_deformation == "undeformed":
                    ops.element("elasticBeamColumn", element_tag, node_tag1, node_tag2, 50., E, 1000., 1000., 2150.,
                                2150., 2, "-mass", mass_x_element, mass_type)

                # Find the coordinates of the nodes and add the Viktor element for the visualization of the beam
                i_node = points[node_tag1 - 1]
                j_node = points[node_tag2 - 1]
                beam = RectangularExtrusion(width=b, height=b, line=Line(i_node, j_node),
                                            material=material)
                beams.append(beam)
                element_tag += 1
                node_tag1 += 1
            node_tag1 += 1  # To get to the column of nodes (in the x,y-plane)

    return beams


def generate_building(params: Munch, mode_of_deformation: str, nodes_with_load: List[Dict] | List,
                      material_nodes: Material, material: Material, material_arrow: Material) \
        -> Tuple[List[Sphere], List, List[Label]]:
    # Initialize structural model
    if mode_of_deformation == "undeformed":
        ops.wipe()
        ops.model("Basic", "-ndm", 3, "-ndf", 6)

    # Adding the base floor, nodes and arrows for the loads.
    base_floor = add_base_floor(params)
    points, nodes, node_labels = add_nodes(params, mode_of_deformation, nodes_with_load, material_nodes)
    arrows = add_arrows(nodes_with_load, points, material_arrow)

    if mode_of_deformation == "undeformed":
        # Defining different transformations for the OpenSees analysis.
        ops.geomTransf(coord_transf, 1, 1, 0, 0)
        ops.geomTransf(coord_transf, 2, 0, 0, 1)

    columns, element_tag = add_columns(params, mode_of_deformation, points, material)  # Adding columns
    beams = add_beams(params, mode_of_deformation, points, element_tag, material)  # Adding beams

    # Create the undeformed building, which is the base floor, the nodes, the columns and the beams
    building_lst = [base_floor, Group(nodes), Group(columns), Group(beams), Group(arrows)]

    return nodes, building_lst, node_labels


class Parametrization(ViktorParametrization):
    step_1 = Step("Create Model", views=["get_geometry"])
    step_1.text1 = Text(
        """## Structural Analysis using OpenSees\n
Welcome to our Structural Analysis App, a tool designed for the analysis of 3D frame buildings. 
Built on the OpenSeesPy framework, which leverages OpenSees (Open System for Earthquake Engineering Simulation), 
widely-used software for simulating the response of structural and geotechnical systems to loads. 

This app allows users to: 

* easily customize building dimensions 📏, 
* apply directional loads to specific nodes ↗️,
* visualize the resulting deformations after running a structural analysis 🏗️. ️\n
The docs of OpenSeesPy can be found on 
[this page](https://openseespydoc.readthedocs.io/).
        """
    )
    step_1.width = NumberField("Width", min=1, default=30, suffix="m", num_decimals=2)
    step_1.length = NumberField("Length", min=1, default=30, suffix="m", num_decimals=2)
    step_1.number_floors = NumberField("Number of floors", variant="slider", min=1, max=40,
                                       default=DEFAULT_NUMBER_FLOORS, num_decimals=0)
    step_1.no_nodes = IntegerField("Number of nodes per side", min=2, default=4)

    step_1.nodes_with_load_array = DynamicArray(
        "Add loads",
        default=[{"magnitude": 100, "direction": "x", "node": f"0 - 0 - {DEFAULT_NUMBER_FLOORS * FLOOR_HEIGHT}"}]
    )
    step_1.nodes_with_load_array.magnitude = NumberField("Load", suffix="kN", num_decimals=2, default=100)
    step_1.nodes_with_load_array.direction = OptionField("Direction", options=["x", "y", "z"], default="x")
    step_1.nodes_with_load_array.node = GeometrySelectField("Select the node to apply a load")

    step_2 = Step("Run Analysis", views=["get_deformed_geometry"], width=30)
    step_2.text = Text("""
## Run the analysis and view the results
To view the deformed building, click on 'Run analysis' in the bottom right 🔄. You can scale the deformation with the 
'Deformation scale factor' below.
    """)
    step_2.deformation_scale = NumberField("Deformation scale factor", min=0, max=1e7, default=1000, num_decimals=2)


class Controller(ViktorController):
    label = "Parametric Building"
    parametrization = Parametrization

    @GeometryView("3D building", duration_guess=1, x_axis_to_right=True)
    def get_geometry(self, params, **kwargs):
        # Generate the undeformed building with its nodes and labels
        undeformed_nodes, undeformed_building_lst, labels = generate_building(
            params, "undeformed", [],
            material_nodes=material_basic_nodes,
            material=material_basic,
            material_arrow=material_undeformed_arrow
        )
        # Find nodes that are selected to have a load
        if len(params.step_1.nodes_with_load_array) != 0:
            for i, node in enumerate(params.step_1.nodes_with_load_array):
                # Check if the information is complete
                if node.magnitude is not None and node.direction is not None:
                    if node.node is not None:
                        # Find the coordinates of the node and check if it is part of the undeformed nodes. If not,
                        # display an error to the user.
                        coords = [float(i) for i in node.node.split("-")]
                        if (Point(coords[0], coords[1], coords[2]) not in
                                [sphere.centre_point for sphere in undeformed_nodes]):
                            raise UserError(f"The selected node for load number {i + 1} is not an existing node, reselect "
                                            f"the node.")

                        # Create the arrow of the load and add it to the building
                        material_load_arrow = Material("Arrow", color=Color(255, 0, 0))
                        load_arrow = create_load_arrow(Point(coords[0], coords[1], coords[2]), node.magnitude,
                                                       node.direction, material=material_load_arrow)
                        undeformed_building_lst.append(load_arrow)
                else:
                    # If the information is not complete, show an error
                    raise UserError(f"Complete the information from load number {i + 1}.")

        return GeometryResult(Group(undeformed_building_lst), labels)

    @GeometryView("Deformed 3D building", duration_guess=10, x_axis_to_right=True, update_label="Run analysis")
    def get_deformed_geometry(self, params, **kwargs):
        # Find loads. If no loads are selected, an error is displayed to the user
        if len(params.step_1.nodes_with_load_array) == 0:
            raise UserError("Select at least one load in the '3D building' view")
        else:
            nodes_with_load = []
            # Loop through the array with loads
            for node in params.step_1.nodes_with_load_array:
                # Check if all the information is given for the load. If not, display an error
                if node.magnitude is None or node.node is None:
                    raise UserError("Fill out all the information of the load in the previous step")
                else:
                    # Find the coordinate of the node. Add it to a dictionary for easy access later
                    coords = [float(i) for i in node.node.split('-')]
                    nodes_with_load.append({"coords": coords, "magnitude": node.magnitude, "direction": node.direction})

        # Get the undeformed model with its nodes and labels.
        undeformed_nodes, undeformed_building_lst, undeformed_labels = generate_building(
            params, "undeformed", nodes_with_load,
            material_nodes=material_undeformed,
            material=material_undeformed,
            material_arrow=material_undeformed_arrow
        )
        undeformed_building = Group(undeformed_building_lst)

        # Run the OpenSees model
        run_opensees_model(nodes_with_load)

        # Get the deformed model with its labels
        deformed_nodes, deformed_building_lst, labels_deformed_nodes = generate_building(
            params, "deformed", nodes_with_load,
            material_nodes=material_deformed,
            material=material_deformed,
            material_arrow=material_deformed_arrow
        )
        deformed_building = Group(deformed_building_lst)

        return GeometryResult([deformed_building, undeformed_building], labels_deformed_nodes)
