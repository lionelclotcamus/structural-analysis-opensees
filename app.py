import math
from typing import List, Dict, Tuple

import numpy as np
import openseespy.opensees as ops

import viktor as vkt

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

material_basic_nodes = vkt.Material("Node", color=vkt.Color.viktor_blue())
material_basic = vkt.Material("Building", color=vkt.Color(165, 165, 165))
material_deformed = vkt.Material("Node", color=vkt.Color.viktor_blue())
material_deformed_arrow = vkt.Material("Arrow", color=vkt.Color(255, 0, 0))
material_undeformed = vkt.Material("Node", color=vkt.Color(220, 220, 220), opacity=0.5)
material_undeformed_arrow = vkt.Material("Arrow", color=vkt.Color(255, 158, 145), opacity=0.5)


class Parametrization(vkt.ViktorParametrization):
    step_1 = vkt.Step("Create Model", views=["get_geometry"])
    step_1.text1 = vkt.Text(
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
    step_1.width = vkt.NumberField("Width", min=1, default=30, suffix="m", num_decimals=2)
    step_1.length = vkt.NumberField("Length", min=1, default=30, suffix="m", num_decimals=2)
    step_1.number_floors = vkt.NumberField("Number of floors", variant="slider", min=1, max=40,
                                           default=DEFAULT_NUMBER_FLOORS, num_decimals=0)
    step_1.no_nodes = vkt.IntegerField("Number of nodes per side", min=2, default=4)

    step_1.nodes_with_load_array = vkt.DynamicArray(
        "Add loads",
        default=[{"magnitude": 100, "direction": "x", "node": f"0 - 0 - {DEFAULT_NUMBER_FLOORS * FLOOR_HEIGHT}"}]
    )
    step_1.nodes_with_load_array.magnitude = vkt.NumberField("Load", suffix="kN", num_decimals=2, default=100)
    step_1.nodes_with_load_array.direction = vkt.OptionField("Direction", options=["x", "y", "z"], default="x")
    step_1.nodes_with_load_array.node = vkt.GeometrySelectField("Select the node to apply a load")

    step_2 = vkt.Step("Run Analysis", views=["get_deformed_geometry"], width=30)
    step_2.text = vkt.Text("""
## Run the analysis and view the results
To view the deformed building, click on 'Run analysis' in the bottom right 🔄. You can scale the deformation with the 
'Deformation scale factor' below.
    """)
    step_2.deformation_scale = vkt.NumberField("Deformation scale factor", min=0, max=1e7, default=1000, num_decimals=2)

class Controller(vkt.ViktorController):
    label = "Parametric Building"
    parametrization = Parametrization

    def create_load_arrow(self, point_node: vkt.Point, magnitude: float, direction: str, material=None) -> vkt.Group:
        """Function to create a load arrow from a selected node"""
        size_arrow = abs(magnitude / 20)
        scale_point = 1.5
        scale_arrow_line = 7

        # Create points for the origin of the arrow point and line, based on the coordinate of the node with the load
        origin_of_arrow_point = vkt.Point(point_node.x - size_arrow - NODE_RADIUS, point_node.y,
                                          point_node.z)
        origin_of_arrow_line = vkt.Point(origin_of_arrow_point.x - size_arrow, origin_of_arrow_point.y, 
                                         origin_of_arrow_point.z)

        # Creating the arrow with Viktor Cone and RectangularExtrusion
        arrow_point = vkt.Cone(size_arrow / scale_point, size_arrow, origin=origin_of_arrow_point,
                               orientation=vkt.Vector(1, 0, 0),
                               material=material)
        arrow_line = vkt.RectangularExtrusion(size_arrow / scale_arrow_line, size_arrow / scale_arrow_line,
                                              vkt.Line(origin_of_arrow_line, origin_of_arrow_point),
                                              material=material)

        arrow = vkt.Group([arrow_point, arrow_line])

        # Rotate the arrow if the direction is not 'x' or if the magnitude is negative
        vector = vkt.Vector(0, 0, 1)
        if direction == "y":
            arrow.rotate(0.5 * math.pi, vector, point=point_node)
        if direction == "z":
            vector = vkt.Vector(0, 1, 0)
            arrow.rotate(0.5 * math.pi, vector, point=point_node)
        if magnitude < 0:
            arrow.rotate(math.pi, vector, point=point_node)

        return arrow

    def run_opensees_model(self, nodes_with_load: List[Dict]) -> None:
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

    def add_base_floor(self, params) -> vkt.RectangularExtrusion:
        """Function to add a base floor to the model for visualization.
        This has the width of the building plus some extra space. The extra space is defined by which direction is the
        largest: the width or the length.
        """
        extra_width_floor = max(params.step_1.width, params.step_1.length) / 2
        base_floor = vkt.RectangularExtrusion(
            width=params.step_1.width + 2 * extra_width_floor,
            height=3 * b,
            line=vkt.Line(vkt.Point(0.5 * params.step_1.width, -extra_width_floor, 0),
                          vkt.Point(0.5 * params.step_1.width, params.step_1.length + extra_width_floor, 0))
        )
        return base_floor

    def get_color_from_displacement(self, displacement: float, max_displacement: float) -> Tuple[int, int, int]:
        """Function to determine the color a node should be based on the amount of deformation. The output is a an
        RGB colorcode"""
        # Normalize the displacement to be between 0 and 1
        normalized_displacement = displacement / max_displacement

        # Color scale: dark blue (0) to red (1)
        # Interpolate between the following colors:
        # dark blue (0, 0, 139) -> blue (0, 0, 255) -> green (0, 255, 0) ->
        # yellow (255, 255, 0) -> orange (255, 165, 0) -> red (255, 0, 0)
        color_scale = [
            (0, 0, 139),  # Dark Blue
            (0, 0, 255),  # Blue
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (255, 165, 0),  # Orange
            (255, 0, 0)  # Red
        ]

        # Interpolate the color
        num_colors = len(color_scale) - 1
        idx = min(int(normalized_displacement * num_colors), num_colors - 1) # Ensure the index is within the range

        # Get the lower and upper bounds for interpolation
        c1 = color_scale[idx]
        c2 = color_scale[idx + 1]

        # Fractional part to interpolate between c1 and c2
        frac = (normalized_displacement * num_colors) - idx

        # Interpolate between two colors
        red = int((1 - frac) * c1[0] + frac * c2[0])
        green = int((1 - frac) * c1[1] + frac * c2[1])
        blue = int((1 - frac) * c1[2] + frac * c2[2])

        return red, green, blue

    def add_color_to_column_or_beam(self, node_tag1: int, node_tag2: int, abs_displacement_nodes: List[float],
                                    max_displacement: float) -> vkt.Material:
        """Function to determine the color of a beam or column based on the amount of displacement. The amount of
        displacement is determined by taking the average of the displacement of the two adjacent nodes."""
        # Find displacement of the adjacent nodes
        displacement_i_node = abs_displacement_nodes[node_tag1 - 1]
        displacement_j_node = abs_displacement_nodes[node_tag2 - 1]
        # Use the average displacement of the 2 nodes to determine the displacement
        displacement_mid_column = 0.5 * (displacement_i_node + displacement_j_node)
        red, green, blue = self.get_color_from_displacement(displacement_mid_column, max_displacement)

        return vkt.Material("Column or beam", color=vkt.Color(red, green, blue))

    def add_nodes(self, params, mode_of_deformation: str, nodes_with_load: List[Dict] | List,
                  material_nodes: vkt.Material) \
            -> Tuple[List[vkt.Point], List[vkt.Sphere], List, float]:
        """Function to add nodes to the OpenSees model and for visualisation.
        The nodes will be added by looping over the building.
        If the mode is 'deformed', the displacement will be considered. If the mode is 'undeformed', nodes will be
        added to the OpenSees model and the node with a load need to be found.
        """
        max_displacement = 0
        abs_displacement_nodes = []
        if mode_of_deformation == "deformed":
            for node_tag in range(1, (params.step_1.number_floors + 1) * params.step_1.no_nodes ** 2 + 1):
                ux = ops.nodeDisp(node_tag, 1)
                uy = ops.nodeDisp(node_tag, 2)
                uz = ops.nodeDisp(node_tag, 3)

                abs_displacement = math.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
                abs_displacement_nodes.append(abs_displacement)
                if abs_displacement > max_displacement:
                    max_displacement = abs_displacement

        # Adding the nodes by looping through the levels, width and length of the building for the number of nodes.
        nodes = []
        points = []
        node_tag = 1
        for z in range(0, (params.step_1.number_floors + 1) * FLOOR_HEIGHT, FLOOR_HEIGHT):
            for x in np.linspace(0, params.step_1.width, params.step_1.no_nodes):
                for y in np.linspace(0, params.step_1.length, params.step_1.no_nodes):
                    if mode_of_deformation == 'deformed':
                        ux = ops.nodeDisp(node_tag, 1) * params.step_2.deformation_scale
                        uy = ops.nodeDisp(node_tag, 2) * params.step_2.deformation_scale
                        uz = ops.nodeDisp(node_tag, 3) * params.step_2.deformation_scale

                        # Determine the color of the node based on the displacement and change the material
                        red, green, blue = self.get_color_from_displacement(abs_displacement_nodes[node_tag - 1],
                                                                            max_displacement)
                        material_nodes = vkt.Material("Node", color=vkt.Color(red, green, blue))

                    else:
                        ux, uy, uz = 0, 0, 0
                    point = vkt.Point(x + ux, y + uy, z + uz)
                    points.append(point)
                    # Create Viktor node to visualize
                    nodes.append(vkt.Sphere(centre_point=point,
                                            radius=NODE_RADIUS,
                                            material=material_nodes,
                                            identifier=f"{x + ux}-{y + uy}-{z + uz}"))

                    if mode_of_deformation == 'undeformed':
                        # Create the OpenSees structural node. The node is identified with a node tag.
                        ops.node(node_tag, x, y, z)
                        ops.mass(node_tag, mass_x_node, mass_x_node, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)

                        # If the node is on the ground floor, it is fixed so this should be added the the OpenSees node.
                        if z == 0:
                            ops.fix(node_tag, 1, 1, 1, 1, 1, 1)

                        # Check if this node is the selected node with a (or multiple) load(s) to find the node tag.
                        coords_lst = [load["coords"] for load in nodes_with_load]
                        if [x, y, z] in coords_lst:
                            # Find all the indices of the nodes_with_load that have x, y, z as coordinates and for all
                            # these indices, add the current node_tag to the dictionary
                            indices = [i for i in range(len(coords_lst)) if coords_lst[i] == [x, y, z]]
                            for index in indices:
                                nodes_with_load[index]["node_tag"] = node_tag

                    node_tag += 1

        return points, nodes, abs_displacement_nodes, max_displacement

    def add_arrows(self, nodes_with_load: List[Dict] | List, points: List[vkt.Point], material_arrow: vkt.Material) \
            -> List[vkt.Group] | List:
        """Function to add an arrow to the building for visualisation.
        To create the load arrows, loop through the nodes with a load.
        """
        arrows = []
        for node in nodes_with_load:
            arrow = self.create_load_arrow(points[node["node_tag"] - 1], node["magnitude"], node["direction"],
                                           material=material_arrow)
            arrows.append(arrow)

        return arrows

    def add_columns(self, params, mode_of_deformation: str, points: List[vkt.Point], material: vkt.Material,
                    abs_displacement_nodes: List, max_displacement: float) \
            -> Tuple[List[vkt.RectangularExtrusion], int]:
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

                    if mode_of_deformation == "undeformed":
                        # Create the OpenSees element
                        # Definition of element in OpenSees docs: ops.element('elasticBeamColumn', eleTag, *eleNodes,
                        # Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag, <'-mass', mass>, <'-cMass'>)
                        ops.element("elasticBeamColumn", element_tag, node_tag1, node_tag2, area, E, G, Jxx, Iy,
                                    Iz, 1, "-mass", mass_x_element, mass_type)

                    i_node = points[node_tag1 - 1]
                    j_node = points[node_tag2 - 1]

                    # If the mode is "deformed", the color of the column will change based on the amount of
                    # displacement of the adjacent nodes. Therefore, the material should be changed
                    if mode_of_deformation == "deformed":
                        material = self.add_color_to_column_or_beam(node_tag1, node_tag2, abs_displacement_nodes,
                                                                    max_displacement)

                    # Create the column to visualize
                    col = vkt.RectangularExtrusion(width=b, height=b, line=vkt.Line(i_node, j_node), material=material)
                    columns.append(col)

                    element_tag += 1
                    node_tag1 += 1

        return columns, element_tag

    def add_beams(self, params, mode_of_deformation: str, points: List[vkt.Point], element_tag: int,
                  material: vkt.Material, abs_displacement_nodes: List, max_displacement: float) \
            -> List[vkt.RectangularExtrusion]:
        """"Function to add beams to the model.
        First in the x-direction, then in the y-direction by looping over the levels and the width and length.
        If the mode is 'undeformed', add the beams as an element to the OpenSees model."""
        beams = []

        # Add beam elements in x-direction. Start on the first floor. Loop in the width over the number of nodes but
        # skip the last node because otherwise the beam would extend outwards of the building. And loop over the length.
        node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes
        for j in range(1, params.step_1.number_floors + 1):
            for i in range(0, (params.step_1.no_nodes - 1)):
                for k in range(0, params.step_1.no_nodes):
                    # To get the other node (one to the right) to which the beam is connected, add the number of nodes.
                    # Add the structural element of the beam.
                    node_tag2 = node_tag1 + params.step_1.no_nodes
                    if mode_of_deformation == "undeformed":
                        ops.element("elasticBeamColumn", element_tag, node_tag1, node_tag2, 50., E, 1000., 1000.,
                                    2150., 2150., 2, "-mass", mass_x_element, mass_type)

                    # Find the coordinates of the nodes and add the Viktor element for the visualization of the beam
                    i_node = points[node_tag1 - 1]
                    j_node = points[node_tag2 - 1]

                    # If the mode is "deformed", the color of the column will change based on the amount of
                    # displacement of the adjacent nodes. Therefore, the material should be changed
                    if mode_of_deformation == "deformed":
                        material = self.add_color_to_column_or_beam(node_tag1, node_tag2, abs_displacement_nodes,
                                                                    max_displacement)

                    beam = vkt.RectangularExtrusion(width=b, height=b, line=vkt.Line(i_node, j_node), material=material)
                    beams.append(beam)

                    element_tag += 1
                    node_tag1 += 1

            node_tag1 += params.step_1.no_nodes  # To go the next column of nodes (in the x,y-plane)
        node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes  # To go to the next floo

        # Add beam elements in y-direction. Start on the first floor. Loop in the width. And loop over the length over
        # the number of nodes but skip the last node because otherwise the beam would extend outwards of the building.
        for j in range(1, params.step_1.number_floors + 1):
            for i in range(0, params.step_1.no_nodes):
                for k in range(0, (params.step_1.no_nodes - 1)):
                    # To get the other node (one forward) to which the beam is connected, add 1. And add the structural
                    # element of the beam.
                    node_tag2 = node_tag1 + 1
                    if mode_of_deformation == "undeformed":
                        ops.element("elasticBeamColumn", element_tag, node_tag1, node_tag2, 50., E, 1000., 1000.,
                                    2150., 2150., 2, "-mass", mass_x_element, mass_type)

                    # Find the coordinates of the nodes and add the Viktor element for the visualization of the beam
                    i_node = points[node_tag1 - 1]
                    j_node = points[node_tag2 - 1]

                    # If the mode is "deformed", the color of the column will change based on the amount of
                    # displacement of the adjacent nodes. Therefore, the material should be changed
                    if mode_of_deformation == "deformed":
                        material = self.add_color_to_column_or_beam(node_tag1, node_tag2, abs_displacement_nodes,
                                                                    max_displacement)

                    beam = vkt.RectangularExtrusion(width=b, height=b, line=vkt.Line(i_node, j_node),
                                                material=material)
                    beams.append(beam)
                    element_tag += 1
                    node_tag1 += 1
                node_tag1 += 1  # To get to the column of nodes (in the x,y-plane)

        return beams

    def generate_building(self, params, mode_of_deformation: str, nodes_with_load: List[Dict] | List,
                          material_nodes: vkt.Material, material: vkt.Material, material_arrow: vkt.Material) \
            -> Tuple[List[vkt.Sphere], List]:
        # Initialize structural model
        if mode_of_deformation == "undeformed":
            ops.wipe()
            ops.model("Basic", "-ndm", 3, "-ndf", 6)

        # Adding the base floor, nodes and arrows for the loads.
        base_floor = self.add_base_floor(params)
        points, nodes, abs_displacement_nodes, max_displacement = self.add_nodes(
            params, mode_of_deformation, nodes_with_load, material_nodes
        )
        arrows = self.add_arrows(nodes_with_load, points, material_arrow)

        if mode_of_deformation == "undeformed":
            # Defining different transformations for the OpenSees analysis.
            ops.geomTransf(coord_transf, 1, 1, 0, 0)
            ops.geomTransf(coord_transf, 2, 0, 0, 1)

        columns, element_tag = self.add_columns(params, mode_of_deformation, points, material, abs_displacement_nodes,
                                                max_displacement)  # Adding columns
        beams = self.add_beams(params, mode_of_deformation, points, element_tag, material, abs_displacement_nodes,
                               max_displacement)  # Adding beams

        # Create the undeformed building, which is the base floor, the nodes, the columns and the beams
        building_lst = [base_floor, vkt.Group(nodes), vkt.Group(columns), vkt.Group(beams), vkt.Group(arrows)]

        return nodes, building_lst

    @vkt.GeometryView("3D building", duration_guess=1, x_axis_to_right=True)
    def get_geometry(self, params, **kwargs):
        # Generate the undeformed building with its nodes
        undeformed_nodes, undeformed_building_lst = self.generate_building(
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
                        if (vkt.Point(coords[0], coords[1], coords[2]) not in
                                [sphere.centre_point for sphere in undeformed_nodes]):
                            raise vkt.UserError(f"The selected node for load number {i + 1} is not an existing node,"
                                            f" reselect the node.")

                        # Create the arrow of the load and add it to the building
                        material_load_arrow = vkt.Material("Arrow", color=vkt.Color(255, 0, 0))
                        load_arrow = self.create_load_arrow(vkt.Point(coords[0], coords[1], coords[2]), node.magnitude,
                                                            node.direction, material=material_load_arrow)
                        undeformed_building_lst.append(load_arrow)
                else:
                    # If the information is not complete, show an error
                    raise vkt.UserError(f"Complete the information from load number {i + 1}.")

        return vkt.GeometryResult(vkt.Group(undeformed_building_lst))

    @vkt.GeometryView("Deformed 3D building", duration_guess=10, x_axis_to_right=True, update_label="Run analysis")
    def get_deformed_geometry(self, params, **kwargs):
        # Find loads. If no loads are selected, an error is displayed to the user
        if len(params.step_1.nodes_with_load_array) == 0:
            raise vkt.UserError("Select at least one load in the '3D building' view")
        else:
            nodes_with_load = []
            # Loop through the array with loads
            for node in params.step_1.nodes_with_load_array:
                # Check if all the information is given for the load. If not, display an error
                if node.magnitude is None or node.node is None:
                    raise vkt.UserError("Fill out all the information of the load in the previous step")
                else:
                    # Find the coordinate of the node. Add it to a dictionary for easy access later
                    coords = [float(i) for i in node.node.split('-')]
                    nodes_with_load.append({"coords": coords, "magnitude": node.magnitude, "direction": node.direction})

        # Get the undeformed model with its nodes.
        undeformed_nodes, undeformed_building_lst = self.generate_building(
            params, "undeformed", nodes_with_load,
            material_nodes=material_undeformed,
            material=material_undeformed,
            material_arrow=material_undeformed_arrow
        )
        undeformed_building = vkt.Group(undeformed_building_lst)

        # Run the OpenSees model
        self.run_opensees_model(nodes_with_load)

        # Get the deformed model
        _, deformed_building_lst = self.generate_building(
            params, "deformed", nodes_with_load,
            material_nodes=material_deformed,
            material=material_deformed,
            material_arrow=material_deformed_arrow
        )
        deformed_building = vkt.Group(deformed_building_lst)

        return vkt.GeometryResult([deformed_building, undeformed_building])
