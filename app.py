import math

import numpy as np
import openseespy.opensees as ops

from viktor.parametrization import ViktorParametrization, NumberField, Text, GeometrySelectField, \
    DynamicArray, IntegerField, Step, OptionField
from viktor import ViktorController, UserError
from viktor.geometry import Material, Color, Group, LinearPattern, Point, RectangularExtrusion, Line, \
    BidirectionalPattern, Sphere, Cone, Vector
from viktor.views import GeometryView, GeometryResult


def create_load_arrow(point_node: Point, size: float, direction: str, material=None) -> Group:
    """Function to create a load arrow from a selected node"""
    size_arrow = size
    scale_point = 1.5
    scale_arrow_line = 7

    origin_of_arrow_point = Point(point_node.x - size_arrow, point_node.y,
                                  point_node.z)
    origin_of_arrow_line = Point(origin_of_arrow_point.x - size_arrow, origin_of_arrow_point.y,
                                 origin_of_arrow_point.z)

    arrow_point = Cone(size_arrow / scale_point, size_arrow, origin=origin_of_arrow_point, orientation=Vector(1, 0, 0),
                       material=material)
    arrow_line = RectangularExtrusion(size_arrow / scale_arrow_line, size_arrow / scale_arrow_line,
                                      Line(origin_of_arrow_line, origin_of_arrow_point),
                                      material=material)

    arrow = Group([arrow_point, arrow_line])

    if direction == 'y':
        arrow.rotate(0.5 * math.pi, Vector(0, 0, 1), point=point_node)
    if direction == 'z':
        arrow.rotate(0.5 * math.pi, Vector(0, 1, 0), point=point_node)

    return arrow


class Parametrization(ViktorParametrization):
    step_1 = Step('Select geometry', views=['get_geometry'])
    step_1.text1 = Text(
        "## Frame analysis with OpenSees\n"
        "Template to use OpenSees for a structural analysis of a 3D frame. Opensees: "
        "https://openseespydoc.readthedocs.io/"
    )
    step_1.width = NumberField('Width', min=0, default=30)
    step_1.length = NumberField('Length', min=0, default=30)
    step_1.number_floors = NumberField("Number of floors", variant='slider', min=5, max=40, default=10)
    step_1.no_nodes = IntegerField("Number of nodes per side", min=2, default=4)

    # step_1.node_with_load = GeometrySelectField('Select the node to apply a load')
    step_1.nodes_with_load_array = DynamicArray('Add loads')
    step_1.nodes_with_load_array.magnitude = NumberField('Magnitude of the load')
    step_1.nodes_with_load_array.direction = OptionField('Direction of the load', options=['x', 'y', 'z'], default='x')
    step_1.nodes_with_load_array.node = GeometrySelectField('Select the node to apply a load')

    step_2 = Step('See deformed geometry', views=['get_deformed_geometry'])
    step_2.deformation_scale = NumberField("Scale the deformation", variant='slider', min=0, max=1e7, default=1e5)


class Controller(ViktorController):
    label = "Parametric Building"
    parametrization = Parametrization

    @GeometryView("3D building", duration_guess=1, x_axis_to_right=True)
    def get_geometry(self, params, **kwargs):
        # Create nodes
        node_material = Material("Node", color=Color.viktor_blue())
        floor_height = 4
        node_radius = 0.5
        nodes = []
        for x in np.linspace(0, params.step_1.width, params.step_1.no_nodes):
            for y in np.linspace(0, params.step_1.length, params.step_1.no_nodes):
                for z in range(0, (params.step_1.number_floors + 1) * floor_height, floor_height):
                    nodes.append(Sphere(centre_point=Point(x, y, z),
                                        radius=node_radius,
                                        material=node_material,
                                        identifier=f"{x}-{y}-{z}"))

        # Create beams
        b = 0.3  # Beam width
        beams = []

        for x in np.linspace(0, params.step_1.width, params.step_1.no_nodes):
            for y in np.linspace(0, params.step_1.length, params.step_1.no_nodes):
                p1 = Point(x, y, floor_height)
                p2 = Point(x, y + (params.step_1.length / (params.step_1.no_nodes - 1)), floor_height)
                p3 = Point(x + (params.step_1.width / (params.step_1.no_nodes - 1)), y, floor_height)

                if not y == params.step_1.length:
                    b1 = RectangularExtrusion(b, b, Line(p1, p2))
                    beams.append(b1)

                if not x == params.step_1.width:
                    b2 = RectangularExtrusion(b, b, Line(p1, p3))
                    beams.append(b2)

        beams = Group(beams)

        # Create columns
        c1 = RectangularExtrusion(b, b, Line(Point(0, 0, 0), Point(0, 0, floor_height)))
        columns = BidirectionalPattern(c1, direction_1=[1, 0, 0], direction_2=[0, 1, 0],
                                       number_of_elements_1=params.step_1.no_nodes,
                                       number_of_elements_2=params.step_1.no_nodes,
                                       spacing_1=params.step_1.width / (params.step_1.no_nodes - 1),
                                       spacing_2=params.step_1.length / (params.step_1.no_nodes - 1))

        floor = Group([columns, beams])
        building = [LinearPattern(floor, direction=[0, 0, 1], number_of_elements=params.step_1.number_floors,
                                  spacing=floor_height)]

        if len(params.step_1.nodes_with_load_array) != 0:
            for node in params.step_1.nodes_with_load_array:
                if node.magnitude is not None and node.direction is not None and node.node is not None:
                    coords = [float(i) for i in node.node.split('-')]
                    material_load_arrow = Material("Arrow", color=Color(255, 0, 0))
                    load_arrow = create_load_arrow(Point(coords[0], coords[1], coords[2]), node.magnitude,
                                                   node.direction, material=material_load_arrow)
                    building.append(load_arrow)

        return GeometryResult([Group(nodes), Group(building)])

    @GeometryView("Deformed 3D building", duration_guess=10, x_axis_to_right=True)
    def get_deformed_geometry(self, params, **kwargs):
        # Set geometry properties
        material_deformed = Material("Node", color=Color.viktor_blue())
        material_deformed_arrow = Material("Arrow", color=Color(255, 0, 0))
        material_undeformed = Material("Node", color=Color(220, 220, 220))
        material_undeformed_arrow = Material("Node", color=Color(255, 158, 145))

        floor_height = 4
        node_radius = 0.5
        b = 0.3  # Width columns and beams

        # Set structural properties
        E = 29500.0
        massX = 0.49
        M = 0.
        coordTransf = "Linear"  # Linear, PDelta, Corotational
        massType = "-lMass"  # -lMass, -cMass

        # Find loads
        if len(params.step_1.nodes_with_load_array) == 0:
            raise UserError("Select at least one load in the '3D building' view")
        else:
            nodes_with_load = []
            for node in params.step_1.nodes_with_load_array:
                if node.magnitude is None or node.node is None:
                    raise UserError("Fill out all the information of the load in the previous step")
                else:
                    coords = [float(i) for i in node.node.split('-')]
                    nodes_with_load.append({'coords': coords, 'magnitude': node.magnitude, 'direction': node.direction})

        # Initialize structural model
        ops.wipe()
        ops.model('Basic', '-ndm', 3, '-ndf', 6)

        # Adding undeformed nodes
        undeformed_nodes = []
        node_tag = 1
        for z in range(0, (params.step_1.number_floors + 1) * floor_height, floor_height):
            for x in np.linspace(0, params.step_1.width, params.step_1.no_nodes):
                for y in np.linspace(0, params.step_1.length, params.step_1.no_nodes):
                    # Geometry node
                    undeformed_nodes.append(Sphere(centre_point=Point(x, y, z),
                                                   radius=node_radius,
                                                   material=material_undeformed,
                                                   identifier=f"{x}-{y}-{z}"))
                    # Structural node
                    ops.node(node_tag, x, y, z)
                    ops.mass(node_tag, massX, massX, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)

                    if z == 0:
                        ops.fix(node_tag, 1, 1, 1, 1, 1, 1)  # Fix nodes if they are on the ground level

                    # Find node_tag for node with load
                    if [x, y, z] in [load['coords'] for load in nodes_with_load]:
                        index = [load['coords'] for load in nodes_with_load].index([x, y, z])
                        nodes_with_load[index]['node_tag'] = node_tag

                    node_tag += 1

        # Adding load arrow
        arrows = []
        for node in nodes_with_load:
            x, y, z = node['coords']
            arrow = create_load_arrow(Point(x, y, z), node['magnitude'], node['direction'],
                                      material=material_undeformed_arrow)
            arrows.append(arrow)

        # Defining transformation
        ops.geomTransf(coordTransf, 1, 1, 0, 0)
        ops.geomTransf(coordTransf, 2, 0, 0, 1)

        # Adding columns
        element_tag = 1
        node_tag1 = 1
        columns_undeformed = []

        for k in range(0, params.step_1.number_floors):
            for i in range(0, params.step_1.no_nodes):
                for j in range(0, params.step_1.no_nodes):
                    node_tag2 = node_tag1 + params.step_1.no_nodes * params.step_1.no_nodes
                    i_node = ops.nodeCoord(node_tag1)
                    j_node = ops.nodeCoord(node_tag2)
                    ops.element('elasticBeamColumn', element_tag, node_tag1, node_tag2, 50., E, 1000., 1000., 2150.,
                                2150., 1, '-mass', M, massType)
                    col = RectangularExtrusion(b, b, Line(i_node, j_node), material=material_undeformed)
                    columns_undeformed.append(col)
                    element_tag += 1
                    node_tag1 += 1

        # Adding beams
        undeformed_beams = []

        # Add beam elements in x-direction
        node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes
        for j in range(1, params.step_1.number_floors + 1):
            for i in range(0, (params.step_1.no_nodes - 1)):
                for k in range(0, params.step_1.no_nodes):
                    node_tag2 = node_tag1 + params.step_1.no_nodes
                    i_node = ops.nodeCoord(node_tag1)
                    j_node = ops.nodeCoord(node_tag2)
                    ops.element('elasticBeamColumn', element_tag, node_tag1, node_tag2, 50., E, 1000., 1000., 2150.,
                                2150., 2, '-mass', M, massType)
                    beam = RectangularExtrusion(b, b, Line(i_node, j_node), material=material_undeformed)
                    undeformed_beams.append(beam)
                    element_tag += 1
                    node_tag1 += 1
            node_tag1 += params.step_1.no_nodes
        node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes

        # add beam elements in y-direction
        for j in range(1, params.step_1.number_floors + 1):
            for i in range(0, params.step_1.no_nodes):
                for k in range(0, (params.step_1.no_nodes - 1)):
                    node_tag2 = node_tag1 + 1
                    i_node = ops.nodeCoord(node_tag1)
                    j_node = ops.nodeCoord(node_tag2)
                    ops.element('elasticBeamColumn', element_tag, node_tag1, node_tag2, 50., E, 1000., 1000., 2150.,
                                2150., 2, '-mass', M, massType)
                    beam = RectangularExtrusion(b, b, Line(i_node, j_node), material=material_undeformed)
                    undeformed_beams.append(beam)
                    element_tag += 1
                    node_tag1 += 1
                node_tag1 += 1

        undeformed_building = Group(
            [Group(undeformed_nodes), Group(columns_undeformed), Group(undeformed_beams), Group(arrows)])

        # Define Static Analysis
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        ops.analysis('Static')

        # Adding loads
        for node in nodes_with_load:
            if node['direction'] == 'x':
                ops.load(node['node_tag'], node['magnitude'], 0, 0, 0, 0, 0)
            elif node['direction'] == 'y':
                ops.load(node['node_tag'], 0, node['magnitude'], 0, 0, 0, 0)
            elif node['direction'] == 'z':
                ops.load(node['node_tag'], 0, 0, node['magnitude'], 0, 0, 0)

        # Run Analysis
        ops.analyze(10)

        # Adding undeformed nodes
        deformed_nodes = []
        displaced_points = []
        node_tag = 1
        for z in range(0, (params.step_1.number_floors + 1) * floor_height, floor_height):
            for x in np.linspace(0, params.step_1.width, params.step_1.no_nodes):
                for y in np.linspace(0, params.step_1.length, params.step_1.no_nodes):
                    ux = ops.nodeDisp(node_tag, 1) * params.step_2.deformation_scale
                    uy = ops.nodeDisp(node_tag, 2) * params.step_2.deformation_scale
                    uz = ops.nodeDisp(node_tag, 3) * params.step_2.deformation_scale
                    displaced_point = Point(x + ux, y + uy, z + uz)
                    deformed_nodes.append(Sphere(centre_point=displaced_point,
                                                 radius=node_radius,
                                                 material=material_deformed,
                                                 identifier=f"{x + ux}-{y + uy}-{z + uz}"))
                    displaced_points.append(displaced_point)

                    node_tag += 1

        # Adding load arrow on displaced building
        arrows = []
        for node in nodes_with_load:
            arrow = create_load_arrow(displaced_points[node['node_tag'] - 1], node['magnitude'], node['direction'],
                                      material=material_deformed_arrow)
            arrows.append(arrow)

        # Adding columns
        element_tag = 1
        node_tag1 = 1
        columns_deformed = []
        b = 0.3

        for k in range(0, params.step_1.number_floors):
            for i in range(0, params.step_1.no_nodes):
                for j in range(0, params.step_1.no_nodes):
                    node_tag2 = node_tag1 + params.step_1.no_nodes * params.step_1.no_nodes
                    i_node = displaced_points[node_tag1 - 1]
                    j_node = displaced_points[node_tag2 - 1]
                    col = RectangularExtrusion(b, b, Line(i_node, j_node), material=material_deformed)
                    columns_deformed.append(col)
                    element_tag += 1
                    node_tag1 += 1

        # Adding beams
        deformed_beams = []

        # Add beam elements in x-direction
        node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes
        for j in range(1, params.step_1.number_floors + 1):
            for i in range(0, (params.step_1.no_nodes - 1)):
                for k in range(0, params.step_1.no_nodes):
                    node_tag2 = node_tag1 + params.step_1.no_nodes
                    i_node = displaced_points[node_tag1 - 1]
                    j_node = displaced_points[node_tag2 - 1]
                    beam = RectangularExtrusion(b, b, Line(i_node, j_node), material=material_deformed)
                    deformed_beams.append(beam)
                    element_tag += 1
                    node_tag1 += 1
            node_tag1 += params.step_1.no_nodes
        node_tag1 = 1 + params.step_1.no_nodes * params.step_1.no_nodes

        # add beam elements in y-direction
        for j in range(1, params.step_1.number_floors + 1):
            for i in range(0, params.step_1.no_nodes):
                for k in range(0, (params.step_1.no_nodes - 1)):
                    node_tag2 = node_tag1 + 1
                    i_node = displaced_points[node_tag1 - 1]
                    j_node = displaced_points[node_tag2 - 1]
                    beam = RectangularExtrusion(b, b, Line(i_node, j_node), material=material_deformed)
                    deformed_beams.append(beam)
                    element_tag += 1
                    node_tag1 += 1
                node_tag1 += 1

        deformed_building = Group([Group(deformed_nodes), Group(columns_deformed), Group(deformed_beams),
                                   Group(arrows)])

        return GeometryResult([deformed_building, undeformed_building])
