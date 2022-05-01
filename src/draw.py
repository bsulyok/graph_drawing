from plotly.graph_objects import Figure, Scatterpolargl
from math import pi
from operator import itemgetter
import numpy as np
from .typing import AdjacencyList, VertexList
from . import geometry
from scipy import sparse


def draw(
    adjacency_list: AdjacencyList,
    vertex_list: VertexList,
    vertex_size_attr: str = 'size',
    vertex_scale: float = 1,
    vertex_name_attr: str = 'name',
    vertex_color_attr: str = 'color',
    edge_color_attr: str = 'color',
    edge_width_attr: str = 'weight',
    edge_scale: float = 1,
    save_file: str = None,
    height: float = 1000,
    width: float = 980,
    title: str = '',
    representation: str = 'native_disk',
    line_type: str = 'euclidean'
) -> None:

    N = len(vertex_list)
    K = sum(len(neigh) for neigh in adjacency_list.values())

    #################################
    # retrieve some drawing options #
    #################################

    if representation == 'native_disk':
        coord_getter = itemgetter('r', 'theta')
        get_coordinates = lambda x: coord_getter(x)
    elif representation == 'hyperboloid':
        coord_getter = itemgetter('t', 'x', 'y')
        get_coordinates = lambda x: geometry.hyperboloid_to_native_disk(coord_getter(x))

    if line_type == 'euclidean':
        line_function = geometry.euclidean_line
        distance_function = geometry.euclidean_distance
    elif line_type == 'hyperbolic':
        line_function = geometry.native_disk_line
        distance_function = geometry.native_disk_distance

    #############################
    # gather vertex information #
    #############################

    vertex_names = []
    vertex_array = np.zeros((N, 4), dtype=float)
    for idx, (v, attr) in enumerate(vertex_list.items()):
        vertex_array[idx, 0: 2] = get_coordinates(attr)
        vertex_array[idx, 2] = attr.get(vertex_size_attr, 1)
        vertex_array[idx, 3] = attr.get(vertex_color_attr, 1)
        vertex_names.append(attr.get(vertex_name_attr, f'vertex_{v}'))
    vertex_array[:, 2] = np.sqrt(vertex_array[:, 2] * vertex_scale * 10000 / np.sum(vertex_array[:, 2]))

    ###########################
    # gather edge information #
    ###########################

    def edge_iterator(adjacency_list):
        for vertex, neighbourhood in adjacency_list.items():
            for neighbour, attributes in neighbourhood.items():
                yield vertex, neighbour, attributes

    vertex_dict = {vertex: idx for idx, vertex in enumerate(vertex_list)}
    edge_array = np.zeros((K, 6), dtype=float)
    for idx, (u, v, attr) in enumerate(edge_iterator(adjacency_list)):
        u, v = vertex_dict[u], vertex_dict[v]
        edge_array[idx, 0: 2] = vertex_array[u, 0: 2]
        edge_array[idx, 2: 4] = vertex_array[v, 0: 2]
        edge_array[idx, 4] = attr.get(edge_width_attr, 1)
        edge_array[idx, 5] = attr.get(edge_color_attr, 1)
    length = distance_function(edge_array[:, :4])
    edge_array[:, 4] *= edge_scale * 2000 / np.sum(length * edge_array[:, 4])

    ##################
    # draw the edges #
    ##################

    fig = Figure()
    for idx, (r_1, theta_1, r_2, theta_2, line_width, color) in enumerate(edge_array):
        r, theta = line_function(r_1, theta_1, r_2, theta_2)
        fig.add_trace(
            Scatterpolargl(
                r=r,
                theta=theta,
                mode='lines',
                line_color=color,
                hoverinfo='skip',
                line_width=line_width,
                thetaunit='radians'
            )
        )

    #####################
    # draw the vertices #
    #####################

    fig.add_trace(
        Scatterpolargl(
            r=vertex_array[:, 0],
            theta=vertex_array[:, 1],
            mode='markers',
            hoverinfo='text+r+theta',
            hovertext=vertex_names,
            marker_cmin=0,
            marker_cmax=1,
            marker_colorscale='HSV_r',
            marker_size=vertex_array[:, 2],
            marker_color=vertex_array[:, 3],
            showlegend=False,
            marker_reversescale=True,
            marker_opacity=1,
            thetaunit='radians'
        )
    )

    ###################
    # layout settings #
    ###################

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        showlegend=False,
        polar=dict(
            angularaxis=dict(
                thetaunit='radians',
                dtick=pi/2,
                showline = True,
                linewidth = 1,
                linecolor='black',
                tickwidth=1,
                tickvals=[],
            ),
            radialaxis=dict(
                showline = False,
                linewidth = 1,
                linecolor='black',
                ticklen=1,
                tickvals=[],
            )
        )
    )

    ###########################
    # print or export to file #
    ###########################
    
    if save_file is None:
        fig.show()
    else:
        fig.write_image(save_file)