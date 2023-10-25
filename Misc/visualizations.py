import networkx as nx
import matplotlib.pyplot as plt
import torch

from Misc.utils import edge_tensor_to_list

colors = ['red', 'orange', 'yellow', 'blue', 'green']
default_color = 'black'

def visualize_from_edge_index(edge_index, id_maps=None, node_types=None):
    if id_maps is not None and node_types is None:
        node_types = [0 for _ in range(int(torch.max(edge_index)+1))]
        

        for dim, dict in enumerate(id_maps):
            for key, node in dict.items():
                node_types[node] = key
    visualize(edge_tensor_to_list(edge_index), node_types)

def visualize(edge_list, node_types = None):
    G = nx.Graph()

    for edge in edge_list:
        G.add_edge(*edge)

    color_map = [default_color for i in G]
    if node_types is not None:
        for i, node in enumerate(G):
            color_map[i] = colors[node_types[node]]

    nx.draw(G, node_color=color_map, with_labels=True, font_weight='bold')
    plt.show()

def visualize_from_data_object(data):
    edge_list = edge_tensor_to_list(data.edge_index)
    
    # Transform the node type to an integer
    node_types = list(map(lambda ls: str(ls), data.node_type.tolist()))
    unique_types = list(set(node_types))
    node_types = list(map(lambda type: unique_types.index(type), node_types))
    
    visualize(edge_list, node_types)
    