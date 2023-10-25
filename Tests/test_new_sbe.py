import os

import torch_geometric
from torch_geometric.data import Data
import torch
#from ogb.graphproppred import PygGraphPropPredDataset

import Misc.config as config
from Misc.visualizations import visualize_from_edge_index, visualize_from_data_object
import GrapTransformations.subgraph_bag_encoding as SBE

triangle_edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                    [1, 0, 2, 1, 2, 0]], dtype=torch.long)
triangle_x = torch.tensor([[0],[1],[2]])
triangle_edge_attr = torch.tensor([[1], [1], [10], [10], [100], [100]])
data =  Data(x=triangle_x, edge_index=triangle_edge_index, edge_attr=triangle_edge_attr)


data =  Data(x=triangle_x, edge_index=triangle_edge_index, edge_attr=triangle_edge_attr)
# print("original graph:", data)

policy = SBE.policy2transform(policy="ego_nets", num_hops=3)
sbt = SBE.SubgraphBagEncodingNoSeparator(policy, dss_message_passing = True, connect_accross_subg = False)
transformed_triangle = sbt(data)
print(transformed_triangle)
print(transformed_triangle.edge_index)
visualize_from_data_object(transformed_triangle)

# molhiv = PygGraphPropPredDataset(root=config.DATA_PATH, name="ogbg-molhiv")
# #data = molhiv[0]

# for data in molhiv:
#     print(data)
#     transformed_data = sbt(data)
#     print(transformed_data)

#     for col in transformed_data.edge_index:
#         for x in col:
#             assert x >= 0 and x <= transformed_data.num_nodes