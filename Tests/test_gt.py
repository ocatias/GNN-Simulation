import os

import torch_geometric
from torch_geometric.data import Data
import torch

import Misc.config as config
from Misc.visualizations import visualize_from_edge_index
import GrapTransformations.subgraph_bag_encoding as SBE

# split_cube = torch.tensor([[0, 1, 1, 2, 0, 2, 0, 3, 2, 3],
#                            [1, 0, 2, 1, 2, 0, 3, 0, 3, 2]], dtype=torch.long)
# split_cube_x = torch.tensor([[0],[1],[2],[3]])
# split_cube_edge_attr = torch.tensor([[0], [0], [10], [10], [100], [100], [1000], [1000], [2000], [2000]])

# data =  Data(x=split_cube_x, edge_index=split_cube, edge_attr=split_cube_edge_attr)
# data =  Data(edge_index=split_cube)

triangle_edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                    [1, 0, 2, 1, 2, 0]], dtype=torch.long)
triangle_x = torch.tensor([[0],[1],[2]])
triangle_edge_attr = torch.tensor([[1], [1], [10], [10], [100], [100]])
data =  Data(x=triangle_x, edge_index=triangle_edge_index, edge_attr=triangle_edge_attr)


data =  Data(x=triangle_x, edge_index=triangle_edge_index, edge_attr=triangle_edge_attr)
# print("original graph:", data)

policy = SBE.policy2transform(policy="edge_deleted", num_hops=4)
sbt = SBE.SubgraphBagEncoding(policy)
transformed_triangle = sbt(data)

# print(sbt)
# print("RESULT:\n", transformed_triangle)
# print(f"max_found in edge: {int(torch.max(transformed_triangle.edge_index))}")
# print(transformed_triangle.x)

# visualize_from_edge_index(transformed_triangle.edge_index, node_types=transformed_triangle.node_types)
x, edge_index, edge_attr = transformed_triangle.x, transformed_triangle.edge_index, transformed_triangle.edge_attr

# Type encoding
for i in range(3):
    assert torch.equal(x[i, 0:3], torch.tensor([1,0,0]))
for i in range(3, 6):
    assert torch.equal(x[i, 0:3], torch.tensor([0,1,0]))
for i in range(6, x.shape[0]):
    assert torch.equal(x[i, 0:3], torch.tensor([0,0,1]))

# Original graph and seperator features
for i in range(3):
    assert torch.equal(x[i, 3:], triangle_x[i])
    assert torch.equal(x[3+i, 3:], triangle_x[i])

# Subgraph features
subgraph_data = policy(Data(x=triangle_x, edge_index=triangle_edge_index, edge_attr=triangle_edge_attr))
subgraph_node_idx =  subgraph_data.subgraph_node_idx

for i, idx in enumerate(subgraph_node_idx.tolist()):
    i = i+6
    assert torch.equal(x[i,3], triangle_x[idx][0])

assert edge_index.shape[1] == 6 + 4*3 + 8*3

# Edges + features in original graph
for i in range(6):
    assert int(edge_attr[i,:]) == int(triangle_edge_attr[i,:])
    assert torch.equal(edge_index[:,i]-3, triangle_edge_index[:,i])

# Edges + features in subgraphs
for i in range(6, 6 + 4*3):
    edge = edge_index[:, i]
    idx1, idx2 = int(edge[0])-6, int(edge[1])-6

    found_edge = False
    for j in range(6):
        # print(triangle_edge_index[:, j], torch.tensor([subgraph_node_idx[idx1], subgraph_node_idx[idx2]]))
        if torch.equal(triangle_edge_index[:, j], torch.tensor([subgraph_node_idx[idx1], subgraph_node_idx[idx2]])):
            found_edge = True
            break

    assert found_edge
    assert int(edge_attr[i,:]) == int(triangle_edge_attr[j,:])

# Edge features involving separators
for i in range(6 + 4*3, edge_index.shape[1]):
    assert int(edge_attr[i,:]) == 0

# Edges involvoing seprators
#   All edges from separators must be to vertices that originate from the same vertex in the original graph
for i in range(3):
    left_side_i = 0
    right_side_i = 0

    for j in range(6 + 4*3, edge_index.shape[1]):
        idx1, idx2 = int(edge_index[0,j]), int(edge_index[1,j])
        if idx1 == i:
            left_side_i += 1
            assert idx2 == i or subgraph_node_idx[idx2-6] == i
        if idx2 == i:
            right_side_i +=1
            assert idx1 == i or subgraph_node_idx[idx1-6] == i

    assert left_side_i == 4
    assert right_side_i == 4

print("All tests passed!")
# ds = torch_geometric.datasets.TUDataset(os.path.join(config.DATA_PATH), name="MUTAG", use_node_attr=True, use_edge_attr=True)
# data = ds[0]
# transformed_data = sbt(data)
# print(transformed_data.x, "\n", transformed_data.edge_index)
# visualize_from_edge_index(transformed_data.edge_index, node_types=transformed_data.node_types)
