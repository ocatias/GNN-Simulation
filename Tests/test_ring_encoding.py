import math

from torch_geometric.data import Data
import torch_geometric

import torch
from GrapTransformations.cell_encoding import CellularRingEncoding, get_rings
from Misc.visualizations import visualize_from_edge_index, visualize_from_data_object

def test(edge_index, name, target):
    rings = get_rings(edge_index)
    print(f"{name}: {rings}")
    assert rings == target


triangle = torch.tensor([[0, 1, 1, 2, 0, 2],
                           [1, 0, 2, 1, 2, 0]], dtype=torch.long)

test(triangle, "Triangle", [(0,1,2)])

split_cube = torch.tensor([[0, 1, 1, 2, 0, 2, 0, 3, 2, 3],
                           [1, 0, 2, 1, 2, 0, 3, 0, 3, 2]], dtype=torch.long)
test(split_cube, "Split_Cube", [(0,1,2),(0,2,3)])

two_cycles = torch.tensor([[0, 1, 1, 2, 0, 2, 0, 3, 3, 4, 4, 5, 0, 5],
                           [1, 0, 2, 1, 2, 0, 3, 0, 4, 3, 5, 4, 5, 0]], dtype=torch.long)
test(two_cycles, "Two cycles", [(0,1,2),(0,3,4,5)])

two_cycles_permuted = torch.tensor([[0, 1, 1, 2, 0, 2, 0, 4, 3, 4, 3, 5, 0, 5],
                           [1, 0, 2, 1, 2, 0, 4, 0, 4, 3, 5, 3, 5, 0]], dtype=torch.long)
test(two_cycles_permuted, "Two cycles (permuted)", [(0,1,2),(0,4,3,5)])



# edge_index =torch.tensor([[0, 1, 1, 2, 0, 2, 0, 3],
#                           [1, 0, 2, 1, 2, 0, 3, 0]], dtype=torch.long)
# x = torch.tensor([[1], [10], [100], [1000], [10000], [1000000]], dtype=torch.float)
# edge_features = torch.tensor([[1], [1], [2], [2], [4], [4], [8], [8]], dtype=torch.float)

# data =  Data(x=x, edge_index=edge_index, edge_attr=edge_features)


# data, id_maps = ring_enc_with_id_maps(data, 7, False, True, True)
# print(data)
# print(id_maps)
# print(data.x)
# print(data.edge_index)
# print(data.edge_attr)
# visualize_from_edge_index(data.edge_index, id_maps)


# edge_index =torch.tensor([[0, 1],
#                           [1, 0]], dtype=torch.long)
# x = torch.tensor([[1], [10],], dtype=torch.float)
# edge_features = torch.tensor([[1], [1], [2], [2]], dtype=torch.float)

# data =  Data(x=x, edge_index=edge_index, edge_attr=edge_features)

# ring_enc = RingEncoding(7, True, True, True)

# data, id_maps = ring_enc.encode_with_id_maps(data)
# print(data)
# print(id_maps)
# print(data.edge_index)
# print(data.edge_attr)
# visualize_from_edge_index(data.edge_index, id_maps)

edge_index =torch.tensor([[0, 1, 1, 2, 0, 2, 0, 3, 3, 4, 4, 5, 0, 5, 2, 5],
                          [1, 0, 2, 1, 2, 0, 3, 0, 4, 3, 5, 4, 5, 0, 5, 2]], dtype=torch.long)
x = torch.tensor([[1], [10], [100], [1000], [10000], [1000000]], dtype=torch.float)
edge_features = torch.tensor([[0], [0], [10], [10], [100], [100], [1000], [1000], [2000], [2000],
    [3000], [3000], [4000], [4000], [10000], [10000]], dtype=torch.float)
data =  Data(x=x, edge_index=edge_index, edge_attr=edge_features)
ring_enc = CellularRingEncoding(200, True, True, True, True)
data_enc, id_maps = ring_enc.encode_with_id_maps(data)

print(data_enc.x.shape)

assert data_enc.x.shape[0] == 17
assert data_enc.edge_index.shape[1] == 16 + 8*4 + 4*2+6*2 + 2*(6+6)
assert data_enc.edge_attr.shape[0] == 16 + 8*4 + 4*2+6*2 + 2*(6+6)

dtype = data_enc.x.dtype

# Check pattern encoding
for i in range(6):
    assert torch.equal(data_enc.x[i,0:3], torch.tensor([1,0,0], dtype=dtype))
for i in range(6, 17-3):
    assert torch.equal(data_enc.x[i,0:3], torch.tensor([0,1,0], dtype=dtype))
for i in range(14, 17):
    assert torch.equal(data_enc.x[i,0:3], torch.tensor([0,0,1], dtype=dtype))

# Check feature aggregation of vertices
# (Edges)
nr_vertices =  data.x.shape[0]
assert torch.equal(data_enc.x[6, 3], (data.x[0, 3]+data.x[1, 3])/2)
assert torch.equal(data_enc.x[7, 3], (data.x[1, 3]+data.x[2, 3])/2)
assert torch.equal(data_enc.x[8, 3], (data.x[0, 3]+data.x[2, 3])/2)
assert torch.equal(data_enc.x[9, 3], (data.x[0, 3]+data.x[3, 3])/2)

# (Rings)
assert torch.equal(data_enc.x[14, 3], (data.x[0, 3]+data.x[1, 3]+data.x[2, 3])/3)
assert torch.equal(data_enc.x[15, 3], (data.x[0, 3]+data.x[3, 3]+data.x[4, 3]+data.x[5, 3])/4)
assert torch.equal(data_enc.x[16, 3], (data.x[0, 3]+data.x[2, 3]+data.x[5, 3])/3)

# Check feature aggregation of edge features in vertices
assert torch.equal(data_enc.x[6, 4], (data.edge_attr[0, 0]))
assert torch.equal(data_enc.x[14, 4], (data.edge_attr[0, 0]+data.edge_attr[2, 0]+data.edge_attr[4, 0])/3)
assert torch.equal(data_enc.x[15, 4], (data.edge_attr[13, 0]+data.edge_attr[11, 0]+data.edge_attr[9, 0]+data.edge_attr[6, 0])/4)

# aggr_edge_atr


assert torch.equal(data_enc.edge_attr[16, 0], data.edge_attr[0, 0])
assert torch.equal(data_enc.edge_attr[17, 0], data.edge_attr[0, 0])
assert torch.equal(data_enc.edge_attr[18, 0], data.edge_attr[0, 0])
assert torch.equal(data_enc.edge_attr[19, 0], data.edge_attr[0, 0])

assert torch.equal(data_enc.edge_attr[24, 0], data.edge_attr[4, 0])
assert torch.equal(data_enc.edge_attr[25, 0], data.edge_attr[5, 0])

def find_edge_idx(v1, v2):
    for idx in range(data_enc.edge_index.shape[1]):
        if set([int(data_enc.edge_index[0, idx]), int(data_enc.edge_index[1, idx])]) == set([v1, v2]):
            return idx

# Vertex(Edge) -> Vertex(Ring)
e1 = find_edge_idx(6, 14)
assert torch.equal(data_enc.edge_attr[e1,0], data.edge_attr[0,0])

# Vertex(Edge) -> Vertex(Edge)
e2 = find_edge_idx(6, 7)
assert torch.equal(data_enc.edge_attr[e2,0], (data.edge_attr[0,0]+data.edge_attr[2,0])/2)

print("All tests passed")

#visualize_from_data_object(data_enc)
