from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from ogb.graphproppred import PygGraphPropPredDataset

import Misc.config as config 
from Misc.visualizations import visualize_from_edge_index
from GrapTransformations.cell_encoding import CellularRingEncoding, get_rings

ds1 = ZINC(root=config.DATA_PATH, subset=True, split="train")
ds2 = GNNBenchmarkDataset(root=config.DATA_PATH, name ="CIFAR10", split="train")
ds3 = PygGraphPropPredDataset(root=config.DATA_PATH, name="ogbg-molpcba")

print(len(ds3))
print("Zinc:")
print(ds1[0])
# for data in ds1:
#     print(data.edge_attr)

print(ds2[0])
print(ds3[0])
print(ds3.num_tasks, ds3.num_classes)

print("ring enc")
ring_enc = CellularRingEncoding(6, True, True, True, True)
data_enc, id_maps = ring_enc.encode_with_id_maps(ds3[0])

print("before: ", ds3[0])
print("after: ", data_enc)
print(ds3[0].edge_index.dtype, data_enc.edge_index.dtype)
visualize_from_edge_index(ds3[0].edge_index)
visualize_from_edge_index(data_enc.edge_index)