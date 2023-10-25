import time

import torch
import os.path as osp

from data.utils import convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset
from torch_geometric.datasets import ZINC

from cwn.data.datasets.zinc import ZincDataset as cwn_zinc

class ZINCCWN(cwn_zinc):
    def __init__(self, root, max_ring_size, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, subset=True, n_jobs=2):
        self.prep_time = 0
        super().__init__(root, max_ring_size, use_edge_features, transform, pre_transform, pre_filter, subset, n_jobs)

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing cell complex dataset for {self.name}")
        train_data = ZINC(self.raw_dir, subset=self._subset, split='train')
        val_data = ZINC(self.raw_dir, subset=self._subset, split='val')
        test_data = ZINC(self.raw_dir, subset=self._subset, split='test')

        data_list = []
        idx = []
        start = 0
        print("Converting the train dataset to a cell complex...")
        time_start = time.time()
        train_complexes, _, _ = convert_graph_dataset_with_rings(
            train_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        data_list += train_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print("Converting the validation dataset to a cell complex...")
        val_complexes, _, _ = convert_graph_dataset_with_rings(
            val_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        data_list += val_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print("Converting the test dataset to a cell complex...")
        test_complexes, _, _ = convert_graph_dataset_with_rings(
            test_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        data_list += test_complexes
        self.prep_time = time.time()-time_start
        idx.append(list(range(start, len(data_list))))

        path = self.processed_paths[0]
        print(f'Saving processed dataset in {path}....')
        torch.save(self.collate(data_list, 2), path)
        
        path = self.processed_paths[1]
        print(f'Saving idx in {path}....')
        torch.save(idx, path)