"""
Generate split index for all dataset for which we use CV.
"""

import os
import json
import numpy as np
from numpy import int32
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch_geometric

import Misc.config as config

datasets = ["MUTAG", "PROTEINS", "NCI1", "NCI109"]
folds = 10
seed = 42

tvt_path = os.path.join(config.SPLITS_PATH, "Train_Val_Test_Splits")

def cv_split(graph_list, seed, fold_idx, nr_folds = 10):
    """
    Split dataset into folds
    Adapted from: https://github.com/weihua916/powerful-gnns/blob/master/util.py
    """
    assert 0 <= fold_idx and fold_idx < nr_folds, "fold_idx must be from 0 to nr_folds."
    skf = StratifiedKFold(n_splits=nr_folds, shuffle = True, random_state = seed)

    idx_list = []
    y = np.concatenate([d.y for d in graph_list]) 

    for idx in skf.split(np.zeros(len(graph_list)), y):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

def stratified_data_split(training_index_y, seed):
    """
    training_index_y must have the shape of (idx, y) where idx is the position in the original dataset
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)

    indices = [d[0] for d in training_index_y]
    y = np.concatenate([d[1] for d in training_index_y]) 
    train_idx_idx, test_idx_idx = list(sss.split(indices, y))[0]
    train_index = [int(indices[idx]) for idx in train_idx_idx]
    test_index = [int(indices[idx]) for idx in test_idx_idx]
    return train_index, test_index

if not os.path.isdir(tvt_path):
    os.mkdir(tvt_path)

dataset = torch_geometric.datasets.TUDataset(root=config.DATA_PATH, name="MUTAG")

# for dataset_name in datasets:
#     dataset = torch_geometric.datasets.TUDataset(root=config.DATA_PATH, name=dataset_name)
    
#     print(dataset[0])
#     for fold in range(folds):
#         train_index, test_index = cv_split(dataset, seed, fold, folds)

#         with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_train.json"), "w") as file:
#             json.dump(list(train_index.tolist()), file)
#         with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_valid.json"), "w") as file:
#             json.dump(list(test_index.tolist()), file)
#         with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_test.json"), "w") as file:
#             json.dump(list(test_index.tolist()), file)