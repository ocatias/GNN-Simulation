import os
import csv
import json

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from GrapTransformations.cell_encoding import CellularRingEncoding, CellularCliqueEncoding
from GrapTransformations.subgraph_bag_encoding import SubgraphBagEncoding, policy2transform, SubgraphBagEncodingNoSeparator
from GrapTransformations.add_zero_edge_attr import AddZeroEdgeAttr
from GrapTransformations.pad_node_attr import PadNodeAttr

from Models.gnn import GNN
# from Models.model import GNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.ESAN.conv import GINConv, OriginalGINConv, GCNConv, ZINCGINConv 
from Models.ESAN.models import DSnetwork, DSSnetwork
from Models.ESAN.models import GNN as ESAN_GNN
from Models.mlp import MLP

import Misc.config as config 

implemented_TU_datasets = ["MUTAG", "PROTEINS", "PTC", "NCI1", "NCI109"]

def load_tvt_indices(dataset_name, fold, folds = 10):
    train_path = os.path.join(config.SPLITS_PATH, "Train_Val_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_train.json")
    valid_path = os.path.join(config.SPLITS_PATH, "Train_Val_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_valid.json")
    test_path = os.path.join(config.SPLITS_PATH, "Train_Val_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_test.json")

    with open(train_path) as file:
        train_idx = json.load(file)
    with open(valid_path) as file:
        valid_idx = json.load(file)
    with open(test_path) as file:
        test_idx = json.load(file)

    return train_idx, valid_idx, test_idx

def get_transform(args):
    transforms = []
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))
    
    if args.use_cliques:
        transforms.append(CellularCliqueEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_expl_type_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices))
    elif args.use_rings:
        transforms.append(CellularRingEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_expl_type_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices))
    elif not args.use_esan and args.policy != "":
        policy = policy2transform(args.policy, args.num_hops)
        # transform = SubgraphBagEncoding(policy, explicit_type_enc=args.use_expl_type_enc)
        print("Using new SBE")
        transforms.append(SubgraphBagEncodingNoSeparator(policy, 
            explicit_type_enc=args.use_expl_type_enc, dss_message_passing = args.use_dss_message_passing, 
            connect_accross_subg = args.use_additonal_gt_con))
        
    elif args.use_esan and args.policy != "":
        transforms.append(policy2transform(args.policy, args.num_hops))
        
    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.dataset.lower() == "csl":
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))
    
    return Compose(transforms)

def load_dataset(args, config):
    transform = get_transform(args)

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "").replace(" ","")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str )

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name ="CIFAR10", root=dir, split=split, pre_transform=Compose([ToUndirected(), transform])) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        dataset = [GNNBenchmarkDataset(name ="CLUSTER", root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset in implemented_TU_datasets:
        dir = os.path.join(config.DATA_PATH, args.dataset)
        
        import shutil
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        
        dataset = torch_geometric.datasets.TUDataset(root=dir, name=args.dataset, pre_transform=transform, use_node_attr=True, use_edge_attr=False)   
        train_idx, val_idx, test_idx = load_tvt_indices(args.dataset, args.split)
        datasets = [dataset[train_idx], dataset[val_idx], dataset[test_idx]]
    elif args.dataset.lower() == "csl":
        all_idx = {}
        for section in ['train', 'val', 'test']:
            with open(os.path.join(config.SPLITS_PATH, "CSL",  f"{section}.index"), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[all_idx["train"][args.split]], dataset[all_idx["val"][args.split]], dataset[all_idx["test"][args.split]]]
    elif args.dataset.lower() in ["exp", "cexp"]:
        dataset = PlanarSATPairsDataset(name=args.dataset, root=dir, pre_transform=transform)
        split_dict = dataset.separate_data(args.seed, args.split)
        datasets = [split_dict["train"], split_dict["valid"], split_dict["test"]]
    else:
        raise NotImplementedError("Unknown dataset")
        
    if args.use_esan:
        print("Using ESAN")
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, follow_batch=['subgraph_idx'])
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class lin_trafo(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(lin_trafo, self).__init__()
        self.linear =  torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x.float())

class lin_trafo_emb_combination(torch.nn.Module):
    def __init__(self, input_dim, output_dim, node_emb, len_emb_feat):
        super(lin_trafo_emb_combination, self).__init__()
        self.linear =  torch.nn.Linear(input_dim, output_dim)
        self.node_emb = node_emb
        self.len_emb_feat = len_emb_feat
        
    def forward(self, x):
        return self.linear(x[:, self.len_emb_feat:].float()) + self.node_emb(x[:, :self.len_emb_feat])

def get_model(args, num_classes, num_vertex_features, num_tasks):
    if not args.use_esan:
        node_feature_dims = []
        bond_feature_dims = []
        
        model = args.model.lower()

        if args.use_expl_type_enc:
            if args.use_rings:
                node_feature_dims = [2,2,2]              
            if args.use_cliques:
                for _ in range(args.max_struct_size):
                    node_feature_dims.append(2)
            elif args.policy != "":
                node_feature_dims = [2,2]

        if args.dataset.lower() == "zinc":
            node_feature_dims += [21]
            if args.edge_attr_in_vertices:
                node_feature_dims += [4]
            
            node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
            edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4])
        elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
            node_feature_dims += get_atom_feature_dims()
            bond_feature_dims =  get_bond_feature_dims()
            if args.edge_attr_in_vertices:
                node_feature_dims += bond_feature_dims
            node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim, feature_dims=bond_feature_dims)
        elif args.dataset in implemented_TU_datasets:            
            if args.dataset == "PROTEINS":
                node_dim = 4
            elif args.dataset == "MUTAG":
                node_dim = 7
            elif args.dataset == "NCI109":
                node_dim = 38
            elif args.dataset == "NCI1":
                node_dim = 37
            elif args.dataset == "PTC":
                node_dim = 64
            else:
                node_dim = 1
                
            if node_feature_dims == []:
                node_encoder = lin_trafo(node_dim, args.emb_dim)
            else:
                node_encoder = lin_trafo_emb_combination(node_dim, args.emb_dim, NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims), len(node_feature_dims))
                
            edge_encoder = lambda x: torch.zeros([1, args.emb_dim], device = args.device)
        else:
            node_encoder, edge_encoder = lambda x: x, lambda x: x
                
        if model in ["gin", "gcn", "gat"]:
            # Cell Encoding
            if args.use_cliques or args.use_rings:
                return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                        gnn_type =  model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.jk, 
                        graph_pooling = args.pooling, type_pooling = args.use_dim_pooling,
                        max_type = 3 if args.use_rings else args.max_struct_size, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                        use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers)
            # Without Cell Encoding
            else:
                return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                        gnn_type = model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.jk, 
                        graph_pooling = args.pooling, type_pooling = args.use_dim_pooling,
                        max_type = 2 if args.policy != "" else 0, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                        use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers)
        elif args.model.lower() == "mlp":
                return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                        num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling)
        else: # Probably don't need other models
            pass

    # ESAN
    else:
        encoder = lambda x: x
        if 'ogb' in args.dataset:
            encoder = AtomEncoder(args.emb_dim) if args.policy != "ego_nets_plus" else EgoEncoder(AtomEncoder(args.emb_dim))
        elif 'ZINC' in args.dataset:
            encoder = ZincAtomEncoder(policy=args.policy, emb_dim=args.emb_dim)
        if 'ogb' in args.dataset or 'ZINC' in args.dataset:
            in_dim = args.emb_dim if args.policy != "ego_nets_plus" else args.emb_dim + 2
        elif args.dataset in ['CSL', 'EXP', 'CEXP']:
            in_dim = 6 if args.policy != "ego_nets_plus" else 6 + 2  # used deg as node feature
        else:
            in_dim = dataset.num_features

        # DSS
        if args.use_dss_message_passing: 
            if args.model == 'GIN':
                GNNConv = GINConv
            elif args.model == 'originalgin':
                GNNConv = OriginalGINConv
            elif args.model == 'graphconv':
                GNNConv = GraphConv
            elif args.model == 'gcn':
                GNNConv = GCNConv
            elif args.model == 'ZINCGIN':
                GNNConv = ZINCGINConv
            else:
                raise ValueError('Undefined GNN type called {}'.format(args.model))
        
            model = DSSnetwork(num_layers=args.num_layers, in_dim=in_dim, emb_dim=args.emb_dim, num_tasks=num_tasks*num_classes,
                            feature_encoder=encoder, GNNConv=GNNConv)
        # DS
        else:
            subgraph_gnn = ESAN_GNN(gnn_type=args.model.lower(), num_tasks=num_tasks*num_classes, num_layer=args.num_layers, in_dim=in_dim,
                           emb_dim=args.emb_dim, drop_ratio=args.drop_out, graph_pooling='sum' if args.model != 'gin' else 'mean', 
                           feature_encoder=encoder)
            model = DSnetwork(subgraph_gnn=subgraph_gnn, channels=[64, 64], num_tasks=num_tasks*num_classes,
                            invariant=args.dataset == 'ZINC')
        return model


def get_optimizer_scheduler(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() == "zinc":
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["cifar10", "csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in implemented_TU_datasets:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method