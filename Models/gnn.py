import torch
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, JumpingKnowledge
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from Models.conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", max_type = 0, 
                    node_encoder = lambda x: x, edge_encoder = lambda x: x, type_pooling = False, use_node_encoder = True, num_mlp_layers = 1):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        
        print("Old GNN implementation.")

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.type_pooling  = type_pooling
        self.max_type = max_type
        self.use_node_encoder = use_node_encoder
        self.num_mlp_layers = num_mlp_layers
        
        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)

        if JK == 'cat':
            initial_in_channels = num_layer * emb_dim
        else:
            initial_in_channels = emb_dim

        ### Pooling function to generate whole-graph embeddings
        print(f"graph_pooling: {graph_pooling}")
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        if self.type_pooling:
            self.lin_per_dim = torch.nn.ModuleList()
            #self.pool = global_add_pool
            self.relu = torch.nn.ReLU()
            for _ in range(self.max_type + 1):
                self.lin_per_dim.append(torch.nn.Linear(initial_in_channels, initial_in_channels))

        # Final linear layer        
        hidden_size = self.emb_dim * 2
        mlp = torch.nn.ModuleList([])
        for i in range(self.num_mlp_layers):
            in_size = hidden_size if i > 0 else initial_in_channels
            out_size = hidden_size if i < self.num_mlp_layers - 1 else self.num_classes*self.num_tasks
            mlp.append(torch.nn.Linear(in_size, out_size))
            
            if i < self.num_mlp_layers - 1:
                mlp.append(torch.nn.ReLU())
                
        self.mlp = mlp
        
        # Jumping Knowledge
        if JK is not None and JK != 'last':
            self.JK = JumpingKnowledge(JK, emb_dim, num_layer)
        else:
            self.JK = lambda ls: ls[-1]
            
    def forward(self, batched_data):
        h_list = self.gnn_node(batched_data)

        h_node = self.JK(h_list)
        if self.type_pooling:
            dimensional_pooling = []
            for dim in range(self.max_type):
                multiplier = torch.unsqueeze(batched_data.node_type[:, dim], dim=1)
                single_dim = h_node * multiplier
                single_dim = self.pool(single_dim, batched_data.batch)
                single_dim = self.relu(self.lin_per_dim[dim](single_dim))
                dimensional_pooling.append(single_dim)
            h_graph = sum(dimensional_pooling)
            
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        x = h_graph
        for module in self.mlp:
            x = module(x)
        
        if self.num_tasks == 1:
            x = x.view(-1, self.num_classes)
        else:
            x.view(-1, self.num_tasks, self.num_classes)
        return x


if __name__ == '__main__':
    GNN(num_tasks = 10)