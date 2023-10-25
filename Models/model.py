import torch
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from torch_geometric.nn import GATv2Conv, GCNConv, GINConv

class GNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", max_type = 0, 
                    node_encoder = lambda x: x, edge_encoder = lambda x: x, type_pooling = False, use_node_encoder = True, num_mlp_layers = 1):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        
        print("New GNN implementation.")
        
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.gnn_type = gnn_type
        self.residual = residual
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        self.JK = JK
        self.max_type = max_type
        self.node_encoder = node_encoder
        self.use_node_encoder = use_node_encoder
        
        self.edge_encoder = edge_encoder
        self.type_pooling = type_pooling
        self.num_mlp_layers = num_mlp_layers
        
        # Main neural nets
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(num_layer):
            if layer == 0 and not self.use_node_encoder:
                in_channels = 6 + max_type
            else:
                in_channels = emb_dim

            if gnn_type == 'gin':
                self.convs.append(GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Linear(64, emb_dim))))

            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_channels = in_channels, out_channels = emb_dim))
            elif gnn_type == 'gat': 
                self.convs.append(GATv2Conv(in_channels = in_channels, out_channels = emb_dim, edge_dim  = 1))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
                    
        # Final linear layer        
        if JK == 'cat':
            initial_in_channels = num_layers * emb_dim
        else:
            initial_in_channels = emb_dim
            
        hidden_size = self.emb_dim * 2
        mlp = ModuleList([])
        for i in range(self.num_mlp_layers):
            in_size = hidden_size if i > 0 else self.emb_dim
            out_size = hidden_size if i < self.num_mlp_layers - 1 else self.num_classes*self.num_tasks
            mlp.append(Linear(in_size, out_size))
            
            if i < self.num_mlp_layers - 1:
                mlp.append(torch.nn.ReLU())
                
        self.mlp = mlp
        
        # Pooling
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
            self.relu = torch.nn.ReLU()
            for _ in range(self.max_type + 1):
                self.lin_per_dim.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
                
        # Jumping Knowledge
        if JK is not None and JK != 'last':
            self.JK = JumpingKnowledge(JK, hidden_channels, num_layers)
        
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        
        # Only works for integer features!!!            
        if x is not None:
            # if self.gnn_type != "gin":
            #     x = x.long()
                
                
            h_list = [self.node_encoder(x)]
        else:
            h_list = [torch.zeros([batched_data.num_nodes, self.emb_dim]).to(edge_index.device)]
        
        
        if edge_attr is not None:
            edge_attr = edge_attr.long()
            edge_attr = self.edge_encoder(edge_attr)
        
        for layer in range(self.num_layer):
            h = self.convs[layer](x = h_list[layer], edge_index = edge_index)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)
            
            
        h = self.jk(h_list) if self.JK != "last" else h_list[-1]
        
        if self.type_pooling:
            dimensional_pooling = []
            # print(batched_data)
            # print(batched_data.node_type)

            for dim in range(self.max_type):
                relevant_nodes = batched_data.node_type[:, dim] == 1
                single_dim = h[relevant_nodes]
                batch = batched_data.batch[relevant_nodes]
                single_dim = self.pool(single_dim, batch, size = batched_data.y.shape[0])
                single_dim = self.relu(self.lin_per_dim[dim](single_dim))
                                                            
                dimensional_pooling.append(single_dim)
                
            h_graph = sum(dimensional_pooling)
        else:
            h_graph = self.pool(h, batched_data.batch)
        
        x = h_graph
        for module in self.mlp:
            x = module(x)
            
        if self.num_tasks == 1:
            x = x.view(-1, self.num_classes)
        else:
            x.view(-1, self.num_tasks, self.num_classes)
        return x
        
        
    # def forward(self, batched_data):
    #     x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        
    #     # Only works for integer features!!!
    #     x = x.long()
    #     edge_attr = edge_attr.long()