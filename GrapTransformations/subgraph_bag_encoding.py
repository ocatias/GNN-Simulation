from typing import Optional, Union, Tuple
from collections import defaultdict

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import coalesce

class SubgraphBagEncodingNoSeparator(BaseTransform):
    """
    Transforms a policy to a Subgraph Bag Encoding transformation:
    Instead of creating subgraphs from a graph, this will instead transform a graph into a graph that encodings the subgraphs
    """

    def __init__(self, policy, explicit_type_enc: bool=True, edge_attr_in_vertices: bool= False, dss_message_passing: bool = True, connect_accross_subg: bool = False):
        """dss_message_passing:    
                False: only add edges corresponding to DS-WL
                True: additional edges will be added corresponding to DSS-WL
            connect_accross_subg:
                Set to true such that all instances of a vertex v will be connected across subgraphs i.e. every instance of v is a neighbor of every other instance
            """
        self.policy = policy
        self.explicit_type_enc = explicit_type_enc 
        self.edge_attr_in_vertices = edge_attr_in_vertices
        self.dss_message_passing = dss_message_passing
        self.connect_accross_subg = connect_accross_subg
        
    def __call__(self, data: Data):
        # print(data)
        data_subgraph = self.policy(data)

        has_vertex_feat =  data.x is not None
        has_edge_attr = data.edge_attr is not None
        edge_attr = data.edge_attr
        subgraph_edge_attr = data_subgraph.edge_attr

        if has_edge_attr and len(edge_attr.shape) == 1:
            edge_attr = edge_attr.view([-1,1])
        if has_edge_attr and len(subgraph_edge_attr.shape) == 1:
            subgraph_edge_attr = subgraph_edge_attr.view([-1,1])

        # Get the number of vertices in the original graph
        # If we have vertex features then we can have isolated vertices!
        if has_vertex_feat:
            nr_og_vertices = data.x.shape[0]
            nr_subgraph_vertices = data_subgraph.x.shape[0]
        else:
            nr_og_vertices = int(torch.max(data.edge_index)) + 1
            nr_subgraph_vertices = int(torch.max(data_subgraph.edge_index)) + 1

        og_graph_feature = torch.stack([torch.tensor([1,0]) for _ in range(nr_og_vertices)])
        subgraph_feature = torch.stack([torch.tensor([0,1]) for _ in range(nr_subgraph_vertices)])
        
        if self.dss_message_passing:
            type_features = torch.cat([og_graph_feature, subgraph_feature])
        else:
            type_features = subgraph_feature

        if has_vertex_feat:
            # Collect features of separator vertices, original graph, subgraphs 
            if self.dss_message_passing:
                x = torch.cat([data.x, data_subgraph.x])
            else:
                x = data_subgraph.x

            if self.explicit_type_enc:
                x = torch.cat((type_features, x), dim = 1)
            data.x = x
        elif self.explicit_type_enc:
            data.x = type_features
            
        subgraph_node_idx = torch.cat([data_subgraph.subgraph_node_idx])
        
        if self.connect_accross_subg:
            # Collect all variants of vertices accross subgraphs
            map_original_vertex_to_subgraph_vertex = defaultdict(list)
            for subgraph_vertex_id, og_vertex_id  in enumerate(subgraph_node_idx):
                if self.dss_message_passing:
                    map_original_vertex_to_subgraph_vertex[int(og_vertex_id)].append(int(subgraph_vertex_id + nr_og_vertices))
                else:
                    map_original_vertex_to_subgraph_vertex[int(og_vertex_id)].append(int(subgraph_vertex_id))

            new_edges_endpoint1, new_edges_endpoint2 = [], []
            for ls in map_original_vertex_to_subgraph_vertex.values():
                for a in ls:
                    for b in ls:
                        if a != b:
                            new_edges_endpoint1.append(torch.tensor(a))
                            new_edges_endpoint2.append(torch.tensor(b))
            edge_index = torch.cat([torch.stack([torch.tensor(new_edges_endpoint1), torch.tensor(new_edges_endpoint2)])], 
                dim= 1)
            
        else:
            edge_index = torch.empty(2,0)
        
                
        # Add original graph + connections to subgraphs
        if self.dss_message_passing:            
            # We need to shift the index of vertices in the subgraphs to accomodate the original graph
            edge_index = torch.cat([edge_index, data.edge_index, data_subgraph.edge_index + nr_og_vertices], dim=1)

            # Add connection between original graph and subgraphs
            new_edges_endpoint1, new_edges_endpoint2 = [], []
            for subgraph_vertex_id, og_vertex_id  in enumerate(subgraph_node_idx):
                # Only need to shift once by max_og_vertex_id, because we already got one shift by adding torch.arange(max_og_vertex_id) to the top of the list
                subgraph_vertex_id =  torch.tensor(subgraph_vertex_id + nr_og_vertices)
                new_edges_endpoint1 += [subgraph_vertex_id, og_vertex_id]
                new_edges_endpoint2 += [og_vertex_id, subgraph_vertex_id]
                
            edge_index = torch.cat([edge_index, 
                torch.stack([torch.tensor(new_edges_endpoint1), torch.tensor(new_edges_endpoint2)])], 
                dim= 1)
             
            if has_edge_attr:
                edge_seperator_features = torch.zeros([len(new_edges_endpoint1), edge_attr.shape[1]])
                edge_attr = torch.cat([edge_attr, subgraph_edge_attr,edge_seperator_features])
                data.edge_attr = edge_attr
               
        # Only add the subgraphs 
        else:
            edge_index = data_subgraph.edge_index
             
            if has_edge_attr:
                data.edge_attr = torch.cat([subgraph_edge_attr])
            
        if has_vertex_feat:
            nr_vertices = x.shape[0]
        else:
            nr_vertices = int(torch.max(edge_index) + 1)
        data.node_type = type_features

        data.edge_index = edge_index
        data.num_nodes = nr_vertices 
        # print(data)
        # quit()
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.policy)}, {self.explicit_type_enc}, {self.edge_attr_in_vertices}, {self.dss_message_passing}, {self.connect_accross_subg})'

class SubgraphBagEncoding(BaseTransform):
    """
    Transforms a policy to a Subgraph Bag Encoding transformation:
    Instead of creating subgraphs from a graph, this will instead transform a graph into a graph that encodings the subgraphs
    """

    def __init__(self, policy, explicit_type_enc: bool=True, edge_attr_in_vertices: bool= False):
        self.policy = policy
        self.explicit_type_enc = explicit_type_enc 
        self.edge_attr_in_vertices = edge_attr_in_vertices

    def __call__(self, data: Data):
        data_subgraph = self.policy(data)

        has_vertex_feat =  data.x is not None
        has_edge_attr = data.edge_attr is not None
        edge_attr = data.edge_attr
        subgraph_edge_attr = data_subgraph.edge_attr

        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr.view([-1,1])
        if len(subgraph_edge_attr.shape) == 1:
            subgraph_edge_attr = subgraph_edge_attr.view([-1,1])

        # Get the number of vertices in the original graph
        # If we have vertex features then we can have isolated vertices!
        if has_vertex_feat:
            nr_og_vertices = data.x.shape[0]
            nr_subgraph_vertices = data_subgraph.x.shape[0]
        else:
            nr_og_vertices = int(torch.max(data.edge_index)) + 1
            nr_subgraph_vertices = int(torch.max(data_subgraph.edge_index)) + 1

        seperator_feature = torch.stack([torch.tensor([1,0,0]) for _ in range(nr_og_vertices)])
        og_graph_feature = torch.stack([torch.tensor([0,1,0]) for _ in range(nr_og_vertices)])
        subgraph_feature = torch.stack([torch.tensor([0,0,1]) for _ in range(nr_subgraph_vertices)])
        type_features = torch.cat([seperator_feature, og_graph_feature, subgraph_feature])

        if has_vertex_feat:
            # Collect features of separator vertices, original graph, subgraphs 
            x = torch.cat([data.x, data.x, data_subgraph.x])

            if self.explicit_type_enc:
                x = torch.cat((type_features, x), dim = 1)
            data.x = x
        elif self.explicit_type_enc:
            data.x = type_features

        # We need to shift the index of vertices in the subgraphs to accomodate the original graph
        edge_index = torch.cat([data.edge_index + nr_og_vertices, data_subgraph.edge_index + 2*nr_og_vertices], dim=1)
        subgraph_node_idx = torch.cat([torch.arange(nr_og_vertices), data_subgraph.subgraph_node_idx])


        # Add edges to separator vertices
        new_edges_endpoint1, new_edges_endpoint2 = [], []
        for subgraph_vertex_id, separator_id  in enumerate(subgraph_node_idx):
            # Only need to shift once by max_og_vertex_id, because we already got one shift by adding torch.arange(max_og_vertex_id) to the top of the list
            subgraph_vertex_id =  torch.tensor(subgraph_vertex_id + nr_og_vertices)
            new_edges_endpoint1 += [subgraph_vertex_id, separator_id]
            new_edges_endpoint2 += [separator_id, subgraph_vertex_id]

        edge_index = torch.cat([    
                                edge_index, 
                                torch.stack([
                                        torch.tensor(new_edges_endpoint1), 
                                        torch.tensor(new_edges_endpoint2)])], 
                                dim= 1)

        if has_edge_attr:
            edge_seperator_features = torch.zeros([len(new_edges_endpoint1), edge_attr.shape[1]])
            edge_attr = torch.cat([edge_attr, subgraph_edge_attr,edge_seperator_features])
            data.edge_attr = edge_attr

        max_vertex_id = int(torch.max(edge_index) + 1)
        data.node_type = type_features

        data.edge_index = edge_index
        data.num_nodes = max_vertex_id 

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.policy)}, {self.explicit_type_enc}, {self.edge_attr_in_vertices})'


ORIG_EDGE_INDEX_KEY = 'original_edge_index'

class SubgraphData(Data):
    def __inc__(self, key, value, store):
        if key == ORIG_EDGE_INDEX_KEY:
            return self.num_nodes_per_subgraph
        else:
            return super().__inc__(key, value)
        
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "subgraph_idx":
            return 0
        else:
           return super().__cat_dim__(key, value, *args, **kwargs)


# TODO: update Pytorch Geometric since this function is on the newest version
def to_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                  num_nodes: Optional[int] = None,
                  reduce: str = "add") -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features. (default: :obj:`"add"`)
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :class:`Tensor`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, reduce)

    if edge_attr is None:
        return edge_index
    else:
        return edge_index, edge_attr


def preprocess(dataset, transform):
    def unbatch_subgraphs(data):
        subgraphs = []
        num_nodes = data.num_nodes_per_subgraph.item()
        for i in range(data.num_subgraphs):
            edge_index, edge_attr = subgraph(torch.arange(num_nodes) + (i * num_nodes),
                                             data.edge_index, data.edge_attr,
                                             relabel_nodes=False, num_nodes=data.x.size(0))
            subgraphs.append(
                Data(
                    x=data.x[i * num_nodes: (i + 1) * num_nodes, :], edge_index=edge_index - (i * num_nodes),
                    edge_attr=edge_attr,
                    subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(num_nodes),
                    num_nodes=num_nodes,
                )
            )

        original_edge_attr = data.original_edge_attr if data.edge_attr is not None else data.edge_attr
        return Data(x=subgraphs[0].x, edge_index=data.original_edge_index, edge_attr=original_edge_attr, y=data.y,
                    subgraphs=subgraphs)

    data_list = [unbatch_subgraphs(data) for data in dataset]

    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    dataset.transform = transform
    return dataset


class Graph2Subgraph:
    def __init__(self, process_subgraphs=lambda x: x, pbar=None):
        self.process_subgraphs = process_subgraphs
        self.pbar = pbar

    def __call__(self, data):
        assert data.is_undirected()

        subgraphs = self.to_subgraphs(data)
        subgraphs = [self.process_subgraphs(s) for s in subgraphs]

        batch = Batch.from_data_list(subgraphs)

        if self.pbar is not None: next(self.pbar)

        return SubgraphData(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                            subgraph_batch=batch.batch,
                            y=data.y, subgraph_idx=batch.subgraph_idx, subgraph_node_idx=batch.subgraph_node_idx,
                            num_subgraphs=len(subgraphs), num_nodes_per_subgraph=data.num_nodes,
                            original_edge_index=data.edge_index, original_edge_attr=data.edge_attr)

    def to_subgraphs(self, data):
        raise NotImplementedError


class EdgeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        # remove one of the bidirectional index
        if data.edge_attr is not None and len(data.edge_attr.shape) == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = data.edge_attr[keep_edge, :] if data.edge_attr is not None else data.edge_attr

        subgraphs = []

        for i in range(edge_index.size(1)):
            subgraph_edge_index = torch.hstack([edge_index[:, :i], edge_index[:, i + 1:]])
            subgraph_edge_attr = torch.vstack([edge_attr[:i], edge_attr[i + 1:]]) \
                if data.edge_attr is not None else data.edge_attr

            if data.edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                        num_nodes=data.num_nodes)
            else:
                subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                    num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        if len(subgraphs) == 0:
            subgraphs = [
                Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                     subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(data.num_nodes),
                     num_nodes=data.num_nodes,
                     )
            ]
        return subgraphs
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class NodeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        subgraphs = []
        all_nodes = torch.arange(data.num_nodes)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1:]])
            subgraph_edge_index, subgraph_edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                                               relabel_nodes=False, num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class EgoNets(Graph2Subgraph):
    def __init__(self, num_hops, add_node_idx=False, process_subgraphs=lambda x: x, pbar=None):
        super().__init__(process_subgraphs, pbar)
        self.num_hops = num_hops
        self.add_node_idx = add_node_idx

    def to_subgraphs(self, data):

        subgraphs = []

        for i in range(data.num_nodes):

            _, _, _, edge_mask = k_hop_subgraph(i, self.num_hops, data.edge_index, relabel_nodes=False,
                                                num_nodes=data.num_nodes)
            subgraph_edge_index = data.edge_index[:, edge_mask]
            subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr

            x = data.x
            if self.add_node_idx:
                # prepend a feature [0, 1] for all non-central nodes
                # a feature [1, 0] for the central node
                ids = torch.arange(2).repeat(data.num_nodes, 1)
                ids[i] = torch.tensor([ids[i, 1], ids[i, 0]])

                x = torch.hstack([ids, data.x]) if data.x is not None else ids.to(torch.float)

            subgraphs.append(
                Data(
                    x=x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs

    # Todo: maybe add process_subgraphs and pbar
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.num_hops)}, {self.add_node_idx})'

def policy2transform(policy: str, num_hops, process_subgraphs=lambda x: x, pbar=None):
    if policy == "edge_deleted":
        return EdgeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "node_deleted":
        return NodeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets":
        return EgoNets(num_hops, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets_plus":
        return EgoNets(num_hops, add_node_idx=True, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "original":
        return process_subgraphs

    raise ValueError("Invalid subgraph policy type")