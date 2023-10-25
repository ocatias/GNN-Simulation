
import torch
import graph_tool as gt
import graph_tool.topology as top
import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

def get_rings(edge_index, max_k=7):
    """
    Adapted from: https://github.com/twitter-research/cwn/
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)

    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles

    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = list(map(lambda isomorphism: tuple(isomorphism.a), sub_isos))
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings

class CellularRingEncoding(BaseTransform):
    r"""Applies Ring encoding to a graph
    Vertex features will have the shape: 
        xxx - Pattern that signales what kind of vertex it is (explicit_pattern_enc)
        ... - Vertex Features
        ... - Edge features (edge_attr_in_vertices)

    Args:
        max_ring_size: The size of the biggest ring that should be encoded
        aggr_edge_atr: True if vertices corresponding should contain the average edge features from the edges in the original graph
        aggr_vertex_feat: True if new vertices should have the mean vertex features of the lower dimensional vertices
        explicit_pattern_enc: True if the pattern should be explicitly encoded in the vertex features, 
            i.e. (1,0,0, ...) signales a vertex, (0,1,0, ...) an edge, and (0,0,1, ...) a ring
        edge_attr_in_vertices: Add the mean of lower dimensional edge attributes at the end of the vertex features.
            
    """
    def __init__(self, max_ring_size: int, aggr_edge_atr: bool = False, 
        aggr_vertex_feat: bool = False, explicit_pattern_enc: bool = False, edge_attr_in_vertices: bool = False):
        self.max_ring_size = max_ring_size
        self.aggr_edge_atr = aggr_edge_atr
        self.aggr_vertex_feat = aggr_vertex_feat
        self.explicit_pattern_enc = explicit_pattern_enc
        self.edge_attr_in_vertices = edge_attr_in_vertices

    def encode_with_id_maps(self, data: Data):  
        edge_index, x, edge_attr = data.edge_index, data.x, data.edge_attr 
        id_maps = [{},{},{}]
        has_edge_attr = edge_attr is not None 
        has_vertex_feat = x is not None

        if has_vertex_feat:
            nr_vertices = data.x.shape[0]
        else:
            nr_vertices = int(torch.max(edge_index)) + 1
        
        for idx in range(nr_vertices):
            id_maps[0][(idx,)] = idx
        vertex_id = len(id_maps[0])

        if has_vertex_feat:
            len_v_feat = x.shape[1]
        else:
            len_v_feat = 0

        len_e_feat = 0
        if has_edge_attr:
            if len(edge_attr.shape) > 1:
                len_e_feat = edge_attr.shape[1]
            else:
                # Ensure that edge_attr has the shape [num_edges, num_features]
                len_e_feat = 1
                edge_attr = edge_attr.view([-1, 1])

        start_v_feat = 0
        if self.explicit_pattern_enc:
            # Add (1, 0, 0) features to every vertex and ensure that vertex features starts after this patterns
            start_v_feat = 3
            vertex_identifiers = torch.stack([torch.tensor([1,0,0]) for _ in range(len(id_maps[0]))])

            if has_vertex_feat:
                x = torch.cat((vertex_identifiers, x), dim = 1)
            else:
                x = vertex_identifiers

        vertex_dim = torch.stack([torch.tensor([1,0,0]) for _ in range(len(id_maps[0]))])

        start_e_feat = start_v_feat + len_v_feat
        if self.edge_attr_in_vertices and has_edge_attr:
            # If we also encode edge values then add space for them to the features of vertices
            dim_edge_features =  edge_attr.shape[1]
            edge_features = torch.stack([torch.zeros([dim_edge_features]) for _ in range(len(id_maps[0]))])
            if x is not None:
                x = torch.cat((x, edge_features), dim = 1)
            else:
                x = edge_features

        rings = get_rings(edge_index, max_k=self.max_ring_size)

        new_edge_endpoints1, new_edge_endpoints2 = [], []
        edge_attr_of_new_edges = {}

        # Add edges as vertices
        for i in range(edge_index.shape[1]):
            p, q = edge_index[0, i], edge_index[1, i]

            # Only add every edge a single time (undirected edges have two entries in edge_index)
            if p > q:
                continue

            new_v_feature = torch.zeros([len_v_feat])
            if self.aggr_vertex_feat and has_vertex_feat:
                new_v_feature = torch.mode(torch.stack([x[p,start_v_feat:start_e_feat], x[q,start_v_feat:start_e_feat]]), dim = 0)[0]

            if self.explicit_pattern_enc:
                # Add (0, 1, 0) to vertex feature to denote that it comes from an edge
                new_v_feature = torch.cat((torch.tensor([0,1,0]), new_v_feature))

            vertex_dim = torch.cat((vertex_dim, torch.tensor([[0,1,0]])), 0)

            if self.edge_attr_in_vertices and has_edge_attr:
                # Todo check other possible edge
                new_v_feature = torch.cat((new_v_feature, edge_attr[i,:]))

            # Add to features
            new_v_feature = torch.unsqueeze(new_v_feature, 0)
            if x is not None:
                x = torch.cat((x, new_v_feature), 0)

            new_edge_endpoints1 += [p, vertex_id, q, vertex_id]
            new_edge_endpoints2 += [vertex_id, p, vertex_id, q]

            if has_edge_attr:
                # Deal with features of newly created edges
                if self.aggr_edge_atr:
                    # Todo check other possible edge
                    new_e_feat = edge_attr[i,:]
                    edge_attr = torch.cat((edge_attr, torch.stack([new_e_feat, new_e_feat, new_e_feat, new_e_feat])), 0)
                    edge_attr_of_new_edges[(vertex_id,)] = new_e_feat
                else:
                    #  Add empty features to edge_attr so we have an edge_attr for every edge
                    edge_attr = torch.cat((edge_attr, torch.zeros([4, len_e_feat])), 0)
                    edge_attr_of_new_edges[(vertex_id,)] = torch.zeros([len_e_feat])


            id_maps[1][(int(p),int(q))] = vertex_id
            vertex_id += 1
       
        # Add rings as vertices
        newly_created_edges = {}
        for ring in rings:
            ring = list(ring)

            if x is not None:
                new_v_feature = torch.zeros_like(x[0, start_v_feat:start_e_feat])

                if self.aggr_vertex_feat and x is not None:
                    # Aggregate features from vertices in the ring
                    v_features = []
                    for vertex in ring:
                        v_features.append(x[id_maps[0][(vertex,)], start_v_feat:start_e_feat])

                    new_v_feature = torch.mode(torch.stack(v_features), dim = 0)[0]

                if self.explicit_pattern_enc:
                    # Add (0, 0, 1) to vertex feature to denote that it comes from a ring
                    new_v_feature = torch.cat((torch.tensor([0,0,1]), new_v_feature))

                vertex_dim = torch.cat((vertex_dim, torch.tensor([[0,0,1]])), 0)
            
             # Add the first element so we can easily find neighbors by just looking at the next vertex in the list
            ring.append(ring[0])

            new_aggr_edge_features_for_vertex = torch.zeros([len_e_feat])      

            # Add edges from edge vertex to ring vertex
            # also collect edge features for the vertex we just created (self.edge_attr_in_vertices)
            edge_vertices = []
            lower_edge_feat = []
            for idx in range(len(ring) - 1):
                p, q = id_maps[0][(ring[idx],)], id_maps[0][(ring[idx+1],)]

                # Ensure that p < q so we can easily find the id of the vertex corresponding to (p,q)
                if p > q:
                    p, q = q, p

                # ToDo: Check why we need this
                if (p,q) not in id_maps[1]:
                    p, q = q, p

                edge_vertex = id_maps[1][(p,q)]
                new_edge_endpoints1 += [edge_vertex, vertex_id]
                new_edge_endpoints2 += [vertex_id, edge_vertex]
                edge_vertices.append(edge_vertex)


                if not has_edge_attr:
                    continue
                # Deal with edge_attr
                lower_dim_edge_feat = edge_attr_of_new_edges[(id_maps[1][(p,q)],)]

                if self.edge_attr_in_vertices:
                    lower_edge_feat.append(lower_dim_edge_feat)
                    new_aggr_edge_features_for_vertex += lower_dim_edge_feat

                if self.aggr_edge_atr:
                    edge_attr = torch.cat((edge_attr, torch.stack([lower_dim_edge_feat,lower_dim_edge_feat])), 0)
                else:
                    #  Add empty features to edge_attr so we have an edge_attr for every edge
                    edge_attr = torch.cat((edge_attr, torch.zeros([2,len_e_feat])), 0)

            
            # Add to features
            if self.edge_attr_in_vertices and x is not None:
                lower_edge_feat_matrix = torch.mode(torch.stack(lower_edge_feat), dim=0)[0]               
                new_v_feature = torch.cat((new_v_feature, lower_edge_feat_matrix), 0)

            
            if x is not None:
                new_v_feature = torch.unsqueeze(new_v_feature, 0)
                x = torch.cat((x, new_v_feature), 0)

            # Add edges from edge vertex to edge_vertex if they are in the same ring
            for idx1 in range(len(edge_vertices)):
                for idx2 in range(len(edge_vertices)):
                    p, q = edge_vertices[idx1], edge_vertices[idx2]

                    # Ensure that p < q (so we only create edges once and avoid self-loops)
                    if p >= q:
                        continue

                    # Ensure we do not create the same edge twice
                    if (p,q) in newly_created_edges or (q,p) in newly_created_edges:
                        continue
                    else:
                        new_edge_endpoints1 += [p,q]
                        new_edge_endpoints2 += [q,p]
                        newly_created_edges[(p,q)] = True
                        newly_created_edges[(q,p)] = True

                        if not has_edge_attr:
                            continue

                        # Deal with edge_attr
                        if self.aggr_edge_atr:
                            lower_dim_edge_feat = torch.mode(torch.stack([edge_attr_of_new_edges[(p,)],edge_attr_of_new_edges[(q,)]]), dim=0)[0]
                            edge_attr = torch.cat((edge_attr, torch.stack([lower_dim_edge_feat,lower_dim_edge_feat])), 0)
                        else:
                            #  Add empty features to edge_attr so we have an edge_attr for every edge
                            edge_attr = torch.cat((edge_attr, torch.zeros([2, len_e_feat])), 0)

            id_maps[2][tuple(ring)] = vertex_id
            vertex_id += 1

        # Combine everything
        edges_endpoints1 = torch.tensor(new_edge_endpoints1)
        edges_endpoints2 = torch.tensor(new_edge_endpoints2)
        edge_index = torch.cat((edge_index, torch.stack([edges_endpoints1, edges_endpoints2])), 1)
        if x is not None and x.shape[1] >= 0:
            data.x = x
        data.edge_index = edge_index
        if has_edge_attr:
            data.edge_attr = edge_attr

        data.node_type = vertex_dim
        data.num_nodes = x.shape[0]

        # return Data(x=x, edge_index=edge_index, vertex_dim=vertex_dim, y=data.y, edge_attr=data.edge_attr), id_maps    
        return data, id_maps    

    def __call__(self, data: Data):
        data, _ = self.encode_with_id_maps(data)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_ring_size}, {self.aggr_edge_atr}, {self.aggr_vertex_feat}, {self.explicit_pattern_enc}, {self.edge_attr_in_vertices})'

class CellularCliqueEncoding(BaseTransform):
    r"""
    Args:

            
    """
    def __init__(self, max_clique_size: int, aggr_edge_atr: bool = False, 
        aggr_vertex_feat: bool = False, explicit_pattern_enc: bool = False, edge_attr_in_vertices: bool = False):
        self.max_clique_size = max_clique_size
        self.aggr_edge_atr = aggr_edge_atr
        self.aggr_vertex_feat = aggr_vertex_feat
        self.explicit_pattern_enc = explicit_pattern_enc
        self.edge_attr_in_vertices = edge_attr_in_vertices

    def encode_with_id_maps(self, data: Data):  
        pass
        # return data, id_maps

    def __call__(self, data: Data):
        data, _ = self.encode_with_id_maps(data)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'