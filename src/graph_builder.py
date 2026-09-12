import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

def build_segment_graph(node_features, similarity_threshold=0.65):
    num_nodes = len(node_features)
    x = torch.from_numpy(node_features).float()
    edges = []

    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    sim_matrix = cosine_similarity(node_features)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_matrix[i, j] > similarity_threshold:
                edges.append([i, j])
                edges.append([j, i])

    if len(edges) == 0:
        edges = [[i, i] for i in range(num_nodes)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
