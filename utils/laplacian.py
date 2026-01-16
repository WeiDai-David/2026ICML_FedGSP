import torch
from torch_geometric.utils import get_laplacian

def compute_laplacian_eig(edge_index, num_nodes, k=64):
    edge_index, edge_weight = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes
    )

    L = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    L[edge_index[0], edge_index[1]] = edge_weight

    lambda_vals, U = torch.linalg.eigh(L)

    return lambda_vals[:k], U[:, :k]


# 第一版先不做 Chebyshev / approximation
