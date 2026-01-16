import torch
from models.spectral_filter import spectral_filter
from models.bandpass import build_bandpass_dictionary
from utils.laplacian import compute_laplacian_eig

@torch.no_grad()
def evaluate(graphs, model, alpha, device, spectral_k=64):
    model.eval()
    alpha.eval()

    correct = 0
    total = 0

    for data in graphs:
        data = data.to(device)

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(
                data.num_nodes, dtype=torch.long, device=device
            )

        lambda_vals, U = compute_laplacian_eig(
            data.edge_index, data.num_nodes, k=spectral_k
        )

        B = build_bandpass_dictionary(
            lambda_vals, M=alpha.logits.numel()
        )

        # before spectral_filter  统一使用 degree / constant feature
        num_nodes = data.num_nodes
        data.x = torch.ones(num_nodes, 1, device=data.edge_index.device)

        Xf = spectral_filter(data.x, U, lambda_vals, B, alpha())
        logits = model(Xf, data.edge_index, data.batch)

        pred = logits.argmax(dim=-1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)

    return correct / total


def train_one_epoch(graphs, model, alpha, optimizer, device, spectral_k=64):
    model.train()
    alpha.train()

    total_loss = 0.0

    for data in graphs:
        data = data.to(device)

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(
                data.num_nodes, dtype=torch.long, device=device
            )

        optimizer.zero_grad()

        lambda_vals, U = compute_laplacian_eig(
            data.edge_index, data.num_nodes, k=spectral_k
        )

        B = build_bandpass_dictionary(
            lambda_vals, M=alpha.logits.numel()
        )
        # before spectral_filter  统一使用 degree / constant feature
        num_nodes = data.num_nodes
        data.x = torch.ones(num_nodes, 1, device=data.edge_index.device)

        Xf = spectral_filter(data.x, U, lambda_vals, B, alpha())
        logits = model(Xf, data.edge_index, data.batch)

        loss = torch.nn.functional.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(graphs)


