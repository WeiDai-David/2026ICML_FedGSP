import torch
from models.spectral_filter import spectral_filter
from models.bandpass import build_bandpass_dictionary
from utils.laplacian import compute_laplacian_eig


def get_node_features(data, alpha, device, spectral_k=64, use_spectral=True):
    # 1. node feature selection 
    if data.x is None:
        # fallback: degree feature (structure-only)
        row, col = data.edge_index
        deg = torch.bincount(row, minlength=data.num_nodes).float().to(device)
        x = deg.view(-1, 1)
    else:
        # use original TU node feature
        x = data.x.to(device).float()

    # 2. no spectral branch
    if not use_spectral:
        return x

    # 3. spectral filtering 
    lambda_vals, U = compute_laplacian_eig(
        data.edge_index, data.num_nodes, k=spectral_k
    )

    B = build_bandpass_dictionary(
        lambda_vals, M=alpha.logits.numel()
    )

    Xf = spectral_filter(x, U, lambda_vals, B, alpha())
    return Xf




@torch.no_grad()
def evaluate(
    graphs,
    model,
    alpha,
    device,
    spectral_k=64,
    use_spectral=True,
):
    model.eval()
    if use_spectral:
        alpha.eval()

    correct, total = 0, 0

    for data in graphs:
        data = data.to(device)

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(
                data.num_nodes, dtype=torch.long, device=device
            )

        x = get_node_features(
            data,
            alpha,
            device=device,
            spectral_k=spectral_k,
            use_spectral=use_spectral,
        )

        logits = model(x, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)

        correct += int((pred == data.y).sum())
        total += data.y.size(0)

    return correct / total


def train_one_epoch(
    graphs,
    model,
    alpha,
    optimizer,
    device,
    spectral_k=64,
    use_spectral=True,
):
    model.train()
    if use_spectral:
        alpha.train()

    total_loss = 0.0

    for data in graphs:
        data = data.to(device)

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(
                data.num_nodes, dtype=torch.long, device=device
            )

        optimizer.zero_grad()

        x = get_node_features(
            data,
            alpha,
            device=device,
            spectral_k=spectral_k,
            use_spectral=use_spectral,
        )

        logits = model(x, data.edge_index, data.batch)
        loss = torch.nn.functional.cross_entropy(logits, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(graphs)

