import torch
import torch.nn.functional as F

from models.spectral_filter import spectral_filter
from models.bandpass import SlidingBandpassDictionary
from utils.laplacian import compute_laplacian_eig


def _ensure_batch(data, device: str):
    """
    Ensure data.batch exists (single-graph fallback).
    If you later switch to PyG DataLoader, it will provide batch automatically.
    """
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    return data


def _get_base_node_features(data, device: str) -> torch.Tensor:
    """
    Base node features:
      - prefer original data.x if present
      - else fallback to degree feature [N,1]
    """
    if getattr(data, "x", None) is None:
        row = data.edge_index[0]
        deg = torch.bincount(row, minlength=data.num_nodes).float().to(device)
        x = deg.view(-1, 1)
    else:
        x = data.x.to(device).float()
    return x


def get_node_features(
    data,
    alpha,
    bandpass: SlidingBandpassDictionary,
    device: str,
    spectral_k: int = 64,
    use_spectral: bool = True,
):
    """
    Build node features (optionally with sliding band-pass spectral filtering).

    Args:
        data: PyG Data
        alpha: ClientAlpha-like module (returns alpha weights via alpha())
        bandpass: SlidingBandpassDictionary instance (learnable u/v)
        device: "cuda" / "cpu"
        spectral_k: number of eigenpairs used
        use_spectral: whether to apply spectral filtering

    Returns:
        x_out: Tensor [N, d]
    """
    x = _get_base_node_features(data, device)

    if not use_spectral:
        return x

    # Laplacian eigendecomposition (K eigenpairs)
    lambda_vals, U = compute_laplacian_eig(
        data.edge_index, data.num_nodes, k=spectral_k
    )

    # Sliding dictionary: B is learnable via (u, v)
    B, mu, sigma = bandpass(lambda_vals)  # B: [M, K]

    # Spectral filtering: g_lambda = sum_m alpha_m * B_m(lambda)
    x_f = spectral_filter(x, U, lambda_vals, B, alpha())
    return x_f


@torch.no_grad()
def evaluate(
    graphs,
    model,
    alpha,
    bandpass: SlidingBandpassDictionary,
    device: str,
    spectral_k: int = 64,
    use_spectral: bool = True,
):
    model.eval()
    if use_spectral:
        alpha.eval()
        bandpass.eval()

    correct, total = 0, 0

    for data in graphs:
        data = data.to(device)
        data = _ensure_batch(data, device)

        x = get_node_features(
            data,
            alpha=alpha,
            bandpass=bandpass,
            device=device,
            spectral_k=spectral_k,
            use_spectral=use_spectral,
        )

        logits = model(x, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)

        correct += int((pred == data.y).sum())
        total += data.y.size(0)

    return correct / max(total, 1)


def train_one_epoch(
    graphs,
    model,
    alpha,
    bandpass: SlidingBandpassDictionary,
    optimizer,
    device: str,
    spectral_k: int = 64,
    use_spectral: bool = True,
    grad_clip: float = None,
):
    model.train()
    if use_spectral:
        alpha.train()
        bandpass.train()

    total_loss = 0.0

    for data in graphs:
        data = data.to(device)
        data = _ensure_batch(data, device)

        optimizer.zero_grad(set_to_none=True)

        x = get_node_features(
            data,
            alpha=alpha,
            bandpass=bandpass,
            device=device,
            spectral_k=spectral_k,
            use_spectral=use_spectral,
        )

        logits = model(x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + (list(alpha.parameters()) if use_spectral else [])
                + (list(bandpass.parameters()) if use_spectral else []),
                max_norm=grad_clip,
            )

        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(graphs), 1)
