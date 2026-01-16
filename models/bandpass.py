import torch

def build_bandpass_dictionary(lambda_vals, M=8):
    """
    Fixed uniform band-pass dictionary over [0, 2]
    """
    device = lambda_vals.device
    Delta = 2.0 / M
    sigma = 0.6 * Delta

    mus = torch.linspace(
        Delta / 2, 2.0 - Delta / 2, M, device=device
    )

    B = []
    for mu in mus:
        B.append(torch.exp(-(lambda_vals - mu) ** 2 / (2 * sigma ** 2)))

    B = torch.stack(B, dim=0)  # [M, K]
    return B
