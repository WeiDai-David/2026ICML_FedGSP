import torch

def spectral_filter(X, U, lambda_vals, B, alpha):
    """
    X: [N, d]
    U: [N, K]
    lambda_vals: [K]
    B: [M, K]
    alpha: [M]
    """
    g_lambda = torch.sum(alpha[:, None] * B, dim=0)  # [K]

    X_hat = U.T @ X              # [K, d]
    X_hat = g_lambda[:, None] * X_hat
    X_out = U @ X_hat            # [N, d]
    return X_out
