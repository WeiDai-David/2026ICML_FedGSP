import torch
import torch.nn as nn
import torch.nn.functional as F

def build_fixedbandpass_dictionary(lambda_vals, M=8):
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

class SlidingBandpassDictionary(nn.Module):
    """
    Sliding Gaussian band-pass dictionary with:
      (1) ordered centers + full coverage
      (2) bounded bandwidth
      (3) diversity-ready structure (for Gram regularization)
    """

    def __init__(
        self,
        M: int,
        lambda_max: float = 2.0,
        sigma_min_ratio: float = 0.15,
        sigma_max_ratio: float = 1.2,
    ):
        """
        Args:
            M: number of band-pass atoms
            lambda_max: max eigenvalue (2.0 for normalized Laplacian)
            sigma_min_ratio / sigma_max_ratio:
                relative to (lambda_max / M)
        """
        super().__init__()
        self.M = M
        self.lambda_max = lambda_max

        # -------- unconstrained learnable parameters --------
        self.u = nn.Parameter(torch.zeros(M))  # for ordered centers
        self.v = nn.Parameter(torch.zeros(M))  # for bandwidths

        # -------- bandwidth bounds --------
        base = lambda_max / M
        self.register_buffer(
            "sigma_min",
            torch.tensor(sigma_min_ratio * base),
        )
        self.register_buffer(
            "sigma_max",
            torch.tensor(sigma_max_ratio * base),
        )

    def forward(self, lambda_vals: torch.Tensor):
        """
        Args:
            lambda_vals: Tensor [K], eigenvalues in [0, 2]

        Returns:
            B:     Tensor [M, K], band-pass dictionary
            mu:    Tensor [M],   band centers
            sigma: Tensor [M],   band widths
        """
        # 约束 1：中心有序 + 覆盖 [0, 2]
        delta = F.softplus(self.u)              # Δ_m > 0
        cum_delta = torch.cumsum(delta, dim=0)
        mu = self.lambda_max * cum_delta / cum_delta[-1]
        # mu: (0, lambda_max), strictly increasing

        # 约束 2：带宽上下界
        sigma = F.softplus(self.v)
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)

        # 构造 Gaussian band-pass
        lambda_vals = lambda_vals.unsqueeze(0)  # [1, K]
        mu = mu.unsqueeze(1)                    # [M, 1]
        sigma = sigma.unsqueeze(1)              # [M, 1]

        B = torch.exp(
            - (lambda_vals - mu) ** 2 / (2.0 * sigma ** 2)
        )  # [M, K]

        return B, mu.squeeze(), sigma.squeeze()
