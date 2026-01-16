import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.lorentz.math as g

class KFLayer(nn.Module):
    def __init__(
        self,
        num_memories: int,
        hopfield_dim: int,
        out_dim: int,
        beta: float = None,
        init_curvature: float = 1.0,
    ):
        super().__init__()

        self.d_k = hopfield_dim                                                     # manifold dimension for queries/keys
        self.d_out = out_dim                                                        # manifold dimension for values

        self.beta = beta if beta is not None else 1.0 / (hopfield_dim ** 0.5)
        self.k = torch.tensor(init_curvature)

        self.W_K = nn.Parameter(torch.randn(num_memories, hopfield_dim) * 0.02)     # (M, d_k)
        self.W_V = nn.Parameter(torch.randn(num_memories, out_dim) * 0.02)          # (M, d_out)

    def _origin(self, batch_shape, dim, device, dtype):
        o = torch.zeros(*batch_shape, dim + 1, device=device, dtype=dtype)          # (S, d_out + 1)
        o[..., 0] = torch.sqrt(self.k)
        return o

    def _to_tangent0(self, x):
        zeros = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)    # (S/M, 1)
        x = torch.cat([zeros, x], dim=-1)                                           # (S/M, d_h/d_out + 1)

        # added clipping
        norm_x = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        factor = torch.clamp(3.0 / (norm_x + 1e-7), max=1.0)
        return x * factor

    def _expmap0(self, x):
        return g.expmap0(self._to_tangent0(x), k=self.k)

    def _karcher_flow(self, weights, values, steps):
        S, M = weights.shape

        z = self._origin(
            batch_shape=(S,),
            dim=self.d_out,
            device=values.device,
            dtype=values.dtype,
        )

        for _ in range(steps):
            log_values = g.logmap(z.unsqueeze(1), values.unsqueeze(0), k=self.k)    # (S, M, d_out + 1)
            euc_mean = torch.sum(weights.unsqueeze(-1) * log_values, dim=1)         # (S, d_out + 1)
            z = g.expmap(z, euc_mean, k=self.k)

        return z

    def forward(self, R, karcher_steps: int = 1):
        R_man = self._expmap0(R)                                                    # (S, d_k + 1)

        W_K_man = self._expmap0(self.W_K)                                           # (M, d_k + 1)
        W_V_man = self._expmap0(self.W_V)                                           # (M, d_out + 1)

        similarities = g.inner(R_man.unsqueeze(1), W_K_man.unsqueeze(0))            # (S, M)
        alpha = F.softmax(- self.beta * similarities, dim=-1)

        Z_man = self._karcher_flow(alpha, W_V_man, steps=karcher_steps)             # (S, d_out + 1)
        return g.logmap0(Z_man, k=self.k)[..., 1:]                                  # (S, d_out)

