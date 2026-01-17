import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.lorentz.math as g


class KFPooling(nn.Module):
    def __init__(
        self,
        state_dim: int,
        memory_dim: int,
        hopfield_dim: int,
        out_dim: int,
        beta: float = None,
        init_curvature: float = 1.0,
    ):
        super().__init__()

        self.d_h = hopfield_dim                                                     # manifold dimension for queries/keys
        self.d_out = out_dim                                                        # manifold dimension for values

        self.beta = beta if beta is not None else 1.0 / (hopfield_dim ** 0.5)
        self.register_buffer("k", torch.tensor(init_curvature))

        self.W_K = nn.Linear(memory_dim, hopfield_dim, bias=False)                  # (d_y, d_h)
        self.W_V = nn.Linear(hopfield_dim, out_dim, bias=False)                     # (d_h, d_out)

    def _to_tangent0(self, x):
        """
        Embed Euclidean vectors into the tangent space at the origin:
        """
        zeros = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)    # (S/M, 1)
        x = torch.cat([zeros, x], dim=-1)                                           # (S/M, d_h/d_out + 1)

        # added clipping
        norm_x = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        factor = torch.clamp(3.5 / (norm_x + 1e-7), max=1.0)
        return x * factor

    def _expmap0(self, x):
        """
        Maps Euclidean vectors to the hyperboloid via Exp_0.
        """
        return g.expmap0(self._to_tangent0(x), k=self.k)

    def _karcher_flow(self, weights, queries, values):
        log_values = g.logmap(queries.unsqueeze(1), values.unsqueeze(0), k=self.k)  # (S, M, d_out + 1)
        tangent_mean = torch.sum(weights.unsqueeze(-1) * log_values, dim=1)         # (S, d_out + 1)
        z = g.expmap(queries, tangent_mean, k=self.k)

        return z

    def forward(self, queries, keys, values):
        Q = self._expmap0(queries)                                                  # (S, d_h + 1)
        K = self._expmap0(self.W_K(keys))                                           # (M, d_h + 1)
        V = self._expmap0(self.W_V(self.W_K(values)))                               # (M, d_out + 1)

        similarities = g.inner(Q.unsqueeze(1), K.unsqueeze(0))                      # (S, M)
        alpha = F.softmax(- self.beta * similarities, dim=-1)

        Z_man = self._karcher_flow(alpha, Q, V)                                     # (S, d_out + 1)
        return g.logmap0(Z_man, k=self.k)[..., 1:]                                  # (S, d_out)