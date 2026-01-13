import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.lorentz.math as g


class HMHN(nn.Module):
    def __init__(
        self,
        query_dim: int,
        stored_dim: int,
        hopfield_dim: int,
        out_dim: int,
        beta: float = None,
        init_curvature: float = 1.0,
        # learnable_curvature: bool = False,
    ):
        super().__init__()

        self.d_k = hopfield_dim                                                     # manifold dimension for queries/keys
        self.d_out = out_dim                                                        # manifold dimension for values

        self.beta = beta if beta is not None else 1.0 / (hopfield_dim ** 0.5)
        self.k = init_curvature

        # curvature parameter
        # init_k = torch.tensor(init_curvature)
        # self.k_logit = nn.Parameter(
        #     torch.log(torch.exp(init_k) - 1.0),
        #     requires_grad=learnable_curvature,
        # )

        self.W_Q = nn.Linear(query_dim, hopfield_dim, bias=False)                   # (d_r, d_k)
        self.W_K = nn.Linear(stored_dim, hopfield_dim, bias=False)                  # (d_y, d_k)
        self.W_V = nn.Linear(hopfield_dim, out_dim, bias=False)                     # (d_k, d_out)


    # @property
    # def k(self):
    #     return F.softplus(self.k_logit)

    def _origin(self, batch_shape, dim, device, dtype):
        """
        Returns the hyperboloid origin for a batch of points:
        [sqrt(k), 0, ..., 0] in H^d_out
        """
        o = torch.zeros(*batch_shape, dim + 1, device=device, dtype=dtype)          # (S, d_out + 1)
        o[..., 0] = torch.sqrt(self.k)
        return o

    def _to_tangent0(self, x):
        """
        Embed Euclidean vectors into the tangent space at the origin:
        """
        zeros = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)    # (S/M, 1)
        return torch.cat([zeros, x], dim=-1)                                        # (S/M, d_h/d_out + 1)

    def _expmap0(self, x):
        """
        Maps Euclidean vectors to the hyperboloid via Exp_0.
        """
        return g.expmap0(self._to_tangent0(x), k=self.k)

    def _karcher_flow(self, weights, values, steps):
        S, M = weights.shape

        # Initialize at the origin of H^{d_out}_k
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

    def forward(self, R, Y, karcher_steps: int = 4):
        Q = self._expmap0(self.W_Q(R))                                              # (S, d_h + 1)
        K = self._expmap0(self.W_K(Y))                                              # (M, d_h + 1)
        V = self._expmap0(self.W_V(self.W_K(Y)))                                    # (M, d_out + 1)

        dists = g.dist(Q.unsqueeze(1), K.unsqueeze(0), k=self.k)                    # (S, M)
        alpha = F.softmax(-self.beta * dists, dim=-1)

        Z_man = self._karcher_flow(alpha, V, steps=karcher_steps)                   # (S, d_out + 1)
        return g.logmap0(Z_man, k=self.k)[..., 1:]                                  # (S, d_out)
