"""LoopFormer-style recurrent Transformer components for TabPFN.

The shared core is conditioned on normalized trajectory time ``t`` and the
current step size ``dt``. A trajectory always starts at zero and its positive
step sizes must sum to one. Consequently, a short trajectory is a learned
coarse route to the same endpoint as the maximum-depth trajectory, rather than
an unconditioned early exit.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from tabpfn.transformer import TransformerModel


def sample_discrete_trajectory(
    max_loops: int,
    num_steps: int,
) -> Tuple[float, ...]:
    """Sample a uniform positive composition of max_loops into steps.

    Selecting the cut points uniformly samples one of the discrete trajectories
    whose increments lie on the 1 / max_loops grid and sum to one.
    """
    if max_loops < 2:
        raise ValueError("max_loops must be at least two.")
    if not 1 <= num_steps < max_loops:
        raise ValueError("num_steps must be in [1, max_loops - 1].")

    if num_steps == 1:
        increments = torch.tensor([max_loops], dtype=torch.long)
    else:
        cuts = torch.randperm(max_loops - 1)[: num_steps - 1] + 1
        cuts = torch.sort(cuts).values
        endpoints = torch.cat(
            (torch.zeros(1, dtype=torch.long), cuts, torch.tensor([max_loops]))
        )
        increments = endpoints[1:] - endpoints[:-1]
    return tuple(increment / max_loops for increment in increments.tolist())


class ScalarFourierEmbedder(nn.Module):
    """Embed a scalar using fixed Fourier features followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        if frequency_embedding_size <= 0 or frequency_embedding_size % 2:
            raise ValueError("frequency_embedding_size must be a positive even integer.")
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def fourier_features(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        arguments = t[:, None].float() * frequencies[None]
        return torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)

    def forward(self, t: Tensor) -> Tensor:
        features = self.fourier_features(t, self.frequency_embedding_size)
        return self.mlp(features.to(dtype=self.mlp[0].weight.dtype))


class LoopFormerEncoderLayer(nn.Module):
    """A TabPFN attention layer with LoopFormer RMSNorm modulation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        recompute_attn: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.recompute_attn = recompute_attn

        # The four outputs are attention/MLP residual gates and RMSNorm scales.
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model),
        )
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def _attention(
        self,
        src: Tensor,
        src_mask,
        src_key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        if isinstance(src_mask, tuple):
            if src_key_padding_mask is not None:
                raise ValueError("A tuple attention mask cannot use src_key_padding_mask.")
            global_mask, train_mask, eval_mask = src_mask
            num_global = global_mask.shape[0]
            num_train = train_mask.shape[0]
            global_src = src[:num_global]
            train_src = src[num_global : num_global + num_train]
            global_and_train_src = src[: num_global + num_train]
            eval_src = src[num_global + num_train :]
            if self.recompute_attn:
                global_out = checkpoint(
                    self.self_attn,
                    global_src,
                    global_and_train_src,
                    global_and_train_src,
                    None,
                    True,
                    global_mask,
                    use_reentrant=False,
                )[0]
                train_out = checkpoint(
                    self.self_attn,
                    train_src,
                    global_src,
                    global_src,
                    None,
                    True,
                    train_mask,
                    use_reentrant=False,
                )[0]
                eval_out = checkpoint(
                    self.self_attn,
                    eval_src,
                    src,
                    src,
                    None,
                    True,
                    eval_mask,
                    use_reentrant=False,
                )[0]
            else:
                global_out = self.self_attn(
                    global_src,
                    global_and_train_src,
                    global_and_train_src,
                    attn_mask=global_mask,
                )[0]
                train_out = self.self_attn(
                    train_src, global_src, global_src, attn_mask=train_mask
                )[0]
                eval_out = self.self_attn(
                    eval_src, src, src, attn_mask=eval_mask
                )[0]
            return torch.cat((global_out, train_out, eval_out), dim=0)

        if isinstance(src_mask, int):
            if src_key_padding_mask is not None:
                raise ValueError("Integer attention masks cannot use src_key_padding_mask.")
            train_src = src[:src_mask]
            eval_src = src[src_mask:]
            train_out = self.self_attn(train_src, train_src, train_src)[0]
            eval_out = self.self_attn(eval_src, train_src, train_src)[0]
            return torch.cat((train_out, eval_out), dim=0)

        if self.recompute_attn:
            return checkpoint(
                self.self_attn,
                src,
                src,
                src,
                src_key_padding_mask,
                True,
                src_mask,
                use_reentrant=False,
            )[0]
        return self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]

    def forward(
        self,
        src: Tensor,
        conditioning: Tensor,
        src_mask=None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        gate_attn, gate_mlp, scale_attn, scale_mlp = self.modulation(
            conditioning
        ).chunk(4, dim=-1)
        # TabPFN tensors are sequence-first, so conditioning broadcasts on dim 0.
        gate_attn = gate_attn.unsqueeze(0)
        gate_mlp = gate_mlp.unsqueeze(0)
        scale_attn = scale_attn.unsqueeze(0)
        scale_mlp = scale_mlp.unsqueeze(0)

        attn_input = self.norm1(src) * (1.0 + scale_attn)
        src = src + gate_attn * self.dropout1(
            self._attention(attn_input, src_mask, src_key_padding_mask)
        )
        mlp_input = self.norm2(src) * (1.0 + scale_mlp)
        mlp_output = self.linear2(self.dropout(self.activation(self.linear1(mlp_input))))
        return src + gate_mlp * self.dropout2(mlp_output)


class LoopFormerEncoder(nn.Module):
    """Repeat a conditioned physical block stack along a unit-time trajectory."""

    def __init__(
        self,
        layers: Sequence[LoopFormerEncoderLayer],
        d_model: int,
        max_loops: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        if max_loops < 1:
            raise ValueError("max_loops must be positive.")
        self.layers = nn.ModuleList(layers)
        self.num_layers = max_loops * len(self.layers)
        self.max_loops = max_loops
        self.time_embedder = ScalarFourierEmbedder(
            d_model, frequency_embedding_size=frequency_embedding_size
        )
        self.step_embedder = ScalarFourierEmbedder(
            d_model, frequency_embedding_size=frequency_embedding_size
        )
        self.norm = nn.RMSNorm(d_model)

    def _trajectory(
        self,
        trajectory: Optional[Sequence[float]],
        num_loops: Optional[int],
    ) -> Tuple[float, ...]:
        if trajectory is not None and num_loops is not None:
            raise ValueError("Provide trajectory or num_loops, not both.")
        if trajectory is None:
            loops = self.max_loops if num_loops is None else int(num_loops)
            if not 1 <= loops <= self.max_loops:
                raise ValueError(f"num_loops must be in [1, {self.max_loops}].")
            return (1.0 / loops,) * loops

        steps = tuple(float(step) for step in trajectory)
        if not 1 <= len(steps) <= self.max_loops:
            raise ValueError(
                f"trajectory length must be in [1, {self.max_loops}]."
            )
        if any(step <= 0.0 for step in steps):
            raise ValueError("All trajectory steps must be positive.")
        if not math.isclose(sum(steps), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Trajectory steps must sum to one.")
        return steps

    def forward(
        self,
        src: Tensor,
        mask=None,
        src_key_padding_mask: Optional[Tensor] = None,
        trajectory: Optional[Sequence[float]] = None,
        num_loops: Optional[int] = None,
    ) -> Tensor:
        steps = self._trajectory(trajectory, num_loops)
        output = src
        batch_size = output.shape[1]
        current_time = torch.zeros(batch_size, dtype=output.dtype, device=output.device)

        for step in steps:
            step_tensor = torch.full_like(current_time, step)
            conditioning = self.time_embedder(current_time) + self.step_embedder(
                step_tensor
            )
            for layer in self.layers:
                output = layer(
                    output,
                    conditioning,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )
            current_time = current_time + step_tensor

        return self.norm(output)


class LoopFormerModel(TransformerModel):
    """TabPFN model with a LoopFormer-conditioned recurrent encoder."""

    def __init__(
        self,
        encoder,
        n_out: int,
        ninp: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float = 0.0,
        loopformer_core_layers: int = 1,
        loopformer_max_loops: int = 12,
        loopformer_frequency_embedding_size: int = 256,
        **kwargs,
    ):
        if nlayers != loopformer_core_layers * loopformer_max_loops:
            raise ValueError(
                "nlayers must equal loopformer_core_layers * loopformer_max_loops."
            )
        activation = kwargs.get("activation", "gelu")
        recompute_attn = kwargs.get("recompute_attn", False)
        super().__init__(
            encoder,
            n_out,
            ninp,
            nhead,
            nhid,
            loopformer_core_layers,
            dropout,
            **kwargs,
        )
        layers = [
            LoopFormerEncoderLayer(
                ninp,
                nhead,
                nhid,
                dropout=dropout,
                activation=activation,
                recompute_attn=recompute_attn,
            )
            for _ in range(loopformer_core_layers)
        ]
        self.transformer_encoder = LoopFormerEncoder(
            layers,
            d_model=ninp,
            max_loops=loopformer_max_loops,
            frequency_embedding_size=loopformer_frequency_embedding_size,
        )
        self.model_type = "LoopFormer"
        self.looped_effective_layers = nlayers
        self.looped_physical_layers = loopformer_core_layers

