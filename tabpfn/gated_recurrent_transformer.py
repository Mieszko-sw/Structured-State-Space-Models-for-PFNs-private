"""Gated recurrent-depth Transformer components for TabPFN.

This implements a prelude/shared-core/coda layout. The prelude output is kept
as a fixed anchor, the shared middle block is recurrently applied, and an
elementwise gate blends every proposed update into the recurrent state.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from tabpfn.transformer import TransformerModel


class GatedRecurrentTransformerEncoder(nn.Module):
    """Untied edge blocks surrounding an anchored, gated recurrent core."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        d_model: int,
        warmup_layers: int,
        core_layers: int,
        exit_layers: int,
        max_loops: int,
        min_loops: int = 1,
        sample_depth_during_training: bool = True,
        state_noise_std: float = 0.1,
        gate_noise_std: float = 0.1,
        gate_temperature: float = 1.0,
        gate_bias: float = 4.0,
        norm=None,
    ):
        super().__init__()
        if warmup_layers < 1:
            raise ValueError("At least one warmup/prelude layer is required.")
        if core_layers < 1:
            raise ValueError("At least one recurrent core layer is required.")
        if exit_layers < 1:
            raise ValueError("At least one exit/coda layer is required.")
        if not 1 <= min_loops <= max_loops:
            raise ValueError("Expected 1 <= min_loops <= max_loops.")
        if state_noise_std < 0 or gate_noise_std < 0:
            raise ValueError("Noise standard deviations must be non-negative.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")

        physical_layers = warmup_layers + core_layers + exit_layers
        if len(layers) != physical_layers:
            raise ValueError(
                f"Expected {physical_layers} physical layers, got {len(layers)}."
            )

        self.layers = nn.ModuleList(layers)
        self.warmup_layers = warmup_layers
        self.core_layers = core_layers
        self.exit_layers = exit_layers
        self.max_loops = max_loops
        self.min_loops = min_loops
        self.sample_depth_during_training = sample_depth_during_training
        self.state_noise_std = float(state_noise_std)
        self.gate_noise_std = float(gate_noise_std)
        self.gate_temperature = float(gate_temperature)
        self.num_layers = warmup_layers + core_layers * max_loops + exit_layers
        self.norm = norm

        self.recurrent_projection = nn.Linear(2 * d_model, d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.anchor_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def _resolve_num_loops(self, num_loops: Optional[int]) -> int:
        if num_loops is None:
            if self.training and self.sample_depth_during_training:
                return int(
                    torch.randint(self.min_loops, self.max_loops + 1, ()).item()
                )
            return self.max_loops

        loops = int(num_loops)
        if not 1 <= loops <= self.max_loops:
            raise ValueError(f"num_loops must be in [1, {self.max_loops}].")
        return loops

    def forward(
        self,
        src: Tensor,
        mask=None,
        src_key_padding_mask: Optional[Tensor] = None,
        num_loops: Optional[int] = None,
    ) -> Tensor:
        output = src
        warmup_end = self.warmup_layers
        core_end = warmup_end + self.core_layers

        for layer in self.layers[:warmup_end]:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        anchor = output

        loops = self._resolve_num_loops(num_loops)
        core = self.layers[warmup_end:core_end]
        for _ in range(loops):
            noisy_state = output
            if self.training and self.state_noise_std:
                noisy_state = noisy_state + self.state_noise_std * torch.randn_like(
                    noisy_state
                )

            proposal = self.recurrent_projection(
                torch.cat((noisy_state, anchor), dim=-1)
            )
            for layer in core:
                proposal = layer(
                    proposal,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )

            gate_logits = self.gate(
                torch.cat(
                    (self.state_norm(output), self.anchor_norm(anchor)), dim=-1
                )
            ) / self.gate_temperature
            if self.training and self.gate_noise_std:
                gate_logits = gate_logits + self.gate_noise_std * torch.randn_like(
                    gate_logits
                )
            copy_gate = torch.sigmoid(gate_logits)
            output = copy_gate * output + (1.0 - copy_gate) * proposal

        for layer in self.layers[core_end:]:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        return self.norm(output) if self.norm is not None else output


class GatedRecurrentTransformerModel(TransformerModel):
    """TabPFN with fixed prelude/coda layers and a gated recurrent middle."""

    def __init__(
        self,
        *args,
        gated_warmup_layers: int = 1,
        gated_core_layers: int = 1,
        gated_exit_layers: int = 1,
        gated_max_loops: int = 10,
        gated_min_loops: int = 1,
        gated_sample_depth_during_training: bool = True,
        gated_state_noise_std: float = 0.1,
        gated_gate_noise_std: float = 0.1,
        gated_gate_temperature: float = 1.0,
        gated_gate_bias: float = 4.0,
        **kwargs,
    ):
        effective_layers = args[5] if len(args) > 5 else kwargs["nlayers"]
        expected_effective_layers = (
            gated_warmup_layers
            + gated_core_layers * gated_max_loops
            + gated_exit_layers
        )
        if effective_layers != expected_effective_layers:
            raise ValueError(
                "nlayers must equal gated_warmup_layers + "
                "gated_core_layers * gated_max_loops + gated_exit_layers "
                f"({expected_effective_layers} for this configuration)."
            )

        physical_layers = (
            gated_warmup_layers + gated_core_layers + gated_exit_layers
        )
        if len(args) > 5:
            args = list(args)
            args[5] = physical_layers
            args = tuple(args)
        else:
            kwargs["nlayers"] = physical_layers

        super().__init__(*args, **kwargs)
        original_encoder = self.transformer_encoder
        self.transformer_encoder = GatedRecurrentTransformerEncoder(
            list(original_encoder.layers),
            d_model=self.ninp,
            warmup_layers=gated_warmup_layers,
            core_layers=gated_core_layers,
            exit_layers=gated_exit_layers,
            max_loops=gated_max_loops,
            min_loops=gated_min_loops,
            sample_depth_during_training=gated_sample_depth_during_training,
            state_noise_std=gated_state_noise_std,
            gate_noise_std=gated_gate_noise_std,
            gate_temperature=gated_gate_temperature,
            gate_bias=gated_gate_bias,
            norm=original_encoder.norm,
        )
        self.model_type = "GatedRecurrentTransformer"
        self.looped_effective_layers = effective_layers
        self.looped_physical_layers = physical_layers

