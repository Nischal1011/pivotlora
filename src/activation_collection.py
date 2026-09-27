"""
Activation sources for the PivotLoRA initialization subspace.

PivotLoRA estimates the row span of the LoRA input factor from activations at
the positions the supervised objective actually reads. Which positions those
are depends on the objective, and the paper treats the choice as an
alignment--estimation trade-off rather than assuming one source is universally
best:

===========================  ==========================================
Objective                    Activation source
===========================  ==========================================
BERT classification          the ``[CLS]`` representation (position 0)
Decoder classification       the sequence representation read by the head
Completion-only reasoning    mean over hidden states that predict a
                             supervised response token
GPT-2 summarization control  sequence mean over all valid positions
===========================  ==========================================

The functions here are dependency-light extractions of the pooling rules used
by the training harnesses. ``response_predictor_mask`` in particular is the
piece most easily got wrong, and it is unit-testable without a model.

In every case a single ``d``-dimensional vector is produced per training
example, so the covariance in ``pivotlora.pca_initialization`` is estimated
from exactly ``N`` vectors -- one per labeled example -- and no example is
weighted by its length.
"""

from __future__ import annotations

from typing import Iterable

import torch

__all__ = [
    "target_position_pool",
    "sequence_mean_pool",
    "response_predictor_mask",
    "response_position_pool",
    "collect_bert_cls_activations",
]


# ---------------------------------------------------------------------------
# Classification: one designated prediction position per example
# ---------------------------------------------------------------------------


def target_position_pool(hidden_states: torch.Tensor, position: int = 0) -> torch.Tensor:
    """Select one position per example.

    ``hidden_states`` is ``[batch, seq_len, d]``; the result is ``[batch, d]``.
    For BERT the designated prediction representation is ``[CLS]``, i.e.
    ``position=0``, which is what ``train_classification.py`` hooks on
    ``attention.self.query``.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be [batch, seq_len, d], received {tuple(hidden_states.shape)}"
        )
    return hidden_states[:, position, :]


def sequence_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over valid (non-padding) positions.

    This is the less targeted source used by the GPT-2 summarization control
    and, in the matched BERT ablation, by the "all non-padding tokens"
    condition. It draws on many more activation vectors than target-position
    pooling but is not tied to the position the supervision acts on.
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denominator


# ---------------------------------------------------------------------------
# Completion-only reasoning: response-predictor positions
# ---------------------------------------------------------------------------


def response_predictor_mask(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Positions ``t`` whose *next* token ``t + 1`` is supervised.

    In a causal language model hidden state ``t`` predicts token ``t + 1``, so
    the activations that carry the supervised signal sit one position to the
    left of the supervised labels:

        P_n = { t : M_{n, t+1} = 1 }

    where ``M`` is the completion mask (``labels != -100``). Prompt tokens are
    masked out of the loss with ``-100``; completion tokens and the appended
    EOS are supervised.

    ``labels`` is ``[batch, seq_len]``. The returned bool mask has the same
    shape, is ``False`` at the final position (nothing follows it), and is
    additionally intersected with ``attention_mask`` so that padded predictor
    positions are excluded.

    Getting the shift wrong -- pooling at the supervised positions rather than
    at their predictors -- silently changes which subspace is estimated, which
    is why this rule is isolated and tested here.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be [batch, seq_len], received {tuple(labels.shape)}")

    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100

    if attention_mask is not None:
        mask &= attention_mask.bool()

    return mask


def response_position_pool(
    hidden_states: torch.Tensor,
    predictor_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean the hidden states over each example's response-predictor positions.

        x_{l,n} = (1 / |P_n|) * sum_{t in P_n} h_{l,n,t}                (Eq. 6)

    ``hidden_states`` is ``[batch, seq_len, d]`` and ``predictor_mask`` is
    ``[batch, seq_len]``; the result is ``[batch, d]``, one vector per example.

    Averaging *within* each example before the covariance is formed prevents a
    long response from dominating the estimate through sheer token count.
    Examples with an empty predictor set (possible after truncation) yield a
    zero vector; the released harness drops them before the PCA rather than
    letting them bias the mean, and callers here should filter on
    ``predictor_mask.any(dim=1)``.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be [batch, seq_len, d], received {tuple(hidden_states.shape)}"
        )
    if predictor_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"predictor_mask {tuple(predictor_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}"
        )

    weights = predictor_mask.unsqueeze(-1).to(hidden_states.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denominator


# ---------------------------------------------------------------------------
# Convenience: full BERT collection pass
# ---------------------------------------------------------------------------


def collect_bert_cls_activations(
    model,
    dataloader: Iterable,
    device: torch.device | str = "cpu",
) -> dict[int, torch.Tensor]:
    """One frozen forward pass collecting ``[CLS]`` adapter inputs per layer.

    Registers a forward pre-hook on ``attention.self.query`` in every encoder
    layer -- query and value share the same adapter input, so one hook per
    layer suffices -- and returns ``{layer_index: [N, d] float32 tensor}``.

    No gradients are computed and the model is left unmodified; this is the
    single frozen pass referred to throughout the paper, including for the
    alignment coefficient rho.
    """
    from collections import defaultdict

    model.eval().to(device)
    collected: dict[int, list[torch.Tensor]] = defaultdict(list)

    def make_pre_hook(layer_index: int):
        def pre_hook(module, inputs):
            collected[layer_index].append(inputs[0].detach()[:, 0, :].float().cpu())
        return pre_hook

    handles = [
        layer.attention.self.query.register_forward_pre_hook(make_pre_hook(index))
        for index, layer in enumerate(model.bert.encoder.layer)
    ]

    try:
        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key not in ("labels", "start_positions", "end_positions")
                }
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {index: torch.cat(parts, dim=0) for index, parts in collected.items()}
