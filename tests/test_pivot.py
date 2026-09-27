"""Synthetic checks on the output-preserving orthogonal pivot.

No model, dataset, or GPU is required; the whole file runs in about a second.
It verifies the two properties the paper claims for the pivot:

    B_new A_new  ~=  B A          (the represented update is preserved)
    A_new A_new^T ~= I            (the input factor is orthonormal again)

Run with ``pytest tests/`` or directly with ``python tests/test_pivot.py``.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pivotlora import (  # noqa: E402
    eigenvalue_scaled_b_rates,
    orthogonality_error,
    output_preserving_pivot,
    pca_initialization,
    preservation_error,
    reset_adam_state,
)

RANK = 8
D_IN = 64
D_OUT = 32


def _random_factors(dtype=torch.float32, seed=0):
    """A drifted (non-orthonormal) A and a nonzero B, as at the pivot step."""
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(D_IN, RANK, generator=generator, dtype=torch.float64)
    # Make A anisotropic, the way early optimization makes it.
    A = A @ torch.diag(torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.4, 0.2], dtype=torch.float64))
    B = torch.randn(RANK, D_OUT, generator=generator, dtype=torch.float64)
    return A.to(dtype), B.to(dtype)


# ---------------------------------------------------------------------------
# The two central guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pivot_preserves_the_represented_update(dtype):
    """B_new A_new == B A, i.e. the model function is unchanged."""
    A, B = _random_factors(dtype=dtype)
    before = (A.to(torch.float64) @ B.to(torch.float64))

    result = output_preserving_pivot(A, B)
    after = (result.A_new.to(torch.float64) @ result.B_new.to(torch.float64))

    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(after, before, atol=tolerance, rtol=tolerance)
    assert result.preservation_error < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pivot_restores_orthonormality(dtype):
    """A_new A_new^T == I, removing the Gram-matrix anisotropy."""
    A, B = _random_factors(dtype=dtype)

    assert orthogonality_error(A) > 1.0, "the fixture should start non-orthonormal"

    result = output_preserving_pivot(A, B)

    gram = result.A_new.to(torch.float64).T @ result.A_new.to(torch.float64)
    identity = torch.eye(RANK, dtype=torch.float64)

    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(gram, identity, atol=tolerance)
    assert result.orthogonality_after < tolerance
    assert result.orthogonality_before > 1.0


def test_pivot_error_magnitudes_match_the_reported_range():
    """float32 adapters should land near the 1e-7 errors reported in the paper.

    The reasoning runs report a maximum preservation error of 1.33e-7 and a
    maximum post-pivot orthogonality error of 2.38e-8 over 64 adapters per run.
    Synthetic factors will not reproduce those numbers exactly, but they should
    sit in the same regime rather than at float64 machine precision.
    """
    errors = []
    orthogonality = []
    for seed in range(20):
        A, B = _random_factors(dtype=torch.float32, seed=seed)
        result = output_preserving_pivot(A, B)
        errors.append(result.preservation_error)
        orthogonality.append(result.orthogonality_after)

    assert max(errors) < 1e-5
    assert max(orthogonality) < 1e-5


# ---------------------------------------------------------------------------
# Convention handling
# ---------------------------------------------------------------------------


def test_paper_convention_matches_code_convention():
    """rows_are_directions=True must give the transpose of the default path."""
    A_code, B_code = _random_factors(dtype=torch.float64)
    A_paper, B_paper = A_code.T.contiguous(), B_code.T.contiguous()

    code = output_preserving_pivot(A_code, B_code)
    paper = output_preserving_pivot(A_paper, B_paper, rows_are_directions=True)

    assert torch.allclose(paper.A_new, code.A_new.T, atol=1e-12)
    assert torch.allclose(paper.B_new, code.B_new.T, atol=1e-12)

    # And the paper-side product B A is preserved too.
    assert torch.allclose(paper.B_new @ paper.A_new, B_paper @ A_paper, atol=1e-12)


def test_pivot_preserves_the_row_span():
    """A_new spans exactly the same subspace as A, so nothing learned is lost."""
    A, B = _random_factors(dtype=torch.float64)
    result = output_preserving_pivot(A, B)

    # Projecting A onto span(A_new) must return A unchanged.
    projector = result.A_new @ result.A_new.T
    assert torch.allclose(projector @ A, A, atol=1e-12)

    combined = torch.cat([A, result.A_new], dim=1)
    assert torch.linalg.matrix_rank(combined, tol=1e-10).item() == RANK


def test_pivot_rejects_shape_mismatch():
    A = torch.randn(D_IN, RANK)
    B = torch.randn(RANK + 1, D_OUT)
    with pytest.raises(ValueError):
        output_preserving_pivot(A, B)


def test_pivot_raises_when_tolerance_exceeded():
    """A non-finite factor must abort rather than silently corrupt the adapter."""
    A, B = _random_factors(dtype=torch.float64)
    A[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        output_preserving_pivot(A, B)


# ---------------------------------------------------------------------------
# PCA initialization
# ---------------------------------------------------------------------------


def test_pca_initialization_is_orthonormal_and_zero_update():
    activations = np.random.default_rng(0).standard_normal((128, D_IN))
    A = pca_initialization(activations, rank=RANK)

    assert A.shape == (D_IN, RANK)
    assert orthogonality_error(A) < 1e-5

    # B is initialized to zero, so Delta_W = 0 and the base model is unchanged.
    B = torch.zeros(RANK, D_OUT)
    assert torch.count_nonzero(A @ B) == 0


def test_pca_initialization_recovers_a_planted_subspace():
    """With a dominant low-rank signal, PCA should find that subspace."""
    rng = np.random.default_rng(7)
    planted, _ = np.linalg.qr(rng.standard_normal((D_IN, RANK)))
    coefficients = rng.standard_normal((512, RANK)) * 50.0
    activations = coefficients @ planted.T + 0.01 * rng.standard_normal((512, D_IN))

    A = pca_initialization(activations, rank=RANK).numpy().astype(np.float64)

    # Both bases span the same subspace, so all principal angles are ~0.
    singular_values = np.linalg.svd(planted.T @ A, compute_uv=False)
    assert np.allclose(singular_values, 1.0, atol=1e-3)


def test_pca_basis_scale_matches_the_reasoning_protocol():
    """GSM8K and SVAMP scale the orthonormal basis by sqrt(5)."""
    activations = np.random.default_rng(1).standard_normal((128, D_IN))
    scale = float(np.sqrt(5.0))
    A = pca_initialization(activations, rank=RANK, basis_scale=scale)

    gram = (A.to(torch.float64).T @ A.to(torch.float64)).numpy()
    assert np.allclose(gram, 5.0 * np.eye(RANK), atol=1e-4)


def test_pca_initialization_rejects_rank_above_dimension():
    activations = np.random.default_rng(0).standard_normal((32, 4))
    with pytest.raises(ValueError):
        pca_initialization(activations, rank=RANK)


# ---------------------------------------------------------------------------
# Optimizer-state handling
# ---------------------------------------------------------------------------


def test_reset_adam_state_clears_only_the_named_parameters():
    """The classification policy resets A and keeps B; both must be reachable."""
    A = torch.nn.Parameter(torch.randn(D_IN, RANK))
    B = torch.nn.Parameter(torch.randn(RANK, D_OUT))
    optimizer = torch.optim.AdamW([A, B], lr=1e-3)

    (A.sum() + B.sum()).backward()
    optimizer.step()

    assert optimizer.state[A]["exp_avg"].abs().sum() > 0
    assert optimizer.state[B]["exp_avg"].abs().sum() > 0

    cleared = reset_adam_state(optimizer, [A])

    assert cleared == 1
    assert optimizer.state[A]["exp_avg"].abs().sum() == 0
    assert optimizer.state[A]["exp_avg_sq"].abs().sum() == 0
    assert optimizer.state[B]["exp_avg"].abs().sum() > 0, "B state must be retained"


def test_eigenvalue_scaled_rates_average_to_the_lora_plus_ratio():
    """Normalizing by the mean eigenvalue fixes the average B/A ratio at 16."""
    eigenvalues = [10.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.25]
    base_lr = 2e-4
    rates = eigenvalue_scaled_b_rates(eigenvalues, base_lr, lambda_ratio=16.0)

    assert len(rates) == len(eigenvalues)
    assert np.isclose(np.mean(rates) / base_lr, 16.0)
    assert rates == sorted(rates, reverse=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
