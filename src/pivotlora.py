"""
PivotLoRA core: PCA initialization and the output-preserving orthogonal pivot.

This module is a dependency-light, self-contained extraction of the two
operations that define PivotLoRA. It is the reference implementation that the
unit tests in ``tests/test_pivot.py`` exercise. The full training harnesses
(``train_classification.py``, ``train_reasoning_gsm8k.py``,
``train_reasoning_svamp.py``) contain the same algebra inline, wrapped in the
Hugging Face ``Trainer`` callbacks that were actually used to produce the
reported numbers; this file exists so that the method itself can be read and
verified without reading a training loop.

Factor convention
-----------------
The paper writes the LoRA update as

    Delta_W = (alpha / r) * B @ A,    A in R^{r x d},  B in R^{m x r}

so the rows of ``A`` are the input directions. The released code stores the
transpose, ``A_code = A^T in R^{d x r}`` and ``B_code = B^T in R^{r x m}``, so
that ``Delta_W^T = A_code @ B_code``. Both conventions are supported here and
are selected with the ``rows_are_directions`` flag. The algebra is identical;
only the orientation of the stored tensors differs.

Pivot
-----
Given factors ``(A, B)`` after ``K`` epochs, the reduced QR decomposition
``A_code = Q R`` yields

    A_new = Q         (orthonormal columns, same column space as A_code)
    B_new = R @ B     (re-encoded output factor)

so that ``A_new @ B_new == A_code @ B_code`` exactly in exact arithmetic. The
represented update -- and therefore the model function -- is unchanged, while
the Gram matrix of the input factor is restored to the identity. QR is computed
in float64 and the result is cast back to the adapter's storage dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

__all__ = [
    "PivotResult",
    "pca_initialization",
    "output_preserving_pivot",
    "preservation_error",
    "orthogonality_error",
    "reset_adam_state",
]


# ---------------------------------------------------------------------------
# Diagnostics (Eqs. 11 and 12 in the paper)
# ---------------------------------------------------------------------------


def preservation_error(
    A_old: torch.Tensor,
    B_old: torch.Tensor,
    A_new: torch.Tensor,
    B_new: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Relative Frobenius error between the old and new represented updates.

    ``|| A_new B_new - A_old B_old ||_F / (|| A_old B_old ||_F + eps)``.

    The fixed scale ``alpha / r`` cancels from this ratio, so it is computed on
    the raw factor product. All four tensors are promoted to float64 before the
    comparison so that the measurement is not limited by the storage dtype.
    """
    A_old = A_old.to(torch.float64)
    B_old = B_old.to(torch.float64)
    A_new = A_new.to(torch.float64)
    B_new = B_new.to(torch.float64)

    old_product = A_old @ B_old
    new_product = A_new @ B_new

    numerator = torch.linalg.norm(new_product - old_product).item()
    denominator = torch.linalg.norm(old_product).item() + eps
    return numerator / denominator


def orthogonality_error(A: torch.Tensor, columns: bool = True) -> float:
    """``|| A^T A - I ||_F`` (columns) or ``|| A A^T - I ||_F`` (rows).

    With the code convention (``A`` is ``d x r``, directions in columns) the
    relevant Gram matrix is ``A^T A``, so ``columns=True`` is the default.
    """
    A = A.to(torch.float64)
    gram = A.T @ A if columns else A @ A.T
    identity = torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device)
    return torch.linalg.norm(gram - identity).item()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def pca_initialization(
    activations: np.ndarray | torch.Tensor,
    rank: int,
    basis_scale: float = 1.0,
    rows_are_directions: bool = False,
) -> torch.Tensor:
    """Top-``rank`` PCA basis of a centered activation matrix.

    Parameters
    ----------
    activations
        ``N x d`` matrix with one activation vector per training example,
        collected at the supervised prediction positions (see
        ``activation_collection.py``).
    rank
        LoRA rank ``r``. Reported experiments use ``r = 8``.
    basis_scale
        Multiplier applied to the orthonormal basis. Classification uses
        ``1.0``; the GSM8K and SVAMP experiments use ``sqrt(5) ~= 2.2360680``,
        which is *not* norm-matched to Kaiming-uniform initialization (see
        Appendix "Training and Pivot Configuration").
    rows_are_directions
        ``False`` (default) returns ``d x r`` with directions in the columns,
        matching the released training code. ``True`` returns ``r x d``,
        matching the paper's ``A`` in Eq. 4.

    Returns
    -------
    torch.Tensor
        float32 tensor holding the scaled PCA basis. ``B`` is initialized to
        zero by the caller, so ``Delta_W = 0`` at initialization and the
        pretrained function is unchanged.

    Notes
    -----
    Covariance is the uncentered-by-N empirical covariance of Eq. 3,
    ``Sigma = (1/N) Xbar^T Xbar``. Eigenvectors are taken in decreasing
    eigenvalue order. When fewer than ``rank`` directions are available
    (``N - 1 < rank``), the deficient columns are filled by orthogonalizing
    against the recovered ones, which keeps the returned basis orthonormal.
    """
    X = torch.as_tensor(np.asarray(activations), dtype=torch.float64)
    if X.ndim != 2:
        raise ValueError(f"activations must be 2-D (N x d), received {tuple(X.shape)}")

    n_examples, dim = X.shape
    if rank < 1:
        raise ValueError(f"rank must be >= 1, received {rank}")
    if rank > dim:
        raise ValueError(f"rank {rank} exceeds activation dimension {dim}")

    X_centered = X - X.mean(dim=0, keepdim=True)
    covariance = (X_centered.T @ X_centered) / n_examples

    # eigh returns ascending eigenvalues for a symmetric matrix; reverse them.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    basis = eigenvectors[:, order[:rank]]

    # Guard against a rank-deficient covariance (N - 1 < rank).
    available = min(rank, max(1, n_examples - 1))
    if available < rank:
        basis = _complete_basis(basis[:, :available], rank)

    basis = basis * basis_scale
    basis = basis.to(torch.float32)
    return basis.T.contiguous() if rows_are_directions else basis.contiguous()


def _complete_basis(partial: torch.Tensor, rank: int) -> torch.Tensor:
    """Extend ``partial`` (``d x k``, orthonormal columns) to ``d x rank``."""
    dim = partial.shape[0]
    generator = torch.Generator().manual_seed(0)
    filler = torch.randn(dim, rank - partial.shape[1], generator=generator, dtype=torch.float64)
    combined = torch.cat([partial, filler], dim=1)
    completed, _ = torch.linalg.qr(combined, mode="reduced")
    # Preserve the sign/orientation of the genuine PCA directions.
    completed[:, : partial.shape[1]] = partial
    return completed


# ---------------------------------------------------------------------------
# Output-preserving orthogonal pivot
# ---------------------------------------------------------------------------


@dataclass
class PivotResult:
    """Pivoted factors plus the numerical diagnostics recorded for each adapter."""

    A_new: torch.Tensor
    B_new: torch.Tensor
    preservation_error: float
    orthogonality_before: float
    orthogonality_after: float
    skipped: bool = False
    reason: Optional[str] = None
    extra: dict = field(default_factory=dict)


def output_preserving_pivot(
    A: torch.Tensor,
    B: torch.Tensor,
    rows_are_directions: bool = False,
    preservation_tolerance: float = 1e-4,
) -> PivotResult:
    """Replace ``(A, B)`` by ``(A_new, B_new)`` with ``A_new`` orthonormal.

    The represented update ``A @ B`` (code convention) or ``B @ A`` (paper
    convention) is preserved up to floating-point error, and the learned row
    span of the input factor is unchanged.

    QR is computed in float64 regardless of the input dtype; the returned
    tensors are cast back to the dtype and device of the inputs. The reported
    diagnostics are measured on the *cast* tensors, so they include the
    round-trip through the adapter's storage precision -- this is why the
    reported errors are on the order of 1e-7 for float32 adapters rather than
    at float64 machine precision.

    Raises
    ------
    RuntimeError
        If the measured preservation error is non-finite or exceeds
        ``preservation_tolerance``. The training harnesses abort a run in this
        case rather than silently continuing with a corrupted adapter.
    """
    if rows_are_directions:
        # Paper convention: A is r x d, B is m x r, update is B @ A.
        # Transpose into the code convention, pivot, and transpose back.
        result = output_preserving_pivot(
            A.T, B.T, rows_are_directions=False,
            preservation_tolerance=preservation_tolerance,
        )
        return PivotResult(
            A_new=result.A_new.T.contiguous(),
            B_new=result.B_new.T.contiguous(),
            preservation_error=result.preservation_error,
            orthogonality_before=result.orthogonality_before,
            orthogonality_after=result.orthogonality_after,
            skipped=result.skipped,
            reason=result.reason,
            extra=dict(result.extra),
        )

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must both be 2-D")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"inner dimensions disagree: A is {tuple(A.shape)}, B is {tuple(B.shape)}"
        )

    store_dtype, store_device = A.dtype, A.device

    A_old_64 = A.detach().to(torch.float64).clone()
    B_old_64 = B.detach().to(torch.float64).clone()

    orth_before = orthogonality_error(A_old_64, columns=True)

    # A_old = Q R  with Q having orthonormal columns spanning col(A_old).
    Q_64, R_64 = torch.linalg.qr(A_old_64, mode="reduced")
    A_new_64 = Q_64
    B_new_64 = R_64 @ B_old_64

    # Cast to the adapter's storage dtype, then measure on what is stored.
    A_new = A_new_64.to(device=store_device, dtype=store_dtype)
    B_new = B_new_64.to(device=store_device, dtype=store_dtype)

    error = preservation_error(A_old_64, B_old_64, A_new, B_new)
    orth_after = orthogonality_error(A_new, columns=True)

    if not math.isfinite(error):
        raise RuntimeError("non-finite pivot preservation error")
    if error > preservation_tolerance:
        raise RuntimeError(
            f"pivot preservation error {error:.3e} exceeds tolerance "
            f"{preservation_tolerance:.3e}"
        )

    return PivotResult(
        A_new=A_new,
        B_new=B_new,
        preservation_error=error,
        orthogonality_before=orth_before,
        orthogonality_after=orth_after,
    )


# ---------------------------------------------------------------------------
# Optimizer-state handling
# ---------------------------------------------------------------------------


def reset_adam_state(optimizer, parameters: Sequence[torch.nn.Parameter]) -> int:
    """Zero the Adam moments and step counter for ``parameters``.

    Optimizer-state handling is an implementation choice and is deliberately
    separate from the algebraic preservation guarantee above. The two released
    policies differ:

    * Classification (``train_classification.py``): reset the moments for the
      input factor ``A`` only, retain those for ``B``, then warm the ``B``
      learning rate over 10 steps to the uniform rate ``16 * eta_A``.
    * Reasoning (``train_reasoning_*.py``): reset the moments for *both*
      transformed factors and leave the global learning-rate schedule running
      from the current global step. No differential or eigenvalue-scaled rates
      and no post-pivot warm-up are used.

    Returns the number of parameters whose state was actually cleared.
    """
    cleared = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not any(param is target for target in parameters):
                continue
            state = optimizer.state.get(param, None)
            if not state:
                continue
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].zero_()
            if "step" in state:
                state["step"] = (
                    torch.zeros_like(state["step"])
                    if isinstance(state["step"], torch.Tensor)
                    else 0
                )
            cleared += 1
    return cleared


def eigenvalue_scaled_b_rates(
    eigenvalues: Sequence[float],
    base_lr: float,
    lambda_ratio: float = 16.0,
) -> list[float]:
    """Per-rank ``B`` learning rates of Eq. 8 (classification protocol only).

        eta_{B,i} = (lambda_i / mean(lambda)) * lambda_ratio * eta_A

    Normalizing by the mean eigenvalue makes the average B-to-A learning-rate
    ratio equal to ``lambda_ratio`` (16, matching the LoRA+ scale). This is an
    optimizer heuristic, not part of the activation-subspace definition; the
    direction-versus-rate controls in the paper separate the two. The GSM8K and
    SVAMP experiments do not use it.
    """
    values = np.asarray(list(eigenvalues), dtype=np.float64)
    if values.size == 0:
        return []
    mean_eigenvalue = float(values.mean())
    if mean_eigenvalue <= 0.0:
        return [lambda_ratio * base_lr] * values.size
    return [float(v / mean_eigenvalue * lambda_ratio * base_lr) for v in values]
