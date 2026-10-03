from __future__ import annotations

import numpy as np


def build_kernel(
    features: np.ndarray,
    quality: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Build a positive-semidefinite L-ensemble kernel.

    ``quality`` is sample importance (larger is better).  Shifted cosine is
    used instead of elementwise clipping because clipping a Gram matrix does
    not, in general, preserve positive semidefiniteness.
    """
    features = np.asarray(features, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {features.shape}")
    if len(features) != len(quality):
        raise ValueError("features and quality must have the same length")
    if not np.isfinite(features).all() or not np.isfinite(quality).all():
        raise ValueError("features and quality must be finite")
    if (quality < 0).any():
        raise ValueError("quality must be non-negative")

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)

    # 0.5 * X X^T + 0.5 * 11^T is PSD and lies in [0, 1].
    similarity = 0.5 * (cosine + 1.0)
    q = np.sqrt(np.maximum(quality, 1e-12))
    kernel = q[:, None] * similarity * q[None, :]
    kernel = 0.5 * (kernel + kernel.T)
    kernel.flat[:: len(kernel) + 1] += jitter
    return kernel


def kdpp_sampling(
    features: np.ndarray,
    quality: np.ndarray,
    k: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Sample exactly k rows from one bounded-size block with DPPy."""
    n = len(features)
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= N, got k={k}, N={n}")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    try:
        from dppy.finite_dpps import FiniteDPP
    except ImportError as exc:
        raise RuntimeError("DPPy is required: pip install dppy") from exc

    rng = np.random.RandomState(seed)
    last_error: Exception | None = None
    for jitter in (1e-8, 1e-6, 1e-4):
        try:
            kernel = build_kernel(features, quality, jitter=jitter)
            dpp = FiniteDPP("likelihood", L=kernel)
            sample = dpp.sample_exact_k_dpp(size=k, random_state=rng)
            result = np.asarray(sample, dtype=np.int64)
            if len(result) != k or len(np.unique(result)) != k:
                raise RuntimeError("DPPy returned an invalid k-DPP sample")
            return result
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            last_error = exc
    raise RuntimeError("k-DPP sampling failed after numerical jitter retries") from last_error

