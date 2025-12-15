from typing import Dict, List

import numpy as np


def _residual_to_confidence(residual: np.ndarray, thr_low: float, thr_high: float) -> np.ndarray:
    conf = np.ones_like(residual, dtype=np.float32)
    if thr_high <= thr_low:
        return conf
    mask_high = residual >= thr_high
    mask_low = residual <= thr_low
    mid = (~mask_high) & (~mask_low)
    conf[mask_high] = 0.0
    if np.any(mid):
        conf[mid] = 1.0 - ((residual[mid] - thr_low) / (thr_high - thr_low))
    return np.clip(conf, 0.0, 1.0)


def _harmonic_mean(stack: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if stack.ndim == 2:
        return stack
    denom = np.sum(1.0 / np.clip(stack, eps, 1.0), axis=0)
    return stack.shape[0] / np.clip(denom, eps, None)


def compute_weight_map(
    r_oof: np.ndarray,
    r_if_stack: List[np.ndarray],
    gap_max: float,
    thr_low: float,
    thr_high: float,
    w_min: float,
    lam_if: float,
    lam_oof: float,
    lam_gap: float,
) -> Dict[str, np.ndarray]:
    r_if = np.stack(r_if_stack, axis=0)
    r_if_mean = np.mean(r_if, axis=0)
    gap = r_oof - r_if_mean
    gap_norm = np.clip(gap / max(gap_max, 1e-6), 0.0, 1.0)
    c_gap = 1.0 - gap_norm
    c_oof = _residual_to_confidence(r_oof, thr_low, thr_high)
    c_if_components = [_residual_to_confidence(r, thr_low, thr_high) for r in r_if_stack]
    c_if = _harmonic_mean(np.stack(c_if_components, axis=0))
    total_lambda = max(lam_if + lam_oof + lam_gap, 1e-6)
    c_bar = (lam_if * c_if + lam_oof * c_oof + lam_gap * c_gap) / total_lambda
    weights = w_min + (1.0 - w_min) * c_bar
    max_if = np.max(r_if, axis=0)
    min_if = np.min(r_if, axis=0)
    rule1 = (np.maximum(max_if, r_oof) < thr_low) & (gap < gap_max)
    rule3 = (r_oof >= thr_high) & (min_if >= thr_high)
    weights = np.where(rule1, 1.0, weights)
    weights = np.where(rule3, 0.0, weights)
    return {
        "weights": weights.astype(np.float32),
        "c_if": c_if.astype(np.float32),
        "c_oof": c_oof.astype(np.float32),
        "c_gap": c_gap.astype(np.float32),
        "gap": gap.astype(np.float32),
        "r_if_mean": r_if_mean.astype(np.float32),
    }
