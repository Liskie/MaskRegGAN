#!/usr/bin/env python3
"""
Inspect residuals/weights across folds for a validation slice.

For a given target fold and validation slice id, this tool gathers:
  1. OOF prediction (target fold model on its validation sample)
  2. IF predictions from the other folds' models (treating this sample as training)
  3. IF mean / residual statistics and derived gap & confidence maps
  4. Final loss-weight map computed from the collected residuals

The inputs are the training artifacts under `output/.../fold_*/stage_*` and the
folded dataset under `data/.../cv_folds/fold_*/{train,val}/B/*.npy`.
No files are modified; the script only reads and visualises existing outputs.

Example:
  python experiment04-1-cross-training-viz.py \\
      --run-root output/SynthRAD-RegGAN-512-keepratio-cross3 \\
      --cv-root data/SynthRAD2023-Task1/cv_folds \\
      --fold 1 --slice 1PA001_0010 \\
      --stages 1 2 3 \\
      --save tmp/fold1_val0010.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_array(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def _ensure_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return path


def _collect_fold_roots(run_root: Path) -> Dict[int, Path]:
    folds: Dict[int, Path] = {}
    for fold_dir in sorted(run_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold_id = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        folds[fold_id] = fold_dir
    if not folds:
        raise RuntimeError(f"No fold_* directories found under {run_root}")
    return folds


def _load_stage_summary(weights_dir: Path) -> Dict[str, float]:
    summary_path = weights_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = data.get("cfg", {})
        return {str(k): float(v) for k, v in cfg.items()}
    except Exception:
        return {}


def _collect_stage_dirs(fold_root: Path) -> Dict[int, Dict[str, Path]]:
    stage_map: Dict[int, Dict[str, Path]] = {}
    for stage_dir in sorted(fold_root.glob("stage_*")):
        if not stage_dir.is_dir():
            continue
        try:
            stage_id = int(stage_dir.name.split("_")[-1])
        except ValueError:
            continue
        stage_map[stage_id] = {
            "root": stage_dir,
            "residual_if": stage_dir / "residuals" / "if" / "maps",
            "residual_oof": stage_dir / "residuals" / "oof" / "maps",
            "weights": stage_dir / "weights",
        }
    if not stage_map:
        raise RuntimeError(f"No stage_* directories under {fold_root}")
    return stage_map


def _available_slice_ids(stage_dirs: Dict[int, Dict[str, Path]]) -> Iterable[str]:
    # Use OOF residuals because we expect the user to provide a validation slice.
    slice_ids = set()
    for info in stage_dirs.values():
        res_dir = info["residual_oof"]
        if not res_dir.is_dir():
            continue
        slice_ids.update(p.stem for p in res_dir.glob("*.npy"))
    return slice_ids


def _load_truth(cv_root: Path, fold_id: int, subset: str, slice_id: str) -> Optional[np.ndarray]:
    base = cv_root / f"fold_{fold_id}" / subset / "B"
    candidates = [base / f"{slice_id}.npy"]
    if "_" in slice_id:
        root, suf = slice_id.split("_", 1)
        candidates.append(base / f"{root}.nii_z{suf}.npy")
    for path in candidates:
        if path.exists():
            return _load_array(path)
    return None


def _resize_keep_ratio_pad(arr: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    if arr is None:
        return None
    th, tw = int(target_shape[0]), int(target_shape[1])
    if arr.shape == (th, tw):
        return arr.astype(np.float32)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    h, w = arr.shape
    scale = min(th / max(h, 1), tw / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    img_resized = img.resize((new_w, new_h), resample=Image.NEAREST)
    canvas = np.full((th, tw), -1.0, dtype=np.float32)
    top = (th - new_h) // 2
    left = (tw - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.array(img_resized, dtype=np.float32)
    return canvas


# --- confidence utilities (mirrors trainer logic) --------------------------------

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
    denom = np.sum(1.0 / np.clip(stack, eps, 1.0), axis=0)
    return stack.shape[0] / np.clip(denom, eps, None)


def _compute_components(r_oof: np.ndarray,
                        r_if_stack: Sequence[np.ndarray],
                        cfg: Dict[str, float]) -> Dict[str, np.ndarray]:
    stack = np.stack(r_if_stack, axis=0)
    r_if_mean = stack.mean(axis=0)
    gap = r_oof - r_if_mean

    thr_low = float(cfg.get("thr_low", 0.05))
    thr_high = float(cfg.get("thr_high", 0.15))
    gap_max = float(cfg.get("gap_max", 0.05))
    lam_if = float(cfg.get("lambda_if", 1.0))
    lam_oof = float(cfg.get("lambda_oof", 2.0))
    lam_gap = float(cfg.get("lambda_gap", 1.0))
    w_min = float(cfg.get("w_min", 0.05))

    c_oof = _residual_to_confidence(r_oof, thr_low, thr_high)
    c_if_components = np.stack([_residual_to_confidence(r, thr_low, thr_high) for r in r_if_stack], axis=0)
    c_if = _harmonic_mean(c_if_components)

    gap_norm = np.clip(gap / max(gap_max, 1e-6), 0.0, 1.0)
    c_gap = 1.0 - gap_norm

    denom = max(lam_if + lam_oof + lam_gap, 1e-6)
    conf_combined = (lam_if * c_if + lam_oof * c_oof + lam_gap * c_gap) / denom
    weights = w_min + (1.0 - w_min) * conf_combined

    return {
        "r_if_mean": r_if_mean,
        "gap": gap,
        "c_if": c_if,
        "c_oof": c_oof,
        "c_gap": c_gap,
        "conf_combined": conf_combined,
        "weight": weights,
    }


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------


def _plot_matrix(stage_ids: List[int],
                 row_defs: List[tuple],
                 stage_data: Dict[int, Dict[str, Optional[np.ndarray]]],
                 fold_id: int,
                 slice_id: str,
                 save_path: Optional[Path],
                 show: bool):
    n_rows = len(row_defs)
    n_cols = len(stage_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    pred_keys = {"pred_oof"} | {f"pred_if_fold{idx:02d}" for idx in range(1, 10)} | {"pred_if_mean"}
    residual_keys = {"residual_oof", "residual_if_mean", "gap"}
    confidence_keys = {"c_if", "c_oof", "c_gap", "conf_combined", "weight"}
    cmap_vlag = sns.color_palette("vlag", as_cmap=True)

    # Determine global bounds per row across all stages
    row_bounds: Dict[str, Optional[tuple]] = {}
    for key, _ in row_defs:
        values = []
        for stage_id in stage_ids:
            arr = stage_data.get(stage_id, {}).get(key)
            if arr is not None:
                values.append(np.asarray(arr, dtype=np.float32))
        if not values:
            row_bounds[key] = None
            continue
        if key in pred_keys:
            row_bounds[key] = (-1.0, 1.0)
        elif key in residual_keys:
            vmax = max(float(np.abs(v).max()) for v in values) or 1.0
            row_bounds[key] = (-vmax, vmax)
        else:
            vmin = min(float(v.min()) for v in values)
            vmax = max(float(v.max()) for v in values)
            if np.isclose(vmax, vmin):
                vmax = vmin + 1e-6
            row_bounds[key] = (vmin, vmax)

    fig.subplots_adjust(left=0.06, right=0.92, top=0.94, bottom=0.08, wspace=0.15, hspace=0.25)

    for col, stage_id in enumerate(stage_ids):
        data = stage_data.get(stage_id, {})
        for row, (key, title) in enumerate(row_defs):
            ax = axes[row, col]
            arr = data.get(key)
            if arr is None:
                ax.axis("off")
                continue
            if key in pred_keys:
                cmap = "gray"
            elif key in residual_keys:
                cmap = cmap_vlag
            else:
                cmap = "gray"
            bounds = row_bounds.get(key)
            if bounds:
                vmin, vmax = bounds
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(arr, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=11)
            ax.set_title(f"Stage {stage_id}", fontsize=11)

    fig.canvas.draw()
    for row, (key, _) in enumerate(row_defs):
        bounds = row_bounds.get(key)
        if not bounds:
            continue
        vmin, vmax = bounds
        if vmax <= vmin:
            vmax = vmin + 1e-6
        if key in pred_keys:
            cmap = "gray"
        elif key in residual_keys:
            cmap = cmap_vlag
        else:
            cmap = "gray"
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax_last = axes[row, -1]
        pos = ax_last.get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Fold {fold_id} | Slice {slice_id}", fontsize=14)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[viz] saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------


def visualise(run_root: Path,
              cv_root: Path,
              fold_id: int,
              stage_ids: Optional[List[int]],
              slice_id: Optional[str],
              save_path: Optional[Path],
              show: bool):
    fold_roots = _collect_fold_roots(run_root)
    if fold_id not in fold_roots:
        raise RuntimeError(f"Fold {fold_id} not found under {run_root}")

    # Gather stage directories for all folds
    stage_infos = {fid: _collect_stage_dirs(root) for fid, root in fold_roots.items()}
    target_stages = stage_infos[fold_id]
    if stage_ids:
        stage_ids = [sid for sid in stage_ids if sid in target_stages]
        if not stage_ids:
            raise RuntimeError("None of the requested stages exist for the target fold.")
    else:
        stage_ids = sorted(target_stages.keys())

    available_ids = sorted(_available_slice_ids(target_stages))
    if not available_ids:
        raise RuntimeError(f"No validation residuals found for fold {fold_id}.")

    if slice_id is None:
        slice_id = random.choice(available_ids)
    elif slice_id not in available_ids:
        raise FileNotFoundError(
            f"Slice {slice_id} not found in fold {fold_id}'s validation residuals "
            f"(available examples: {available_ids[:5]} ...)"
        )

    other_folds = [fid for fid in sorted(fold_roots.keys()) if fid != fold_id]
    truths: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for fid in fold_roots:
        train_gt = _load_truth(cv_root, fid, "train", slice_id)
        val_gt = _load_truth(cv_root, fid, "val", slice_id)
        if train_gt is None and val_gt is None:
            print(f"[warn] GT missing: fold {fid} train/val GT not found for slice {slice_id}")
        truths[fid] = {"train": train_gt, "val": val_gt}

    stage_data: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for stage_id in stage_ids:
        data: Dict[str, Optional[np.ndarray]] = {}
        target_info = target_stages[stage_id]

        # --- OOF residual / prediction (target fold on validation sample)
        oof_path = target_info["residual_oof"] / f"{slice_id}.npy"
        if not oof_path.exists():
            print(f"[warn] OOF residual missing: {oof_path}")
        r_oof = _load_array(oof_path)
        data["residual_oof"] = r_oof

        # --- IF residuals from other folds
        r_if_stack_raw: List[np.ndarray] = []
        residuals_by_fold: Dict[int, np.ndarray] = {}
        for idx, other_fold in enumerate(other_folds):
            other_stage = stage_infos[other_fold].get(stage_id, None)
            key_pred = f"pred_if_fold{other_fold:02d}"
            if other_stage is None:
                print(f"[warn] stage {stage_id} missing for fold {other_fold}")
                data[key_pred] = None
                continue
            if_path = other_stage["residual_if"] / f"{slice_id}.npy"
            if not if_path.exists():
                print(f"[warn] IF residual missing: {if_path}")
            r_if = _load_array(if_path)
            if r_if is not None:
                r_if_stack_raw.append(r_if)
                residuals_by_fold[other_fold] = r_if
            data[key_pred] = None  # placeholder (will assign after resizing)

        if not r_if_stack_raw:
            # no IF residuals collected → skip remaining rows
            stage_data[stage_id] = data
            continue

        target_shape = r_oof.shape if r_oof is not None else r_if_stack_raw[0].shape
        gt_target = truths[fold_id]["val"]
        if gt_target is None:
            gt_target = truths[fold_id]["train"]
        gt_val_target = _resize_keep_ratio_pad(gt_target, target_shape)
        if r_oof is not None and gt_val_target is not None:
            data["pred_oof"] = gt_val_target + r_oof
        else:
            data["pred_oof"] = None

        aligned_if_stack: List[np.ndarray] = []
        for other_fold, r_if in residuals_by_fold.items():
            aligned = r_if if r_if.shape == target_shape else _resize_keep_ratio_pad(r_if, target_shape)
            aligned_if_stack.append(aligned)
            key_pred = f"pred_if_fold{other_fold:02d}"
            if gt_val_target is not None:
                pred = gt_val_target + aligned
                data[key_pred] = pred
            else:
                data[key_pred] = None

        # --- Compute aggregate statistics (IF mean, gap, confidence, weights)
        cfg = _load_stage_summary(
            next((stage_infos[f][stage_id]["weights"] for f in other_folds if stage_id in stage_infos[f]), target_info["weights"])
        )
        components = _compute_components(r_oof if r_oof is not None else aligned_if_stack[0], aligned_if_stack, cfg)

        r_if_mean = components["r_if_mean"]
        if r_if_mean.shape != target_shape:
            r_if_mean = _resize_keep_ratio_pad(r_if_mean, target_shape)
        data["residual_if_mean"] = r_if_mean
        if gt_val_target is not None:
            data["pred_if_mean"] = gt_val_target + r_if_mean
        else:
            data["pred_if_mean"] = None
        data["residual_oof"] = r_oof if r_oof is None or r_oof.shape == target_shape else _resize_keep_ratio_pad(r_oof, target_shape)
        data["gap"] = components["gap"]
        data["c_if"] = components["c_if"]
        data["c_oof"] = components["c_oof"]
        data["c_gap"] = components["c_gap"]
        data["conf_combined"] = components["conf_combined"]
        data["weight"] = components["weight"]

        stage_data[stage_id] = data

    # --- Row definitions (ordered as requested)
    row_defs = [
        ("pred_oof", "OOF Prediction"),
        (f"pred_if_fold{other_folds[0]:02d}", f"IF Prediction fold{other_folds[0]:02d}") if len(other_folds) > 0 else (None, ""),
        (f"pred_if_fold{other_folds[1]:02d}", f"IF Prediction fold{other_folds[1]:02d}") if len(other_folds) > 1 else (None, ""),
        ("pred_if_mean", "IF Mean Prediction"),
        ("residual_oof", "OOF Residual"),
        ("residual_if_mean", "IF Residual Mean"),
        ("gap", "Gap Map"),
        ("c_if", "Confidence IF"),
        ("c_oof", "Confidence OOF"),
        ("c_gap", "Confidence GAP"),
        ("conf_combined", "Combined Confidence"),
        ("weight", "Loss Weight"),
    ]
    row_defs = [(k, t) for k, t in row_defs if k is not None]

    _plot_matrix(stage_ids, row_defs, stage_data, fold_id, slice_id, save_path, show)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Stage-wise residual/weight inspection for validation slices.")
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Root directory containing fold_* training outputs.")
    parser.add_argument("--cv-root", type=Path, required=True,
                        help="Root directory containing cv_folds/fold_*/{train,val}/B datasets.")
    parser.add_argument("--fold", type=int, required=True,
                        help="Target fold id whose validation slice you want to inspect (1-based).")
    parser.add_argument("--stages", type=int, nargs="*", default=None,
                        help="Specific stage numbers to plot; defaults to all available in the target fold.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Validation slice id (e.g. 1PA001_0010). If omitted a random slice is chosen.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional path to save the figure as PNG.")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure interactively.")
    args = parser.parse_args()

    visualise(
        run_root=args.run_root,
        cv_root=args.cv_root,
        fold_id=args.fold,
        stage_ids=args.stages,
        slice_id=args.slice,
        save_path=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()

"""
python experiment04-1-cross-training-viz.py \
  --run-root output/SynthRAD-RegGAN-512-keepratio-cross3 \
  --cv-root data/SynthRAD2023-Task1/cv_folds \
  --fold 1 --slice 1PA059_0110 \
  --stages 1 2 3 4 5 6 7 8 9 10 11\
  --save experiment-results/04-cross-training-viz/fold1_1PA059_0110.png
  
python experiment04-1-cross-training-viz.py \
  --run-root output/SynthRAD-RegGAN-512-keepratio-cross3 \
  --cv-root data/SynthRAD2023-Task1/cv_folds \
  --fold 1 --slice 1PA026_0030 \
  --stages 1 2 3 4 5 6 7 8 9 10 11\
  --save experiment-results/04-cross-training-viz/fold1_1PA026_0030.png
"""
