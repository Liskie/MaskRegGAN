#!/usr/bin/env python3
"""
Visualise per-stage residuals, gap maps, and weight maps for cross-fold training.

Requirements
------------
- numpy
- matplotlib
- pillow (for saving PNGs)

Usage
-----
python experiment04-cross-training-viz.py \
    --run-root ./output/SynthRAD-RegGAN-512-keepratio-cross3 \
    --fold 1 \
    --slice 1PA001_0010 \
    --stages 1 2 3 4 \
    --save ./tmp/fold1_slice0010.png

If --slice is omitted, the script will randomly choose one that is present in all
selected stages for the given fold.  The script does *not* modify any training
artifacts; it only reads the saved residuals / weight files.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from typing import Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise staged residual / gap / weight maps.")
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Root directory that contains fold_* outputs (e.g., ./output/...cross3).")
    parser.add_argument("--cv-root", type=Path, required=True,
                        help="Dataset root containing fold_*/train and fold_*/val directories (e.g., data/.../cv_folds).")
    parser.add_argument("--fold", type=int, required=True, help="Fold id to inspect (1-based).")
    parser.add_argument("--stages", type=int, nargs="*", default=None,
                        help="List of stage numbers to plot. Default: all available stages.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice identifier (e.g., 1PA001_0010). If omitted, pick one randomly.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional path to save the visualisation as PNG.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    return parser.parse_args()


def _ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected path does not exist: {path}")
    return path


def _load_np(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _load_truth(data_root: Path, subset: str, slice_name: str) -> Optional[np.ndarray]:
    base = data_root / subset / "B"
    if not base.is_dir():
        return None
    npy_path = base / f"{slice_name}.npy"
    if not npy_path.exists():
        return None
    return _load_np(npy_path)


def _load_stage_summary(weights_dir: Path) -> Dict:
    summary_path = weights_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data.get("cfg", {})


def _collect_stage_paths(fold_root: Path) -> Dict[int, Dict[str, Path]]:
    stages: Dict[int, Dict[str, Path]] = {}
    for stage_dir in sorted(fold_root.glob("stage_*")):
        if not stage_dir.is_dir():
            continue
        try:
            stage_id = int(stage_dir.name.split("_")[-1])
        except ValueError:
            continue
        stages[stage_id] = {
            "stage_dir": stage_dir,
            "weights_dir": stage_dir / "weights",
            "components_dir": stage_dir / "weights" / "components",
            "residual_if_dir": stage_dir / "residuals" / "if" / "maps",
            "residual_oof_dir": stage_dir / "residuals" / "oof" / "maps",
        }
    if not stages:
        raise RuntimeError(f"No stage_* directories found under {fold_root}")
    return stages


def _collect_all_fold_roots(run_root: Path) -> Dict[int, Path]:
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


def _collect_truths(cv_root: Path, fold_ids: List[int], slice_name: str) -> Dict[int, Dict[str, Optional[np.ndarray]]]:
    truths: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for fid in fold_ids:
        fold_dir = _ensure_exists(cv_root / f"fold_{fid:01d}")
        truths[fid] = {
            "train": _load_truth(fold_dir, "train", slice_name),
            "val": _load_truth(fold_dir, "val", slice_name),
        }
    return truths


def _slice_exists(stage_info: Dict[str, Path], slice_name: str) -> bool:
    weights_dir = stage_info["weights_dir"]
    if (weights_dir / f"{slice_name}.npy").exists():
        return True
    if (stage_info["residual_if_dir"] / f"{slice_name}.npy").exists():
        return True
    if (stage_info["residual_oof_dir"] / f"{slice_name}.npy").exists():
        return True
    if (stage_info["components_dir"] / f"{slice_name}.npz").exists():
        return True
    return False


def _select_slice(stages: Dict[int, Dict[str, Path]], stage_ids: List[int], explicit: Optional[str]) -> str:
    if explicit:
        for stage_id in stage_ids:
            if not _slice_exists(stages[stage_id], explicit):
                raise FileNotFoundError(
                    f"slice {explicit} not found in stage {stage_id} (weights/residual directories)."
                )
        return explicit
    # otherwise pick a random slice present in all requested stages
    common: Optional[set[str]] = None
    for stage_id in stage_ids:
        weights_dir = stages[stage_id]["weights_dir"]
        residual_if_dir = stages[stage_id]["residual_if_dir"]
        residual_oof_dir = stages[stage_id]["residual_oof_dir"]
        slices = {
            p.stem for p in weights_dir.glob("*.npy")
        } | {
            p.stem for p in residual_if_dir.glob("*.npy")
        } | {
            p.stem for p in residual_oof_dir.glob("*.npy")
        }
        if not slices:
            continue
        if common is None:
            common = slices
        else:
            common &= slices
    if not common:
        raise RuntimeError("No common slice found across the requested stages.")
    return random.choice(sorted(common))


def _to_uint8(arr: np.ndarray, diverging: bool = False) -> np.ndarray:
    a = np.squeeze(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array for visualisation, got shape {a.shape}")
    if diverging:
        vmax = float(np.max(np.abs(a))) or 1.0
        norm = (a / (2 * vmax)) + 0.5
    else:
        amin, amax = float(np.min(a)), float(np.max(a))
        if amax > amin:
            norm = (a - amin) / (amax - amin + 1e-6)
        else:
            norm = np.zeros_like(a, dtype=np.float32)
    return np.clip(norm, 0.0, 1.0) * 255.0


def _load_stage_artifacts(stage_info: Dict[str, Path], slice_name: str) -> Dict[str, Optional[np.ndarray]]:
    weights_dir = stage_info["weights_dir"]
    components_dir = stage_info["components_dir"]
    residual_if_dir = stage_info["residual_if_dir"]
    residual_oof_dir = stage_info["residual_oof_dir"]

    result: Dict[str, Optional[np.ndarray]] = {}

    weight_path = weights_dir / f"{slice_name}.npy"
    if weight_path.exists():
        print(f"[debug] {stage_info['stage_dir'].name}: weight → {weight_path}")
        result["weight"] = _load_np(weight_path)
    else:
        print(f"[debug] {stage_info['stage_dir'].name}: weight missing for {slice_name}")
        result["weight"] = None

    comp_path = components_dir / f"{slice_name}.npz"
    if comp_path.exists():
        print(f"[debug] {stage_info['stage_dir'].name}: components → {comp_path}")
        comp = np.load(comp_path)
        result["gap"] = comp.get("gap")
        result["c_if"] = comp.get("c_if")
        result["c_oof"] = comp.get("c_oof")
        result["c_gap"] = comp.get("c_gap")
        result["r_if_mean"] = comp.get("r_if_mean")
        result["r_oof"] = comp.get("r_oof")
    else:
        print(f"[debug] {stage_info['stage_dir'].name}: components missing for {slice_name}")
        result["gap"] = None
        result["c_if"] = None
        result["c_oof"] = None
        result["c_gap"] = None
        result["r_if_mean"] = None
        result["r_oof"] = None

    if residual_if_dir.exists():
        if_path = residual_if_dir / f"{slice_name}.npy"
        if if_path.exists():
            print(f"[debug] {stage_info['stage_dir'].name}: residual_if → {if_path}")
            result["residual_if"] = _load_np(if_path)
        else:
            print(f"[debug] {stage_info['stage_dir'].name}: residual_if missing for {slice_name}")
            result["residual_if"] = None
    else:
        result["residual_if"] = None

    if residual_oof_dir.exists():
        oof_path = residual_oof_dir / f"{slice_name}.npy"
        if oof_path.exists():
            print(f"[debug] {stage_info['stage_dir'].name}: residual_oof → {oof_path}")
            result["residual_oof"] = _load_np(oof_path)
        else:
            print(f"[debug] {stage_info['stage_dir'].name}: residual_oof missing for {slice_name}")
            result["residual_oof"] = None
    else:
        result["residual_oof"] = None

    return result

def _build_stage_data(stage_id: int,
                      target_stage: Dict[str, Path],
                      stage_infos_all: Dict[int, Dict[int, Dict[str, Path]]],
                      target_fold: int,
                      other_folds: List[int],
                      truths: Dict[int, Dict[str, Optional[np.ndarray]]],
                      cfg: Dict[str, float],
                      slice_name: str) -> Dict[str, Optional[np.ndarray]]:
    data: Dict[str, Optional[np.ndarray]] = {}
    artefacts = _load_stage_artifacts(target_stage, slice_name)

    # Ground truth references
    truth_target_val = truths[target_fold].get("val")
    truth_target_train = truths[target_fold].get("train")
    truth_target_ref = truth_target_val if truth_target_val is not None else truth_target_train

    residual_oof = artefacts.get("residual_oof")
    if residual_oof is not None and truth_target_val is not None:
        data["pred_oof"] = truth_target_val + residual_oof
    else:
        data["pred_oof"] = None
    data["residual_oof"] = residual_oof

    r_if_mean = artefacts.get("r_if_mean")
    if r_if_mean is not None and truth_target_ref is not None:
        data["pred_if_mean"] = truth_target_ref + r_if_mean
    else:
        data["pred_if_mean"] = None
    data["residual_if_mean"] = r_if_mean

    data["gap"] = artefacts.get("gap")
    data["c_if"] = artefacts.get("c_if")
    data["c_oof"] = artefacts.get("c_oof")
    data["c_gap"] = artefacts.get("c_gap")
    data["weight"] = artefacts.get("weight")

    # Combined confidence if available
    c_if = data.get("c_if")
    c_oof = data.get("c_oof")
    c_gap = data.get("c_gap")
    lam_if = float(cfg.get("lambda_if", 1.0))
    lam_oof = float(cfg.get("lambda_oof", 2.0))
    lam_gap = float(cfg.get("lambda_gap", 1.0))
    denom = lam_if + lam_oof + lam_gap
    if c_if is not None and c_oof is not None and c_gap is not None and denom > 0:
        data["conf_combined"] = (lam_if * c_if + lam_oof * c_oof + lam_gap * c_gap) / denom
    else:
        data["conf_combined"] = None

    # Other folds IF predictions
    for idx, ofid in enumerate(other_folds[:2]):
        stage_map = stage_infos_all.get(ofid, {})
        stage_info = stage_map.get(stage_id)
        key_pred = f"pred_if_fold{ofid:02d}"
        if stage_info is None:
            print(f"[debug] fold {ofid:02d} stage {stage_id:02d} missing for IF prediction")
            data[key_pred] = None
            continue
        if_dir = stage_info["residual_if_dir"]
        if_path = if_dir / f"{slice_name}.npy"
        if not if_path.exists():
            print(f"[debug] fold {ofid:02d} stage {stage_id:02d} residual_if missing for {slice_name}")
            data[key_pred] = None
            continue
        res_if = _load_np(if_path)
        print(f"[debug] fold {ofid:02d} stage {stage_id:02d} residual_if → {if_path}")
        truth_other = truths[ofid].get("train") or truths[ofid].get("val")
        if truth_other is not None:
            data[key_pred] = truth_other + res_if
        else:
            data[key_pred] = None

    return data


def visualise(run_root: Path, cv_root: Path, fold_id: int, stage_ids: List[int], slice_name: str,
              save_path: Optional[Path], show: bool):
    fold_roots = _collect_all_fold_roots(run_root)
    if fold_id not in fold_roots:
        raise RuntimeError(f"Fold {fold_id} not found under {run_root}")
    stage_infos_all = {fid: _collect_stage_paths(root) for fid, root in fold_roots.items()}
    target_stages = stage_infos_all[fold_id]
    if not stage_ids:
        stage_ids = sorted(target_stages.keys())
    slice_name = _select_slice(target_stages, stage_ids, slice_name)
    other_folds = [fid for fid in sorted(stage_infos_all.keys()) if fid != fold_id]

    # Preload truths
    truths = _collect_truths(cv_root, list(stage_infos_all.keys()), slice_name)

    print(f"[viz] fold={fold_id} slice={slice_name} stages={stage_ids} others={other_folds}")

    stage_datas = {}
    for stage_id in stage_ids:
        target_stage = target_stages.get(stage_id)
        if target_stage is None:
            continue
        cfg = _load_stage_summary(target_stage["weights_dir"])
        stage_data = _build_stage_data(
            stage_id,
            target_stage,
            stage_infos_all,
            fold_id,
            other_folds,
            truths,
            cfg,
            slice_name,
        )
        stage_datas[stage_id] = stage_data

    # Decide rows dynamically based on available other folds
    row_def: List[Tuple[str, str]] = [("pred_oof", "OOF Prediction")]
    for idx, ofid in enumerate(other_folds[:2]):
        row_def.append((f"pred_if_fold{ofid:02d}", f"IF Pred (fold {ofid:02d})"))
    row_def.extend([
        ("pred_if_mean", "IF Mean Prediction"),
        ("residual_oof", "OOF Residual"),
        ("residual_if_mean", "IF Residual (mean)"),
        ("gap", "Gap Map"),
        ("c_if", "Confidence IF"),
        ("c_oof", "Confidence OOF"),
        ("c_gap", "Confidence GAP"),
        ("conf_combined", "Confidence Combined"),
        ("weight", "Loss Weight"),
    ])

    n_rows = len(row_def)
    n_cols = len(stage_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, stage_id in enumerate(stage_ids):
        if stage_id not in stage_datas:
            continue
        artefacts = stage_datas[stage_id]
        for row, (key, title) in enumerate(row_def):
            ax = axes[row, col]
            arr = artefacts.get(key)
            if arr is None:
                ax.axis("off")
                continue
            diverging = key in {"gap", "residual_oof", "residual_if_mean"}
            img = _to_uint8(arr, diverging=diverging)
            cmap = "gray" if key not in {"gap"} else "coolwarm"
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"Stage {stage_id}", fontsize=12)

    fig.suptitle(f"Fold {fold_id} | Slice {slice_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[viz] saved figure to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = _parse_args()
    visualise(
        run_root=args.run_root,
        cv_root=args.cv_root,
        fold_id=args.fold,
        stage_ids=args.stages,
        slice_name=args.slice,
        save_path=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()

"""
python experiment04-cross-training-viz.py \
   --run-root output/SynthRAD-RegGAN-512-keepratio-cross3 \
   --cv-root data/SynthRAD2023-Task1/cv_folds \
   --fold 1 --stages 1 2 3 4 5 6 7 8 9 10 11 --slice 1PA035_0102 \
   --save tmp/fold1_slice0010.png
   
python experiment04-cross-training-viz.py \
   --run-root output/SynthRAD-RegGAN-512-keepratio-cross3 \
   --cv-root data/SynthRAD2023-Task1/cv_folds \
   --fold 1 --stages 1 2 3 4 5 6 7 8 9 10 11 --slice 1PA059_0110 \
   --save tmp/fold1_slice0010.png
"""
