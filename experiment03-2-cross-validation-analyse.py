#!/usr/bin/env python3
"""Analyse cross-validation residuals to highlight learnable vs misaligned regions.

Given directories containing in-fold (training) residual maps ``r_if`` and the
corresponding out-of-fold residual maps ``r_oof`` (evaluated with a model that
has *not* seen the sample), this script classifies each pixel into three
categories:

1. **Learnable (Type 1)** — model generalises well (``r_oof`` low and close to
   ``r_if``).
2. **Overfit noise (Type 2)** — training residual low but generalisation gap
   high (``r_oof`` significantly larger than ``r_if``).
3. **Unlearnt / misaligned (Type 3)** — both residuals stay high.

For every slice we generate a colour overlay (green / yellow / red masks) on top
of the reference target image, store the per-pixel ratios in a CSV, and persist
optional artefacts such as the gap map. When multiple in-fold residual folders
are provided, the script aggregates them (mean by default; median/min/max
available) before comparing to the OOF residual. Thresholds are configurable so
the experiment can be tuned to different residual scales. A minimum-area filter
can optionally remove tiny Type-3 islands that often arise from noise.

Example usage (fold-3 analysis)::

    python experiment03-2-cross-validation-analyse.py \
        --if-residual-dirs /path/to/fold3/modelA_if,/path/to/fold3/modelB_if \
        --oof-residual-dir /path/to/fold3/oof_residuals \
        --target-dir data/SynthRAD2023-Task1/train2D-foreground/B \
        --output-dir analysis/fold3 \
        --if-threshold 0.03 --oof-threshold 0.05 --gap-threshold 0.03 \
        --min-type3-pixels 100

The script relies on ``fire.Fire`` for the CLI and ``rich.progress`` for
feedback.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fire
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

try:
    import cv2
except Exception as exc:  # pragma: no cover - OpenCV should be available in this project
    raise RuntimeError("OpenCV (cv2) is required for visualisation.") from exc

try:
    from matplotlib import cm as mpl_cm
except Exception:
    mpl_cm = None

try:
    import seaborn as sns
except Exception:
    sns = None

_OPENCV_CMAPS: Dict[str, int] = {
    name.replace('COLORMAP_', '').lower(): getattr(cv2, name)
    for name in dir(cv2)
    if name.startswith('COLORMAP_') and isinstance(getattr(cv2, name), int)
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_array(path: Path) -> np.ndarray:
    """Load a 2D array from .npy/.npz/.png/.jpg/.jpeg."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if not data.files:
                raise ValueError(f"Empty npz file: {path}")
            arr = data[data.files[0]]
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = img
    else:
        raise ValueError(f"Unsupported file format for {path}")

    arr = np.asarray(arr).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {path}, got shape {arr.shape}")
    return arr


def _normalise_to_uint8(
    arr: np.ndarray,
    eps: float = 1e-6,
    mask: Optional[np.ndarray] = None,
    diverging: bool = False,
) -> np.ndarray:
    """Scale array to [0, 255] for visualisation."""

    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    if mask is not None:
        valid &= mask.astype(bool)
    if not np.any(valid):
        return np.full_like(a, 128, dtype=np.uint8)

    subset = a[valid]
    if diverging:
        max_abs = float(np.max(np.abs(subset)))
        if max_abs < eps:
            norm = np.full_like(a, 0.5, dtype=np.float32)
        else:
            norm = (a / (2.0 * max_abs)) + 0.5
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.where(valid, norm, 0.5)
        return (norm * 255.0).astype(np.uint8)

    amin, amax = float(np.min(subset)), float(np.max(subset))
    if math.isclose(amin, amax, abs_tol=eps):
        return np.zeros_like(a, dtype=np.uint8)
    norm = (a - amin) / (amax - amin)
    norm = np.clip(norm, 0.0, 1.0)
    norm = np.where(valid, norm, 0.0)
    return (norm * 255.0).astype(np.uint8)


def _apply_colormap(
    arr: np.ndarray,
    cmap: int = cv2.COLORMAP_TURBO,
    cmap_name: Optional[str] = None,
    diverging: bool = False,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    norm = _normalise_to_uint8(arr, mask=valid_mask, diverging=diverging)

    if cmap_name is not None:
        name = cmap_name.lower()
        if name in _OPENCV_CMAPS:
            return cv2.applyColorMap(norm, _OPENCV_CMAPS[name])
        if sns is not None:
            try:
                palette = np.asarray(sns.color_palette(name, 256), dtype=np.float32)
                rgb = palette[norm]
                bgr = np.clip(rgb[..., ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
                return bgr
            except Exception:
                pass
        if mpl_cm is not None:
            cmap_obj = mpl_cm.get_cmap(cmap_name)
            rgba = cmap_obj(norm.astype(np.float32) / 255.0, bytes=True)
            rgb = rgba[..., :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        raise RuntimeError(
            f"Colormap '{cmap_name}' is not available. Install matplotlib/seaborn or choose an OpenCV colormap."
        )

    colored = cv2.applyColorMap(norm, cmap)
    return colored


def _compose_overlay(
    base: np.ndarray,
    masks: Sequence[np.ndarray],
    colours: Sequence[Tuple[int, int, int]],
    alpha: float = 0.15,
) -> np.ndarray:
    """Blend coloured masks onto a base grayscale image."""

    base_uint8 = _normalise_to_uint8(base)
    base_rgb = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    out = base_rgb.copy()
    for mask, colour in zip(masks, colours):
        if mask is None:
            continue
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not np.any(mask):
            continue
        colour_arr = np.array(colour, dtype=np.float32) / 255.0
        mask_f = mask.astype(np.float32)
        out = out * (1.0 - alpha * mask_f[..., None]) + colour_arr * (alpha * mask_f[..., None])

    out_uint8 = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out_uint8


def _filter_small_components(
    mask: np.ndarray,
    min_pixels: int,
    connectivity: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove connected components smaller than ``min_pixels``.

    Returns the filtered mask and a boolean mask of the removed pixels.
    """

    if min_pixels <= 0:
        return mask.astype(bool), np.zeros_like(mask, dtype=bool)

    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity)
    if num_labels <= 1:
        return mask.astype(bool), np.zeros_like(mask, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    small_labels = np.where(areas < min_pixels)[0] + 1
    if small_labels.size == 0:
        return mask.astype(bool), np.zeros_like(mask, dtype=bool)

    removed = np.isin(labels, small_labels)
    filtered = mask.astype(bool) & ~removed
    return filtered, removed


def _resolve_target_path(slice_name: str, target_dir: Optional[Path], search_exts: Sequence[str]) -> Optional[Path]:
    if target_dir is None:
        return None
    candidates = []
    # 1. Exact filename match
    for ext in search_exts:
        candidates.append(target_dir / f"{slice_name}{ext}")
    # 2. If slice_name already includes an extension (e.g. 'foo.npy'), try as-is
    path_with_ext = target_dir / slice_name
    candidates.append(path_with_ext)

    for cand in candidates:
        if cand.exists():
            return cand
    return None


@dataclass
class SliceMetrics:
    name: str
    pixels: int
    ratio_type1: float
    ratio_type2: float
    ratio_type3: float
    mean_if: float
    mean_oof: float
    mean_gap: float
    median_gap: float
    max_gap: float
    std_gap: float
    mean_if_std: float
    max_if_std: float


def _classify_pixels(
    r_if: np.ndarray,
    r_oof: np.ndarray,
    valid_mask: np.ndarray,
    if_threshold: float,
    oof_threshold: float,
    gap_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return masks for Type1, Type2, Type3 (no leftovers)."""

    if r_if.shape != r_oof.shape:
        raise ValueError("r_if and r_oof must share the same shape")

    gap = r_oof - r_if

    valid = valid_mask.astype(bool)

    type2 = (gap >= gap_threshold) & valid
    base_mask = ~type2 & valid
    type1 = (r_if <= if_threshold) & (r_oof <= oof_threshold) & base_mask
    type3 = valid & ~(type1 | type2)
    return type1, type2, type3


def _write_csv(path: Path, rows: List[SliceMetrics]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "slice",
            "pixels_valid",
            "ratio_type1",
            "ratio_type2",
            "ratio_type3",
            "mean_r_if",
            "mean_r_oof",
            "mean_gap",
            "median_gap",
            "max_gap",
            "std_gap",
            "mean_if_std",
            "max_if_std",
        ])
        for row in rows:
            writer.writerow([
                row.name,
                row.pixels,
                f"{row.ratio_type1:.6f}",
                f"{row.ratio_type2:.6f}",
                f"{row.ratio_type3:.6f}",
                f"{row.mean_if:.6f}",
                f"{row.mean_oof:.6f}",
                f"{row.mean_gap:.6f}",
                f"{row.median_gap:.6f}",
                f"{row.max_gap:.6f}",
                f"{row.std_gap:.6f}",
                f"{row.mean_if_std:.6f}",
                f"{row.max_if_std:.6f}",
            ])

    # Optional overall summary row
    if rows:
        total_pixels = sum(r.pixels for r in rows)
        if total_pixels > 0:
            type1_pixels = sum(r.ratio_type1 * r.pixels for r in rows)
            type2_pixels = sum(r.ratio_type2 * r.pixels for r in rows)
            type3_pixels = sum(r.ratio_type3 * r.pixels for r in rows)
            with path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([])
                writer.writerow([
                    "overall",
                    total_pixels,
                    f"{type1_pixels / total_pixels:.6f}",
                    f"{type2_pixels / total_pixels:.6f}",
                    f"{type3_pixels / total_pixels:.6f}",
                    "", "", "", "", "", "", "",
                ])


# -----------------------------------------------------------------------------
# Main analysis entry-point
# -----------------------------------------------------------------------------


def _parse_if_dirs(if_residual_dirs: Iterable[str]) -> List[Path]:
    if isinstance(if_residual_dirs, str):
        iterable = [if_residual_dirs]
    else:
        iterable = list(if_residual_dirs)

    dirs: List[Path] = []
    for entry in iterable:
        if entry is None:
            continue
        parts = []
        if isinstance(entry, str) and ("," in entry or os.pathsep in entry):
            for token in entry.replace(os.pathsep, ",").split(","):
                token = token.strip()
                if token:
                    parts.append(token)
        elif isinstance(entry, str):
            parts.append(entry.strip())
        else:
            parts.append(str(entry))
        for p in parts:
            if p:
                dirs.append(Path(p))
    if not dirs:
        raise ValueError("At least one in-fold residual directory must be provided.")
    return dirs


def analyse(
    if_residual_dirs: Iterable[str],
    oof_residual_dir: str,
    target_dir: Optional[str],
    output_dir: str,
    if_threshold: float = 0.03,
    oof_threshold: float = 0.05,
    gap_threshold: float = 0.03,
    min_type3_pixels: int = 0,
    background_val: float = -1.0,
    alpha: float = 0.15,
    mask_exts: Sequence[str] = (".npy", ".npz", ".png", ".jpg", ".jpeg"),
    residual_exts: Sequence[str] = (".npy", ".npz"),
    aggregate: str = "mean",
    save_if_mean: bool = False,
    save_if_std: bool = False,
    save_visuals: bool = False,
    colormap: str = "turbo",
    save_masks: bool = False,
    save_gap: bool = False,
) -> str:
    """Analyse residuals for one fold and export overlays + CSV summary.

    Args:
        if_residual_dirs: Comma/``os.pathsep``-separated string or iterable of
            directories containing in-fold residual maps (each produced by a
            model that has seen the sample).
        oof_residual_dir: Directory with OOF residual maps (``r_oof``).
        target_dir: Directory containing the ground-truth target slices (for
            visualisation). Pass ``None`` to skip background imagery.
        output_dir: Root where overlays/CSVs will be written.
        if_threshold, oof_threshold, gap_threshold: thresholds controlling the
            three-way classification (in absolute residual units).
        min_type3_pixels: drop Type3 connected components smaller than this
            size (pixel count) before computing metrics/overlays.
        background_val: sentinel value representing background in target slices.
        alpha: overlay opacity for the coloured masks.
        mask_exts: extensions to probe when resolving target slices.
        aggregate: how to collapse multiple in-fold residual maps ('mean',
            'median', 'min', 'max').
        save_if_mean: save the aggregated in-fold residual map per slice.
        save_if_std: save the per-pixel standard deviation across in-fold maps.
        save_visuals: if True, also export colourised PNGs for gap and in-fold
            mean residuals using ``colormap``.
        colormap: OpenCV colormap name (e.g. 'turbo', 'viridis', 'jet') used for
            visualisations when ``save_visuals`` is enabled.
        save_masks: if True, persist per-slice boolean masks for each type.
        save_gap: if True, persist ``gap = r_oof - r_if`` as .npy alongside the
            overlays.
    """

    if_dirs = _parse_if_dirs(if_residual_dirs)
    oof_dir = Path(oof_residual_dir)
    tgt_dir = Path(target_dir) if target_dir else None
    out_root = Path(output_dir)
    overlay_dir = out_root / "overlays"
    masks_dir = out_root / "masks"
    gap_dir = out_root / "gaps"
    if_mean_dir = out_root / "if_mean"
    if_std_dir = out_root / "if_std"
    vis_dir = out_root / "visuals"
    summary_csv = out_root / "slice_metrics.csv"

    min_type3_pixels = max(int(min_type3_pixels), 0)

    for if_dir in if_dirs:
        if not if_dir.is_dir():
            raise FileNotFoundError(f"In-fold residual directory not found: {if_dir}")
    if not oof_dir.is_dir():
        raise FileNotFoundError(f"OOF residual directory not found: {oof_dir}")
    if tgt_dir is not None and not tgt_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {tgt_dir}")

    aggregate_mode = str(aggregate).strip().lower()
    allowed_modes = {"mean", "median", "min", "max"}
    if aggregate_mode not in allowed_modes:
        raise ValueError(f"Unsupported aggregate mode '{aggregate}'. Choose from {sorted(allowed_modes)}.")

    overlay_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
    if save_gap:
        gap_dir.mkdir(parents=True, exist_ok=True)
    if save_if_mean:
        if_mean_dir.mkdir(parents=True, exist_ok=True)
    if save_if_std:
        if_std_dir.mkdir(parents=True, exist_ok=True)
    if save_visuals:
        vis_dir.mkdir(parents=True, exist_ok=True)
        (vis_dir / "gap").mkdir(parents=True, exist_ok=True)
        (vis_dir / "if_mean").mkdir(parents=True, exist_ok=True)

    cmap_name = str(colormap or "turbo")
    cmap_cv2 = _OPENCV_CMAPS.get(cmap_name.lower(), cv2.COLORMAP_TURBO)

    def _list_files(root: Path, allowed_exts: Sequence[str]) -> Dict[str, Path]:
        allowed = tuple(e.lower() for e in allowed_exts)
        out: Dict[str, Path] = {}
        for p in root.glob("*"):
            if not p.is_file():
                continue
            if allowed and p.suffix.lower() not in allowed:
                continue
            out[p.name] = p
        return out

    if_file_maps = [_list_files(d, residual_exts) for d in if_dirs]
    oof_files = _list_files(oof_dir, residual_exts)

    common_names = set(oof_files.keys())
    for fmap in if_file_maps:
        common_names &= set(fmap.keys())
    common_names = sorted(common_names)
    if not common_names:
        raise RuntimeError("No overlapping filenames between in-fold and OOF residual directories.")

    rows: List[SliceMetrics] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Analysing slices", total=len(common_names))
        for filename in common_names:
            progress.update(task, description=f"Processing {filename}")

            stacked_if = []
            for fmap in if_file_maps:
                stacked_if.append(_load_array(fmap[filename]))
            r_if_stack = np.stack(stacked_if, axis=0)
            r_oof = _load_array(oof_files[filename])
            if r_if_stack.shape[1:] != r_oof.shape:
                raise ValueError(
                    f"Shape mismatch for {filename}: in-fold {r_if_stack.shape[1:]} vs OOF {r_oof.shape}"
                )

            if aggregate_mode == "median":
                r_if = np.median(r_if_stack, axis=0)
            elif aggregate_mode == "max":
                r_if = np.max(r_if_stack, axis=0)
            elif aggregate_mode == "min":
                r_if = np.min(r_if_stack, axis=0)
            else:
                r_if = np.mean(r_if_stack, axis=0)

            r_if_std = r_if_stack.std(axis=0)

            base = None
            if tgt_dir is not None:
                base_path = _resolve_target_path(filename, tgt_dir, mask_exts)
                if base_path is None:
                    base_path = _resolve_target_path(Path(filename).stem, tgt_dir, mask_exts)
                if base_path is not None:
                    base = _load_array(base_path)

            if base is None:
                # fallback: visualise r_oof to provide context
                base = r_oof

            valid_mask = np.ones_like(r_if, dtype=bool)
            if base is not None and np.issubdtype(base.dtype, np.floating):
                valid_mask &= base > background_val

            type1, type2, type3 = _classify_pixels(
                r_if,
                r_oof,
                valid_mask,
                if_threshold=float(if_threshold),
                oof_threshold=float(oof_threshold),
                gap_threshold=float(gap_threshold),
            )

            removed_small = None
            if min_type3_pixels > 0:
                type3, removed_small = _filter_small_components(type3, min_type3_pixels)
                if np.any(removed_small):
                    valid_mask = valid_mask & ~removed_small
                    type1 = type1 & valid_mask
                    type2 = type2 & valid_mask
                    type3 = type3 & valid_mask

            overlay = _compose_overlay(
                base,
                (type1, type2, type3),
                colours=((0, 255, 0), (0, 255, 255), (0, 0, 255)),  # BGR: green, yellow, red
                alpha=float(alpha),
            )
            cv2.imwrite(str(overlay_dir / f"{Path(filename).stem}_overlay.png"), overlay)

            gap = r_oof - r_if
            if save_masks:
                np.save(masks_dir / f"{Path(filename).stem}_type1.npy", type1.astype(np.uint8))
                np.save(masks_dir / f"{Path(filename).stem}_type2.npy", type2.astype(np.uint8))
                np.save(masks_dir / f"{Path(filename).stem}_type3.npy", type3.astype(np.uint8))

            if save_gap:
                np.save(gap_dir / f"{Path(filename).stem}_gap.npy", gap.astype(np.float32))

            if save_if_mean:
                np.save(if_mean_dir / f"{Path(filename).stem}_if_mean.npy", r_if.astype(np.float32))

            if save_if_std:
                np.save(if_std_dir / f"{Path(filename).stem}_if_std.npy", r_if_std.astype(np.float32))

            if save_visuals:
                gap_vis = _apply_colormap(gap, cmap=cmap_cv2, cmap_name=cmap_name, diverging=True, valid_mask=valid_mask)
                if_mean_vis = _apply_colormap(r_if, cmap=cmap_cv2, cmap_name=cmap_name, valid_mask=valid_mask)
                bg_idx = ~valid_mask
                if np.any(bg_idx):
                    gap_vis[bg_idx] = (255, 255, 255)
                    if_mean_vis[bg_idx] = (255, 255, 255)
                cv2.imwrite(str(vis_dir / "gap" / f"{Path(filename).stem}_gap.png"), gap_vis)
                cv2.imwrite(str(vis_dir / "if_mean" / f"{Path(filename).stem}_if_mean.png"), if_mean_vis)

            valid_pixels = int(valid_mask.sum())
            if valid_pixels == 0:
                ratios = (0.0, 0.0, 0.0)
            else:
                ratios = (
                    float(type1.sum() / valid_pixels),
                    float(type2.sum() / valid_pixels),
                    float(type3.sum() / valid_pixels),
                )

            masked_gap = gap[valid_mask]
            mean_if = float(r_if[valid_mask].mean()) if valid_pixels > 0 else 0.0
            mean_oof = float(r_oof[valid_mask].mean()) if valid_pixels > 0 else 0.0
            mean_gap = float(masked_gap.mean()) if valid_pixels > 0 else 0.0
            median_gap = float(np.median(masked_gap)) if valid_pixels > 0 else 0.0
            max_gap = float(masked_gap.max()) if valid_pixels > 0 else 0.0
            std_gap = float(masked_gap.std()) if valid_pixels > 0 else 0.0
            mean_if_std = float(r_if_std[valid_mask].mean()) if valid_pixels > 0 else 0.0
            max_if_std = float(r_if_std[valid_mask].max()) if valid_pixels > 0 else 0.0

            rows.append(
                SliceMetrics(
                    name=Path(filename).stem,
                    pixels=valid_pixels,
                    ratio_type1=ratios[0],
                    ratio_type2=ratios[1],
                    ratio_type3=ratios[2],
                    mean_if=mean_if,
                    mean_oof=mean_oof,
                    mean_gap=mean_gap,
                    median_gap=median_gap,
                    max_gap=max_gap,
                    std_gap=std_gap,
                    mean_if_std=mean_if_std,
                    max_if_std=max_if_std,
                )
            )

            progress.advance(task)

    _write_csv(summary_csv, rows)

    return f"Analysis completed. Outputs stored in {out_root}"


def main(**kwargs):
    return analyse(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)

"""
python experiment03-2-cross-validation-analyse.py \
    --if-residual-dirs 'output/SynthRAD-RegGAN-512-keepratio-bestparams-foreground-nomask-fold1/img-val-fold3-without-mask/residual','output/SynthRAD-RegGAN-512-keepratio-bestparams-foreground-nomask-fold2/img-val-fold3-without-mask/residual' \
    --oof-residual-dir 'output/SynthRAD-RegGAN-512-keepratio-bestparams-foreground-nomask-fold3/img-val-fold3-without-mask/residual' \
    --target-dir data/SynthRAD2023-Task1/train2D-foreground/B \
    --output-dir experiment-results/03-cross-validation/fold3-as-if \
    --if-threshold 0.06 --oof-threshold 0.06 --gap-threshold 0.04 \
    --min-type3-pixels 150 \
    --save-masks --save-gap --save-if-mean --save-if-std --save-visuals --colormap vlag
"""
