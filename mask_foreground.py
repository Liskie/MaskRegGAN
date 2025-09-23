#!/usr/bin/env python3
"""Utility to convert near-background pixels in medical slices to the sentinel value -1."""

import argparse
import sys
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TextColumn
from skimage.morphology import remove_small_holes, remove_small_objects, reconstruction
from scipy.ndimage import binary_fill_holes, binary_dilation


def _build_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def detect_foreground(
    arr: np.ndarray,
    threshold: float = None,
    method: str = "otsu",
    percentile: float = 5.0,
    open_radius: int = 0,
    close_radius: int = 0,
    hyst_high: float = 0.6,
    hyst_low_ratio: float = 0.5,
    hyst_min_area_frac: float = 0.001,
    hyst_max_components: int = 2,
) -> np.ndarray:
    """Return a boolean mask indicating foreground pixels."""

    data = np.asarray(arr, dtype=np.float32)
    # Flatten spatial dims but keep channels if present
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]

    norm = np.clip((data + 1.0) * 0.5, 0.0, 1.0)
    valid = np.isfinite(norm)
    if not np.any(valid):
        return np.zeros_like(norm, dtype=bool)

    if method == "body_hysteresis":
        mask_bool = _detect_body_hysteresis(
            norm,
            valid,
            high_percentile=hyst_high,
            low_ratio=hyst_low_ratio,
            min_area_frac=hyst_min_area_frac,
            max_components=hyst_max_components,
        )
    else:
        if threshold is None:
            if method == "percentile":
                perc = float(percentile)
                thr = np.percentile(norm[valid], perc)
            else:  # default to otsu
                thr = threshold_otsu(norm[valid]) if np.any(valid) else 0.0
        else:
            thr = float(threshold)

        mask_bool = (norm > thr) & valid

    mask = mask_bool.astype(np.uint8)
    if open_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _build_kernel(open_radius))
    if close_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _build_kernel(close_radius))

    return mask.astype(bool)


def _detect_body_hysteresis(
    norm_img: np.ndarray,
    valid_mask: np.ndarray,
    high_percentile: float = 0.6,
    low_ratio: float = 0.5,
    min_area_frac: float = 0.001,
    max_components: int = 2,
) -> np.ndarray:
    """Detect foreground via smoothed hysteresis thresholding focused on the largest body component."""

    # Smooth to reduce noise and small dark spots
    smoothed = cv2.GaussianBlur(norm_img.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)

    # Estimate thresholds from valid pixels
    vals = smoothed[valid_mask]
    if vals.size == 0:
        return np.zeros_like(norm_img, dtype=bool)
    high_percent = float(np.clip(high_percentile, 0.0, 1.0)) * 100.0
    high_thr = np.percentile(vals, high_percent)
    low_thr = high_thr * float(np.clip(low_ratio, 0.05, 0.95))

    # Hysteresis: strong seeds above high_thr, grow into weak pixels above low_thr when connected
    strong = smoothed >= high_thr
    weak = smoothed >= low_thr
    strong &= valid_mask
    weak &= valid_mask

    if not np.any(strong):
        strong = smoothed >= np.percentile(vals, max(5.0, high_percent * 0.5))
        strong &= valid_mask

    recon = reconstruction(strong.astype(np.float32), weak.astype(np.float32), method='dilation')
    body_mask = recon > 0

    # Remove small objects and keep the largest connected component
    min_area = max(64, int(body_mask.size * float(np.clip(min_area_frac, 0.0, 0.1))))
    body_mask = remove_small_objects(body_mask, min_size=min_area)

    # Fill holes inside body (any cavities not connected to background)
    body_mask = binary_fill_holes(body_mask)
    body_mask = remove_small_holes(body_mask, area_threshold=min_area)

    if body_mask.ndim == 2 and np.any(body_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(body_mask.astype(np.uint8), connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            order = np.argsort(areas)[::-1]
            keep = []
            max_components = max(1, int(max_components))
            for idx in order:
                lbl = 1 + idx
                if areas[idx] >= min_area or not keep:
                    keep.append(lbl)
                if len(keep) >= max_components:
                    break
            mask = np.zeros_like(labels, dtype=bool)
            for lbl in keep:
                mask |= labels == lbl
            body_mask = mask
    return body_mask.astype(bool)


def apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        mask_b = mask[None, ...]
    else:
        mask_b = mask
    out = np.where(mask_b, out, -1.0)
    return out.astype(np.float32, copy=False)


def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*.npy")):
        if p.is_file():
            yield p


def save_visualization(rel_path: Path, arr: np.ndarray, mask: np.ndarray, out_dir: Path, alpha: float = 0.3):
    target = (out_dir / rel_path).with_suffix('.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    norm = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[0] == 1:
        img_rgb = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = np.repeat(img[..., None], 3, axis=-1)

    overlay = np.zeros_like(img_rgb, dtype=np.float32)
    overlay[..., 0] = 255.0  # R
    overlay[..., 1] = 255.0  # G
    # convert mask to HxW
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask_2d = mask[0]
    else:
        mask_2d = mask
    mask_inv = (~mask_2d).astype(np.float32)
    overlay = overlay * mask_inv[..., None]

    alpha_clamped = np.clip(alpha, 0.0, 1.0)
    weight = alpha_clamped * mask_inv[..., None]
    blended = img_rgb.astype(np.float32) * (1.0 - weight) + overlay * weight
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    cv2.imwrite(str(target), blended[:, :, ::-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert low-intensity background pixels to -1 sentinel")
    p.add_argument("input", type=Path, help="Path to a .npy slice or a directory of slices")
    p.add_argument("output", type=Path, help="Directory to save processed slices")
    p.add_argument("--viz-dir", type=Path, default=None, help="Directory to save visualization PNGs")
    p.add_argument("--viz-alpha", type=float, default=0.3, help="Alpha for yellow overlay on background regions")
    p.add_argument("--threshold", type=float, default=None, help="Manual threshold in [0,1] after normalization")
    p.add_argument("--method", choices=["otsu", "percentile", "body_hysteresis"], default="otsu", help="Foreground detection method")
    p.add_argument("--percentile", type=float, default=5.0, help="Percentile used when method=percentile")
    p.add_argument("--open-radius", type=int, default=0, help="Morphological opening radius in pixels")
    p.add_argument("--close-radius", type=int, default=1, help="Morphological closing radius in pixels")
    p.add_argument("--dilate-radius", type=int, default=0, help="Dilate foreground mask by this radius (pixels)")
    p.add_argument("--hyst-high", type=float, default=0.6, help="High percentile (0-1) for body_hysteresis seeds")
    p.add_argument("--hyst-low-ratio", type=float, default=0.4, help="Low/High threshold ratio for hysteresis growth")
    p.add_argument("--hyst-min-area-frac", type=float, default=0.001, help="Minimum component area fraction for body mask")
    p.add_argument("--hyst-max-components", type=int, default=2, help="Maximum number of body components to keep")
    p.add_argument("--bad-cases", type=Path, default=Path("data/SynthRAD2023-Task1/bad_cases.txt"), help="Path to file listing slices to skip (PID_slice)")
    return p.parse_args()


def main():
    args = parse_args()
    src_path = args.input
    out_root = args.output
    if src_path.is_dir():
        out_root.mkdir(parents=True, exist_ok=True)
        base_path = src_path
    elif src_path.is_file():
        out_root.mkdir(parents=True, exist_ok=True)
        base_path = src_path.parent
    else:
        raise FileNotFoundError(f"Input path not found: {src_path}")

    if args.viz_dir is not None:
        args.viz_dir.mkdir(parents=True, exist_ok=True)

    bad_cases = set()
    if args.bad_cases is not None and args.bad_cases.exists():
        with open(args.bad_cases, 'r') as fh:
            for line in fh:
                entry = line.strip()
                if entry:
                    bad_cases.add(entry)

    files = list(iter_input_files(src_path))
    total = len(files)
    processed = 0

    progress_columns = (
        TextColumn("[cyan]mask_foreground[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with Progress(*progress_columns) as progress:
        task = progress.add_task("process", total=total if total else None)
        for src in files:
            arr = np.load(src)
            name = src.name
            stem = src.stem
            matched = False
            if stem in bad_cases:
                matched = True
            else:
                m = re.search(r'([A-Za-z0-9]+)(?:\.nii)?_z(\d+)', name)
                if m:
                    key = f"{m.group(1)}_z{m.group(2)}"
                    matched = key in bad_cases
            if matched:
                progress.advance(task, 1)
                continue
            mask = detect_foreground(
                arr,
                threshold=args.threshold,
                method=args.method,
                percentile=args.percentile,
                open_radius=args.open_radius,
                close_radius=args.close_radius,
                hyst_high=args.hyst_high,
                hyst_low_ratio=args.hyst_low_ratio,
                hyst_min_area_frac=args.hyst_min_area_frac,
                hyst_max_components=args.hyst_max_components,
            )
            if args.dilate_radius > 0:
                structure = (_build_kernel(args.dilate_radius) > 0).astype(bool)
                mask = binary_dilation(mask, structure=structure)
            cleaned = apply_mask(arr, mask)

            rel_path = src.relative_to(base_path)
            dst = out_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, cleaned)

            if args.viz_dir is not None:
                save_visualization(rel_path, arr, mask, args.viz_dir, alpha=args.viz_alpha)

            processed += 1
            progress.advance(task, 1)

    print(f"Processed {processed} file(s).", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
python mask_foreground.py 'data/SynthRAD2023-Task1/train2D/A' 'data/SynthRAD2023-Task1/train2D-foreground/A' \
    --viz-dir 'data/SynthRAD2023-Task1/train2D-foreground/A-viz' --viz-alpha 0.3 --method body_hysteresis \
    --hyst-high 0.65 --hyst-low-ratio 0.35 --hyst-min-area-frac 0.05 --dilate-radius 3

python mask_foreground.py 'data/SynthRAD2023-Task1/train2D/B' 'data/SynthRAD2023-Task1/train2D-foreground/B' \
    --viz-dir 'data/SynthRAD2023-Task1/train2D-foreground/B-viz' --viz-alpha 0.3 --method body_hysteresis \
    --hyst-high 0.65 --hyst-low-ratio 0.35 --hyst-min-area-frac 0.05 --dilate-radius 3
    
python mask_foreground.py 'data/SynthRAD2023-Task1/test2D/A' 'data/SynthRAD2023-Task1/test2D-foreground/A' \
    --viz-dir 'data/SynthRAD2023-Task1/test2D-foreground/A-viz' --viz-alpha 0.3 --method body_hysteresis \
    --hyst-high 0.65 --hyst-low-ratio 0.35 --hyst-min-area-frac 0.05 --dilate-radius 3
    
python mask_foreground.py 'data/SynthRAD2023-Task1/test2D/B' 'data/SynthRAD2023-Task1/test2D-foreground/B' \
    --viz-dir 'data/SynthRAD2023-Task1/test2D-foreground/B-viz' --viz-alpha 0.3 --method body_hysteresis \
    --hyst-high 0.65 --hyst-low-ratio 0.35 --hyst-min-area-frac 0.05 --dilate-radius 3
"""
