#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotation tool for unregable regions.
--------------------------------------
- Loads validation dataset images (y_true).
- Allows the user to draw multiple polygons per slice and assign categories.
- Saves all annotations for a slice into a .npz file (patches, masks, categories, meta).
- Also saves a visualization PNG with polygons overlaid.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import fire

from trainer.utils import ResizeKeepRatioPad
from trainer.datasets import ValDataset

PATTERN_TYPES = [
    "gas_ring",
    "bowel_wall_patch",
    "interleg_gray",
    "outer_fat_bulge",
    "bladder_shape",
]

# Fixed color map for categories
CATEGORY_COLOR = {
    "gas_ring": (255, 0, 0),          # red
    "bowel_wall_patch": (255, 165, 0),# orange
    "interleg_gray": (0, 255, 255),   # cyan
    "outer_fat_bulge": (255, 0, 255), # magenta
    "bladder_shape": (255, 255, 0),   # yellow
}

class MultiPolygonAnnotator:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        # Draw polygon with red edge and red handles (compatible with Matplotlib PolygonSelector props)
        selector_props = dict(color='r', linestyle='-', linewidth=2, alpha=0.8)
        handle_props = dict(markeredgecolor='r', markerfacecolor='r')
        self.current_selector = PolygonSelector(
            ax,
            self.onselect,
            useblit=False,
            props=selector_props,
            handle_props=handle_props,
        )
        self.current_mask = None
        self.current_verts = None

    def onselect(self, verts):
        path = Path(verts)
        ny, nx = self.image.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
        mask = path.contains_points(coords).reshape((ny, nx))
        self.current_mask = mask
        self.current_verts = np.array(verts, dtype=np.float32)
        print(f"[info] Polygon mask created, area={mask.sum()} pixels")

    def get_current_mask(self):
        return self.current_mask

    def get_current_verts(self):
        return self.current_verts

def annotate_slice(image_y, image_x, slice_id, out_dir):
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(10, 5))
    ax_x.imshow(image_x, cmap='gray')
    ax_x.set_title(f"x (A)")
    ax_x.axis('off')
    ax_y.imshow(image_y, cmap='gray')
    ax_y.set_title(f"y_true (B) – Annotate here: {slice_id}")
    ax_y.axis('off')

    patches, masks, categories = [], [], []
    bboxes = []
    full_masks = []  # 512x512 (same as image canvas) uint8 masks per polygon
    verts_all = []   # list of Nx2 float32 polygon vertices in canvas coords
    keep_going = True
    while keep_going:
        annotator = MultiPolygonAnnotator(ax_y, image_y)
        plt.show()  # block until window closed
        mask = annotator.get_current_mask()
        verts = annotator.get_current_verts()
        if mask is None:
            print("[warn] No polygon drawn. End annotation for this slice.")
            break
        if verts is None:
            print("[warn] No vertices captured for polygon; skipping.")
            continue
        # category choice
        print("Select pattern type:")
        for i, pt in enumerate(PATTERN_TYPES, 1):
            print(f"{i}. {pt}")
        choice = input("Enter number (or ENTER to skip/finish): ").strip()
        if choice == "":
            break
        try:
            choice = int(choice)
        except ValueError:
            print("[err] invalid input, skip polygon.")
            continue
        if choice < 1 or choice > len(PATTERN_TYPES):
            print("[err] invalid choice, skip polygon.")
            continue
        pattern_type = PATTERN_TYPES[choice - 1]

        coords = np.where(mask)
        y0, y1 = coords[0].min(), coords[0].max()
        x0, x1 = coords[1].min(), coords[1].max()
        patch = image_y[y0:y1 + 1, x0:x1 + 1]
        mask_crop = mask[y0:y1 + 1, x0:x1 + 1]

        patches.append(patch.astype(np.float32))
        masks.append(mask_crop.astype(np.uint8))
        categories.append(pattern_type)
        bboxes.append((int(y0), int(x0), int(y1), int(x1)))
        full_masks.append(mask.astype(np.uint8))
        verts_all.append(verts.astype(np.float32))

        cont = input("Add another polygon for this slice? (y/n): ").strip().lower()
        if cont == "y":
            keep_going = True
        else:
            keep_going = False

    if not patches:
        print("[info] No annotations for this slice.")
        return

    out_path = os.path.join(out_dir)
    os.makedirs(out_path, exist_ok=True)
    npz_path = os.path.join(out_path, f"{slice_id}.npz")
    meta = dict(slice_id=slice_id,
                bboxes=np.array(bboxes, dtype=np.int32).tolist(),
                areas=[int(m.sum()) for m in masks],
                canvas_size=(int(image_y.shape[0]), int(image_y.shape[1])))
    np.savez_compressed(npz_path,
                        patches=patches,
                        masks=masks,
                        full_masks=full_masks,
                        categories=categories,
                        verts=verts_all,
                        meta=meta)
    print(f"[save] annotations saved to {npz_path}")

    # visualization (ensure uint8 RGB 0..255)
    # If image is in [-1,1] or arbitrary float, normalize to 0..255 first
    if image_y.dtype != np.uint8:
        if image_y.min() >= -1.01 and image_y.max() <= 1.01:
            base = (image_y + 1.0) * 0.5 * 255.0
        else:
            imin, imax = float(image_y.min()), float(image_y.max())
            base = (image_y - imin) / (imax - imin + 1e-6) * 255.0
        base = np.clip(base, 0, 255).astype(np.uint8)
    else:
        base = image_y

    vis = np.stack([base, base, base], axis=-1).astype(np.uint8)

    # Overlay each full canvas mask with fixed color
    for (mask_full, cat) in zip(full_masks, categories):
        color = np.array(CATEGORY_COLOR.get(cat, (0, 255, 0)), dtype=np.uint8)
        m = mask_full.astype(bool)
        vis[m] = (0.6 * vis[m] + 0.4 * color).astype(np.uint8)

    png_path = os.path.join(out_path, f"{slice_id}.png")
    # Draw polygon edges for clarity
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(vis)
    # annotate legend-like labels
    for (mask_full, cat) in zip(full_masks, categories):
        try:
            import cv2
            cnts, _ = cv2.findContours(mask_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                cnt = cnt.squeeze(1)
                if cnt.ndim != 2 or len(cnt) < 3:
                    continue
                xs = cnt[:,0]
                ys = cnt[:,1]
                ax2.plot(xs, ys, '-', linewidth=1.5, color=np.array(CATEGORY_COLOR.get(cat,(0,255,0)))/255.0)
        except Exception:
            pass
    ax2.set_axis_off()
    fig2.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    print(f"[save] visualization saved to {png_path}")

def annotate_cmd(config_valroot: str, out_dir: str, size: int = 256, batch_size: int = 1):
    """Run the interactive annotation loop.
    Args:
        config_valroot: Path to validation dataset root (val_dataroot in config).
        out_dir: Where to save annotated patches.
        size: Target size for keepratio resize.
        batch_size: Dataloader batch size.
    """
    transforms_ = [ToTensor(), ResizeKeepRatioPad(size_tuple=(size, size), fill=-1)]
    dataset = ValDataset(config_valroot, transforms_=transforms_, unaligned=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in enumerate(loader):
        x_np = batch['A'][0, 0].numpy()
        y_np = batch['B'][0, 0].numpy()
        pid = batch.get('patient_id', ['unknown'])[0]
        sid = batch.get('slice_id', ['0000'])[0]
        slice_id = f"{pid}_{sid}"
        print(f"Showing slice {i}: {slice_id}")
        annotate_slice(y_np, x_np, slice_id, out_dir)

def inspect_cmd(npz_path: str, show_first: bool = False, save_preview: str | None = None):
    """Inspect a saved annotation .npz file and print summary.
    Args:
        npz_path: Path to the .npz produced by this tool.
        show_first: If True, show the first patch+mask overlay in a window.
        save_preview: If provided, save the first patch+mask overlay to this PNG path.
    """
    if not os.path.exists(npz_path):
        print(f"[ERR] file not found: {npz_path}")
        return
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)
    print("Keys:", keys)

    patches = data.get('patches', None)
    masks = data.get('masks', None)
    categories = data.get('categories', None)
    meta = data.get('meta', None)
    full_masks = data.get('full_masks', None)
    verts = data.get('verts', None)
    if meta is not None:
        try:
            meta = meta.item()
        except Exception:
            pass

    n = len(patches) if patches is not None else 0
    print(f"Num polygons: {n}")
    if categories is not None:
        print("Categories:", list(categories))
    if meta is not None:
        print("Meta:", meta)
    if full_masks is not None:
        print("Has full_masks (canvas):", True)
    if verts is not None:
        print("Has verts:", True)

    # Per-polygon shapes and basic stats
    for i in range(n):
        p = patches[i]
        m = masks[i]
        cat = categories[i] if categories is not None else 'NA'
        fm_shape = full_masks[i].shape if full_masks is not None else None
        print(f"[{i}] cat={cat} patch_shape={p.shape} mask_shape={m.shape} full_mask_shape={fm_shape} area={int(m.sum())}")

    # Optional quick viz for the first item
    if show_first or save_preview:
        if n == 0:
            print("[info] no polygons to visualize")
        else:
            if full_masks is not None:
                m0 = full_masks[0].astype(bool)
                # Build base from the full canvas using the patch pasted back is complex; for preview, create a blank canvas and mark mask
                # Or simply reuse the cropped preview as before if you prefer texture context.
                # Keep cropped preview for texture context:
                p0 = patches[0]
                m_crop0 = masks[0].astype(bool)
                # Normalize patch for display
                if p0.dtype != np.uint8:
                    pmin, pmax = float(p0.min()), float(p0.max())
                    if pmin >= -1.01 and pmax <= 1.01:
                        base = (p0 + 1.0) * 0.5 * 255.0
                    else:
                        base = (p0 - pmin) / (pmax - pmin + 1e-6) * 255.0
                    base = np.clip(base, 0, 255).astype(np.uint8)
                else:
                    base = p0
                rgb = np.stack([base, base, base], axis=-1)
                color = np.array((255, 0, 0), dtype=np.uint8)
                rgb[m_crop0] = (0.6 * rgb[m_crop0] + 0.4 * color).astype(np.uint8)
            else:
                # Fallback to cropped preview (same as old)
                p0 = patches[0]
                m0 = masks[0].astype(bool)
                if p0.dtype != np.uint8:
                    pmin, pmax = float(p0.min()), float(p0.max())
                    if pmin >= -1.01 and pmax <= 1.01:
                        base = (p0 + 1.0) * 0.5 * 255.0
                    else:
                        base = (p0 - pmin) / (pmax - pmin + 1e-6) * 255.0
                    base = np.clip(base, 0, 255).astype(np.uint8)
                else:
                    base = p0
                rgb = np.stack([base, base, base], axis=-1)
                color = np.array((255, 0, 0), dtype=np.uint8)
                rgb[m0] = (0.6 * rgb[m0] + 0.4 * color).astype(np.uint8)
            if save_preview:
                plt.imsave(save_preview, rgb)
                print(f"[save] preview saved to {save_preview}")
            if show_first:
                plt.figure(); plt.imshow(rgb); plt.title('first patch overlay'); plt.axis('off'); plt.show()

from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter
import glob


def stats_cmd(dir_path: str, save_csv: Optional[str] = None, verbose: bool = False):
    """Summarize all annotation .npz files in a directory.

    Args:
        dir_path: Directory containing .npz annotation files.
        save_csv: Optional path to save a CSV summary.
        verbose: If True, print per-file details.
    """
    dir_path = os.path.expanduser(dir_path)
    files = sorted(glob.glob(os.path.join(dir_path, '*.npz')))
    if not files:
        print(f"[ERR] no .npz files found under: {dir_path}")
        return

    # Aggregators
    per_cat_counts: Counter = Counter()
    per_cat_area_sum: Dict[str, float] = defaultdict(float)
    per_cat_area_list: Dict[str, List[float]] = defaultdict(list)
    per_cat_frac_sum: Dict[str, float] = defaultdict(float)  # area fraction in canvas
    per_cat_patch_area_sum: Dict[str, float] = defaultdict(float)
    per_cat_files: Dict[str, int] = defaultdict(int)  # number of files that contain at least one of this cat

    n_files = 0
    for fp in files:
        try:
            data = np.load(fp, allow_pickle=True)
        except Exception as e:
            print(f"[warn] failed to load {fp}: {e}")
            continue
        n_files += 1
        categories = data.get('categories', None)
        masks = data.get('masks', None)
        full_masks = data.get('full_masks', None)
        meta = data.get('meta', None)
        if meta is not None:
            try:
                meta = meta.item()
            except Exception:
                pass
        canvas_hw = None
        if meta and 'canvas_size' in meta:
            canvas_hw = tuple(meta['canvas_size'])
        elif full_masks is not None and len(full_masks) > 0:
            canvas_hw = full_masks[0].shape
        # fallback if nothing known
        if canvas_hw is None:
            canvas_hw = (512, 512)
        H, W = canvas_hw
        canvas_area = float(H * W)

        if categories is None or masks is None or len(categories) != len(masks):
            if verbose:
                print(f"[warn] malformed file (categories/masks mismatch): {fp}")
            continue

        file_cats_seen = set()
        for i in range(len(categories)):
            cat = str(categories[i])
            m = masks[i]
            # area from meta if available, otherwise compute from mask
            if meta and 'areas' in meta and i < len(meta['areas']):
                area = float(meta['areas'][i])
            else:
                area = float(np.asarray(m, dtype=bool).sum())

            per_cat_counts[cat] += 1
            per_cat_area_sum[cat] += area
            per_cat_area_list[cat].append(area)
            per_cat_frac_sum[cat] += (area / canvas_area)
            per_cat_patch_area_sum[cat] += float(np.prod(m.shape))
            file_cats_seen.add(cat)

        for cat in file_cats_seen:
            per_cat_files[cat] += 1

        if verbose:
            sid = meta.get('slice_id', os.path.basename(fp)) if isinstance(meta, dict) else os.path.basename(fp)
            areas_str = ','.join(str(int(a)) for a in (meta.get('areas', []) if isinstance(meta, dict) else []))
            print(f"[file] {sid}: cats={list(file_cats_seen)} areas=[{areas_str}] canvas={H}x{W}")

    # Build summary table
    cats_sorted = sorted(per_cat_counts.keys())
    rows = []
    header = [
        'category', 'count', 'files_with_cat',
        'area_mean', 'area_std', 'area_median', 'area_min', 'area_max',
        'area_frac_mean', 'patch_pixels_mean'
    ]

    def _safe_stats(vals: List[float]) -> Tuple[float, float, float, float, float]:
        import math
        if not vals:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(vals, dtype=float)
        return (float(arr.mean()), float(arr.std(ddof=0)), float(np.median(arr)), float(arr.min()), float(arr.max()))

    for cat in cats_sorted:
        cnt = per_cat_counts[cat]
        mean_area, std_area, med_area, min_area, max_area = _safe_stats(per_cat_area_list[cat])
        mean_frac = per_cat_frac_sum[cat] / max(cnt, 1)
        mean_patch_pix = per_cat_patch_area_sum[cat] / max(cnt, 1)
        rows.append([
            cat, cnt, per_cat_files[cat],
            int(round(mean_area)), int(round(std_area)), int(round(med_area)), int(round(min_area)), int(round(max_area)),
            round(mean_frac, 6), int(round(mean_patch_pix))
        ])

    # Print nicely
    colw = [max(len(str(h)), max(len(str(r[i])) for r in rows) if rows else len(str(h))) for i, h in enumerate(header)]
    def _fmt_row(r):
        return '  '.join(str(v).ljust(colw[i]) for i, v in enumerate(r))
    print(_fmt_row(header))
    for r in rows:
        print(_fmt_row(r))

    # Optionally save CSV
    if save_csv:
        import csv
        with open(save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"[save] stats CSV written to {save_csv}")

if __name__ == "__main__":
    fire.Fire({
        'annotate': annotate_cmd,
        'inspect': inspect_cmd,
        'stats': stats_cmd,
    })

"""
# Run annotation
python annotate_unregable.py annotate \
  --config_valroot /Users/liskie/Downloads/test2D \
  --out_dir /Users/liskie/Downloads/test2D/anno \
  --size 512

# Inspect a saved .npz
python annotate_unregable.py inspect \
  /Users/liskie/Downloads/test2D/anno/1PA177_0000.npz \
  --show_first True
  
python annotate_unregable.py stats /Users/liskie/Downloads/test2D/anno --save_csv /Users/liskie/Downloads/test2D/anno_stats.csv
"""