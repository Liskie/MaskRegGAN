#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 02: Build a curated set of (x, y_true, y_pred) triplets
---------------------------------------------------------------
Step 1: Given a TXT file listing manually selected slice IDs like "1PA177_0008",
filter ValDataset to only those slices, run the generator to produce y_pred,
and save (x, y_true, y_pred) as .npy for downstream experiments.

Usage:
  python experiment02-unregable-area.py --config path/to/config.yaml --keep_txt path/to/keep.txt --out_dir out/exp02_triplets

Notes:
- Requires that ValDataset returns 'patient_id' and 'slice_id' in its samples.
  (We already added this in datasets.py.)
- File naming falls back to "batchIndex_sliceIndex" if IDs are missing.
"""

import os
import sys
import csv
import argparse
import numpy as np
import glob
import random
import cv2

# For grid search utilities
from itertools import product as _it_product

# --- Add matplotlib for plotting utilities ---
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script usage
import matplotlib.pyplot as plt

# --- Add rich.progress for progress bars ---
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, TextColumn

from typing import Tuple, List, Dict

# Try SciPy's erf; fall back to a NumPy-only approximation if unavailable
try:
    from scipy.special import erf as _sp_erf
except Exception:
    _sp_erf = None

from skimage.metrics import structural_similarity  # (not used here yet, reserved for later steps)
from skimage.morphology import binary_dilation, remove_small_objects, opening, closing, binary_erosion, disk
from skimage.measure import label, regionprops

from trainer.utils import plot_composite


def _resize_to(img: np.ndarray, size: int = 512, mode: str = 'keepratio', fill: float = -1.0) -> np.ndarray:
    """Resize to a square canvas of side `size`.
    mode='resize'  : direct resize to (size,size) ignoring aspect ratio.
    mode='keepratio': keep aspect ratio and center-pad with `fill`.
    Expects grayscale 2D; returns float32 in input range.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        img = np.squeeze(img)
    h, w = img.shape[:2]
    if h == size and w == size:
        return img.astype(np.float32)
    if mode == 'resize':
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return resized
    # default: keepratio + pad
    scale = min(size / float(h), size / float(w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp).astype(np.float32)
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    out = np.full((size, size), fill, dtype=np.float32)
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out

def _scan_val_pairs(val_root: str):
    pa = os.path.join(val_root, 'A')
    pb = os.path.join(val_root, 'B')
    if not (os.path.isdir(pa) and os.path.isdir(pb)):
        raise FileNotFoundError(f"Expect subfolders A and B under {val_root}")

    def _stem(p):
        s = os.path.basename(p)
        return os.path.splitext(s)[0]

    A_list = sorted(glob.glob(os.path.join(pa, '*')))
    B_list = sorted(glob.glob(os.path.join(pb, '*')))
    A_map = {_stem(p): p for p in A_list}
    B_map = {_stem(p): p for p in B_list}
    names = sorted(set(A_map.keys()) & set(B_map.keys()))
    pairs = [(n, A_map[n], B_map[n]) for n in names]
    if not pairs:
        raise RuntimeError(f"No intersection between A and B under {val_root}")
    return pairs


def _read_gray_any(path: str) -> np.ndarray:
    if path.lower().endswith('.npy'):
        arr = np.load(path)
        return np.squeeze(arr).astype(np.float32)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    # assume uint8 [0,255] -> scale to [-1,1]
    return (img.astype(np.float32) / 127.5) - 1.0

# --- Helper: normalize ValDataset stem to y_pred stem variants ---
def _pred_stem_variants(stem: str) -> List[str]:
    """Return possible filename stems for y_pred based on a ValDataset stem.
    Examples:
      '1PA185.nii_z0070' -> ['1PA185.nii_z0070', '1PA185_0070', '1PA185nii_z0070', '1PA185_nii_z0070']
    We keep it conservative but cover common cases observed.
    """
    cands = [stem]
    # Replace the common pattern ".nii_z" -> "_"
    if '.nii_z' in stem:
        cands.append(stem.replace('.nii_z', '_'))
    # Generic dot -> underscore fallback
    if '.' in stem:
        cands.append(stem.replace('.', '_'))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in cands:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

# --- Helper: filter pairs by metrics.csv and PSNR threshold ---
def _filter_pairs_by_metrics_csv(pairs, metrics_csv: str, psnr_min: float) -> List[tuple]:
    if not metrics_csv or not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")
    keep_ids = _select_from_metrics(metrics_csv, psnr_min)
    if not keep_ids:
        print(f"[warn] No slices in metrics_csv with PSNR >= {psnr_min}")
        return []
    kept = []
    for name, pathA, pathB in pairs:
        # name is ValDataset stem like '1PA185.nii_z0070'; map to variants and compare
        variants = _pred_stem_variants(name)
        if any(v in keep_ids for v in variants):
            kept.append((name, pathA, pathB))
    print(f"[select] filtered by metrics_csv: {len(kept)}/{len(pairs)} pairs with PSNR >= {psnr_min}")
    return kept


def inject_eval_from_val(val_root: str, anno_dir: str, ypred_dir: str, out_dir: str,
                         num_pairs: int = 100, seed: int = 42, alpha: float = 1.0,
                         rd_cfg: Dict = None, metrics_csv: str = None, psnr_min: float = 28.0,
                         progress: Progress = None, task_id: int = None, draw: bool = True,
                         size: int = 512, resize_mode: str = 'keepratio', fill: float = -1.0,
                         anno_canvas: int = 512):
    random.seed(seed);
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    pairs = _scan_val_pairs(val_root)
    if metrics_csv:
        pairs = _filter_pairs_by_metrics_csv(pairs, metrics_csv, float(psnr_min))
        if not pairs:
            print(f"[ERR] No pairs left after PSNR filtering (>= {psnr_min}).")
            # still write empty CSV with header and average row of NaNs
            csv_path = os.path.join(out_dir, 'metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['slice', 'iou', 'dice'])
                w.writerow(['average_over_all_samples', '', ''])
            print(f"[done] saved 0 pairs (from ValDataset folders), CSV at {csv_path}")
            return
    anno_files = sorted(glob.glob(os.path.join(anno_dir, '*.npz')))
    if not anno_files:
        raise FileNotFoundError(f"No annotations under {anno_dir}")

    rd_defaults = {
        'rd_fdr_q': 0.10,
        'rd_Tl': 1.5,
        'rd_seed_smooth_sigma': 1.0,
        'rd_seed_min_area_frac': 0.0003,
        'rd_seed_open_r': 1,
        'rd_seed_percentile': 99.5,
        'rd_min_area_frac': 0.0005,
        'rd_bridge_r': 3,
        'rd_open_r': 2,
        'rd_close_r': 2,
        'rd_topk': -1,                # -1: keep all connected components
        'rd_seed_dilate': 1,          # light dilation to bridge thin gaps
        'rd_weight_mode': 'sigmoid',  # 'sigmoid' or 'masked'
        'rd_weight_alpha': 1.5,
        'rd_allow_empty': True,       # allow empty when no FDR seeds
        'rd_th_seed_hi': 0.0,         # hard-evidence threshold (|z|)
        'rd_min_highz_pixels': 2,     # min high-|z| pixel count per component
    }
    if rd_cfg: rd_defaults.update(rd_cfg)

    rows = []
    diag_rows = []
    if not pairs:
        print(f"[ERR] No available pairs to sample from.")
        # still write empty CSV with header and average row of NaNs
        csv_path = os.path.join(out_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['slice', 'iou', 'dice'])
            w.writerow(['average_over_all_samples', '', ''])
        print(f"[done] saved 0 pairs (from ValDataset folders), CSV at {csv_path}")
        return
    for i in range(num_pairs):
        name, pathA, pathB = random.choice(pairs)
        try:
            x = _read_gray_any(pathA)
            y_true = _read_gray_any(pathB)
        except Exception as e:
            print(f"[skip] fail load A/B for {name}: {e}")
            # progress hook for outer rich progress (if provided)
            if progress is not None and task_id is not None:
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass
            continue
        # y_pred by name stem
        try:
            y_pred = _read_pred(ypred_dir, name)
        except Exception as e:
            print(f"[skip] no y_pred for {name}: {e}")
            # progress hook for outer rich progress (if provided)
            if progress is not None and task_id is not None:
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass
            continue
        # Ensure same canvas according to configured size/mode
        y_true = _resize_to(y_true, size=size, mode=resize_mode, fill=fill)
        y_pred = _resize_to(y_pred, size=size, mode=resize_mode, fill=fill)
        # pick annotation
        adata = np.load(random.choice(anno_files), allow_pickle=True)
        patches = adata['patches'];
        masks = adata['masks']
        meta = adata['meta'].item() if isinstance(adata.get('meta', None), np.ndarray) else {}
        bboxes = meta.get('bboxes', None)
        k = random.randrange(len(patches))
        patch = patches[k].astype(np.float32)
        mask_crop = masks[k].astype(np.uint8)
        if bboxes is not None and k < len(bboxes):
            bbox = tuple(map(int, bboxes[k]))
        else:
            h, w = mask_crop.shape;
            bbox = (0, 0, h - 1, w - 1)
        # Map annotations (e.g., on 512×512) to current size, then align bbox to patch & canvas
        patch, mask_crop, bbox = _maybe_rescale_patch_and_bbox(patch, mask_crop, bbox, anno_canvas, size)
        Hc, Wc = y_true.shape
        bbox = _fix_bbox_for_patch(patch, bbox, Hc, Wc)

        y_inj, mask_true = _inject_region(y_true, patch, mask_crop, bbox, alpha=alpha)
        mask_pred, rd_data = _extract_mask_from_residual(gt=y_inj, pred=y_pred, cfg=rd_defaults)
        iou, dice = _iou_dice(mask_true, mask_pred)
        rows.append([name, iou, dice])
        empty = int(not np.any(mask_pred))
        pred_area_frac = float(rd_data.get('pred_area_frac', 0.0))
        kept_cc_cnt = int(rd_data.get('kept_cc_cnt', 0))
        cnt_hi_list = rd_data.get('cnt_hi_list', [])
        cnt_hi_str = ';'.join(str(int(v)) for v in cnt_hi_list) if isinstance(cnt_hi_list, (list, tuple)) else ''
        diag_rows.append([name, empty, pred_area_frac, kept_cc_cnt, cnt_hi_str])

        if draw:
            overlay = _overlay_vis(y_inj, mask_true, mask_pred)
            cv2.imwrite(os.path.join(out_dir, f"{name}_overlay.png"), overlay[:, :, ::-1])

            # Save composite figure with overlay in the 3rd row
            comp_path = os.path.join(out_dir, f"{name}_composite.png")
            plot_composite(inp=x, gt=y_inj, pred=y_pred, residual=(y_pred - y_inj),
                           metrics=None, rd_data=rd_data, uncertainty=None,
                           use_reg=False, save_path=comp_path, overlay_rgb=overlay)

        # progress hook for outer rich progress (if provided)
        if progress is not None and task_id is not None:
            try:
                progress.update(task_id, advance=1)
            except Exception:
                pass

    # Compute averages
    if rows:
        avg_iou = float(np.mean([r[1] for r in rows]))
        avg_dice = float(np.mean([r[2] for r in rows]))
        print(f"[avg] IoU={avg_iou:.4f}, Dice={avg_dice:.4f}")
    else:
        avg_iou = avg_dice = None

    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f);
        w.writerow(['slice', 'iou', 'dice'])
        if rows:
            w.writerow(['average_over_all_samples', avg_iou, avg_dice])
        w.writerows(rows)
    print(f"[done] saved {len(rows)} pairs (from ValDataset folders), CSV at {csv_path}")

    # Diagnostics outputs (per-sample and summary)
    diag_csv = os.path.join(out_dir, 'diagnostics.csv')
    with open(diag_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slice','empty','pred_area_frac','kept_cc_cnt','cnt_hi_list'])
        w.writerows(diag_rows)

    empty_rate = float(np.mean([r[1] for r in diag_rows])) if diag_rows else 0.0
    mean_pred_area_frac = float(np.mean([r[2] for r in diag_rows])) if diag_rows else 0.0
    mean_kept_cc = float(np.mean([r[3] for r in diag_rows])) if diag_rows else 0.0

    diag_sum_csv = os.path.join(out_dir, 'diagnostics_summary.csv')
    with open(diag_sum_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['empty_rate','mean_pred_area_frac','mean_kept_cc','n'])
        w.writerow([empty_rate, mean_pred_area_frac, mean_kept_cc, len(diag_rows)])
    print(f"[diag] empty_rate={empty_rate:.3f}, area_frac={mean_pred_area_frac:.4f}, kept_cc={mean_kept_cc:.2f}")


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _parse_keep_list(txt_path):
    keep = set()
    with open(txt_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Allow "1PA177_0008" or "1PA177 0008"
            s = s.replace(' ', '_')
            keep.add(s)
    return keep


def _select_from_metrics(csv_path, psnr_min):
    keep = set()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return keep
    header = rows[0]
    # 找到 psnr_mean 列
    try:
        psnr_col = next(i for i, h in enumerate(header) if 'psnr_mean' in h)
    except StopIteration:
        psnr_col = 3 if len(header) > 3 else 1  # 兜底
    slice_col = 0
    for r in rows[1:]:
        if not r or len(r) <= max(slice_col, psnr_col):
            continue
        sid = (r[slice_col] or '').strip()
        if not sid or sid.lower().startswith('average_over_all_samples'):
            continue
        try:
            psnr_val = float(r[psnr_col])
        except Exception:
            continue
        if psnr_val >= psnr_min:
            keep.add(sid)
    return keep


# ===== Residual-detection helpers (ported from CycTrainer) =====

def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF Φ(x).
    Uses SciPy's erf when available; otherwise falls back to a fast NumPy approximation
    (Abramowitz & Stegun 7.1.26).
    """
    if _sp_erf is not None:
        return 0.5 * (1.0 + _sp_erf(x / np.sqrt(2.0)))
    # Fallback approximation of erf (vectorized, no SciPy required)
    z = x / np.sqrt(2.0)
    sign = np.sign(z)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * np.abs(z))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(z * z))
    erf_approx = sign * y
    return 0.5 * (1.0 + erf_approx)


def _robust_zscore(residual: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    a = np.asarray(residual, dtype=np.float32)
    if mask is not None:
        a = np.where(mask, a, np.nan)
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-8:
        sigma = np.nanstd(a)
    if not np.isfinite(sigma) or sigma <= 1e-8:
        sigma = 1.0
    z = (residual - med) / (sigma + 1e-8)
    if mask is not None:
        z = np.where(mask, z, 0.0)
    return z.astype(np.float32)


def _bh_fdr_mask(p: np.ndarray, q: float = 0.10, mask: np.ndarray = None) -> np.ndarray:
    # Benjamini–Hochberg FDR
    if mask is not None:
        p_use = p[mask]
    else:
        p_use = p.ravel()
    if p_use.size == 0:
        return np.zeros_like(p, dtype=bool)
    order = np.argsort(p_use)
    p_sorted = p_use[order]
    m = float(p_use.size)
    thresh = (np.arange(1, p_use.size + 1, dtype=np.float64) / m) * float(q)
    le = p_sorted <= thresh
    if not np.any(le):
        out = np.zeros_like(p_use, dtype=bool)
    else:
        k = np.max(np.nonzero(le)[0])
        p_th = p_sorted[k]
        out = (p_use <= p_th)
    if mask is not None:
        result = np.zeros_like(mask, dtype=bool)
        result[mask] = out
        return result
    else:
        return out.reshape(p.shape)





def _hysteresis_from_seeds(z: np.ndarray, seeds: np.ndarray, Tl: float = 2.0, max_iters: int = 1024) -> np.ndarray:
    band = (np.abs(z) >= Tl)
    grown = seeds.astype(bool).copy()
    if not np.any(grown):
        return grown
    it = 0
    while True:
        it += 1
        nxt = binary_dilation(grown) & band
        if np.array_equal(nxt, grown):
            break
        grown = nxt
        if it >= max_iters:
            break
    return grown


def _morph_clean(mask: np.ndarray, min_area: int = 64, open_r: int = 1, close_r: int = 2,
                 bridge_r: int = 0) -> np.ndarray:
    m = mask.astype(bool)
    if not np.any(m): return m
    m = remove_small_objects(m, min_size=int(max(1, min_area)))
    if bridge_r and int(bridge_r) > 0:
        se = disk(int(bridge_r))
        m = binary_dilation(m, se)
        m = binary_erosion(m, se)
    if open_r and int(open_r) > 0:
        m = opening(m, disk(int(open_r)))
    if close_r and int(close_r) > 0:
        m = closing(m, disk(int(close_r)))
    m = remove_small_objects(m, min_size=int(max(1, min_area)))
    return m


def _extract_mask_from_residual(gt: np.ndarray, pred: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict]:
    body = (gt != -1)
    residual = pred - gt
    zmap = _robust_zscore(residual, mask=body)
    absz = np.abs(zmap)

    q_fdr = float(cfg.get('rd_fdr_q', 0.10))
    Tl_fix = float(cfg.get('rd_Tl', 1.5))
    area_fr = float(cfg.get('rd_min_area_frac', 0.0005))
    open_r = int(cfg.get('rd_open_r', 2))
    close_r = int(cfg.get('rd_close_r', 2))
    topk = int(cfg.get('rd_topk', -1))
    seed_dilate = int(cfg.get('rd_seed_dilate', 1))
    rd_allow_empty = bool(cfg.get('rd_allow_empty', True))
    Th_hard = float(cfg.get('rd_th_seed_hi', 0.0))
    K_hard = int(cfg.get('rd_min_highz_pixels', 2))
    seed_sigma = float(cfg.get('rd_seed_smooth_sigma', 1.0))
    seed_min_area_frac = float(cfg.get('rd_seed_min_area_frac', 0.0003))
    seed_open_r = int(cfg.get('rd_seed_open_r', 1))
    bridge_r = int(cfg.get('rd_bridge_r', 3))
    weight_mode = str(cfg.get('rd_weight_mode', 'sigmoid')).lower()
    weight_alpha = float(cfg.get('rd_weight_alpha', 1.5))

    absz_s = cv2.GaussianBlur(absz.astype(np.float32), (0, 0), seed_sigma, seed_sigma) if seed_sigma > 0 else absz
    pvals = 2.0 * (1.0 - _normal_cdf(absz_s))
    seeds = _bh_fdr_mask(pvals, q=q_fdr, mask=body)

    body_area = int(np.count_nonzero(body))
    if np.any(seeds):
        min_area_seed = max(1, int(seed_min_area_frac * max(1, body_area)))
        seeds = remove_small_objects(seeds.astype(bool), min_size=min_area_seed)
        if seed_open_r > 0:
            seeds = opening(seeds, disk(int(seed_open_r)))

    nonempty = bool(np.any(seeds))
    if not nonempty:
        if rd_allow_empty:
            seeds = np.zeros_like(body, dtype=bool)
        else:
            pctl = float(cfg.get('rd_seed_percentile', 99.5))
            thr = max(Tl_fix, np.percentile(absz[body], pctl)) if np.any(body) else Tl_fix
            seeds = (absz >= thr) & body
            nonempty = bool(np.any(seeds))

    if nonempty and seed_dilate > 0:
        seeds = binary_dilation(seeds, disk(int(seed_dilate))) & body

    final_mask = np.zeros_like(seeds, dtype=bool)
    weight_map = np.zeros_like(absz, dtype=np.float32)
    if nonempty:
        grown = _hysteresis_from_seeds(zmap, seeds, Tl=Tl_fix)
        lab = label(grown, connectivity=2)
        if lab.max() > 0:
            regs = regionprops(lab, intensity_image=np.abs(zmap).astype(np.float32))
            regs = [r for r in regs if r.area >= max(1, int(area_fr * max(1, body_area)))]
            if regs:
                high_z = (np.abs(zmap) >= Th_hard)
                valid_labels = []
                for r in regs:
                    lbl = r.label
                    cnt_hi = int(np.count_nonzero(high_z & (lab == lbl)))
                    if cnt_hi >= max(0, K_hard):
                        valid_labels.append(lbl)
                if valid_labels:
                    if topk is not None and int(topk) >= 1:
                        scored = sorted([r for r in regs if r.label in valid_labels],
                                        key=lambda r: float(r.mean_intensity) * np.log1p(float(r.area)),
                                        reverse=True)
                        keep = [r.label for r in scored[:int(topk)]]
                    else:
                        keep = valid_labels
                    final_mask = np.isin(lab, keep)
                    final_mask = _morph_clean(final_mask,
                                              min_area=max(1, int(area_fr * max(1, body_area))),
                                              open_r=open_r, close_r=close_r, bridge_r=bridge_r)

    # === Diagnostics: component/high-z stats & area fraction ===
    cc_cnt = 0
    kept_cc_cnt = 0
    cnt_hi_list = []
    try:
        # 'lab' and 'regs' were computed above when nonempty
        if 'regs' in locals():
            cc_cnt = len(regs)
            if lab.max() > 0 and np.any(final_mask):
                kept_labels = np.unique(lab[final_mask])
                kept_labels = kept_labels[kept_labels > 0]
                kept_cc_cnt = int(len(kept_labels))
                high_z = (np.abs(zmap) >= Th_hard)
                for lbl in kept_labels:
                    cnt_hi = int(np.count_nonzero(high_z & (lab == lbl)))
                    cnt_hi_list.append(cnt_hi)
    except Exception:
        pass

    pred_area = int(np.count_nonzero(final_mask))
    pred_area_frac = float(pred_area) / float(max(1, body_area))

    # Compose weight map for diagnostics/output
    if weight_mode == 'masked':
        weight_map = final_mask.astype(np.float32)
    else:  # default 'sigmoid'
        # Sigmoid centered at Tl; sharpness controlled by weight_alpha
        w = 1.0 / (1.0 + np.exp(-weight_alpha * (np.abs(zmap) - Tl_fix)))
        weight_map = (w * body.astype(np.float32)).astype(np.float32)
    return final_mask.astype(bool), {
        'zmap': zmap,
        'seeds': seeds,
        'weight_map': weight_map,
        'final_mask': final_mask,
        'cc_cnt': cc_cnt,
        'kept_cc_cnt': kept_cc_cnt,
        'cnt_hi_list': cnt_hi_list,
        'pred_area_frac': pred_area_frac,
        'body_area': body_area,
    }


# ===== Affine injection + evaluation utilities =====

def _reconstruct_src_canvas(patch: np.ndarray, mask_crop: np.ndarray, bbox: Tuple[int, int, int, int],
                            canvas_hw=(512, 512)):
    y0, x0, y1, x1 = bbox
    H, W = canvas_hw
    canvas = np.zeros((H, W), dtype=patch.dtype)
    mfull = np.zeros((H, W), dtype=np.uint8)
    canvas[y0:y1 + 1, x0:x1 + 1] = patch
    mfull[y0:y1 + 1, x0:x1 + 1] = mask_crop.astype(np.uint8)
    return canvas, mfull


def _random_affine(H: int, W: int, angle_deg=15.0, scale_min=0.9, scale_max=1.1, shift=10) -> np.ndarray:
    cx, cy = W / 2.0, H / 2.0
    ang = random.uniform(-angle_deg, angle_deg)
    sc = random.uniform(scale_min, scale_max)
    tx = random.uniform(-shift, shift)
    ty = random.uniform(-shift, shift)
    M = cv2.getRotationMatrix2D((cx, cy), ang, sc)
    M[:, 2] += [tx, ty]
    return M

# --- Helper: scale bbox defined on a square annotation canvas ---
def _scale_bbox_square(bbox: Tuple[int,int,int,int], s: float) -> Tuple[int,int,int,int]:
    y0, x0, y1, x1 = map(float, bbox)
    y0 *= s; x0 *= s; y1 *= s; x1 *= s
    return int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1))

# --- Helper: optionally rescale patch/mask and bbox from annotation canvas to current size ---
def _maybe_rescale_patch_and_bbox(patch: np.ndarray, mask_crop: np.ndarray, bbox: Tuple[int,int,int,int],
                                  anno_canvas: int, size: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """If annotations were made on a square canvas (e.g., 512x512) and current size differs,
    uniformly rescale patch/mask and bbox by s=size/anno_canvas.
    """
    if int(anno_canvas) <= 0 or int(anno_canvas) == int(size):
        return patch, mask_crop, bbox
    s = float(size) / float(anno_canvas)
    ph, pw = patch.shape[:2]
    new_w = max(1, int(round(pw * s)))
    new_h = max(1, int(round(ph * s)))
    patch2 = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(patch.dtype)
    mask2 = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    bbox2 = _scale_bbox_square(bbox, s)
    return patch2, mask2, bbox2

# --- Helper: ensure bbox matches patch size and stays within current canvas ---
def _fix_bbox_for_patch(patch: np.ndarray, bbox: Tuple[int, int, int, int], H: int, W: int) -> Tuple[int, int, int, int]:
    """Ensure bbox size matches the patch shape and fits within (H,W).
    Keeps the top-left (y0,x0) from the provided bbox (clamped into canvas),
    then sets (y1,x1) to y0+ph-1, x0+pw-1, with clipping if needed.
    """
    ph, pw = map(int, patch.shape[:2])
    y0, x0, y1, x1 = map(int, bbox)
    # Clamp top-left inside canvas
    y0 = max(0, min(y0, max(0, H - 1)))
    x0 = max(0, min(x0, max(0, W - 1)))
    # Compute bottom-right based on patch size
    y1 = y0 + ph - 1
    x1 = x0 + pw - 1
    # If overflow, shift back so the patch fits
    if y1 >= H:
        y0 = max(0, H - ph)
        y1 = y0 + ph - 1
    if x1 >= W:
        x0 = max(0, W - pw)
        x1 = x0 + pw - 1
    return int(y0), int(x0), int(y1), int(x1)


def _inject_region(y_true: np.ndarray, patch: np.ndarray, mask_crop: np.ndarray, bbox: Tuple[int, int, int, int],
                   alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    H, W = y_true.shape
    src_canvas, src_mask_full = _reconstruct_src_canvas(patch, mask_crop, bbox, (H, W))
    M = _random_affine(H, W)
    warped_src = cv2.warpAffine(src_canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
    warped_mk = cv2.warpAffine(src_mask_full, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    warped_mk = (warped_mk > 0)
    out = y_true.copy()
    if alpha >= 1.0:
        out[warped_mk] = warped_src[warped_mk]
    else:
        out[warped_mk] = alpha * warped_src[warped_mk] + (1.0 - alpha) * out[warped_mk]
    return out.astype(np.float32), warped_mk.astype(bool)


def _read_pred(ypred_dir: str, slice_name: str) -> np.ndarray:
    # Try exact stem, then normalized variants
    stems = _pred_stem_variants(slice_name)
    for st in stems:
        p_npy = os.path.join(ypred_dir, f"{st}.npy")
        p_png = os.path.join(ypred_dir, f"{st}.png")
        if os.path.exists(p_npy):
            arr = np.load(p_npy)
            return np.squeeze(arr).astype(np.float32)
        if os.path.exists(p_png):
            img = cv2.imread(p_png, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # assume uint8 [0,255] -> [-1,1]
            return (img.astype(np.float32) / 127.5) - 1.0
    raise FileNotFoundError(f"No y_pred found for {slice_name} (tried: {stems}) in {ypred_dir}")


def _overlay_vis(base: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    a = base
    if a.dtype != np.uint8:
        amin, amax = float(a.min()), float(a.max())
        if amin >= -1.01 and amax <= 1.01:
            vis = ((a + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            vis = ((a - amin) / (amax - amin + 1e-6) * 255.0).astype(np.uint8)
    else:
        vis = a
    rgb = np.stack([vis, vis, vis], axis=-1)
    mT = mask_true.astype(bool)
    mP = mask_pred.astype(bool)
    rgb[mT] = (0.6 * rgb[mT] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)  # red True
    rgb[mP] = (0.6 * rgb[mP] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)  # green Pred
    rgb[mT & mP] = (0.6 * rgb[mT & mP] + 0.4 * np.array([255, 255, 0])).astype(np.uint8)  # overlap yellow
    return rgb

# --- Helper: build config key string for grid search ---
def _cfg_slug(d: Dict, keys: List[str]) -> str:
    parts = []
    for k in keys:
        v = d.get(k)
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "+".join(parts)

# --- Helper: ensure a value is a list ---
def _ensure_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def _iou_dice(m1: np.ndarray, m2: np.ndarray) -> Tuple[float, float]:
    m1 = m1.astype(bool)
    m2 = m2.astype(bool)
    inter = float(np.logical_and(m1, m2).sum())
    u = float(np.logical_or(m1, m2).sum())
    iou = inter / (u + 1e-6)
    dice = 2 * inter / (float(m1.sum()) + float(m2.sum()) + 1e-6)
    return iou, dice

# --- Helper to read metrics.csv and compute stats for plotting ---
def _read_metrics_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    ious, dices, names = [], [], []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return names, np.array([]), np.array([])
    header = rows[0]
    for r in rows[1:]:
        if not r or len(r) < 3:
            continue
        sid = (r[0] or '').strip()
        if not sid:
            continue
        if sid.lower().startswith('average_over_all_samples'):
            continue
        try:
            iou = float(r[1]); dice = float(r[2])
        except Exception:
            continue
        names.append(sid)
        ious.append(iou)
        dices.append(dice)
    return names, np.array(ious, dtype=np.float32), np.array(dices, dtype=np.float32)

# --- Plotting function for IoU/Dice histograms ---
def plot_histograms(csv_path: str, out_dir: str, bins: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    names, ious, dices = _read_metrics_csv(csv_path)
    if ious.size == 0:
        print(f"[plot] No valid rows in {csv_path}")
        return
    def _basic_stats(x):
        return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                    p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))
    s_iou = _basic_stats(ious); s_dice = _basic_stats(dices)
    print(f"[plot] IoU stats:  mean={s_iou['mean']:.4f}, median={s_iou['median']:.4f}, p10={s_iou['p10']:.4f}, p90={s_iou['p90']:.4f}")
    print(f"[plot] Dice stats: mean={s_dice['mean']:.4f}, median={s_dice['median']:.4f}, p10={s_dice['p10']:.4f}, p90={s_dice['p90']:.4f}")

    # IoU histogram
    plt.figure(figsize=(6,4))
    plt.hist(ious, bins=bins, range=(0.0, 1.0), alpha=0.8)
    plt.axvline(s_iou['mean'], linestyle='--', linewidth=2)
    plt.axvline(s_iou['median'], linestyle=':', linewidth=2)
    plt.xlabel('IoU'); plt.ylabel('Count'); plt.title('IoU Distribution')
    plt.tight_layout()
    iou_path = os.path.join(out_dir, 'hist_iou.png')
    plt.savefig(iou_path, dpi=160)
    plt.close()

    # Dice histogram
    plt.figure(figsize=(6,4))
    plt.hist(dices, bins=bins, range=(0.0, 1.0), alpha=0.8)
    plt.axvline(s_dice['mean'], linestyle='--', linewidth=2)
    plt.axvline(s_dice['median'], linestyle=':', linewidth=2)
    plt.xlabel('Dice'); plt.ylabel('Count'); plt.title('Dice Distribution')
    plt.tight_layout()
    dice_path = os.path.join(out_dir, 'hist_dice.png')
    plt.savefig(dice_path, dpi=160)
    plt.close()

    # Save a small stats CSV
    stats_csv = os.path.join(out_dir, 'metrics_stats.csv')
    with open(stats_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','mean','median','p10','p90','n'])
        w.writerow(['IoU', s_iou['mean'], s_iou['median'], s_iou['p10'], s_iou['p90'], int(ious.size)])
        w.writerow(['Dice', s_dice['mean'], s_dice['median'], s_dice['p10'], s_dice['p90'], int(dices.size)])
    print(f"[plot] Saved: {iou_path}, {dice_path}, {stats_csv}")



if __name__ == '__main__':
    ap2 = argparse.ArgumentParser()
    sub = ap2.add_subparsers(dest='cmd')

    p2 = sub.add_parser('inject_eval_from_val', help='Run injection + evaluation reading A/B from ValDataset root')
    p2.add_argument('--val_root', type=str, required=True)
    p2.add_argument('--anno_dir', type=str, required=True)
    p2.add_argument('--ypred_dir', type=str, required=True)
    p2.add_argument('--out_dir', type=str, required=True)
    p2.add_argument('--num_pairs', type=int, default=100)
    p2.add_argument('--alpha', type=float, default=1.0)
    p2.add_argument('--seed', type=int, default=42)
    p2.add_argument('--metrics_csv', type=str, required=False, help='Path to metrics.csv for PSNR-based filtering')
    # Residual-detect hyperparameters
    p2.add_argument('--rd_fdr_q', type=float, default=0.10)
    p2.add_argument('--rd_Tl', type=float, default=1.5)
    p2.add_argument('--rd_seed_smooth_sigma', type=float, default=1.0)
    p2.add_argument('--rd_seed_min_area_frac', type=float, default=0.0003)
    p2.add_argument('--rd_seed_open_r', type=int, default=1)
    p2.add_argument('--rd_seed_percentile', type=float, default=99.5)
    p2.add_argument('--rd_min_area_frac', type=float, default=0.0005)
    p2.add_argument('--rd_bridge_r', type=int, default=3)
    p2.add_argument('--rd_open_r', type=int, default=2)
    p2.add_argument('--rd_close_r', type=int, default=2)
    p2.add_argument('--rd_topk', type=int, default=-1)
    p2.add_argument('--rd_seed_dilate', type=int, default=1)
    p2.add_argument('--rd_weight_mode', type=str, default='sigmoid', choices=['sigmoid','masked'])
    p2.add_argument('--rd_weight_alpha', type=float, default=1.5)
    p2.add_argument('--rd_allow_empty', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True)
    p2.add_argument('--rd_th_seed_hi', type=float, default=0.0)
    p2.add_argument('--rd_min_highz_pixels', type=int, default=2)
    p2.add_argument('--psnr_min', type=float, default=28.0, help='Minimum PSNR to keep a slice (used with --metrics_csv)')
    p2.add_argument('--no_draw', action='store_true', default=False, help='Do not draw and save overlay/composite images')
    p2.add_argument('--size', type=int, default=512, help='Final square canvas size for x/gt/pred')
    p2.add_argument('--resize_mode', type=str, default='keepratio', choices=['resize','keepratio'],
                   help='resize: force to size×size (ignore aspect); keepratio: keep aspect and pad')
    p2.add_argument('--fill', type=float, default=-1.0, help='Pad value when resize_mode=keepratio')
    p2.add_argument('--anno_canvas', type=int, default=512, help='Square canvas size used during annotation (e.g., 512)')

    p3 = sub.add_parser('plot_hist', help='Plot IoU/Dice histograms from a metrics.csv')
    p3.add_argument('--csv', type=str, required=True)
    p3.add_argument('--out_dir', type=str, required=True)
    p3.add_argument('--bins', type=int, default=20)

    # --- tune_rd subcommand for grid/random search over RD params ---
    p4 = sub.add_parser('tune_rd', help='Grid/random scan RD hyperparameters and summarize results')
    p4.add_argument('--val_root', type=str, required=True)
    p4.add_argument('--anno_dir', type=str, required=True)
    p4.add_argument('--ypred_dir', type=str, required=True)
    p4.add_argument('--out_dir', type=str, required=True)
    p4.add_argument('--metrics_csv', type=str, required=False)
    p4.add_argument('--psnr_min', type=float, default=28.0)
    p4.add_argument('--num_pairs', type=int, default=200)
    p4.add_argument('--alpha', type=float, default=1.0)
    p4.add_argument('--seed', type=int, default=42)
    p4.add_argument('--max_runs', type=int, default=0, help='Cap total runs (0=unlimited)')
    p4.add_argument('--no_draw', action='store_true', default=False, help='Do not draw and save overlay/composite images for speed')
    # RD params as lists
    p4.add_argument('--rd_fdr_q', type=float, nargs='+', default=[0.10])
    p4.add_argument('--rd_Tl', type=float, nargs='+', default=[1.5])
    p4.add_argument('--rd_seed_smooth_sigma', type=float, nargs='+', default=[1.0])
    p4.add_argument('--rd_seed_min_area_frac', type=float, nargs='+', default=[0.0003])
    p4.add_argument('--rd_seed_open_r', type=int, nargs='+', default=[1])
    p4.add_argument('--rd_seed_percentile', type=float, nargs='+', default=[99.5])
    p4.add_argument('--rd_min_area_frac', type=float, nargs='+', default=[0.0005])
    p4.add_argument('--rd_bridge_r', type=int, nargs='+', default=[3])
    p4.add_argument('--rd_open_r', type=int, nargs='+', default=[2])
    p4.add_argument('--rd_close_r', type=int, nargs='+', default=[2])
    p4.add_argument('--rd_topk', type=int, nargs='+', default=[-1])
    p4.add_argument('--rd_seed_dilate', type=int, nargs='+', default=[1])
    p4.add_argument('--rd_weight_mode', type=str, nargs='+', default=['sigmoid'])
    p4.add_argument('--rd_weight_alpha', type=float, nargs='+', default=[1.5])
    p4.add_argument('--rd_allow_empty', type=str, nargs='+', default=['true'], help='true/false values')
    p4.add_argument('--rd_th_seed_hi', type=float, nargs='+', default=[0.0])
    p4.add_argument('--rd_min_highz_pixels', type=int, nargs='+', default=[2])
    p4.add_argument('--size', type=int, default=512)
    p4.add_argument('--resize_mode', type=str, default='keepratio', choices=['resize','keepratio'])
    p4.add_argument('--fill', type=float, default=-1.0)
    p4.add_argument('--anno_canvas', type=int, default=512)

    args2 = ap2.parse_args()
    if args2.cmd == 'inject_eval_from_val':
        rd_cfg = {
            'rd_fdr_q': args2.rd_fdr_q,
            'rd_Tl': args2.rd_Tl,
            'rd_seed_smooth_sigma': args2.rd_seed_smooth_sigma,
            'rd_seed_min_area_frac': args2.rd_seed_min_area_frac,
            'rd_seed_open_r': args2.rd_seed_open_r,
            'rd_seed_percentile': args2.rd_seed_percentile,
            'rd_min_area_frac': args2.rd_min_area_frac,
            'rd_bridge_r': args2.rd_bridge_r,
            'rd_open_r': args2.rd_open_r,
            'rd_close_r': args2.rd_close_r,
            'rd_topk': args2.rd_topk,
            'rd_seed_dilate': args2.rd_seed_dilate,
            'rd_weight_mode': args2.rd_weight_mode,
            'rd_weight_alpha': args2.rd_weight_alpha,
            'rd_allow_empty': args2.rd_allow_empty,
            'rd_th_seed_hi': args2.rd_th_seed_hi,
            'rd_min_highz_pixels': args2.rd_min_highz_pixels,
        }
        draw = not args2.no_draw
        inject_eval_from_val(args2.val_root, args2.anno_dir, args2.ypred_dir, args2.out_dir,
                             num_pairs=args2.num_pairs, seed=args2.seed, alpha=args2.alpha,
                             rd_cfg=rd_cfg, metrics_csv=args2.metrics_csv, psnr_min=args2.psnr_min,
                             draw=draw, size=args2.size, resize_mode=args2.resize_mode, fill=args2.fill,
                             anno_canvas=args2.anno_canvas)
    elif args2.cmd == 'plot_hist':
        plot_histograms(args2.csv, args2.out_dir, bins=args2.bins)
    elif args2.cmd == 'tune_rd':
        # Build grid
        bool_l = lambda s: str(s).lower() in ['1','true','yes','y']
        grid_keys = [
            'rd_fdr_q','rd_Tl','rd_seed_smooth_sigma','rd_seed_min_area_frac','rd_seed_open_r',
            'rd_seed_percentile','rd_min_area_frac','rd_bridge_r','rd_open_r','rd_close_r','rd_topk',
            'rd_seed_dilate','rd_weight_mode','rd_weight_alpha','rd_allow_empty','rd_th_seed_hi','rd_min_highz_pixels']
        grid_lists = [
            _ensure_list(args2.rd_fdr_q), _ensure_list(args2.rd_Tl), _ensure_list(args2.rd_seed_smooth_sigma),
            _ensure_list(args2.rd_seed_min_area_frac), _ensure_list(args2.rd_seed_open_r), _ensure_list(args2.rd_seed_percentile),
            _ensure_list(args2.rd_min_area_frac), _ensure_list(args2.rd_bridge_r), _ensure_list(args2.rd_open_r),
            _ensure_list(args2.rd_close_r), _ensure_list(args2.rd_topk), _ensure_list(args2.rd_seed_dilate),
            _ensure_list(args2.rd_weight_mode), _ensure_list(args2.rd_weight_alpha), _ensure_list(args2.rd_allow_empty),
            _ensure_list(args2.rd_th_seed_hi), _ensure_list(args2.rd_min_highz_pixels),
        ]
        # Normalize booleans represented as strings
        for i,k in enumerate(grid_keys):
            if k == 'rd_allow_empty':
                grid_lists[i] = [bool_l(v) for v in grid_lists[i]]
        combos = list(_it_product(*grid_lists))
        if args2.max_runs and len(combos) > args2.max_runs:
            combos = combos[:args2.max_runs]
        os.makedirs(args2.out_dir, exist_ok=True)
        summary_rows = []
        summary_csv = os.path.join(args2.out_dir, 'tune_summary.csv')
        print(f"[tune] total runs: {len(combos)}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as prog:
            task_runs = prog.add_task("tune-runs", total=len(combos))
            for ridx, vals in enumerate(combos):
                rd_cfg = dict(zip(grid_keys, vals))
                slug_keys = ['rd_Tl','rd_fdr_q','rd_th_seed_hi','rd_min_highz_pixels','rd_min_area_frac','rd_open_r','rd_close_r']
                subdir = os.path.join(args2.out_dir, _cfg_slug(rd_cfg, slug_keys))
                # sub-task for per-slice progress
                task_one = prog.add_task(f"run {ridx+1}/{len(combos)}", total=args2.num_pairs)
                try:
                    inject_eval_from_val(args2.val_root, args2.anno_dir, args2.ypred_dir, subdir,
                                         num_pairs=args2.num_pairs, seed=args2.seed, alpha=args2.alpha,
                                         rd_cfg=rd_cfg, metrics_csv=args2.metrics_csv, psnr_min=args2.psnr_min,
                                         progress=prog, task_id=task_one, draw=(not args2.no_draw),
                                         size=args2.size, resize_mode=args2.resize_mode, fill=args2.fill,
                                         anno_canvas=args2.anno_canvas)
                    # Read back average row
                    mcsv = os.path.join(subdir, 'metrics.csv')
                    avg_iou = avg_dice = None
                    with open(mcsv, 'r', newline='') as f:
                        rr = list(csv.reader(f))
                    for r in rr[1:3]:
                        if r and r[0].strip().lower().startswith('average_over_all_samples'):
                            try:
                                avg_iou = float(r[1]); avg_dice = float(r[2])
                            except Exception:
                                pass
                            break
                    # Read diagnostics summary
                    dsum = os.path.join(subdir, 'diagnostics_summary.csv')
                    empty_rate = mean_pred_area_frac = mean_kept_cc = None
                    if os.path.exists(dsum):
                        with open(dsum, 'r', newline='') as f:
                            rr2 = list(csv.reader(f))
                        if len(rr2) >= 2:
                            try:
                                empty_rate = float(rr2[1][0])
                                mean_pred_area_frac = float(rr2[1][1])
                                mean_kept_cc = float(rr2[1][2])
                            except Exception:
                                pass
                    summary_rows.append([
                        subdir,
                        rd_cfg.get('rd_Tl'), rd_cfg.get('rd_fdr_q'), rd_cfg.get('rd_th_seed_hi'), rd_cfg.get('rd_min_highz_pixels'),
                        rd_cfg.get('rd_min_area_frac'), rd_cfg.get('rd_open_r'), rd_cfg.get('rd_close_r'),
                        rd_cfg.get('rd_seed_smooth_sigma'), rd_cfg.get('rd_seed_open_r'), rd_cfg.get('rd_seed_dilate'),
                        rd_cfg.get('rd_bridge_r'), rd_cfg.get('rd_weight_mode'), rd_cfg.get('rd_weight_alpha'),
                        bool(rd_cfg.get('rd_allow_empty')), rd_cfg.get('rd_seed_percentile'),
                        avg_iou, avg_dice, empty_rate, mean_pred_area_frac, mean_kept_cc
                    ])
                except Exception as e:
                    print(f"[tune][FAIL] {subdir}: {e}")
                finally:
                    # close sub-task and advance the main task
                    try:
                        prog.update(task_runs, advance=1)
                        prog.remove_task(task_one)
                    except Exception:
                        pass
        # Write summary CSV
        with open(summary_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run_dir','rd_Tl','rd_fdr_q','rd_th_seed_hi','rd_min_highz_pixels','rd_min_area_frac','rd_open_r','rd_close_r',
                        'rd_seed_smooth_sigma','rd_seed_open_r','rd_seed_dilate','rd_bridge_r','rd_weight_mode','rd_weight_alpha',
                        'rd_allow_empty','rd_seed_percentile','avg_iou','avg_dice','empty_rate','mean_pred_area_frac','mean_kept_cc'])
            w.writerows(summary_rows)
        # Print top-5 by avg_dice
        summary_rows2 = [r for r in summary_rows if (r[-1] is not None)]
        summary_rows2.sort(key=lambda r: (r[-1], r[-2] if r[-2] is not None else -1), reverse=True)
        print("[tune] Top-5 by avg_dice:")
        for r in summary_rows2[:5]:
            print(f"  dice={r[-1]:.4f}, iou={(r[-2] if r[-2] is not None else float('nan')):.4f}  -> {r[0]}")
        print(f"[tune] summary saved to {summary_csv}")
    else:
        ap2.print_help(sys.stderr)
        sys.exit(2)

"""
python experiment02-unregable-area.py inject_eval_from_val \
  --val_root  data/SynthRAD2023-Task1/test2D/ \
  --anno_dir  data/SynthRAD2023-Task1/test2D/anno \
  --ypred_dir output/SynthRAD_CycleGAN_noise0_bigbatch_keepratio_512/NC+R/img-residual/pred \
  --metrics_csv output/SynthRAD_CycleGAN_noise0_bigbatch_keepratio_512/NC+R/img-residual/metrics.csv \
  --out_dir   experiment-results/02-unregable-area \
  --psnr_min  28 \
  --num_pairs 1000 --alpha 1.0 --seed 0
  --size 512 --resize_mode keepratio --anno_canvas 512
  
python experiment02-unregable-area.py plot_hist \
  --csv experiment-results/02-unregable-area/metrics.csv \
  --out_dir experiment-results/02-unregable-area \
  --bins 40
  
python experiment02-unregable-area.py tune_rd \
  --val_root  data/SynthRAD2023-Task1/test2D/ \
  --anno_dir  data/SynthRAD2023-Task1/test2D/anno \
  --ypred_dir output/SynthRAD_CycleGAN_noise0_bigbatch_keepratio_512/NC+R/img-residual/pred \
  --metrics_csv output/SynthRAD_CycleGAN_noise0_bigbatch_keepratio_512/NC+R/img-residual/metrics.csv \
  --out_dir   experiment-results/02-unregable-area/tune \
  --psnr_min  28 \
  --no_draw \
  --num_pairs 200 --alpha 1.0 --seed 42 \
  --rd_Tl 1.2 1.5 1.8 \
  --rd_fdr_q 0.05 0.10 \
  --rd_th_seed_hi 0 2.5 \
  --rd_min_highz_pixels 2 5 \
  --size 512 --resize_mode keepratio --anno_canvas 512
  
python experiment02-unregable-area.py tune_rd \
  --val_root  data/SynthRAD2023-Task1/test2D/ \
  --anno_dir  data/SynthRAD2023-Task1/test2D/anno \
  --ypred_dir output/SynthRAD_CycleGAN_noise0_bigbatch_keepratio_512/NC+R/img-residual/pred \
  --metrics_csv output/SynthRAD_CycleGAN_noise0_bigbatch_keepratio_512/NC+R/img-residual/metrics.csv \
  --out_dir   experiment-results/02-unregable-area/tune_A \
  --psnr_min  28 --no_draw --num_pairs 2000 --seed 42 \
  --rd_Tl 1.6 1.7 1.8 1.9 2.0 \
  --rd_fdr_q 0.03 0.05 0.08 \
  --rd_th_seed_hi 0 \
  --rd_min_highz_pixels 2 \
  --size 512 --resize_mode keepratio --anno_canvas 512
  
python experiment02-unregable-area.py tune_rd \
  --val_root  data/SynthRAD2023-Task1/test2D/ \
  --anno_dir  data/SynthRAD2023-Task1/test2D/anno \
  --ypred_dir output/SynthRAD_CycleGAN_noise0_bigbatch/NC+R/img-residual/pred \
  --metrics_csv output/SynthRAD_CycleGAN_noise0_bigbatch/NC+R/img-residual/metrics.csv \
  --out_dir   experiment-results/02-unregable-area/tune_256_resize \
  --size 256 --resize_mode resize \
  --psnr_min  27 \
  --no_draw \
  --num_pairs 1000 --alpha 1.0 --seed 42 \
  --rd_Tl 1.2 1.5 1.8 \
  --rd_fdr_q 0.05 0.10 \
  --rd_th_seed_hi 0 2.5 \
  --rd_min_highz_pixels 2 5
"""
