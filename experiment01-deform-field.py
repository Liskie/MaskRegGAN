

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment — DF error & calibration vs. noise level (BRATS val/test)
-------------------------------------------------------------------
- Loads trained Generator (netG_A2B.pth) and Registrator (R_A.pt/pth).
- On the validation/test split, constructs controlled RandomAffine noise
  ONLY on the target (B) side to form y_noisy and a ground-truth DF_true.
- Runs either R-only (moving=y, fixed=y_noisy) or E2E (moving=G(x), fixed=y_noisy)
  to get DF_pred, then computes:
    * Figure A: x = |DF_true| (mean or P95), y = |DF_pred - DF_true| (EPE)
    * Figure B: x = |DF_true|,              y = |DF_pred|
  Fit a line and report slope/intercept.

Usage example:
  python experiment01-deform-field.py \
      --config CycleGAN-noise3-NC+R(RegGAN)-trial.yaml \
      --levels 0,1,2,3,4,5 \
      --mode e2e \
      --metric p95 \
      --spacing-mm 0 0 \
      --out_dir ./output/CycleGAN_noise3_trial/NC+R/df_eval/

Notes:
- Results are in mm if --spacing-mm sy sx > 0; otherwise in pixels.
- Background is assumed to be exactly -1.0 (same as your pipeline). We use B_noisy > -1 as ROI.
- Noise follows your training recipe: degrees=±L°, translate=±2%·L, scale=1±2%·L.
- Internally we invert the predicted backward flow to a forward (moving→fixed) field so EPE is computed against DF_true under the same convention.
"""
import argparse
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Rich progress (optional)
try:
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn, MofNCompleteColumn, \
        TimeRemainingColumn
except Exception:
    Progress = None

# Project modules
from trainer.utils import Resize, ToTensor
from trainer.datasets import ValDataset
from models.CycleGan import Generator
from trainer.reg import Reg
from trainer.transformer import Transformer_2D


# -----------------------------
# Helpers
# -----------------------------

def get_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def seed_all(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_levels(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip() != '']


def voxel_to_mm(flow: torch.Tensor, spacing_mm: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """Convert 2D flow (B,2,H,W) from voxels to mm given (sy, sx)."""
    if spacing_mm is None:
        return flow
    sy, sx = spacing_mm
    if sy <= 0 or sx <= 0:
        return flow
    scale = torch.tensor([sx, sy], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
    return flow * scale


def mask_from_target(t: torch.Tensor) -> torch.Tensor:
    """Foreground mask (B,1,H,W) using -1 background convention (no extra dim)."""
    return (t > -1.0)


def mag_stats(flow_2_bhw: torch.Tensor, mask_b1hw: torch.Tensor) -> Tuple[float, float]:
    """Return mean and P95 of |flow| over ROI."""
    mag = torch.linalg.vector_norm(flow_2_bhw, ord=2, dim=1)  # (B,H,W)
    vals = mag[mask_b1hw.squeeze(1)]
    if vals.numel() == 0:
        return float('nan'), float('nan')
    return vals.mean().item(), torch.quantile(vals, 0.95).item()


def epe_stats(flow_pred: torch.Tensor, flow_true: torch.Tensor, mask_b1hw: torch.Tensor) -> Tuple[float, float]:
    epe = torch.linalg.vector_norm(flow_pred - flow_true, ord=2, dim=1)  # (B,H,W)
    vals = epe[mask_b1hw.squeeze(1)]
    if vals.numel() == 0:
        return float('nan'), float('nan')
    return vals.mean().item(), torch.quantile(vals, 0.95).item()

# --- Numerically stable linear fit y ≈ a*x + b ---
def safe_linear_fit(x, y, tiny: float = 1e-12):
    """Numerically stable linear fit y ≈ a*x + b.
    Filters non-finite values, works in float64; if var(x)~0, returns (0, mean(y)).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan, np.nan
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_d = x - x_mean
    var_x = float(np.dot(x_d, x_d))
    if var_x <= tiny:
        return 0.0, y_mean
    cov_xy = float(np.dot(x_d, (y - y_mean)))
    a = cov_xy / var_x
    b = y_mean - a * x_mean
    return float(a), float(b)


def sample_affine_params(level: int, H: int, W: int):
    """Return angle(deg), translate(px tuple), scale(scalar), shear(0,0)."""
    # Match your training recipe: degrees=±L; translate=±0.02*L (fraction of size); scale=1±0.02*L
    ang = random.uniform(-level, level)
    tx = random.uniform(-0.02 * level * W, 0.02 * level * W)
    ty = random.uniform(-0.02 * level * H, 0.02 * level * H)
    smin, smax = 1 - 0.02 * level, 1 + 0.02 * level
    sc = random.uniform(smin, smax)
    return ang, (tx, ty), sc, (0.0, 0.0)


def forward_affine_displacement(H: int, W: int, angle_deg: float, translate_px: Tuple[float, float], scale: float,
                                 center_xy: Tuple[float, float]) -> torch.Tensor:
    """
    Compute forward displacement field (2,H,W) in *pixels* for the given affine
    (rotation+isotropic scale+translation) around center.
    """
    device = 'cpu'
    dtype = torch.float32
    cx, cy = center_xy  # (x, y)
    # Build grid of absolute pixel coords (x,y)
    # X: (H,W) from 0..W-1, Y: 0..H-1
    xs = torch.arange(W, dtype=dtype)
    ys = torch.arange(H, dtype=dtype)
    X, Y = torch.meshgrid(ys, xs, indexing='ij')  # X=Y indices; careful: first is row (y), second is col (x)
    X = X.to(dtype)  # (H,W) rows (y)
    Y = Y.to(dtype)  # (H,W) cols (x)
    # Convert to (x,y)
    x = Y
    y = X
    # Shift to center
    x_c = x - cx
    y_c = y - cy

    theta = angle_deg * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    # Apply rotation+scale
    x_p = scale * (c * x_c - s * y_c)
    y_p = scale * (s * x_c + c * y_c)
    # Shift back and translate
    tx, ty = translate_px
    x_new = x_p + cx + tx
    y_new = y_p + cy + ty

    dx = x_new - x
    dy = y_new - y
    disp = torch.stack([dx, dy], dim=0)  # (2,H,W)
    return disp



def apply_affine_tensor(img: torch.Tensor, angle_deg: float, translate_px: Tuple[float, float], scale: float,
                        center_xy: Tuple[float, float]) -> torch.Tensor:
    """Apply affine to a single tensor image (1,H,W) using torchvision F.affine with fill=-1."""
    return TF.affine(
        img, angle=angle_deg, translate=(translate_px[0], translate_px[1]), scale=scale, shear=(0.0, 0.0),
        interpolation=InterpolationMode.BILINEAR, center=center_xy, fill=-1.0
    )


# --- Field conversion: backward (fixed→moving) to forward (moving→fixed) ---

def _make_pixel_grid(n: int, h: int, w: int, device, dtype):
    """Return base pixel grid of shape (N,H,W,2) with columns (x,y)."""
    xs = torch.arange(w, device=device, dtype=dtype)
    ys = torch.arange(h, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')  # Y=row, X=col
    grid = torch.stack([X, Y], dim=-1)  # (H,W,2) = (x,y)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)   # (N,H,W,2)
    return grid


def _pixels_to_norm(grid_xy: torch.Tensor, h: int, w: int, align_corners: bool = False) -> torch.Tensor:
    """Convert pixel coordinates to normalized [-1,1] coords for grid_sample.
    grid_xy: (N,H,W,2) with (x,y) in pixels.
    Returns (N,H,W,2) normalized.
    """
    x = grid_xy[..., 0]
    y = grid_xy[..., 1]
    if align_corners:
        x_n = 2.0 * x / (w - 1) - 1.0
        y_n = 2.0 * y / (h - 1) - 1.0
    else:
        x_n = 2.0 * (x + 0.5) / w - 1.0
        y_n = 2.0 * (y + 0.5) / h - 1.0
    return torch.stack([x_n, y_n], dim=-1)


def invert_backward_flow(bw: torch.Tensor, n_iter: int = 20, align_corners: bool = False) -> torch.Tensor:
    """Invert a backward displacement field b (fixed→moving, defined on fixed grid)
    to a forward field f (moving→fixed, defined on moving grid) by fixed-point iteration:
        f(y) = - b(y + f(y)).
    Args:
        bw: (N,2,H,W) in *pixels*.
        n_iter: number of fixed-point iterations.
        align_corners: matches the align_corners used in grid_sample for sampling b.
    Returns:
        fwd: (N,2,H,W) forward field in pixels (on moving grid).
    """
    assert bw.dim() == 4 and bw.size(1) == 2
    N, _, H, W = bw.shape
    device, dtype = bw.device, bw.dtype

    # Base pixel grid (N,H,W,2) with (x,y)
    base_xy = _make_pixel_grid(N, H, W, device, dtype)

    # Initialize f ≈ -b as a starting guess (on moving grid)
    f = -bw.clone()

    for _ in range(max(1, n_iter)):
        # Positions in fixed grid where to evaluate b: y + f(y)
        pos_xy = base_xy + f.permute(0, 2, 3, 1)  # (N,H,W,2)
        grid_norm = _pixels_to_norm(pos_xy, H, W, align_corners=align_corners)
        # Sample b at these positions
        b_at = F.grid_sample(
            bw, grid_norm, mode='bilinear', padding_mode='border', align_corners=align_corners
        )  # (N,2,H,W)
        # Fixed-point update: f := - b(y + f)
        f = -b_at

    return f

# --- Extra helpers for qualitative visualization ---

def warp_with_backward(src_bchw: torch.Tensor, bw_b2hw: torch.Tensor, *, align_corners: bool = False) -> torch.Tensor:
    """Warp `src` from moving→fixed using a backward field `bw` defined on fixed grid.
    Uses the same geometry convention as training (`align_corners=False`).
    """
    N, C, H, W = src_bchw.shape
    base_xy = _make_pixel_grid(N, H, W, src_bchw.device, src_bchw.dtype)  # (N,H,W,2)
    pos_xy  = base_xy + bw_b2hw.permute(0, 2, 3, 1)                        # y + b(y)
    grid    = _pixels_to_norm(pos_xy, H, W, align_corners=align_corners)
    return F.grid_sample(src_bchw, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)


def flow_to_rgb(flow_2hw: torch.Tensor, clip_percentile: float = 95.0):
    """Encode a 2D flow (2,H,W) to an RGB image via HSV: hue=angle, value≈magnitude.
    Returns (rgb_uint8, clip_mag).
    """
    from matplotlib.colors import hsv_to_rgb
    assert flow_2hw.dim() == 3 and flow_2hw.size(0) == 2
    u = flow_2hw[0].detach().cpu()
    v = flow_2hw[1].detach().cpu()
    mag = torch.sqrt(u * u + v * v)
    ang = torch.atan2(v, u)  # [-pi, pi]
    if mag.numel() == 0:
        mclip = 1.0
    else:
        mclip = float(torch.quantile(mag.flatten(), clip_percentile / 100.0).item())
        mclip = max(mclip, 1e-6)
    mag_n = (mag / mclip).clamp(0, 1).numpy()
    hue   = ((ang.numpy() + np.pi) / (2.0 * np.pi))  # [0,1)
    sat   = np.ones_like(hue)
    val   = mag_n
    hsv   = np.stack([hue, sat, val], axis=-1)
    rgb   = (hsv_to_rgb(hsv) * 255.0).astype(np.uint8)
    return rgb, mclip


def add_grid(ax, H: int, W: int, step: int = 16, color: str = 'w', alpha: float = 0.25, lw: float = 0.6):
    for x in range(0, W, step):
        ax.axvline(x, color=color, alpha=alpha, lw=lw)
    for y in range(0, H, step):
        ax.axhline(y, color=color, alpha=alpha, lw=lw)


def overlay_quiver(ax, flow_2hw: torch.Tensor, stride: int = 16, color: str = 'k', alpha: float = 0.7, scale: float = 1.0):
    """Overlay sparse arrows for a (2,H,W) flow on an image axis (image coords: +y is down)."""
    u = flow_2hw[0].detach().cpu().numpy()
    v = flow_2hw[1].detach().cpu().numpy()
    H, W = u.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    uu = u[::stride, ::stride]
    vv = v[::stride, ::stride]
    # 注意：图像坐标 +y 已经向下，所以这里不要再取反
    ax.quiver(xx, yy, uu, vv, angles='xy', scale_units='xy', scale=1.0/scale,
              color=color, alpha=alpha, width=0.003)


# --- Flow legend helpers ---
def _make_flow_colorwheel(size: int = 96) -> np.ndarray:
    """Return an RGBA image of an HSV color wheel (hue=angle, full sat/val)."""
    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    rr = np.sqrt(xx*xx + yy*yy)
    ang = np.arctan2(yy, xx)  # [-pi, pi]
    hue = (ang + np.pi) / (2*np.pi)
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    from matplotlib.colors import hsv_to_rgb
    rgb = (hsv_to_rgb(np.stack([hue, sat, val], axis=-1)) * 255.0).astype(np.uint8)
    # alpha: 1 inside unit circle, 0 outside
    alpha = (rr <= 1.0).astype(np.uint8) * 255
    rgba = np.dstack([rgb, alpha])
    return rgba


def add_flow_legend(ax, clip_mag: float):
    """Add a hue-direction color wheel and a magnitude bar (0..P95≈clip_mag px) as insets."""
    # Color wheel inset (top-left)
    wheel = _make_flow_colorwheel(96)
    ax_w = inset_axes(ax, width=1.2, height=1.2, loc='upper left', borderpad=0.6)
    ax_w.imshow(wheel)
    ax_w.set_xticks([]); ax_w.set_yticks([]); ax_w.set_title('dir', fontsize=8)
    for spine in ax_w.spines.values():
        spine.set_linewidth(0.5)

    # Magnitude bar inset (right)
    from matplotlib.colors import hsv_to_rgb
    H = 1; W = 128
    h = np.ones((H, W), dtype=np.float32) * 0.0  # hue fixed (red), sat=1
    s = np.ones((H, W), dtype=np.float32)
    v = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    rgb = (hsv_to_rgb(np.stack([h, s, v], axis=-1)) * 255.0).astype(np.uint8)
    ax_b = inset_axes(ax, width=2.2, height=0.25, loc='lower right', borderpad=0.8)
    ax_b.imshow(rgb, aspect='auto')
    ax_b.set_xticks([0, W-1])
    ax_b.set_xticklabels(['0', f'{clip_mag:.1f} px'], fontsize=8)
    ax_b.set_yticks([])
    ax_b.set_title('|u| (≈P95 scale)', fontsize=8)
    for spine in ax_b.spines.values():
        spine.set_linewidth(0.5)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--levels', type=str, default='0,1,2,3,4,5')
    ap.add_argument('--mode', type=str, choices=['e2e', 'ronly'], default='e2e',
                   help='e2e: DF_pred from R(G(x), y_noisy). ronly: DF_pred from R(y, y_noisy).')
    ap.add_argument('--metric', type=str, choices=['mean', 'p95'], default='p95',
                   help='x-axis aggregation for |DF_true| and y aggregation as described.')
    ap.add_argument('--spacing-mm', type=float, nargs=2, default=(0.0, 0.0), help='(sy sx) spacing in mm/px (0 0 = pixel)')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--limit', type=int, default=0, help='Max number of batches (0=all)')
    ap.add_argument('--envelope', type=str, choices=['linear', 'rss', 'power', 'none'], default='linear',
                    help='Draw EPE upper/lower envelopes in Fig A using |DF_pred| vs |DF_true| calibration.\n'
                         ' linear: use yB ≈ aB*x + bB;  rss: use y ≈ √((k x)^2 + β^2);  power: use y ≈ c·x^p + b;  none: disable.')
    ap.add_argument('--rss-fit', type=str, choices=['nls', 'squared_ols'], default='nls',
                    help='How to fit the RSS curve y ≈ √((k x)^2 + β^2): '
                         'nls = nonlinear least squares on (x,y); '
                         'squared_ols = linear OLS on (y^2 vs x^2).')
    ap.add_argument('--viz-all-points', action='store_true',
                    help='If set, render a 2x3 panel for every data point (batch) contributing to the scatter.')
    ap.add_argument('--viz-sample-index', type=int, default=0,
                    help='Sample index within a batch to visualize for each point (default: 0).')
    ap.add_argument('--viz-subdir', type=str, default='panels_all',
                    help='Subdirectory under out_dir to save per-point panels.')
    args = ap.parse_args()

    seed_all(args.seed)
    cfg = get_config(args.config)

    use_cuda = bool(cfg.get('cuda', True)) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    size = int(cfg['size'])
    batch_size = int(cfg.get('batchSize', 1))
    num_workers = int(cfg.get('n_cpu', 0))
    val_root = cfg.get('val_dataroot', cfg.get('dataroot'))

    # Models
    G = Generator(cfg['input_nc'], cfg['output_nc']).to(device)
    R = Reg(cfg['size'], cfg['size'], cfg['input_nc'], cfg['input_nc']).to(device)
    G.eval(); R.eval()

    # Extra warper for side-by-side comparison (align_corners=True internally; expects flow as (y,x))
    T2D = Transformer_2D().to(device)
    T2D.eval()

    g_path = os.path.join(cfg['save_root'], 'netG_A2B.pth')
    r_path = os.path.join(cfg['save_root'], 'R_A.pt')
    if not os.path.exists(r_path):
        alt = os.path.join(cfg['save_root'], 'R_A.pth')
        if os.path.exists(alt):
            r_path = alt
    if not (os.path.exists(g_path) and os.path.exists(r_path)):
        raise FileNotFoundError(f"Missing weights: {g_path} or {r_path}")

    G.load_state_dict(torch.load(g_path, map_location=device))
    R.load_state_dict(torch.load(r_path, map_location=device))

    # Data
    val_tf = [ToTensor(), Resize(size_tuple=(size, size))]
    val_ds = ValDataset(val_root, transforms_=val_tf, unaligned=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    levels = parse_levels(args.levels)
    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, args.viz_subdir)
    os.makedirs(viz_dir, exist_ok=True)

    # Accumulators for plots
    x_true_all = []   # aggregation of |DF_true|
    y_epe_all = []    # aggregation of EPE = |DF_pred - DF_true|
    y_pred_all = []   # aggregation of |DF_pred|
    level_tags = []
    # For bound diagnostics (saved to CSV)
    env_rows = []  # list of dicts per batch: level, batch_idx, x(|true|), r(|pred|), e(EPE), lower, upper, d_lower, d_upper, tag

    metric_name = args.metric
    spacing = tuple(args.spacing_mm) if (args.spacing_mm[0] > 0 and args.spacing_mm[1] > 0) else None
    unit = 'mm' if spacing else 'px'

    print(f"Running evaluation on {val_root} | mode={args.mode} | metric={metric_name} | unit={unit}")
    print("[EPE] Scheme A active: comparing FORWARD fields (moving→fixed); backward from R assumed (y,x), inverted with align_corners=True.")

    # Progress bar over total batches (levels × per-level limit)
    prog = None
    prog_task = None
    try:
        if Progress is not None:
            # We can estimate total batches to process
            len_loader = len(val_loader)
            limit_per_level = args.limit if args.limit > 0 else len_loader
            total_batches_est = limit_per_level * len(levels)
            prog = Progress(
                TextColumn("[bold cyan]viz"), BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn()
            )
            prog.start()
            prog_task = prog.add_task("render", total=total_batches_est)
    except Exception:
        prog = None

    with torch.no_grad():
        for L in levels:
            print(f"\n==> Noise level {L}")
            random.seed(args.seed + L)
            np.random.seed(args.seed + L)
            torch.manual_seed(args.seed + L)

            os.makedirs(os.path.join(viz_dir, f'L{L}'), exist_ok=True)

            n_batches = 0
            for i, batch in enumerate(val_loader):
                if args.limit and i >= args.limit:
                    break
                real_A = batch['A'].to(device, non_blocking=True)
                real_B = batch['B'].to(device, non_blocking=True)

                B, H, W = real_B.shape[1], real_B.shape[2], real_B.shape[3]  # B=channels (should be 1), H,W
                assert B == 1, "Expect single-channel MR slices"

                # Per-sample affine sampling & application on CPU/GPU tensor using TF.affine
                # We need per-sample DF_true as well
                batch_true_list = []
                batch_noisy_list = []
                centers = []
                angles = []
                trans_list = []
                scales = []

                for k in range(real_B.size(0)):
                    # Tensor image (1,H,W)
                    imgB = real_B[k]
                    # Center definition consistent with TF.affine
                    cx = (W - 1) / 2.0
                    cy = (H - 1) / 2.0
                    ang, (tx, ty), sc, _ = sample_affine_params(L, H, W)

                    # Apply affine to B only
                    B_noisy = apply_affine_tensor(imgB, ang, (tx, ty), sc, (cx, cy))

                    # Build DF_true in pixels
                    df_true = forward_affine_displacement(H, W, ang, (tx, ty), sc, (cx, cy))  # (2,H,W) CPU
                    batch_true_list.append(df_true.unsqueeze(0))       # (1,2,H,W)
                    batch_noisy_list.append(B_noisy.unsqueeze(0))      # (1,1,H,W)
                    centers.append((cx, cy))
                    angles.append(ang)
                    trans_list.append((tx, ty))
                    scales.append(sc)

                DF_true = torch.cat(batch_true_list, dim=0).to(device)   # (N,2,H,W)
                B_noisy = torch.cat(batch_noisy_list, dim=0).to(device)  # (N,1,H,W)

                # Moving image for R
                if args.mode == 'e2e':
                    fake_B = G(real_A)
                    moving = fake_B
                else:
                    moving = real_B

                # Predict flow (in voxels/pixels)
                DF_pred = R(moving, B_noisy)  # (N,2,H,W)

                # --- Diagnostics: check magnitude/units of predicted flow (raw backward) ---
                # q_raw = torch.quantile(torch.linalg.vector_norm(DF_pred, 2, dim=1), 0.95).item()
                # print(f"[diag] L={L} i={i}  P95(|DF_pred_raw(backward)|)={q_raw:.6f}   H,W={H},{W}")
                DF_pred_bw = DF_pred  # keep backward flow (as produced by R); convention: (y, x), align_corners=True (T2D)

                # Scheme A: Convert backward (fixed→moving) to forward (moving→fixed) under the SAME convention as Transformer_2D
                # 1) Channel order: (y,x) → (x,y) for our inversion utility
                bw_xy = DF_pred_bw[:, [1, 0], ...]
                # 2) Use align_corners=True during sampling in inversion to match T2D's normalization
                DF_pred = invert_backward_flow(bw_xy, n_iter=20, align_corners=True)

                # --- Diagnostics: after inversion to forward (moving→fixed) ---
                # q_fwd = torch.quantile(torch.linalg.vector_norm(DF_pred, 2, dim=1), 0.95).item()
                # q_true = torch.quantile(torch.linalg.vector_norm(DF_true, 2, dim=1), 0.95).item()
                # print(f"[diag] L={L} i={i}  P95(|DF_pred_fwd|)={q_fwd:.6f}   P95(|DF_true|)={q_true:.6f}   unit={unit}")

                # ROI for forward-domain comparison: intersection of moving & fixed foregrounds
                roi_moving = mask_from_target(moving)
                roi_fixed  = mask_from_target(B_noisy)
                roi = roi_moving & roi_fixed

                # Convert to mm if requested
                DF_true_eff = voxel_to_mm(DF_true, spacing)
                DF_pred_eff = voxel_to_mm(DF_pred, spacing)

                # Aggregate stats per batch (averaged across samples in batch)
                mean_true, p95_true = mag_stats(DF_true_eff, roi)
                mean_pred, p95_pred = mag_stats(DF_pred_eff, roi)
                mean_epe, p95_epe = epe_stats(DF_pred_eff, DF_true_eff, roi)

                if metric_name == 'p95':
                    x_val = p95_true
                    yA_val = p95_epe
                    yB_val = p95_pred
                else:
                    x_val = mean_true
                    yA_val = mean_epe
                    yB_val = mean_pred

                x_true_all.append(x_val)
                y_epe_all.append(yA_val)
                y_pred_all.append(yB_val)
                level_tags.append(L)

                # --- Theoretical bounds per batch point (using actual r=|pred| and t=|true| under the chosen metric) ---
                lower_bound = abs(yB_val - x_val)
                upper_bound = yB_val + x_val
                d_lower = abs(yA_val - lower_bound)
                d_upper = abs(yA_val - upper_bound)
                tag = 'near-lower' if d_lower <= d_upper else 'near-upper'
                env_rows.append({
                    'level': L,
                    'batch_idx': i,
                    'x_true': x_val,
                    'r_pred': yB_val,
                    'e_epe': yA_val,
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'd_lower': d_lower,
                    'd_upper': d_upper,
                    'tag': tag,
                })

                # ===== 2x3 qualitative panel per data point =====
                if args.viz_all_points or i == 0:
                    k = min(max(0, args.viz_sample_index), moving.size(0) - 1)
                    Hk, Wk = H, W

                    # Row 1: y_pred (moving), DF_pred (forward), y_resample (warp by DF_pred_bw)
                    y_pred = moving[k:k + 1, ...]  # (1,1,H,W), in [-1,1]
                    df_pred_bw = DF_pred_bw[k, ...]  # (2,H,W), backward
                    # y_resample = warp_with_backward(y_pred, df_pred_bw.unsqueeze(0), align_corners=False)[0, 0]

                    # Transformer_2D warp for side-by-side comparison
                    # NOTE: Transformer_2D expects flow as (y,x) and uses align_corners=True internally
                    # flow_yx = df_pred_bw[[1, 0], ...].unsqueeze(0)
                    # y_resample_t2d = T2D(y_pred, flow_yx)[0, 0]
                    y_resample = T2D(y_pred, df_pred_bw.unsqueeze(0))[0, 0]

                    # Row 2: y_true, DF_true (forward), y_noise (fixed)
                    y_true = real_B[k, 0]  # (H,W)
                    df_true_fw = DF_true[k, ...]  # (2,H,W), forward (pixels)
                    df_pred_fw = DF_pred[k, ...]  # (2,H,W), forward (pixels)
                    y_noise = B_noisy[k, 0]       # (H,W)

                    # Flow visualizations (forward fields)
                    rgb_pred, pred_clip = flow_to_rgb(df_pred_fw)
                    rgb_true, true_clip = flow_to_rgb(df_true_fw)

                    # Prepare images to [0,1]
                    def to01(x: torch.Tensor) -> np.ndarray:
                        return ((x.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).numpy()

                    y_pred_img = to01(y_pred[0, 0])
                    y_res_img  = to01(y_resample)
                    y_true_img = to01(y_true)
                    y_noise_img= to01(y_noise)

                    fig = plt.figure(figsize=(12.8, 7))
                    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.55], wspace=0.15, hspace=0.25)

                    ax_y_pred  = fig.add_subplot(gs[0, 0])
                    ax_y_res   = fig.add_subplot(gs[0, 1])
                    ax_df_pred = fig.add_subplot(gs[0, 2])
                    ax_y_true  = fig.add_subplot(gs[1, 0])
                    ax_y_noise = fig.add_subplot(gs[1, 1])
                    ax_df_true = fig.add_subplot(gs[1, 2])
                    ax_wheel   = fig.add_subplot(gs[0, 3])
                    ax_mag     = fig.add_subplot(gs[1, 3])

                    # (1,1) y_pred + grid
                    ax = ax_y_pred
                    ax.imshow(y_pred_img, cmap='gray', vmin=0, vmax=1)
                    add_grid(ax, Hk, Wk, step=16, color='w', alpha=0.25, lw=0.6)
                    ax.set_title('y_pred (moving)')
                    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
                    ax.set_xticks(list(range(0, Wk, 32)) + [Wk-1])
                    ax.set_yticks(list(range(0, Hk, 32)) + [Hk-1])
                    ax.set_aspect('equal')

                    # (1,2) y_resample = warp(y_pred, DF_pred_bw)
                    ax = ax_y_res
                    ax.imshow(y_res_img, cmap='gray', vmin=0, vmax=1)
                    add_grid(ax, Hk, Wk, step=16, color='w', alpha=0.25, lw=0.6)
                    ax.set_title('y_resample = warp(y_pred, DF_pred_bw)')
                    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
                    ax.set_xticks(list(range(0, Wk, 32)) + [Wk-1])
                    ax.set_yticks(list(range(0, Hk, 32)) + [Hk-1])
                    ax.set_aspect('equal')

                    # (1,3) DF_pred (forward)
                    ax = ax_df_pred
                    ax.imshow(rgb_pred)
                    overlay_quiver(ax, df_pred_fw, stride=16, color='k', alpha=0.85, scale=1.0)
                    ax.set_title(f'DF_pred (forward)  hue=dir  val=|u|/P95\nP95(|u|)≈{pred_clip:.2f} {unit}')

                    # (2,1) y_true + grid
                    ax = ax_y_true
                    ax.imshow(y_true_img, cmap='gray', vmin=0, vmax=1)
                    add_grid(ax, Hk, Wk, step=16, color='w', alpha=0.25, lw=0.6)
                    ax.set_title('y_true')
                    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
                    ax.set_xticks(list(range(0, Wk, 32)) + [Wk-1])
                    ax.set_yticks(list(range(0, Hk, 32)) + [Hk-1])
                    ax.set_aspect('equal')

                    # (2,2) y_noise (fixed) + grid
                    ax = ax_y_noise
                    ax.imshow(y_noise_img, cmap='gray', vmin=0, vmax=1)
                    add_grid(ax, Hk, Wk, step=16, color='w', alpha=0.25, lw=0.6)
                    ax.set_title('y_noise (fixed)')
                    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
                    ax.set_xticks(list(range(0, Wk, 32)) + [Wk-1])
                    ax.set_yticks(list(range(0, Hk, 32)) + [Hk-1])
                    ax.set_aspect('equal')

                    # (2,3) DF_true (forward)
                    ax = ax_df_true
                    ax.imshow(rgb_true)
                    overlay_quiver(ax, df_true_fw, stride=16, color='k', alpha=0.85, scale=1.0)
                    ax.set_title(f'DF_true (forward)  hue=dir  val=|u|/P95\nP95(|u|)≈{true_clip:.2f} {unit}')

                    # External legends
                    wheel = _make_flow_colorwheel(128)
                    ax_wheel.imshow(wheel)
                    ax_wheel.set_title('Direction (Hue)', fontsize=10)
                    ax_wheel.set_xticks([]); ax_wheel.set_yticks([])
                    for spine in ax_wheel.spines.values():
                        spine.set_linewidth(0.6)

                    from matplotlib.colors import hsv_to_rgb
                    Hbar, Wbar = 256, 28
                    h = np.zeros((Hbar, Wbar), dtype=np.float32)
                    s = np.ones((Hbar, Wbar), dtype=np.float32)
                    v = np.linspace(0, 1, Hbar, dtype=np.float32)[:, None]
                    v = np.repeat(v, Wbar, axis=1)
                    rgb_bar = (hsv_to_rgb(np.stack([h, s, v], axis=-1)) * 255.0).astype(np.uint8)
                    clip_max = max(pred_clip, true_clip)

                    ax_mag.set_title('|u| (Value)', fontsize=10)
                    ax_mag.set_xticks([]); ax_mag.set_yticks([])
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
                    ax_mag_bar = _inset_axes(ax_mag, width="28%", height="92%", loc='center', borderpad=0.2)
                    ax_mag_bar.imshow(rgb_bar, aspect='auto', origin='lower')
                    ax_mag_bar.set_xticks([])
                    ax_mag_bar.set_yticks([0, Hbar-1])
                    ax_mag_bar.set_yticklabels(['0', f'{clip_max:.1f} {unit}'], fontsize=9)
                    for spine in ax_mag_bar.spines.values():
                        spine.set_linewidth(0.6)

                    # Annotate point metrics & bound proximity
                    ax_mag.text(0.5, -0.10,
                                (f'|true|({metric_name})={x_val:.3f} {unit}\n'
                                 f'|pred|({metric_name})={yB_val:.3f} {unit}\n'
                                 f'EPE({metric_name})={yA_val:.3f} {unit}\n'
                                 f'lower=|r−t|={lower_bound:.3f}, upper=r+t={upper_bound:.3f}\n'
                                 f'→ {tag} (Δlower={d_lower:.3f}, Δupper={d_upper:.3f})'),
                                ha='center', va='top', transform=ax_mag.transAxes, fontsize=9)

                    # plt.tight_layout()
                    panel_path = os.path.join(viz_dir, f'L{L}', f"panel_b{i}_k{k}_{tag}.png")
                    plt.savefig(panel_path, dpi=220)
                    plt.close(fig)
                    # print(f"[panel] saved {panel_path}")

                # Progress advance per processed batch
                if prog is not None and prog_task is not None:
                    prog.advance(prog_task, 1)

                n_batches += 1

            print(f"Level {L}: processed {n_batches} batches")

    # Convert to numpy arrays
    x_arr = np.asarray(x_true_all)
    yA_arr = np.asarray(y_epe_all)
    yB_arr = np.asarray(y_pred_all)
    L_arr = np.asarray(level_tags)

    # Pre-compute calibration for |DF_pred| vs |DF_true| to build envelopes for Fig A
    maskB_env = np.isfinite(x_arr) & np.isfinite(yB_arr)
    if maskB_env.sum() >= 2:
        # Linear fit on (x, y)
        lin_aB, lin_bB = safe_linear_fit(x_arr[maskB_env], yB_arr[maskB_env])

        # RSS fit: y ≈ sqrt((k x)^2 + beta^2)
        if args.rss_fit == 'squared_ols':
            # Old behavior: fit y^2 ≈ (k^2) x^2 + (β^2)
            x2 = (x_arr[maskB_env] ** 2)
            y2 = (yB_arr[maskB_env] ** 2)
            rss_m, rss_c = np.polyfit(x2, y2, 1)
            rss_k = float(np.sqrt(max(rss_m, 0.0)))
            rss_beta = float(np.sqrt(max(rss_c, 0.0)))
        else:
            # Nonlinear least squares directly on (x, y) to avoid mean-vs-RMS mismatch
            x_fit_nls = x_arr[maskB_env]
            y_fit_nls = yB_arr[maskB_env]
            # Search ranges
            k_hi = max(2.0, 2.0 * float(max(lin_aB, 0.0)))
            b_hi = max(2.0, 2.0 * float(np.percentile(y_fit_nls, 95)))
            k_grid = np.linspace(0.0, k_hi, 201)
            b_grid = np.linspace(0.0, b_hi, 201)
            best = (np.inf, 0.0, 0.0)
            # Coarse grid search
            for kk in k_grid:
                rhat = np.sqrt((kk * x_fit_nls) ** 2 + (b_grid[0] ** 2))  # warm-up allocate
                for bb in b_grid:
                    rhat = np.sqrt((kk * x_fit_nls) ** 2 + (bb ** 2))
                    sse = np.sum((y_fit_nls - rhat) ** 2)
                    if sse < best[0]:
                        best = (sse, kk, bb)
            rss_k, rss_beta = float(best[1]), float(best[2])

        # Power-law fit: y ≈ c * x^p + b  (grid over p,b; solve c by least squares)
        power_c = power_p = power_b = np.nan
        try:
            x_fit_pw = x_arr[maskB_env]
            y_fit_pw = yB_arr[maskB_env]
            if np.all(np.isfinite(x_fit_pw)) and np.all(np.isfinite(y_fit_pw)):
                p_grid = np.linspace(0.25, 1.50, 101)  # exponent search
                b_hi = max(0.0, float(np.percentile(y_fit_pw, 50)))
                b_grid = np.linspace(0.0, max(1e-6, b_hi), 101)
                best = (np.inf, 0.0, 1.0, 0.0)  # (sse, c, p, b)
                x_fit_pw_clip = np.clip(x_fit_pw, 0.0, None)
                for pp in p_grid:
                    xpow = x_fit_pw_clip ** pp
                    denom = float(np.dot(xpow, xpow))
                    if denom <= 0:
                        continue
                    for bb in b_grid:
                        num = float(np.dot(xpow, (y_fit_pw - bb)))
                        cc = max(0.0, num / denom)  # constrain c ≥ 0
                        yhat = cc * xpow + bb
                        sse = float(np.sum((y_fit_pw - yhat) ** 2))
                        if sse < best[0]:
                            best = (sse, cc, pp, bb)
                power_c, power_p, power_b = float(best[1]), float(best[2]), float(best[3])
        except Exception:
            power_c = power_p = power_b = np.nan
    else:
        lin_aB = lin_bB = np.nan
        rss_k = rss_beta = np.nan
        power_c = power_p = power_b = np.nan

    # ----- Figure A: Error vs |DF_true| -----
    # Fit y = a*x + b
    mask = np.isfinite(x_arr) & np.isfinite(yA_arr)
    x_fit, yA_fit = x_arr[mask], yA_arr[mask]
    if len(x_fit) >= 2:
        aA, bA = safe_linear_fit(x_fit, yA_fit)
    else:
        aA, bA = np.nan, np.nan

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x_arr, yA_arr, c=L_arr, s=16, alpha=0.5, cmap='viridis')
    if np.isfinite(aA) and np.isfinite(bA):
        xx = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 100)
        plt.plot(xx, aA * xx + bA, linewidth=2)
        plt.title(f"Fig A — Error vs |DF_true| ({metric_name}, unit={unit})\nSlope={aA:.3f}, Intercept={bA:.3f}")
        # Draw theoretical envelopes for EPE using triangle inequality
        # EPE ∈ [ | |a| - |b| |, |a| + |b| ], where |a|≈|DF_pred|, |b|=|DF_true|
        xx_env = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 200)
        if args.envelope == 'linear' and np.isfinite(lin_aB) and np.isfinite(lin_bB):
            rhat = np.maximum(0.0, lin_aB * xx_env + lin_bB)  # estimated |DF_pred|
            y_lower = np.abs(rhat - xx_env)
            y_upper = rhat + xx_env
            plt.plot(xx_env, y_lower, linestyle='--', linewidth=1.5, label='lower ≈ |(aB−1)x + bB|')
            plt.plot(xx_env, y_upper, linestyle='--', linewidth=1.5, label='upper ≈ (aB+1)x + bB')
        elif args.envelope == 'rss' and np.isfinite(rss_k) and np.isfinite(rss_beta):
            rhat = np.sqrt((rss_k * xx_env) ** 2 + (rss_beta ** 2))
            y_lower = np.abs(rhat - xx_env)
            y_upper = rhat + xx_env
            plt.plot(xx_env, y_lower, linestyle='--', linewidth=1.5,
                     label=f'lower (RSS-{args.rss_fit}) ≈ |√((k x)^2+β^2) − x|')
            plt.plot(xx_env, y_upper, linestyle='--', linewidth=1.5,
                     label=f'upper (RSS-{args.rss_fit}) ≈ √((k x)^2+β^2) + x')
        elif args.envelope == 'power' and np.isfinite(power_c) and np.isfinite(power_p) and np.isfinite(power_b):
            rhat = np.maximum(0.0, power_c * (xx_env ** power_p) + power_b)
            y_lower = np.abs(rhat - xx_env)
            y_upper = rhat + xx_env
            plt.plot(xx_env, y_lower, linestyle='--', linewidth=1.5,
                     label=f'lower (Power) ≈ |{power_c:.3g}·x^{power_p:.2f}+{power_b:.3g} − x|')
            plt.plot(xx_env, y_upper, linestyle='--', linewidth=1.5,
                     label=f'upper (Power) ≈ {power_c:.3g}·x^{power_p:.2f}+{power_b:.3g} + x')
        if args.envelope != 'none':
            plt.legend(loc='best')
    else:
        plt.title(f"Fig A — Error vs |DF_true| ({metric_name}, unit={unit})")
    plt.xlabel(f"|DF_true| ({metric_name}, {unit})")
    plt.ylabel(f"|DF_pred - DF_true| ({metric_name}, {unit})")
    cbar = plt.colorbar(sc); cbar.set_label('noise level L')
    figA_path = os.path.join(args.out_dir, f"figA_error_vs_true_{args.mode}_{metric_name}.png")
    plt.tight_layout(); plt.savefig(figA_path, dpi=200)
    plt.close()

    # ----- Per-level Figure A (one plot per noise level) -----
    levels_unique = np.unique(L_arr)
    if len(levels_unique) > 1:
        Lmin, Lmax = int(np.min(levels_unique)), int(np.max(levels_unique))
        for Lv in levels_unique:
            maskL = (L_arr == Lv)
            x_sub = x_arr[maskL]
            yA_sub = yA_arr[maskL]

            plt.figure(figsize=(7, 5))
            sc = plt.scatter(x_sub, yA_sub, c=L_arr[maskL], s=16, alpha=0.5, cmap='viridis',
                             vmin=Lmin, vmax=Lmax)

            # Per-level linear fit (optional if enough points)
            mask_fit = np.isfinite(x_sub) & np.isfinite(yA_sub)
            if mask_fit.sum() >= 2:
                a_sub, b_sub = safe_linear_fit(x_sub[mask_fit], yA_sub[mask_fit])
                xx = np.linspace(max(0.0, float(np.nanmin(x_sub))), float(np.nanmax(x_sub)), 100)
                plt.plot(xx, a_sub * xx + b_sub, linewidth=2)
                plt.title(f"Fig A (L={int(Lv)}) — Error vs |DF_true| ({metric_name}, unit={unit})\nSlope={a_sub:.3f}, Intercept={b_sub:.3f}")
            else:
                plt.title(f"Fig A (L={int(Lv)}) — Error vs |DF_true| ({metric_name}, unit={unit})")

            plt.xlabel(f"|DF_true| ({metric_name}, {unit})")
            plt.ylabel(f"|DF_pred - DF_true| ({metric_name}, {unit})")
            cbar = plt.colorbar(sc); cbar.set_label('noise level L')

            figA_L_path = os.path.join(args.out_dir, f"figA_error_vs_true_{args.mode}_{metric_name}_L{int(Lv)}.png")
            plt.tight_layout(); plt.savefig(figA_L_path, dpi=200)
            plt.close()

    # ----- Figure B: |DF_pred| vs |DF_true| -----
    mask = np.isfinite(x_arr) & np.isfinite(yB_arr)
    x_fit, yB_fit = x_arr[mask], yB_arr[mask]
    if len(x_fit) >= 2:
        aB, bB = safe_linear_fit(x_fit, yB_fit)
    else:
        aB, bB = np.nan, np.nan

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x_arr, yB_arr, c=L_arr, s=16, alpha=0.5, cmap='viridis')
    if np.isfinite(aB) and np.isfinite(bB):
        xx = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 100)
        plt.plot(xx, aB * xx + bB, linewidth=2)
        plt.title(f"Fig B — |DF_pred| vs |DF_true| ({metric_name}, unit={unit})\nSlope={aB:.3f}, Intercept={bB:.3f}")
        # Optional RSS overlay for |DF_pred| vs |DF_true|
        if args.envelope == 'rss' and np.isfinite(rss_k) and np.isfinite(rss_beta):
            xx2 = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 200)
            rhat_rss = np.sqrt((rss_k * xx2) ** 2 + (rss_beta ** 2))
            plt.plot(xx2, rhat_rss, linestyle='--', linewidth=1.5,
                     label=f'RSS ({args.rss_fit}): √((k x)^2 + β^2), k={rss_k:.3f}, β={rss_beta:.3f}')
            plt.legend(loc='best')
        # Optional Power overlay for |DF_pred| vs |DF_true|
        if args.envelope == 'power' and np.isfinite(power_c) and np.isfinite(power_p) and np.isfinite(power_b):
            xx2 = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 200)
            yhat_power = np.maximum(0.0, power_c * (xx2 ** power_p) + power_b)
            plt.plot(xx2, yhat_power, linestyle='--', linewidth=1.5,
                     label=f'Power: {power_c:.3g}·x^{power_p:.2f}+{power_b:.3g}')
            plt.legend(loc='best')
    else:
        plt.title(f"Fig B — |DF_pred| vs |DF_true| ({metric_name}, unit={unit})")
        # Optional RSS overlay for |DF_pred| vs |DF_true|
        if args.envelope == 'rss' and np.isfinite(rss_k) and np.isfinite(rss_beta):
            xx2 = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 200)
            rhat_rss = np.sqrt((rss_k * xx2) ** 2 + (rss_beta ** 2))
            plt.plot(xx2, rhat_rss, linestyle='--', linewidth=1.5,
                     label=f'RSS ({args.rss_fit}): √((k x)^2 + β^2), k={rss_k:.3f}, β={rss_beta:.3f}')
            plt.legend(loc='best')
        # Optional Power overlay for |DF_pred| vs |DF_true|
        if args.envelope == 'power' and np.isfinite(power_c) and np.isfinite(power_p) and np.isfinite(power_b):
            xx2 = np.linspace(max(0.0, float(x_arr.min())), float(x_arr.max()), 200)
            yhat_power = np.maximum(0.0, power_c * (xx2 ** power_p) + power_b)
            plt.plot(xx2, yhat_power, linestyle='--', linewidth=1.5,
                     label=f'Power: {power_c:.3g}·x^{power_p:.2f}+{power_b:.3g}')
            plt.legend(loc='best')
    plt.xlabel(f"|DF_true| ({metric_name}, {unit})")
    plt.ylabel(f"|DF_pred| ({metric_name}, {unit})")
    cbar = plt.colorbar(sc); cbar.set_label('noise level L')
    figB_path = os.path.join(args.out_dir, f"figB_pred_vs_true_{args.mode}_{metric_name}.png")
    plt.tight_layout(); plt.savefig(figB_path, dpi=200)
    plt.close()

    # Save CSV
    csv_path = os.path.join(args.out_dir, f"summary_{args.mode}_{metric_name}.csv")
    import csv
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["level", f"|DF_true|_{metric_name}_{unit}", f"EPE_{metric_name}_{unit}", f"|DF_pred|_{metric_name}_{unit}"])
        for L, x, ya, yb in zip(L_arr.tolist(), x_arr.tolist(), yA_arr.tolist(), yB_arr.tolist()):
            w.writerow([L, x, ya, yb])

    if prog is not None:
        try:
            prog.stop()
        except Exception:
            pass

    # Save per-point bound diagnostics
    env_csv = os.path.join(args.out_dir, f"envelope_metrics_{args.mode}_{metric_name}.csv")
    import csv as _csv
    with open(env_csv, 'w', newline='') as fenv:
        w = _csv.writer(fenv)
        w.writerow(['level', 'batch_idx', f'|DF_true|_{metric_name}_{unit}', f'|DF_pred|_{metric_name}_{unit}',
                    f'EPE_{metric_name}_{unit}', 'lower=|r−t|', 'upper=r+t', 'Δlower', 'Δupper', 'tag'])
        for row in env_rows:
            w.writerow([row['level'], row['batch_idx'], row['x_true'], row['r_pred'], row['e_epe'],
                        row['lower'], row['upper'], row['d_lower'], row['d_upper'], row['tag']])
    print(f"Saved envelope diagnostics: {env_csv}")

    print(f"Saved:\n  {figA_path}\n  {figB_path}\n  {csv_path}")


if __name__ == '__main__':
    main()