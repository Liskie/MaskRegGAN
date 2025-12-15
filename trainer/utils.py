import random
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
import wandb
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage
# 放在 utils.py 顶部（如未导入）
import matplotlib.pyplot as plt
import os
import math
from skimage.metrics import structural_similarity


class Resize():
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])

        tensor = tensor.squeeze(0)

        return tensor  # 1, 64, 128, 128


class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0).copy()
        return torch.from_numpy(tensor)


def tensor2image(tensor):
    image = (127.5 * (tensor.cpu().float().numpy())) + 127.5
    image1 = image[0]
    for i in range(1, tensor.shape[0]):
        image1 = np.hstack((image1, image[i]))

    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    # print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


def smooth_weight_map_tv(weights: np.ndarray, strength: float = 0.2, n_iters: int = 15,
                         mask: np.ndarray = None) -> np.ndarray:
    """
    Lightweight TV-like smoothing for 2D weight maps to suppress speckle.
    Args:
        weights: 2D array in [0,1].
        strength: step size for the diffusion (>=0). Higher → stronger smoothing.
        n_iters: number of iterations.
        mask: optional binary mask to preserve background (e.g., body region); if given,
              smoothing is confined to masked pixels.
    """
    w = np.array(weights, dtype=np.float32, copy=True)
    if mask is not None:
        w = w * mask.astype(np.float32)
    strength = float(max(strength, 0.0))
    if strength <= 0.0 or n_iters <= 0:
        return np.clip(w, 0.0, 1.0)
    for _ in range(int(n_iters)):
        # 4-neighbor Laplacian
        lap = (-4.0 * w +
               np.roll(w, 1, axis=0) + np.roll(w, -1, axis=0) +
               np.roll(w, 1, axis=1) + np.roll(w, -1, axis=1))
        w = w + strength * lap
        if mask is not None:
            w = w * mask
    return np.clip(w, 0.0, 1.0)


def resolve_model_path(config: dict, name_key: str, default_name: str) -> str:
    """Return the checkpoint path for a model, honoring overrides in the config.

    If the config provides an absolute or explicit path in ``name_key`` it is used
    as-is. Otherwise the value (or ``default_name`` when unset) is joined with the
    ``save_root`` directory so legacy configs keep working out-of-the-box.
    """

    save_root = os.path.expanduser(config.get('save_root', '') or '')
    override = config.get(name_key)

    def _materialize(value: str) -> str:
        value = os.path.expanduser(str(value))
        if os.path.isabs(value):
            return value
        if save_root:
            return os.path.normpath(os.path.join(save_root, value))
        return os.path.normpath(value)

    if override:
        return _materialize(override)

    return _materialize(default_name)


class Logger:
    """
    Minimal, explicit logger:
      - log_step(losses): per-step (e.g., every 50 steps) metrics -> 'train_step/...'
      - log_epoch(means, epoch): per-epoch averaged metrics -> 'train_epoch/...' + {'epoch': epoch}
      - optional W&B configuration handling via wandb_settings / wandb_config
    """

    def __init__(self, env_name, ports, n_epochs, batches_epoch, is_main: bool = True,
                 wandb_settings: dict = None, wandb_config: dict = None):
        import os, time, wandb
        try:
            env_rank = int(os.environ.get('RANK')) if os.environ.get('RANK') is not None else None
            self.is_main = (env_rank == 0) if env_rank is not None else bool(is_main)
        except Exception:
            self.is_main = bool(is_main)

        self._wandb_enabled = False
        if self.is_main:
            settings = wandb_settings or {}
            _project = settings.get('project') or (env_name if isinstance(env_name, str) and len(env_name) > 0 else 'reg-gan')
            _run_name = settings.get('run_name')
            if not _run_name:
                _run_name = f"{_project}-e{n_epochs}-b{batches_epoch}-{int(time.time())}"
            base_config = {
                'n_epochs': n_epochs,
                'batches_epoch': batches_epoch,
            }
            if isinstance(wandb_config, dict):
                try:
                    merged = base_config.copy()
                    merged.update(wandb_config)
                    base_config = merged
                except Exception:
                    pass
            init_kwargs = {
                'project': _project,
                'name': _run_name,
                'reinit': False,
                'config': base_config,
            }
            for key in ('entity', 'group', 'job_type', 'notes'):
                val = settings.get(key, None)
                if val:
                    init_kwargs[key] = val
            tags = settings.get('tags', None)
            if tags:
                init_kwargs['tags'] = tags
            try:
                if wandb.run is None:
                    wandb.init(**init_kwargs)
                elif wandb_config:
                    try:
                        wandb.config.update(wandb_config, allow_val_change=True)
                    except Exception as update_err:
                        print(f"[wandb] config update failed: {update_err}")
                self._wandb_enabled = True
            except Exception as e:
                print(f"[wandb] init failed: {e}")
                self._wandb_enabled = False
        else:
            self._wandb_enabled = False

    def log_step(self, losses=None):
        import wandb
        if not (self.is_main and self._wandb_enabled) or not losses:
            return
        log_dict = {}
        for k, v in (losses or {}).items():
            if v is None:
                continue
            try:
                val = v.item() if hasattr(v, 'item') else float(v)
            except Exception:
                try:
                    val = float(v)
                except Exception:
                    continue
            log_dict[f"train_step/{k}"] = val
        if log_dict:
            try:
                wandb.log(log_dict)
            except Exception as e:
                print(f"[wandb] step log failed: {e}")

    def log_epoch(self, epoch_means: dict, epoch: int):
        import wandb
        if not (self.is_main and self._wandb_enabled) or not epoch_means:
            return
        log_dict = {f"train_epoch/{k}": float(v) for k, v in epoch_means.items() if v is not None}
        log_dict["epoch"] = int(epoch)
        try:
            wandb.log(log_dict)
        except Exception as e:
            print(f"[wandb] epoch log failed: {e}")


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d


# --- Aspect-ratio preserving resize with padding ---
class ResizeKeepRatioPad:
    """Resize to target size without distortion (keep aspect ratio) and pad borders with a fill value.
    Supports torch.Tensor (C,H,W). Intended to be used *after* ToTensor().
    """

    def __init__(self, size_tuple=(256, 256), fill=-1):
        self.target_h, self.target_w = size_tuple
        self.fill = float(fill)

    def __call__(self, img):
        # Expect torch.Tensor with shape (C,H,W) or (H,W); convert if needed
        if isinstance(img, PILImage.Image):
            # Convert PIL to tensor (H,W) or (C,H,W) in float32
            arr = np.array(img, dtype=np.float32)
            img = torch.from_numpy(arr)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if img.ndim == 2:
            img = img.unsqueeze(0)  # (1,H,W)
        assert img.ndim == 3, f"Expected (C,H,W), got {tuple(img.shape)}"
        if not img.is_floating_point():
            img = img.float()

        C, H, W = img.shape
        # Compute scale to fit inside target
        scale = min(self.target_h / max(H, 1), self.target_w / max(W, 1))
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))

        # Resize with bilinear (for 2D)
        # img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0)

        # Compute symmetric padding
        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Pad (left, right, top, bottom) on last two dims
        out = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=self.fill)
        return out


import numpy as _np


def _to_vis_uint8(_a):
    if _a is None:
        return None
    if isinstance(_a, (_np.integer, _np.floating)):
        _a = _np.array(_a)
    if _a.dtype == _np.uint8:
        return _a
    amin, amax = float(_np.nanmin(_a)), float(_np.nanmax(_a))
    if amin >= -1.01 and amax <= 1.01:
        return ((_a + 1.0) * 127.5).clip(0, 255).astype(_np.uint8)
    return ((_a - amin) / (amax - amin + 1e-6) * 255.0).astype(_np.uint8)


def to_uint8_image(arr):
    """Convert a tensor/array (C,H,W) or (H,W,C) in [-1,1] or arbitrary range to uint8 for visualization."""
    if arr is None:
        return None
    a = _np.asarray(arr)
    a = _np.squeeze(a)
    if a.ndim == 3:
        if a.shape[0] in (1, 3):
            if a.shape[0] == 1:
                a = a[0]
            else:
                a = _np.transpose(a, (1, 2, 0))
        elif a.shape[-1] == 1:
            a = a[..., 0]
    elif a.ndim > 3:
        # best effort squeeze to 3 or fewer dims
        while a.ndim > 3:
            a = _np.squeeze(a, axis=0)
    amin = float(_np.nanmin(a))
    amax = float(_np.nanmax(a))
    if amin >= -1.01 and amax <= 1.01:
        a = ((a + 1.0) * 127.5).clip(0, 255)
    else:
        a = ((a - amin) / (amax - amin + 1e-6) * 255.0).clip(0, 255)
    return a.astype(_np.uint8)


def _imshow_ax(ax, img, title=None, vmin=None, vmax=None, cmap='gray'):
    if img is None:
        ax.axis('off');
        return
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)


def plot_composite(*, inp, gt, pred, residual, metrics, rd_data=None, uncertainty=None, use_reg=False,
                   save_path=None, overlay_rgb=None, plot_dpi: int = 120, plot_colorbar: bool = True):
    """
    Unified figure renderer (utils version).
    - metrics: dict with keys {'mae','psnr','ssim'}
    - rd_data: None or dict with keys {'zmap','seeds','final_mask','weight_map','q_fdr'}
    - uncertainty: None or 2D np.ndarray (std map in [-1,1] scale)
    """

    def _norm01(x):
        x = np.asarray(x)
        mn = float(x.min())
        mx = float(x.max())
        if mx > mn:
            return (x - mn) / (mx - mn + 1e-8)
        return np.zeros_like(x, dtype=np.float32)

    has_rd = rd_data is not None
    has_u = uncertainty is not None

    def _to_disp(arr, force_gray=False):
        a = np.asarray(arr)
        a = np.squeeze(a)
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        if force_gray and a.ndim == 3 and a.shape[2] == 3:
            a = a.mean(axis=2)
        return a

    inp_v = _to_disp(inp)
    gt_v = _to_disp(gt)
    pred_v = _to_disp(pred)
    residual_v = _to_disp(residual, force_gray=True)
    overlay_v = _to_disp(overlay_rgb) if overlay_rgb is not None else None

    # 布局：与 CycTrainer._plot_composite 保持一致
    if has_rd:
        nrows, ncols = (2, 5) if has_u else (2, 4)
    else:
        nrows, ncols = (2, 4) if has_u else (1, 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])  # 统一二维索引

    # 第 1 行：input / truth / pred / residual (+ colorbar)
    axes[0, 0].imshow(_norm01(inp_v), cmap='gray')
    axes[0, 0].set_title('Input');
    axes[0, 0].axis('off')

    axes[0, 1].imshow(_norm01(gt_v), cmap='gray')
    axes[0, 1].set_title('Truth');
    axes[0, 1].axis('off')

    axes[0, 2].imshow(_norm01(pred_v), cmap='gray')
    if overlay_v is not None:
        axes[0, 2].imshow(overlay_v, alpha=0.3)  # 半透明叠加
    axes[0, 2].set_title('Prediction');
    axes[0, 2].axis('off')

    # 关键改进：固定残差显示范围，跨图可比；使用发散色图
    im_res = axes[0, 3].imshow(residual_v, cmap='seismic', vmin=-2.0, vmax=2.0)
    axes[0, 3].set_title('Residual (Pred - Truth)');
    axes[0, 3].axis('off')

    if plot_colorbar:
        try:
            cbar = fig.colorbar(im_res, ax=axes[0, 3], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        except Exception:
            pass

    if ncols > 4:
        axes[0, 4].axis('off')  # 有 RD 且有不确定度时的占位

    # 第 2 行（若有）
    if nrows == 2:
        if has_rd:
            zmap = _to_disp(rd_data.get('zmap', None), force_gray=True) if rd_data.get('zmap', None) is not None else None
            seeds = _to_disp(rd_data.get('seeds', None), force_gray=True) if rd_data.get('seeds', None) is not None else None
            final_mask = _to_disp(rd_data.get('final_mask', None), force_gray=True) if rd_data.get(
                'final_mask', None) is not None else None
            weight_map = _to_disp(rd_data.get('weight_map', None), force_gray=True) if rd_data.get(
                'weight_map', None) is not None else None
            q_fdr = rd_data.get('q_fdr', 0.10)

            z_vis = _norm01(zmap) if zmap is not None else np.zeros_like(residual_v)
            axes[1, 0].imshow(z_vis, cmap='magma')
            axes[1, 0].set_title('z-map (robust)');
            axes[1, 0].axis('off')

            axes[1, 1].imshow((seeds.astype(np.uint8) if seeds is not None else np.zeros_like(residual_v)),
                              cmap='gray')
            axes[1, 1].set_title(f'FDR seeds (two-sided, q={q_fdr})');
            axes[1, 1].axis('off')

            axes[1, 2].imshow((final_mask.astype(np.uint8) if final_mask is not None else np.zeros_like(residual_v)),
                              cmap='gray')
            axes[1, 2].set_title('mask (connected)');
            axes[1, 2].axis('off')

            w_vis = weight_map if (weight_map is not None and np.max(weight_map) > 0) else np.zeros_like(residual_v,
                                                                                                         dtype=np.float32)
            if np.max(w_vis) > 0:
                w_vis = w_vis / (np.max(w_vis) + 1e-6)
            axes[1, 3].imshow(w_vis, cmap='viridis')
            axes[1, 3].set_title('weight (soft, 0..1)');
            axes[1, 3].axis('off')

            if has_u:
                u_v = _to_disp(uncertainty, force_gray=True) if uncertainty is not None else None
                u_vis = u_v / (u_v.max() + 1e-8) if (u_v is not None and float(np.max(u_v)) > 0) else np.zeros_like(
                    residual_v)
                axes[1, 4].imshow(u_vis, cmap='inferno')
                axes[1, 4].set_title('uncertainty (std)');
                axes[1, 4].axis('off')

            if final_mask is not None and not np.any(final_mask):
                axes[1, 0].set_title('z-map (robust) — no region')
                axes[1, 1].set_title('FDR seeds (none)')
                axes[1, 2].set_title('mask (empty)')
                axes[1, 3].set_title('weight (empty)')
        else:
            # 没有 RD：若有不确定度则在 [1,0] 显示
            if has_u:
                u_v = _to_disp(uncertainty, force_gray=True) if uncertainty is not None else None
                u_vis = u_v / (u_v.max() + 1e-8) if (u_v is not None and float(np.max(u_v)) > 0) else np.zeros_like(
                    residual_v)
                axes[1, 0].imshow(u_vis, cmap='inferno')
                axes[1, 0].set_title('uncertainty (std)');
                axes[1, 0].axis('off')
            for j in range(1, ncols):
                axes[1, j].axis('off')

    tag = ' (after registration)' if use_reg else ''
    mae_i = float(metrics.get('mae', 0.0))
    psnr_i = float(metrics.get('psnr', 0.0))
    ssim_i = float(metrics.get('ssim', 0.0))
    fig.suptitle(f'MAE={mae_i:.4f} | PSNR={psnr_i:.2f} dB | SSIM={ssim_i:.4f}{tag}')

    if save_path is not None:
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(save_path, dpi=plot_dpi)
    plt.close(fig)


# =========
# Shared stateless helpers (moved from CycTrainer)
# =========
import math as _math
import cv2 as _cv2
import os as _os2


def norm01(x):
    x = np.asarray(x)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx > mn:
        return (x - mn) / (mx - mn + 1e-8)
    return np.zeros_like(x, dtype=np.float32)


def robust_zscore(arr, mask=None, eps=1e-6):
    a = np.asarray(arr)
    if mask is not None:
        a_sel = a[mask]
    else:
        a_sel = a
    if a_sel.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    med = np.median(a_sel)
    mad = np.median(np.abs(a_sel - med)) + eps
    z = (a - med) / (1.4826 * mad)
    return z.astype(np.float32)


def normal_cdf(x):
    # standard normal CDF via erf (no SciPy dependency)
    vec_erf = np.vectorize(lambda t: _math.erf(float(t) / (_math.sqrt(2.0))))
    return 0.5 * (1.0 + vec_erf(x))


def bh_fdr_mask(pvals, q=0.05, mask=None):
    """
    Benjamini–Hochberg FDR selector.
    Returns a boolean mask of same shape as pvals; True where significant.
    If 'mask' is provided, FDR runs only on those locations.
    """
    p = np.asarray(pvals).astype(np.float64)
    if mask is not None:
        idx = np.flatnonzero(mask.ravel())
    else:
        idx = np.arange(p.size)
    if idx.size == 0:
        return np.zeros_like(p, dtype=bool)

    p_sel = p.ravel()[idx]
    m = p_sel.size
    order = np.argsort(p_sel)
    p_sorted = p_sel[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    crit = (ranks / m) * q
    le = p_sorted <= crit
    if not np.any(le):
        return np.zeros_like(p, dtype=bool)
    k = np.max(np.nonzero(le)[0])  # last True index
    p_th = p_sorted[k]
    sel = p <= p_th
    if mask is not None:
        sel = sel & mask
    return sel


def hysteresis_from_seeds(z, seeds, Tl=2.0, max_iters=1024):
    """
    Grow from 'seeds' within the low-threshold band z>=Tl using binary dilation until convergence.
    """
    from skimage.morphology import binary_dilation
    band = (np.abs(z) >= Tl)
    grown = seeds.copy()
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


def morph_clean(mask, min_area=64, open_r=2, close_r=3, bridge_r=0):
    """
    Morphological clean-up for binary masks.
    """
    from skimage.morphology import remove_small_objects, opening, closing, binary_dilation, binary_erosion, disk
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not np.any(mask):
        return mask
    mask = remove_small_objects(mask, min_size=int(max(1, min_area)))
    if bridge_r and int(bridge_r) > 0:
        se = disk(int(bridge_r))
        mask = binary_dilation(mask, se)
        mask = binary_erosion(mask, se)
    if open_r and int(open_r) > 0:
        mask = opening(mask, disk(int(open_r)))
    if close_r and int(close_r) > 0:
        mask = closing(mask, disk(int(close_r)))
    mask = remove_small_objects(mask, min_size=int(max(1, min_area)))
    return mask


def save_gray_png(path, img):
    """
    Save a single-channel array to PNG in [0,255] (uint8). If values are constant, writes zeros.
    """
    a = np.asarray(img)
    if a.ndim != 2:
        a = np.squeeze(a)
    amin, amax = float(np.min(a)), float(np.max(a))
    if amax > amin:
        vis = ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)
    else:
        vis = np.zeros_like(a, dtype=np.uint8)
    _cv2.imwrite(path, vis)


def compose_slice_name(batch, batch_index: int, b: int) -> str:
    """Compose a per-slice name. Prefer patient_id + slice_id from batch; fallback to index-based."""
    try:
        B = int(batch['B'].shape[0])
    except Exception:
        B = None
    pid_list = batch.get('patient_id', None)
    sid_list = batch.get('slice_id', None)
    pid = None
    sid = None
    try:
        if pid_list is not None:
            if isinstance(pid_list, (list, tuple)):
                pid = pid_list[b] if b < len(pid_list) else None
            elif hasattr(pid_list, 'shape'):
                pid = pid_list[b]
        if sid_list is not None:
            if isinstance(sid_list, (list, tuple)):
                sid = sid_list[b] if b < len(sid_list) else None
            elif hasattr(sid_list, 'shape'):
                sid = sid_list[b]
    except Exception:
        pid, sid = None, None
    if pid is not None and sid is not None:
        return f"{pid}_{sid}"
    return f"{batch_index:05d}_{b:02d}"


def load_weight_or_mask_for_slice(slice_name: str, target_2d: np.ndarray,
                                  rd_input_type: str, rd_mask_dir: str, rd_weights_dir: str, rd_w_min: float,
                                  ignore_background: bool = False):
    """
    Stateless loader for per-slice guidance as a weight map (float32, [0,1]).
    """
    H, W = int(target_2d.shape[-2]), int(target_2d.shape[-1])
    if ignore_background:
        body = np.ones((H, W), dtype=np.float32)
    else:
        body = (target_2d != -1).astype(np.float32)

    # Choose source directory
    if rd_input_type == 'weights':
        src_dir = rd_weights_dir or rd_mask_dir
    else:
        src_dir = rd_mask_dir
    if not src_dir:
        return body  # no masking configured → all ones on body

    base = _os2.path.join(src_dir, slice_name)
    arr = None
    npy_path = base + '.npy'
    if arr is None and _os2.path.exists(npy_path):
        try:
            arr = np.load(npy_path)
        except Exception:
            arr = None
    png_path = base + '.png'
    if arr is None and _os2.path.exists(png_path):
        try:
            im = _cv2.imread(png_path, _cv2.IMREAD_UNCHANGED)
            if im is not None:
                if im.ndim == 3:
                    im = _cv2.cvtColor(im, _cv2.COLOR_BGR2GRAY)
                arr = im
        except Exception:
            arr = None

    if arr is None:
        return body

    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] == 3:
            arr = arr.mean(axis=0)
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] == 3:
            arr = arr.mean(axis=-1)
    if arr.shape != (H, W):
        try:
            arr = _cv2.resize(arr.astype(np.float32), (W, H), interpolation=_cv2.INTER_NEAREST)
        except Exception:
            return body

    if rd_input_type == 'weights':
        # 输入的 weights 语义：高值=不对齐=应剔除 → 转成 keep = 1 - weight
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = (arr.astype(np.float32) / 255.0)
        mis = np.clip(arr.astype(np.float32), 0.0, 1.0)
        w = 1.0 - mis
    else:
        if arr.dtype != np.bool_:
            arr = (arr > 0.5).astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        w = 1.0 - arr

    w = (w * body).astype(np.float32)
    if rd_w_min > 0.0:
        w = np.maximum(w, float(rd_w_min)) * (body > 0).astype(np.float32)
    return w

def rd_file_exists_for_slice(slice_name: str, rd_input_type: str, rd_mask_dir: str, rd_weights_dir: str) -> bool:
    """
    仅根据文件是否存在来判断某 slice 是否有对应的 RD 文件（不会触发读取或加载）。
    优先检查 .npy，其次 .png。
    """
    if rd_input_type == 'weights':
        src_dir = rd_weights_dir or rd_mask_dir
    else:
        src_dir = rd_mask_dir
    if not src_dir:
        return False
    base = os.path.join(src_dir, slice_name)
    return os.path.exists(base + '.npy') or os.path.exists(base + '.png')

def load_weights_for_batch(batch, batch_index: int, target_B: torch.Tensor,
                           rd_input_type: str, rd_mask_dir: str, rd_weights_dir: str, rd_w_min: float,
                           ignore_background: bool = False):
    """Build a (B,1,H,W) weight tensor for current batch based on rd_* settings."""
    target_np = target_B.detach().cpu().numpy()
    if target_np.ndim == 4 and target_np.shape[1] == 1:
        target_np_2d = target_np[:, 0, ...]
    else:
        target_np_2d = np.squeeze(target_np)
        if target_np_2d.ndim == 2:
            target_np_2d = target_np[None, ...]
    B = target_np_2d.shape[0]
    weights = []
    for b in range(B):
        slice_name = compose_slice_name(batch, batch_index, b)
        w2d = load_weight_or_mask_for_slice(
            slice_name, target_np_2d[b], rd_input_type, rd_mask_dir, rd_weights_dir, rd_w_min,
            ignore_background=ignore_background
        )
        weights.append(w2d)
    Wst = np.stack(weights, axis=0)[:, None, ...]  # (B,1,H,W)
    return torch.from_numpy(Wst).to(target_B.device, dtype=target_B.dtype)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean |pred-target| weighted by `weight` in (B,1,H,W). If sum(weight)<eps, fall back to unweighted."""
    diff = (pred - target).abs()
    if weight is None:
        return diff.mean()
    num = (weight * diff).sum()
    den = weight.sum().clamp_min(eps)
    return num / den


def _squeeze_to_image(arr):
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1, 1).astype(np.float64)
    # Iteratively drop singleton dims while preserving last two dims
    while a.ndim > 2:
        if a.shape[0] == 1:
            a = a[0]
        else:
            a = np.squeeze(a)
            if a.ndim > 2 and a.shape[0] != 1:
                break
    a = np.squeeze(a)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        a = np.atleast_2d(a)
    return a.astype(np.float64)


def _prepare_for_ssim(arr):
    a = np.asarray(arr)
    channel_axis = None
    if a.ndim == 3:
        if a.shape[0] in (1, 3):
            if a.shape[0] == 1:
                a = a[0]
            else:
                a = np.transpose(a, (1, 2, 0))
                channel_axis = -1
        elif a.shape[-1] in (1, 3):
            if a.shape[-1] == 1:
                a = a[..., 0]
            else:
                channel_axis = -1
        else:
            # fallback: squeeze if possible
            squeezed = np.squeeze(a)
            if squeezed.ndim == 2:
                a = squeezed
            else:
                raise ValueError(f"Unsupported array shape for SSIM: {a.shape}")
    elif a.ndim > 3:
        a = np.squeeze(a)
        return _prepare_for_ssim(a)
    return a.astype(np.float64, copy=False), channel_axis


def _build_metric_weights(target, mask=None, background_val=-1.0, eps=1e-6):
    tgt = _squeeze_to_image(target)
    body = (tgt != background_val).astype(np.float64)
    if mask is None:
        weights = body
    else:
        m = _squeeze_to_image(mask).astype(np.float64)
        if m.shape != body.shape:
            try:
                m = np.broadcast_to(m, body.shape)
            except ValueError:
                raise ValueError(f"Mask shape {m.shape} does not match target shape {body.shape}")
        weights = np.clip(m, 0.0, 1.0) * body
    weight_sum = float(np.sum(weights))
    if weight_sum <= eps:
        weight_sum = float(np.sum(body))
        weights = body if weight_sum > eps else None
    return weights, weight_sum


def _ssim_effective_window_size(image_shape):
    """Mirror skimage's window selection: default 7, shrink on small inputs, keep odd."""
    if not image_shape:
        return 1
    h, w = int(image_shape[-2]), int(image_shape[-1])
    min_dim = min(h, w)
    if min_dim <= 1:
        return 1
    win = 7 if min_dim >= 7 else max(1, min_dim)
    if win % 2 == 0:
        win = max(1, win - 1)
    return max(1, win)


def _windowed_fraction_map(base_mask, body_mask, win_size):
    """Average mask coverage inside each SSIM window, zeroing out true background."""
    base = np.asarray(base_mask, dtype=np.float64)
    body = np.asarray(body_mask, dtype=np.float64)
    if base.shape != body.shape:
        base = np.broadcast_to(base, body.shape)
    # If a channel dimension is present, collapse it, handling both CHW and HWC layouts.
    if base.ndim == 3:
        if base.shape[0] in (1, 3) and base.shape[-1] not in (1, 3):
            base = base.mean(axis=0)  # CHW -> HW
        elif base.shape[-1] in (1, 3):
            base = base.mean(axis=-1)  # HWC -> HW
        else:
            base = base.mean(axis=0)
    if body.ndim == 3:
        if body.shape[0] in (1, 3) and body.shape[-1] not in (1, 3):
            body = body.mean(axis=0)
        elif body.shape[-1] in (1, 3):
            body = body.mean(axis=-1)
        else:
            body = body.mean(axis=0)
    if win_size <= 1:
        cov = base
    else:
        pad = win_size // 2
        padded = np.pad(base, ((pad, pad), (pad, pad)), mode='reflect')
        view = np.lib.stride_tricks.sliding_window_view(padded, (win_size, win_size))
        cov = view.sum(axis=(-2, -1)) / float(win_size * win_size)
    return cov * body


def compute_mae(pred, target, mask=None, background_val=-1.0, eps=1e-6):
    """Mean absolute error in [0,1] space with optional mask (0..1 weights)."""
    fake = _squeeze_to_image(pred)
    real = _squeeze_to_image(target)
    weights, weight_sum = _build_metric_weights(real, mask, background_val, eps)
    abs_err = np.abs(fake - real) / 2.0
    if weights is not None and weight_sum > eps:
        return float(np.sum(abs_err * weights) / weight_sum)
    return float(np.mean(abs_err))


def compute_psnr(pred, target, mask=None, background_val=-1.0, eps=1e-6):
    """PSNR in dB using masked MSE in [0,1] space."""
    fake = _squeeze_to_image(pred)
    real = _squeeze_to_image(target)
    weights, weight_sum = _build_metric_weights(real, mask, background_val, eps)
    diff01 = ((fake + 1.0) * 0.5) - ((real + 1.0) * 0.5)
    mse = None
    if weights is not None and weight_sum > eps:
        mse = float(np.sum(weights * (diff01 * diff01)) / weight_sum)
    else:
        mse = float(np.mean(diff01 * diff01))
    if mse < 1.0e-10:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def compute_ssim(pred, target, mask=None, background_val=-1.0, eps=1e-6):
    """SSIM averaged over masked region (background excluded by default)."""
    fake_raw, channel_axis = _prepare_for_ssim(pred)
    real_raw, _ = _prepare_for_ssim(target)
    fake = fake_raw if channel_axis is None else fake_raw
    real = real_raw if channel_axis is None else real_raw
    if channel_axis is not None and channel_axis != -1:
        # channel axis could be at position 0 for HWC conversion; ensure final axis is channel
        if channel_axis == 0 and fake.ndim == 3:
            fake = np.transpose(fake, (1, 2, 0))
            real = np.transpose(real, (1, 2, 0))
            channel_axis = -1
    weights, weight_sum = _build_metric_weights(real, mask, background_val, eps)
    body = (real != background_val).astype(np.float64)
    base_mask = weights if weights is not None else body
    try:
        ssim_val, ssim_map = structural_similarity(fake, real, data_range=2.0, full=True,
                                                   channel_axis=(None if fake.ndim == 2 else -1))
    except Exception:
        ssim_val, ssim_map = structural_similarity(fake.astype(np.float64),
                                                   real.astype(np.float64),
                                                   data_range=2.0, full=True,
                                                   channel_axis=(None if fake.ndim == 2 else -1))
    ssim_map = _squeeze_to_image(ssim_map)
    if ssim_map.shape != real.shape:
        try:
            ssim_map = np.broadcast_to(ssim_map, real.shape)
        except ValueError:
            raise ValueError(f"SSIM map shape {ssim_map.shape} mismatches target {real.shape}")
    win_size = _ssim_effective_window_size(real.shape)
    coverage = _windowed_fraction_map(base_mask, body, win_size)
    # Align coverage shape to SSIM map for weighted averaging
    if coverage.ndim != ssim_map.ndim:
        try:
            coverage = np.broadcast_to(coverage, ssim_map.shape)
        except ValueError:
            # Last resort: average SSIM map channels if present
            if ssim_map.ndim == 3 and ssim_map.shape[-1] in (1, 3):
                ssim_map = ssim_map.mean(axis=-1)
                coverage = np.broadcast_to(coverage, ssim_map.shape)
            else:
                raise ValueError(f"Cannot broadcast coverage {coverage.shape} to SSIM map {ssim_map.shape}")
    coverage_sum = float(np.sum(coverage))
    if coverage_sum <= eps:
        if weights is not None and weight_sum > eps:
            if weights.shape != ssim_map.shape:
                try:
                    weights_b = np.broadcast_to(weights, ssim_map.shape)
                except ValueError:
                    raise ValueError(f"SSIM map shape {ssim_map.shape} mismatches weights {weights.shape}")
                weights = weights_b
            return float(np.sum(ssim_map * weights) / weight_sum)
        return float(ssim_val)
    return float(np.sum(ssim_map * coverage) / coverage_sum)
