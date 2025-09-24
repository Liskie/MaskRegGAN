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

    # 布局：与 CycTrainer._plot_composite 保持一致
    if has_rd:
        nrows, ncols = (2, 5) if has_u else (2, 4)
    else:
        nrows, ncols = (2, 4) if has_u else (1, 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])  # 统一二维索引

    # 第 1 行：input / truth / pred / residual (+ colorbar)
    axes[0, 0].imshow(_norm01(inp), cmap='gray')
    axes[0, 0].set_title('Input');
    axes[0, 0].axis('off')

    axes[0, 1].imshow(_norm01(gt), cmap='gray')
    axes[0, 1].set_title('Truth');
    axes[0, 1].axis('off')

    axes[0, 2].imshow(_norm01(pred), cmap='gray')
    if overlay_rgb is not None:
        axes[0, 2].imshow(overlay_rgb, alpha=0.3)  # 半透明叠加
    axes[0, 2].set_title('Prediction');
    axes[0, 2].axis('off')

    # 关键改进：固定残差显示范围，跨图可比；使用发散色图
    im_res = axes[0, 3].imshow(residual, cmap='seismic', vmin=-2.0, vmax=2.0)
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
            zmap = rd_data.get('zmap', None)
            seeds = rd_data.get('seeds', None)
            final_mask = rd_data.get('final_mask', None)
            weight_map = rd_data.get('weight_map', None)
            q_fdr = rd_data.get('q_fdr', 0.10)

            z_vis = _norm01(zmap) if zmap is not None else np.zeros_like(pred)
            axes[1, 0].imshow(z_vis, cmap='magma')
            axes[1, 0].set_title('z-map (robust)');
            axes[1, 0].axis('off')

            axes[1, 1].imshow((seeds.astype(np.uint8) if seeds is not None else np.zeros_like(pred)), cmap='gray')
            axes[1, 1].set_title(f'FDR seeds (two-sided, q={q_fdr})');
            axes[1, 1].axis('off')

            axes[1, 2].imshow((final_mask.astype(np.uint8) if final_mask is not None else np.zeros_like(pred)),
                              cmap='gray')
            axes[1, 2].set_title('mask (connected)');
            axes[1, 2].axis('off')

            w_vis = weight_map if (weight_map is not None and np.max(weight_map) > 0) else np.zeros_like(pred,
                                                                                                         dtype=np.float32)
            if np.max(w_vis) > 0:
                w_vis = w_vis / (np.max(w_vis) + 1e-6)
            axes[1, 3].imshow(w_vis, cmap='viridis')
            axes[1, 3].set_title('weight (soft, 0..1)');
            axes[1, 3].axis('off')

            if has_u:
                u_vis = uncertainty / (uncertainty.max() + 1e-8) if float(np.max(uncertainty)) > 0 else np.zeros_like(
                    pred)
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
                u_vis = uncertainty / (uncertainty.max() + 1e-8) if float(np.max(uncertainty)) > 0 else np.zeros_like(
                    pred)
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
                                  rd_input_type: str, rd_mask_dir: str, rd_weights_dir: str, rd_w_min: float):
    """
    Stateless loader for per-slice guidance as a weight map (float32, [0,1]).
    """
    H, W = int(target_2d.shape[-2]), int(target_2d.shape[-1])
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
    if arr.shape != (H, W):
        print(f'Mask shape mismatch: {arr.shape}, should be {(H, W)}') # DEBUG
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
                           rd_input_type: str, rd_mask_dir: str, rd_weights_dir: str, rd_w_min: float):
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
        w2d = load_weight_or_mask_for_slice(slice_name, target_np_2d[b], rd_input_type, rd_mask_dir, rd_weights_dir,
                                            rd_w_min)
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
