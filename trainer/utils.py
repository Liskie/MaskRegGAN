import random
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
from visdom import Visdom
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage


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
        tensor = np.expand_dims(tensor, 0)
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


class Logger():
    def __init__(self, env_name, ports, n_epochs, batches_epoch):
        self.viz = Visdom(port=ports, env=env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        if images is not None:
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
                else:
                    self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                                   opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


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
        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

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

# =====================
# Reusable composite plotting (for CycTrainer & experiment02)
# =====================
import os as _os
import numpy as _np
import matplotlib.pyplot as _plt

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
        ax.axis('off'); return
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)

def plot_composite(*, inp=None, gt=None, pred=None, residual=None,
                   metrics=None, rd_data=None, uncertainty=None,
                   use_reg=False, save_path=None, overlay_rgb=None):
    """
    通用 3x4 可视化面板：
    Row 1: input x, y_true, y_pred, residual
    Row 2: |z|, seeds, final_mask, weight_map (只在提供时显示)
    Row 3: overlay (True=R, Pred=G) + 3 个空位
    """
    x_vis  = _to_vis_uint8(inp)
    gt_vis = _to_vis_uint8(gt)
    pd_vis = _to_vis_uint8(pred)
    rs_vis = _to_vis_uint8(residual)

    zmap = rd_data.get('zmap') if isinstance(rd_data, dict) else None
    seeds = rd_data.get('seeds') if isinstance(rd_data, dict) else None
    final_mask = rd_data.get('final_mask') if isinstance(rd_data, dict) else None
    weight_map = rd_data.get('weight_map') if isinstance(rd_data, dict) else None

    fig, axes = _plt.subplots(3, 4, figsize=(12, 8))

    # Row 1
    _imshow_ax(axes[0,0], x_vis, 'x (input)')
    _imshow_ax(axes[0,1], gt_vis, 'y_true')
    _imshow_ax(axes[0,2], pd_vis, 'y_pred' + (' [reg]' if use_reg else ''))
    title_r = 'residual'
    if isinstance(metrics, dict):
        tbits = []
        for k in ('mae', 'psnr', 'ssim'):
            if k in metrics:
                try:
                    tbits.append(f"{k}={metrics[k]:.3f}")
                except Exception:
                    pass
        if tbits:
            title_r += ' (' + ', '.join(tbits) + ')'
    _imshow_ax(axes[0,3], rs_vis, title_r)

    # Row 2
    if zmap is not None:
        zabs = _np.abs(zmap)
        vmax = float(_np.nanpercentile(zabs, 99)) if _np.isfinite(zabs).any() else None
        _imshow_ax(axes[1,0], zabs, '|z|', vmin=0, vmax=vmax)
    else:
        axes[1,0].axis('off')
    if seeds is not None:
        _imshow_ax(axes[1,1], seeds.astype(_np.uint8) * 255, 'seeds', vmin=0, vmax=255)
    else:
        axes[1,1].axis('off')
    if final_mask is not None:
        _imshow_ax(axes[1,2], final_mask.astype(_np.uint8) * 255, 'final_mask', vmin=0, vmax=255)
    else:
        axes[1,2].axis('off')
    if (weight_map is not None) and _np.any(weight_map):
        _imshow_ax(axes[1,3], weight_map, 'weight_map', vmin=0, vmax=1.0, cmap='viridis')
    else:
        axes[1,3].axis('off')

    # Row 3
    _imshow_ax(axes[2,0], overlay_rgb, 'overlay (True=R, Pred=G)')
    axes[2,1].axis('off'); axes[2,2].axis('off'); axes[2,3].axis('off')

    _plt.tight_layout()
    if save_path:
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    _plt.close(fig)