#!/usr/bin/python3

import itertools
import os

from torchvision.transforms import RandomAffine, ToPILImage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import math
import numpy as np
import cv2
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TextColumn

from .utils import Resize, ToTensor, smooothing_loss, LambdaLR, Logger, ReplayBuffer, plot_composite
from .datasets import ImageDataset, ValDataset
from .reg import Reg
from .transformer import Transformer_2D
from models.CycleGan import *

import time
from collections import defaultdict


def _enable_mc_dropout(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()  # activate dropout during inference


# --- Lightweight timer context for profiling ---
class _Timer:
    def __init__(self, name, prof_dict):
        self.name = name
        self.prof = prof_dict

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        self.prof[self.name] += (t1 - self.t0)


class LogRange:
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, x):
        try:
            if isinstance(x, torch.Tensor):
                vmin = float(x.min().item())
                vmax = float(x.max().item())
                dtype = str(x.dtype)
                shape = tuple(x.shape)
            elif isinstance(x, np.ndarray):
                vmin = float(np.min(x))
                vmax = float(np.max(x))
                dtype = str(x.dtype)
                shape = x.shape
            elif isinstance(x, PILImage.Image):
                arr = np.array(x)
                vmin = float(arr.min())
                vmax = float(arr.max())
                dtype = f"PIL[{x.mode}]/{arr.dtype}"
                shape = arr.shape
            else:
                dtype = type(x).__name__
                shape = getattr(x, 'size', None)
                vmin = float('nan')
                vmax = float('nan')
            print(f"[{self.tag}] type={type(x).__name__} shape={shape} dtype={dtype} min={vmin:.6g} max={vmax:.6g}",
                  flush=True)
        except Exception as e:
            print(f"[{self.tag}] (failed to compute stats: {e})", flush=True)
        return x


class ToPILWithLog:
    def __init__(self, tag: str = "ToPILImage"):
        self._op = ToPILImage()
        self._before = LogRange(f"before {tag}")
        self._after = LogRange(f"after {tag}")

    def __call__(self, x):
        self._before(x)
        y = self._op(x)
        self._after(y)
        return y


class Cyc_Trainer():
    @staticmethod
    def _norm01(x):
        x = np.asarray(x)
        mn = float(x.min())
        mx = float(x.max())
        if mx > mn:
            return (x - mn) / (mx - mn + 1e-8)
        return np.zeros_like(x, dtype=np.float32)

    def _plot_composite(self, *, inp, gt, pred, residual, metrics, rd_data=None, uncertainty=None, use_reg=False,
                        save_path=None):
        """
        Unified figure renderer.
        - metrics: dict with keys {'mae','psnr','ssim'}
        - rd_data: None or dict with keys {'zmap','seeds','final_mask','weight_map','q_fdr'}
        - uncertainty: None or 2D np.ndarray (std map in [-1,1] scale)
        """
        import matplotlib.pyplot as plt
        plot_dpi = int(self.config.get('plot_dpi', 120))
        plot_colorbar = bool(self.config.get('plot_colorbar', False))
        has_rd = rd_data is not None
        has_u = uncertainty is not None

        # Decide grid shape
        if has_rd:
            nrows, ncols = (2, 5) if has_u else (2, 4)
        else:
            nrows, ncols = (2, 4) if has_u else (1, 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if nrows == 1:
            axes = np.array([axes])  # make 2D for uniform indexing

        # Row 1: input / truth / pred / residual (+ colorbar)
        axes[0, 0].imshow(self._norm01(inp), cmap='gray');
        axes[0, 0].set_title('Input');
        axes[0, 0].axis('off')
        axes[0, 1].imshow(self._norm01(gt), cmap='gray');
        axes[0, 1].set_title('Truth');
        axes[0, 1].axis('off')
        axes[0, 2].imshow(self._norm01(pred), cmap='gray');
        axes[0, 2].set_title('Prediction');
        axes[0, 2].axis('off')
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
            axes[0, 4].axis('off')  # filler when has_rd and has_u

        # Row 2 (if present)
        if nrows == 2:
            if has_rd:
                zmap = rd_data.get('zmap', None)
                seeds = rd_data.get('seeds', None)
                final_mask = rd_data.get('final_mask', None)
                weight_map = rd_data.get('weight_map', None)
                q_fdr = rd_data.get('q_fdr', 0.10)

                z_vis = self._norm01(zmap) if zmap is not None else np.zeros_like(pred)
                axes[1, 0].imshow(z_vis, cmap='magma');
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
                axes[1, 3].imshow(w_vis, cmap='viridis');
                axes[1, 3].set_title('weight (soft, 0..1)');
                axes[1, 3].axis('off')

                if has_u:
                    u_vis = uncertainty / (uncertainty.max() + 1e-8) if float(
                        np.max(uncertainty)) > 0 else np.zeros_like(pred)
                    axes[1, 4].imshow(u_vis, cmap='inferno');
                    axes[1, 4].set_title('uncertainty (std)');
                    axes[1, 4].axis('off')

                if final_mask is not None and not np.any(final_mask):
                    axes[1, 0].set_title('z-map (robust) — no region')
                    axes[1, 1].set_title('FDR seeds (none)')
                    axes[1, 2].set_title('mask (empty)')
                    axes[1, 3].set_title('weight (empty)')

            else:
                # no RD: show uncertainty at [1,0] if provided, blanks otherwise
                if has_u:
                    u_vis = uncertainty / (uncertainty.max() + 1e-8) if float(
                        np.max(uncertainty)) > 0 else np.zeros_like(pred)
                    axes[1, 0].imshow(u_vis, cmap='inferno');
                    axes[1, 0].set_title('uncertainty (std)');
                    axes[1, 0].axis('off')
                for j in range(1, ncols):
                    axes[1, j].axis('off')

        tag = ' (after registration)' if use_reg else ''
        mae_i, psnr_i, ssim_i = metrics['mae'], metrics['psnr'], metrics['ssim']
        fig.suptitle(f'MAE={mae_i:.4f} | PSNR={psnr_i:.2f} dB | SSIM={ssim_i:.4f}{tag}')
        if save_path is not None:
            plt.tight_layout(rect=[0, 0, 1, 0.94]);
            plt.savefig(save_path, dpi=plot_dpi)
        plt.close(fig)

    # === Residual-based non-correspondence detection helpers ===
    @staticmethod
    def _robust_zscore(arr, mask=None, eps=1e-6):
        if mask is not None:
            a = np.asarray(arr)[mask]
        else:
            a = np.asarray(arr)
        if a.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        med = np.median(a)
        mad = np.median(np.abs(a - med)) + eps
        z = (arr - med) / (1.4826 * mad)
        return z.astype(np.float32)

    @staticmethod
    def _normal_cdf(x):
        # standard normal CDF via erf
        return 0.5 * (1.0 + np.vectorize(lambda t: math.erf(float(t) / (math.sqrt(2.0))))(x))

    @staticmethod
    def _bh_fdr_mask(pvals, q=0.05, mask=None):
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
            # no rejections
            out = np.zeros_like(p, dtype=bool)
            return out
        k = np.max(np.nonzero(le)[0])  # last True index
        p_th = p_sorted[k]
        sel = p <= p_th
        if mask is not None:
            sel = sel & mask
        return sel

    @staticmethod
    def _hysteresis_from_seeds(z, seeds, Tl=2.0, max_iters=1024):
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
            if np.array_equal(nxt, grown):  # converged
                break
            grown = nxt
            if it >= max_iters:
                break
        return grown

    @staticmethod
    def _morph_clean(mask, min_area=64, open_r=2, close_r=3, bridge_r=0):
        """
        Morphological clean-up for binary masks.
        - remove_small_objects(min_area)  → 去除小碎片
        - (optional) bridging via dilation→erosion with radius=bridge_r  → 让彼此接近的区域连成一片
        - opening(close)                  → 圆滑边缘/填小孔
        - remove_small_objects(min_area)  → 再次去小碎片（防止中间过程产生的小屑块）
        """
        from skimage.morphology import remove_small_objects, opening, closing, binary_dilation, binary_erosion, disk
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not np.any(mask):
            return mask
        # 1) 先去小碎片
        mask = remove_small_objects(mask, min_size=int(max(1, min_area)))
        # 2) 若需要，将相近的组件“搭桥”连成片（不会无限长大，基本等价于较强的 closing）
        if bridge_r and int(bridge_r) > 0:
            se = disk(int(bridge_r))
            mask = binary_dilation(mask, se)
            mask = binary_erosion(mask, se)
        # 3) 圆滑边界/填小孔
        if open_r and int(open_r) > 0:
            mask = opening(mask, disk(int(open_r)))
        if close_r and int(close_r) > 0:
            mask = closing(mask, disk(int(close_r)))
        # 4) 再次去小碎片（防止第 2/3 步产生的微小残留）
        mask = remove_small_objects(mask, min_size=int(max(1, min_area)))
        return mask

    @staticmethod
    def _save_gray_png(path, img):
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
        cv2.imwrite(path, vis)

    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                                lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level
        # --- Resize mode config ---
        resize_mode = str(config.get('resize_mode', 'resize')).lower()
        if resize_mode not in ('resize', 'keepratio'):
            resize_mode = 'resize'
        if resize_mode == 'keepratio':
            from .utils import ResizeKeepRatioPad
            last_tf = ResizeKeepRatioPad(size_tuple=(config['size'], config['size']), fill=-1)
        else:
            last_tf = Resize(size_tuple=(config['size'], config['size']))

        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level], fill=-1),
                        ToTensor(),
                        last_tf]

        transforms_2 = [ToPILImage(),
                        RandomAffine(degrees=1, translate=[0.02, 0.02], scale=[0.98, 1.02], fill=-1),
                        ToTensor(),
                        last_tf]

        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2,
                         unaligned=False),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'],
            pin_memory=config['cuda'],  # Accelerate
            persistent_workers=True,  # Accelerate
            prefetch_factor=2,  # Accelerate
        )

        val_transforms = [ToTensor(),
                          last_tf]

        self.val_data = DataLoader(
            ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
            batch_size=config['batchSize'],
            shuffle=False,
            num_workers=config['n_cpu'],
            pin_memory=config['cuda'],  # Accelerate
            persistent_workers=True,  # Accelerate
            prefetch_factor=2,  # Accelerate
        )

        # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs'], len(self.dataloader))

    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):

                # Set model input
                # real_A = Variable(self.input_A.copy_(batch['A']))
                # real_B = Variable(self.input_B.copy_(batch['B']))
                real_A = batch['A'].to('cuda', non_blocking=True)  # Accelerate
                real_B = batch['B'].to('cuda', non_blocking=True)  # Accelerate

                if self.config['bidirect']:  # C dir
                    if self.config['regist']:  # C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################

                    else:  # only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################



                else:  # s dir :NC
                    if self.config['regist']:  # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, torch.ones_like(pred_fake0))
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss + adv_loss + SR_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, torch.zeros_like(pred_fake0)) + \
                                   self.config[
                                       'Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))

                        loss_D_B.backward()
                        self.optimizer_D_B.step()



                    else:  # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################

                if i % 50 == 0:
                    self.logger.log({'loss_D_B': loss_D_B, 'SR_loss': SR_loss},
                                    images={'real_A': real_A, 'real_B': real_B,
                                            'fake_B': fake_B})  # ,'SR':SysRegist_A2B
                else:
                    self.logger.log({'loss_D_B': loss_D_B, 'SR_loss': SR_loss})

            #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            torch.save(self.R_A.state_dict(), self.config['save_root'] + 'R_A.pth')
            # torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            # torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')

            #############val###############
            if epoch % self.config['val_freq'] == 0:
                with torch.no_grad():
                    MAE = 0
                    num = 0
                    for i, batch in enumerate(self.val_data):
                        # 异步拷到 GPU
                        real_A = batch['A'].to('cuda', non_blocking=True)
                        real_Bt = batch['B'].to('cuda', non_blocking=True)

                        # 前向在 GPU 上
                        fake_Bt = self.netG_A2B(real_A)

                        # 评估指标在 CPU/numpy 上
                        real_B = real_Bt.detach().cpu().numpy().squeeze()
                        fake_B = fake_Bt.detach().cpu().numpy().squeeze()

                        mae = self.MAE(fake_B, real_B)
                        MAE += mae
                        num += 1

                    print('Val MAE:', MAE / num)

    def test(self, ):
        # --- Read injection config and quality thresholds ---
        synth_cfg = dict(self.config.get('inject_nc', {})) if isinstance(self.config.get('inject_nc', {}), dict) else {}
        synth_enable_global = bool(synth_cfg.get('enable', False))
        quality_ssim_min = float(synth_cfg.get('quality_ssim_min', 0.85))
        quality_mae_max = float(synth_cfg.get('quality_mae_max', 0.12))

        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        os.makedirs(self.config['image_save'], exist_ok=True)
        pred_save_root = self.config.get('pred_save', os.path.join(self.config['image_save'], 'pred'))
        os.makedirs(pred_save_root, exist_ok=True)
        # === Create output dirs for MC mean and uncertainty maps ===
        mean_save_root = self.config.get('pred_mean_save', os.path.join(self.config['image_save'], 'pred_mean'))
        os.makedirs(mean_save_root, exist_ok=True)
        uncert_save_root = self.config.get('uncert_save', os.path.join(self.config['image_save'], 'uncert'))
        os.makedirs(uncert_save_root, exist_ok=True)
        import csv
        # Enable evaluation-after-registration if configured and registration net exists
        use_reg = bool(self.config.get('eval_with_registration', False)) and self.config.get('regist',
                                                                                             False) and hasattr(self,
                                                                                                                'R_A') and hasattr(
            self, 'spatial_transform')
        if use_reg:
            ra_path = os.path.join(self.config['save_root'], 'R_A.pth')
            if os.path.exists(ra_path):
                self.R_A.load_state_dict(torch.load(ra_path))
            else:
                print(f"[test] R_A weights not found at {ra_path}; disabling eval_with_registration.")
                use_reg = False
        # --- Monte Carlo inference settings ---
        mc_runs = int(self.config.get('mc_runs', 1))  # e.g., 10
        mc_mode = str(self.config.get('mc_mode', 'dropout')).lower()  # 'dropout' or 'input_noise'
        mc_input_noise_sigma = float(
            self.config.get('mc_input_noise_sigma', 0.0))  # for 'input_noise' mode (in [-1,1] scale)

        # Plotting controls
        save_composite = bool(self.config.get('save_composite', True))  # master switch to save composite figures
        plot_every_n = int(self.config.get('plot_every_n', 1))          # save 1 of every N slices (1 = all)

        # Accumulators for prediction mean/std per slice (in [-1,1] space)
        pred_sum = {}
        pred_sumsq = {}

        # Keep eval() globally, then selectively enable dropout
        self.netG_A2B.eval()
        if mc_runs > 1 and mc_mode == 'dropout':
            _enable_mc_dropout(self.netG_A2B)
            if use_reg and hasattr(self, 'R_A'):
                self.R_A.eval()
                _enable_mc_dropout(self.R_A)
        with torch.no_grad():
            prof = defaultdict(float)
            counters = defaultdict(int)
            MAE = 0.0
            PSNR = 0.0
            SSIM = 0.0
            N = 0
            # For MC runs accumulate per-slice metrics
            per_slice_values = {}  # key: slice_name -> {'mae':[], 'psnr':[], 'ssim':[]}

            # Defer composite plotting until std (uncertainty) is available
            composite_cache = {}
            # 不再单独保存不确定性图；如需恢复，置 True
            save_uncert_outputs = bool(self.config.get('save_uncert_outputs', False))

            # Set up progress bar
            try:
                total_slices = len(self.val_data.dataset)
            except Exception:
                try:
                    total_slices = len(self.val_data) * self.config.get('batchSize', 1)
                except Exception:
                    total_slices = None

            progress = Progress(
                TextColumn("[bold blue]Testing[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )

            total_processed_slices = 0

            with progress:
                task_id = progress.add_task("eval", total=total_slices if (
                        isinstance(total_slices, int) and total_slices > 0) else None)

                for i, batch in enumerate(self.val_data):
                    with _Timer('batch_total', prof):
                        # Move to GPU
                        real_A = batch['A'].to('cuda', non_blocking=True)
                        real_Bt = batch['B'].to('cuda', non_blocking=True)
                        real_Bnp = real_Bt.detach().cpu().numpy()

                        B = real_Bnp.shape[0]
                        # Try to get patient_id and slice_id from batch, fallback to None
                        patient_ids = batch.get('patient_id', [None] * B)
                        slice_ids = batch.get('slice_id', [None] * B)
                        for t in range(mc_runs):
                            # Prepare possibly perturbed input
                            if mc_runs > 1 and mc_mode == 'input_noise' and mc_input_noise_sigma > 0:
                                noise = torch.randn_like(real_A) * mc_input_noise_sigma
                                real_A_in = torch.clamp(real_A + noise, -1.0, 1.0)
                            else:
                                real_A_in = real_A

                            # Forward pass (dropout randomness already enabled if selected)
                            with _Timer('forward', prof):
                                fake_Bt = self.netG_A2B(real_A_in)
                            counters['forward_calls'] += 1

                            # Optionally register prediction to GT for evaluation
                            fake_eval = fake_Bt
                            if use_reg:
                                with _Timer('registration', prof):
                                    Trans = self.R_A(fake_Bt, real_Bt)
                                    fake_eval = self.spatial_transform(fake_Bt, Trans)
                                counters['registration_calls'] += 1

                            # To numpy (B,1,H,W)
                            with _Timer('to_numpy', prof):
                                fake_Bnp = fake_eval.detach().cpu().numpy()
                                fake_B_rawnp = fake_Bt.detach().cpu().numpy() if use_reg else None
                            counters['to_numpy_calls'] += 1

                            for b in range(B):
                                # Compose slice_name using patient_id and slice_id if available
                                patient_id = patient_ids[b] if len(patient_ids) > b else None
                                slice_id = slice_ids[b] if len(slice_ids) > b else None
                                if patient_id is not None and slice_id is not None:
                                    slice_name = f"{patient_id}_{slice_id}"
                                else:
                                    slice_name = f"{i:05d}_{b:02d}"
                                r = real_Bnp[b]
                                fe = fake_Bnp[b]
                                fr = fake_B_rawnp[b] if use_reg else None

                                # Squeeze single-channel if present
                                if r.ndim == 3 and r.shape[0] == 1:
                                    r = r[0]
                                else:
                                    r = np.squeeze(r)
                                if fe.ndim == 3 and fe.shape[0] == 1:
                                    fe = fe[0]
                                else:
                                    fe = np.squeeze(fe)
                                if use_reg:
                                    if fr.ndim == 3 and fr.shape[0] == 1:
                                        fr = fr[0]
                                    else:
                                        fr = np.squeeze(fr)

                                # Metrics per-slice (baseline alignment check)
                                with _Timer('metrics', prof):
                                    mae_i = self.MAE(fe, r)
                                    psnr_i = self.PSNR(fe, r, mode='correct')
                                    try:
                                        ssim_i = structural_similarity(fe, r, data_range=2.0)
                                    except Exception:
                                        ssim_i = structural_similarity(fe.astype(np.float64), r.astype(np.float64),
                                                                       data_range=2.0)
                                counters['metric_points'] += 1

                                # Progress: count every slice once on first MC run (avoid over/under counting)
                                if t == 0:
                                    total_processed_slices += 1
                                    progress.advance(task_id, 1)

                                # --- Injection gating ---
                                pair_ok = (ssim_i >= quality_ssim_min) and (mae_i <= quality_mae_max)

                                # If injection is enabled and pair is not well-aligned, SKIP this slice entirely
                                if synth_enable_global:
                                    if not pair_ok:
                                        # Skip: do not cache, do not save, do not accumulate
                                        continue
                                # If injection is disabled, always proceed (legacy behavior)

                                # === Accumulate per-slice sums for mean/std (fe is in [-1,1]) ===
                                acc_s = pred_sum.get(slice_name)
                                if acc_s is None:
                                    pred_sum[slice_name] = fe.astype(np.float32).copy()
                                    pred_sumsq[slice_name] = (fe.astype(np.float32) ** 2)
                                else:
                                    pred_sum[slice_name] += fe.astype(np.float32)
                                    pred_sumsq[slice_name] += (fe.astype(np.float32) ** 2)

                                # Accumulate global sums for overall averages (over MC and slices)
                                MAE += mae_i;
                                PSNR += psnr_i;
                                SSIM += ssim_i;
                                N += 1

                                # Accumulate per-slice list for mean/std
                                d = per_slice_values.setdefault(slice_name, {'mae': [], 'psnr': [], 'ssim': []})
                                d['mae'].append(mae_i);
                                d['psnr'].append(psnr_i);
                                d['ssim'].append(ssim_i)

                                # Only on first MC run, produce visualization and save pred image
                                if t == 0:
                                    inp = real_A[b].detach().cpu().numpy().squeeze()
                                    gt = r
                                    pred = fe
                                    residual = pred - gt
                                    # === Residual-based non-correspondence detection (判空 + 连续区域提取) ===
                                    try:
                                        rd_enable = bool(self.config.get('residual_detect', True))
                                    except Exception:
                                        rd_enable = True

                                    zmap = seeds = final_mask = weight_map = None
                                    if rd_enable:
                                        with _Timer('residual_detect_total', prof):
                                            q_fdr = float(self.config.get('rd_fdr_q', 0.10))
                                            Tl_fix = float(self.config.get('rd_Tl', 1.5))
                                            area_fr = float(self.config.get('rd_min_area_frac', 0.0003))
                                            open_r = int(self.config.get('rd_open_r', 1))
                                            close_r = int(self.config.get('rd_close_r', 2))
                                            topk = int(self.config.get('rd_topk', -1))
                                            seed_dilate = int(self.config.get('rd_seed_dilate', 1))
                                            weight_mode = str(self.config.get('rd_weight_mode', 'sigmoid'))
                                            weight_alpha = float(self.config.get('rd_weight_alpha', 1.5))
                                            # --- New config for residual-detection ---
                                            rd_allow_empty = bool(self.config.get('rd_allow_empty',
                                                                                  True))  # allow empty result if no seeds after FDR
                                            Th_hard = float(self.config.get('rd_th_seed_hi',
                                                                            2.5))  # hard-evidence z threshold (>|z|)
                                            K_hard = int(self.config.get('rd_min_highz_pixels',
                                                                         5))  # minimum count of hard-evidence pixels per component

                                            body = (gt != -1)
                                            zmap = self._robust_zscore(residual, mask=body)
                                            absz = np.abs(zmap)
                                            # 两侧检验
                                            # --- Precompute body area for seed post-filtering ---
                                            body_area = int(np.count_nonzero(body))

                                            # 1) (optional) smooth |z| to suppress isolated pixels before FDR
                                            seed_sigma = float(self.config.get('rd_seed_smooth_sigma', 0.0))
                                            if seed_sigma > 0:
                                                absz_s = cv2.GaussianBlur(absz.astype(np.float32), (0, 0),
                                                                          sigmaX=seed_sigma,
                                                                          sigmaY=seed_sigma)
                                            else:
                                                absz_s = absz

                                            with _Timer('rd_fdr', prof):
                                                pvals = 2.0 * (1.0 - self._normal_cdf(absz_s))
                                                seeds = self._bh_fdr_mask(pvals, q=q_fdr, mask=body)

                                            # 3) Remove tiny seed islands and optionally open to de-spur
                                            from skimage.morphology import remove_small_objects, opening, disk
                                            seed_min_area_frac = float(
                                                self.config.get('rd_seed_min_area_frac', 0.0002))  # fraction of body
                                            seed_open_r = int(self.config.get('rd_seed_open_r', 0))
                                            if np.any(seeds):
                                                with _Timer('rd_seed_post', prof):
                                                    min_area_seed = max(1, int(seed_min_area_frac * max(1, body_area)))
                                                    seeds = remove_small_objects(seeds.astype(bool),
                                                                                 min_size=min_area_seed)
                                                    if seed_open_r > 0:
                                                        seeds = opening(seeds, disk(int(seed_open_r)))

                                            # 4) Optional fallback: allow empty (preferred) or percentile fallback (legacy)
                                            if not np.any(seeds):
                                                with _Timer('rd_seed_post', prof):
                                                    if rd_allow_empty:
                                                        # Do NOT force-create seeds; leave empty and mark as nonempty=False
                                                        nonempty = False
                                                        seeds = np.zeros_like(body, dtype=bool)
                                                    else:
                                                        pctl = float(self.config.get('rd_seed_percentile',
                                                                                     99.5))  # e.g., 99.0~99.9
                                                        thr = max(Tl_fix, np.percentile(absz[body], pctl)) if np.any(
                                                            body) else Tl_fix
                                                        seeds = (absz >= thr) & body

                                            min_area = max(1, int(area_fr * max(1, body_area)))
                                            # If fallback above set nonempty, keep; otherwise recompute here
                                            if 'nonempty' not in locals():
                                                nonempty = bool(seeds.any())
                                            # Seed dilation as before
                                            if nonempty and seed_dilate > 0:
                                                from skimage.morphology import binary_dilation, disk
                                                with _Timer('rd_seed_post', prof):
                                                    seeds = binary_dilation(seeds, disk(int(seed_dilate))) & body

                                            final_mask = np.zeros_like(seeds, dtype=bool)
                                            if nonempty:
                                                with _Timer('rd_grow_cc', prof):
                                                    grown = self._hysteresis_from_seeds(zmap, seeds, Tl=Tl_fix)
                                                    from skimage.measure import label, regionprops
                                                    lab = label(grown, connectivity=2)
                                                if lab.max() > 0:
                                                    regs = regionprops(lab,
                                                                       intensity_image=np.abs(zmap).astype(np.float32))
                                                    # --- HARD-EVIDENCE FILTER: each component must contain at least K_hard pixels with |z| ≥ Th_hard ---
                                                    regs = [r for r in regs if r.area >= min_area]
                                                    if regs:
                                                        high_z = (np.abs(zmap) >= Th_hard)
                                                        valid_regs = []
                                                        for r in regs:
                                                            lbl = r.label
                                                            cnt_hi = int(np.count_nonzero(high_z & (lab == lbl)))
                                                            if cnt_hi >= max(0, K_hard):
                                                                valid_regs.append((r, cnt_hi))
                                                        if valid_regs:
                                                            # sort by original score; ties implicitly consider cnt_hi via mean_intensity/area
                                                            scored = sorted((vr[0] for vr in valid_regs),
                                                                            key=lambda r: float(
                                                                                r.mean_intensity) * np.log1p(
                                                                                float(r.area)), reverse=True)
                                                            keep_labels = [r.label for r in scored] if (
                                                                        topk is None or int(topk) < 0) else [r.label for
                                                                                                             r in
                                                                                                             scored[
                                                                                                                 :max(1,
                                                                                                                      topk)]]
                                                            with _Timer('rd_filter_morph', prof):
                                                                final_mask = np.isin(lab, keep_labels)
                                                                final_mask = self._morph_clean(
                                                                    final_mask,
                                                                    min_area=min_area,
                                                                    open_r=open_r,
                                                                    close_r=close_r,
                                                                    bridge_r=int(self.config.get('rd_bridge_r', 0))
                                                                )
                                                        else:
                                                            nonempty = False
                                                    else:
                                                        nonempty = False
                                                else:
                                                    nonempty = False

                                            if nonempty:
                                                with _Timer('rd_weight', prof):
                                                    if weight_mode.lower() == 'sigmoid':
                                                        w = 1.0 / (1.0 + np.exp(
                                                            -weight_alpha * (np.abs(zmap) - Tl_fix)))
                                                        weight_map = (w * body.astype(np.float32)).astype(np.float32)
                                                    else:
                                                        z_in = zmap[final_mask]
                                                        if z_in.size > 0 and float(z_in.max()) > float(z_in.min()):
                                                            w = (zmap - float(z_in.min())) / (
                                                                    float(z_in.max()) - float(z_in.min()) + 1e-6)
                                                        else:
                                                            w = (zmap >= Tl_fix).astype(np.float32)
                                                        weight_map = (w * final_mask.astype(np.float32)).astype(
                                                            np.float32)
                                            else:
                                                final_mask = np.zeros_like(zmap, dtype=bool)
                                                weight_map = np.zeros_like(zmap, dtype=np.float32)
                                            counters['rd_calls'] += 1

                                    # 缓存供“MC 结束后 + 有了std”再统一绘图
                                    composite_cache[slice_name] = {
                                        'inp': inp, 'gt': gt, 'pred': pred, 'residual': residual,
                                        'rd_enable': rd_enable,
                                        'zmap': zmap, 'seeds': seeds, 'final_mask': final_mask,
                                        'weight_map': weight_map,
                                        'mae_i': mae_i, 'psnr_i': psnr_i, 'ssim_i': ssim_i
                                    }

                                    # Defer plotting to after-MC when mc_runs>1 and enabled; otherwise plot now
                                    # When no MC accumulation, optionally plot immediately
                                    if mc_runs == 1 and save_composite:
                                        # sampling: only plot 1 of every N slices to speed up
                                        plot_ok = (plot_every_n <= 1) or ((counters.get('plot_enqueued', 0) % max(1, plot_every_n)) == 0)
                                        if plot_ok:
                                            metrics = {'mae': mae_i, 'psnr': psnr_i, 'ssim': ssim_i}
                                            rd_pack = None
                                            if rd_enable:
                                                rd_pack = {'zmap': zmap, 'seeds': seeds, 'final_mask': final_mask,
                                                           'weight_map': weight_map,
                                                           'q_fdr': self.config.get('rd_fdr_q', 0.10)}
                                            save_path = os.path.join(self.config['image_save'], f"{slice_name}.png")
                                            with _Timer('plot_immediate', prof):
                                                plot_composite(inp=inp, gt=gt, pred=pred, residual=residual,
                                                               metrics=metrics, rd_data=rd_pack, uncertainty=None,
                                                               use_reg=use_reg, save_path=save_path, overlay_rgb=None)
                                            counters['plots'] += 1
                                        counters['plot_enqueued'] = counters.get('plot_enqueued', 0) + 1

                                    pred_img = ((pred + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                                    pred_only_path = os.path.join(pred_save_root, f"{slice_name}.png")
                                    with _Timer('save_pred_png', prof):
                                        cv2.imwrite(pred_only_path, pred_img)
                                    counters['save_pred_png'] += 1
                                    if use_reg:
                                        pred_pre_img = ((fr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                                        pred_only_pre_path = os.path.join(pred_save_root, f"{slice_name}_preReg.png")
                                        with _Timer('save_pred_pre_png', prof):
                                            cv2.imwrite(pred_only_pre_path, pred_pre_img)
                                        counters['save_pred_pre_png'] += 1

                        # === After finishing MC runs for this batch: save mean pred and uncertainty (std) per slice ===
                        for b in range(B):
                            patient_id = patient_ids[b] if len(patient_ids) > b else None
                            slice_id = slice_ids[b] if len(slice_ids) > b else None
                            if patient_id is not None and slice_id is not None:
                                slice_name = f"{patient_id}_{slice_id}"
                            else:
                                slice_name = f"{i:05d}_{b:02d}"
                            if slice_name not in pred_sum:
                                continue
                            s = pred_sum.pop(slice_name)
                            ss = pred_sumsq.pop(slice_name)
                            # Mean and std in [-1,1]
                            mean_pred = s / float(mc_runs)
                            var = np.maximum(ss / float(mc_runs) - mean_pred ** 2, 0.0)
                            std_pred = np.sqrt(var, dtype=np.float32)

                            # 若需要单独保存，再受开关控制（默认不保存）
                            if save_uncert_outputs:
                                with _Timer('save_uncert', prof):
                                    mean_img = ((mean_pred + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                                    mean_path = os.path.join(mean_save_root, f"{slice_name}_mean.png")
                                    cv2.imwrite(mean_path, mean_img)
                                    std_raw_path = os.path.join(uncert_save_root, f"{slice_name}_std.npy")
                                    np.save(std_raw_path, std_pred.astype(np.float32))
                                    std_vis = (std_pred / (std_pred.max() + 1e-8) * 255.0).astype(
                                        np.uint8) if std_pred.max() > 0 else \
                                        np.zeros_like(mean_img, dtype=np.uint8)
                                    std_vis_path = os.path.join(uncert_save_root, f"{slice_name}_std.png")
                                    cv2.imwrite(std_vis_path, std_vis)
                                counters['save_uncert'] += 1

                            # Render unified composite (only if mc_runs > 1 and enabled), now with uncertainty available
                            if (mc_runs > 1) and save_composite and (slice_name in composite_cache):
                                c = composite_cache.pop(slice_name)
                                inp, gt, pred, residual = c['inp'], c['gt'], c['pred'], c['residual']
                                rd_enable = c['rd_enable']
                                rd_pack = None
                                if rd_enable:
                                    rd_pack = {'zmap': c['zmap'], 'seeds': c['seeds'], 'final_mask': c['final_mask'],
                                               'weight_map': c['weight_map'],
                                               'q_fdr': self.config.get('rd_fdr_q', 0.10)}
                                metrics = {'mae': c['mae_i'], 'psnr': c['psnr_i'], 'ssim': c['ssim_i']}
                                save_path = os.path.join(self.config['image_save'], f"{slice_name}.png")
                                plot_ok = (plot_every_n <= 1) or ((counters.get('plot_enqueued_after', 0) % max(1, plot_every_n)) == 0)
                                if plot_ok:
                                    with _Timer('plot_after_mc', prof):
                                        plot_composite(inp=inp, gt=gt, pred=pred, residual=residual,
                                                       metrics=metrics, rd_data=rd_pack, uncertainty=None,
                                                       use_reg=use_reg, save_path=save_path, overlay_rgb=None)
                                    counters['plots'] += 1
                                counters['plot_enqueued_after'] = counters.get('plot_enqueued_after', 0) + 1

            # Compute overall averages over all (MC × slices) for quick reference
            avg_mae = MAE / N if N > 0 else float('nan')
            avg_psnr = PSNR / N if N > 0 else float('nan')
            avg_ssim = SSIM / N if N > 0 else float('nan')

            # --- Timing summary ---
            print("\n==== Timing (seconds) ====")
            total_time = sum(prof.values())
            for k in sorted(prof.keys()):
                print(f"{k:>22s}: {prof[k]:.3f}")
            print(f"{'TOTAL (sum above)':>22s}: {total_time:.3f}")
            if total_processed_slices > 0:
                print(
                    f"{'per-slice (approx)':>22s}: {total_time / total_processed_slices:.4f} s/slice (processed={total_processed_slices})")
            # occurrence counters
            print("---- Counters ----")
            for k in sorted(counters.keys()):
                print(f"{k:>22s}: {counters[k]}")
            print("==== End Timing ====")

            # Build mean/std table per slice
            slice_rows = []  # (slice, mae_mean, mae_std, psnr_mean, psnr_std, ssim_mean, ssim_std)
            for s, vals in per_slice_values.items():
                mae_arr = np.array(vals['mae'], dtype=np.float64)
                psnr_arr = np.array(vals['psnr'], dtype=np.float64)
                ssim_arr = np.array(vals['ssim'], dtype=np.float64)
                row = (
                    s,
                    float(mae_arr.mean()), float(mae_arr.std(ddof=1) if mae_arr.size > 1 else 0.0),
                    float(psnr_arr.mean()), float(psnr_arr.std(ddof=1) if psnr_arr.size > 1 else 0.0),
                    float(ssim_arr.mean()), float(ssim_arr.std(ddof=1) if ssim_arr.size > 1 else 0.0),
                )
                slice_rows.append(row)

            # Sort by psnr_mean then ssim_mean (both desc)
            slice_rows.sort(key=lambda x: (x[3], x[5]), reverse=True)

            csv_path = os.path.join(self.config['image_save'], "metrics.csv")
            with open(csv_path, "w", newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                reg_note = ' (after registration)' if use_reg else ''
                mc_note = f" | MC runs={mc_runs} mode={mc_mode} (mean/std over runs)" if mc_runs > 1 else ''
                # If injection is enabled, include extra columns
                if synth_enable_global:
                    csv_writer.writerow([f'slice', f'mae_mean{reg_note}', 'mae_std', f'psnr_mean{reg_note}', 'psnr_std',
                                         f'ssim_mean{reg_note}', 'ssim_std', 'iou_gt', 'dice_gt'])
                else:
                    csv_writer.writerow([f'slice', f'mae_mean{reg_note}', 'mae_std', f'psnr_mean{reg_note}', 'psnr_std',
                                         f'ssim_mean{reg_note}', 'ssim_std'])
                csv_writer.writerow(
                    ['average_over_all_samples', f"{avg_mae:.6f}", '', f"{avg_psnr:.6f}", '', f"{avg_ssim:.6f}",
                     f"{mc_note}"])
                for row in slice_rows:
                    if synth_enable_global:
                        # Only rows for injected slices (pair_ok) are present, so just output NA for iou_gt/dice_gt for now.
                        csv_writer.writerow(
                            [row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}",
                             f"{row[5]:.6f}",
                             f"{row[6]:.6f}", 'NA', 'NA']
                        )
                    else:
                        csv_writer.writerow(
                            [row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}",
                             f"{row[5]:.6f}",
                             f"{row[6]:.6f}"]
                        )

            print('MAE (mean over all MC×slices):', avg_mae)
            print('PSNR (mean over all MC×slices):', avg_psnr)
            print('SSIM (mean over all MC×slices):', avg_ssim)

    def PSNR(self, fake, real, mode: str = 'correct'):
        """
        Compute PSNR in dB.

        Parameters
        ----------
        fake : np.ndarray-like
        real : np.ndarray-like
        mode : str
            'correct' (default): correct foreground-masked PSNR using boolean mask (real != -1).
            'reggan-original': reproduce the original RegGAN-style computation requested by user.
            'full-image': PSNR computed over the entire image without masking.
        """
        fake = np.asarray(fake)
        real = np.asarray(real)

        if mode == 'reggan-original':
            # User-requested original (incorrect) variant
            # Foreground by coordinates, then compute MSE in [0,1]
            x, y = np.where(real != -1)
            mse = np.mean(((fake[x][y] + 1) / 2. - (real[x][y] + 1) / 2.) ** 2)

            if mse < 1.0e-10:
                psnr = 100.0
            else:
                PIXEL_MAX = 1.0
                psnr = 20.0 * np.log10(PIXEL_MAX / np.sqrt(mse))

        elif mode == 'correct':
            # ---- Correct method (default) ----
            # Compute squared error in [0,1] space
            mse = ((fake + 1.0) / 2.0 - (real + 1.0) / 2.0) ** 2

            # Foreground mask: real != -1 (same shape as data)
            mask = (real != -1)

            # Try to align dims in case of squeezes/extra singleton dims
            try:
                if mask.shape != mse.shape:
                    mask = np.squeeze(mask)
                    mse = np.squeeze(mse)
            except Exception:
                pass

            if mask.any():
                mse = mse[mask].mean()
            else:
                mse = mse.mean()

            if mse < 1.0e-10:
                psnr = 100.0
            else:
                PIXEL_MAX = 1.0
                psnr = 20.0 * np.log10(PIXEL_MAX / np.sqrt(mse))
        elif mode == 'full-image':
            psnr = peak_signal_noise_ratio((real + 1.0) / 2.0, (fake + 1.0) / 2.0, data_range=1.0)
        elif mode == 'skimage':
            # Use skimage's implementation (foreground-masked)
            mask = (real != -1)
            if mask.any():
                psnr = peak_signal_noise_ratio((real[mask] + 1.0) / 2.0, (fake[mask] + 1.0) / 2.0, data_range=1.0)
            else:
                psnr = peak_signal_noise_ratio((real + 1.0) / 2.0, (fake + 1.0) / 2.0, data_range=1.0)
        else:
            raise ValueError(f"Invalid mode for PSNR: {mode}")

        return psnr

    def MAE(self, fake, real):
        fake = np.asarray(fake)
        real = np.asarray(real)

        # Absolute error in [-1,1] space then rescale to [0,1]
        abs_err = np.abs(fake - real) / 2.0

        # Foreground mask: real != -1
        mask = (real != -1)

        # Align dims if batch/channel dims exist
        try:
            if mask.shape != abs_err.shape:
                mask = np.squeeze(mask)
                abs_err = np.squeeze(abs_err)
        except Exception:
            pass

        if mask.any():
            return float(abs_err[mask].mean())
        else:
            return float(abs_err.mean())

    def save_deformation(self, defms, root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max, x_min = dir_x.max(), dir_x.min()
        y_max, y_min = dir_y.max(), dir_y.min()
        dir_x = ((dir_x - x_min) / (x_max - x_min)) * 255
        dir_y = ((dir_y - y_min) / (y_max - y_min)) * 255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy)
