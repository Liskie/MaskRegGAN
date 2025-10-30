#!/usr/bin/python3

import itertools
import os
import random
from datetime import datetime
import math
import time
import copy
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from torchvision.transforms import RandomAffine, ToPILImage
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import cv2
from PIL import Image as PILImage
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from torch.autograd import Variable
import wandb
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TextColumn

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:
    LearnedPerceptualImagePatchSimilarity = None

from .utils import (
    Resize, ToTensor, smooothing_loss, LambdaLR, Logger, ReplayBuffer, plot_composite,
    ResizeKeepRatioPad,
    norm01, robust_zscore, normal_cdf, bh_fdr_mask, hysteresis_from_seeds, morph_clean, save_gray_png,
    compose_slice_name, load_weight_or_mask_for_slice, load_weights_for_batch, masked_l1,
    compute_mae, compute_psnr, compute_ssim, resolve_model_path
)
from .datasets import ImageDataset, ValDataset
from .reg import Reg
from .transformer import Transformer_2D
from models.CycleGan import *
from .common_utils import _Timer
from .distrib_utils import (
    _dist_is_available_and_initialized,
    _get_world_size,
    _get_rank,
    _is_main_process,
    _setup_ddp_if_needed,
    _reduce_tensor_sum,
    _enable_mc_dropout,
)

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._config_snapshots = {}
        self._config_logged_stages = set()
        self._wandb_settings = self._build_wandb_settings(config)
        self._progress_disable = bool(config.get('disable_progress', False))
        ## def networks
        upsample_mode = str(config.get('generator_upsample_mode', 'resize')).lower()
        self.netG_A2B = Generator(config['input_nc'], config['output_nc'], upsample_mode=upsample_mode).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc'], upsample_mode=upsample_mode).cuda()
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

        # --- Residual-detection configuration (must be defined BEFORE building datasets) ---
        raw_mode = config.get('rd_mode', config.get('rd_input_type', 'none'))
        mode = str(raw_mode).strip().lower() if raw_mode is not None else 'none'
        if mode in ('none', 'false', 'off', '0', ''):
            self.rd_mode = None
        elif mode in ('mask', 'weights'):
            self.rd_mode = mode
        else:
            print(f"[train][warn] Unknown rd_mode '{mode}', disabling residual guidance")
            self.rd_mode = None

        # Fixed mask/weights directory (Scheme B)
        self.rd_mask_dir = str(config.get('rd_mask_dir', '') or '').strip()
        # Optional directory for soft weights (0..1 float maps); if empty, fallback to rd_mask_dir
        self.rd_weights_dir = str(config.get('rd_weights_dir', '') or '').strip()
        # Minimum weight floor to avoid vanishing loss when using soft weights
        self.rd_w_min = float(config.get('rd_w_min', 0.0))

        if self.rd_mode is None:
            self.rd_input_type = None
            print('[train] Residual guidance disabled (rd_mode=none).')
        else:
            self.rd_input_type = self.rd_mode
            if self.rd_mode == 'weights':
                if not self.rd_weights_dir:
                    raise ValueError("rd_mode='weights' requires rd_weights_dir to be set")
                print(f"[train] Using residual WEIGHTS from: {self.rd_weights_dir} (w_min={self.rd_w_min})")
            elif self.rd_mode == 'mask':
                if not self.rd_mask_dir:
                    raise ValueError("rd_mode='mask' requires rd_mask_dir to be set")
                print(f"[train] Using residual MASKS from: {self.rd_mask_dir}")

        # === Validation metrics masking (val-time) ===
        # Defaults to metrics_use_rd if unspecified
        self.metrics_use_rd = bool(config.get('metrics_use_rd', False)) if not hasattr(self, 'metrics_use_rd') else self.metrics_use_rd
        self.val_metrics_use_rd = bool(config.get('val_metrics_use_rd', config.get('metrics_use_rd', False)))
        self.val_save_keep_masks = bool(config.get('val_save_keep_masks', False))
        self.val_keep_masks_dir = None
        if self.val_save_keep_masks:
            default_keep_dir = config.get('val_keep_masks_dir',
                                          os.path.join(config.get('image_save', config.get('save_root', '.')),
                                                       'val_metrics_keep'))
            try:
                os.makedirs(default_keep_dir, exist_ok=True)
                self.val_keep_masks_dir = default_keep_dir
            except Exception as _mk_err:
                print(f"[val] Failed to create val_keep_masks_dir '{default_keep_dir}': {_mk_err}")
                self.val_keep_masks_dir = None

        # Dataset loader
        level = config['noise_level']  # set noise level
        # --- DataLoader runtime controls ---
        self._n_workers = int(config.get('n_cpu', 0))
        self._dl_timeout = int(config.get('dataloader_timeout', 0))  # 0 = no timeout
        self._dl_persistent = bool(config.get('persistent_workers', True))
        self._dl_prefetch = int(config.get('prefetch_factor', 2))
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

        dl_kwargs = dict(
            batch_size=config['batchSize'],
            shuffle=True,
            num_workers=self._n_workers,
            pin_memory=config['cuda'],
            timeout=self._dl_timeout,
        )
        if self._n_workers > 0:
            dl_kwargs.update(
                persistent_workers=self._dl_persistent,
                prefetch_factor=max(1, self._dl_prefetch),
            )
        self.train_data = DataLoader(
            ImageDataset(
                config['dataroot'], level,
                transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False,
                rd_input_type=self.rd_input_type,
                rd_mask_dir=self.rd_mask_dir,
                rd_weights_dir=self.rd_weights_dir,
                rd_w_min=self.rd_w_min,
                cache_mode=config.get('cache_mode', 'mmap'),  # ← 新增
                rd_cache_weights=config.get('rd_cache_weights', False),
            ),
            **dl_kwargs
        )

        val_transforms = [ToTensor(),
                          last_tf]

        vdl_kwargs = dict(
            batch_size=config['batchSize'],
            shuffle=False,
            num_workers=self._n_workers,
            pin_memory=config['cuda'],
            timeout=self._dl_timeout,
        )
        if self._n_workers > 0:
            vdl_kwargs.update(
                persistent_workers=self._dl_persistent,
                prefetch_factor=max(1, self._dl_prefetch),
            )
        self.val_data = DataLoader(
            ValDataset(
                config['val_dataroot'],
                transforms_=val_transforms,
                unaligned=False,
                rd_input_type=self.rd_input_type,
                rd_mask_dir=self.rd_mask_dir,
                rd_weights_dir=self.rd_weights_dir,
                rd_w_min=self.rd_w_min,
                cache_mode=config.get('cache_mode', 'mmap'),  # ← 新增
                rd_cache_weights=config.get('rd_cache_weights', False),
            ),
            **vdl_kwargs
        )

        # Debug info: count how many dataset samples include rd_weight/mask (with progress & optional limit)
        try:
            do_scan = bool(self.config.get('rd_scan_on_init', True))
            scan_limit = int(self.config.get('rd_scan_max_samples', 0))  # 0/neg = no limit
            if do_scan and _is_main_process():
                def _count_with_progress(ds, tag: str):
                    try:
                        n = len(ds)
                    except Exception:
                        n = None
                    count = 0
                    # Only main process renders progress to avoid clutter
                    use_bar = _is_main_process()
                    pg = Progress(
                        TextColumn(f"[yellow]{tag} mask-scan[/]"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                    ) if use_bar else None
                    if pg is not None:
                        task = pg.add_task("scan", total=n)
                    from contextlib import nullcontext
                    with (pg if pg is not None else nullcontext()):
                        iterable = None
                        try:
                            if hasattr(ds, '__getitem__') and hasattr(ds, '__len__'):
                                iterable = (ds[i] for i in range(len(ds)))
                            else:
                                iterable = iter(ds)
                        except Exception:
                            iterable = iter([])

                        i = 0
                        for s in iterable:
                            if isinstance(s, dict) and bool(s.get('rd_has_file', False)):
                                count += 1
                            if pg is not None:
                                pg.advance(task, 1)
                            i += 1
                            if isinstance(scan_limit, int) and scan_limit > 0 and i >= scan_limit:
                                break
                        return count
                tr_cnt = _count_with_progress(self.train_data.dataset, 'train') if hasattr(self, 'train_data') and hasattr(self.train_data, 'dataset') else 0
                va_cnt = _count_with_progress(self.val_data.dataset, 'val') if hasattr(self, 'val_data') and hasattr(self.val_data, 'dataset') else 0
                # Add a concise one-line summary after bars finish
                print(f"[init] mask/weight presence → train: {tr_cnt} / {len(self.train_data.dataset) if hasattr(self.train_data, 'dataset') else '?'} | val: {va_cnt} / {len(self.val_data.dataset) if hasattr(self.val_data, 'dataset') else '?'} (limit={scan_limit if scan_limit>0 else 'all'})")
            else:
                print("[init] mask-count scan skipped (rd_scan_on_init=False)")
        except Exception as e:
            print(f"[init] mask-count scan failed: {e}")

        # Loss plot (defer Logger init until after DDP setup to avoid multi-init on non-main ranks)
        self.logger = None

        # Best-validation tracking & checkpoint
        self.best_val_mae = float('inf')
        self.best_ckpt_path = os.path.join(self.config.get('save_root', './'), 'best.pth')
        # How many validation samples to upload to wandb (images)
        self.val_sample_images = int(self.config.get('val_sample_images', 50))  # 0 = disable, default 50

        # Fixed validation tracking (log the same K triplets every validation)
        self.val_track_enable = bool(self.config.get('val_track_enable', True))
        self.val_track_fixed_n = int(self.config.get('val_track_fixed_n', 50))
        try:
            ds_len = len(self.val_data.dataset)
        except Exception:
            ds_len = 0
        k = max(0, min(self.val_track_fixed_n, ds_len))
        self.val_track_indices = list(range(k))  # deterministic: first K samples
        if self.val_track_enable and k > 0:
            print(f"[val-track] Tracking {k} fixed validation samples each eval.")

        # Cache initial configuration snapshot for logging/reporting
        self._config_snapshots['init'] = self._effective_config_snapshot(stage='init')

        # === 周期性保存配置 ===
        try:
            self._save_every = int(self.config.get('save_every_n_epochs', 0) or 0)
        except Exception:
            self._save_every = 0
        # 约定保存根目录（若已有 best/last 的保存目录，这里沿用）
        self.save_root = self.config.get('save_root')
        os.makedirs(self.save_root, exist_ok=True)
        # 运行标识，便于调试（不影响文件名的去重逻辑）
        self._run_tag = self.config.get('run_tag', datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._val_metric_support = None


    # ------------------------------------------------------------------
    # Helper utilities for configuration reporting / logging
    # ------------------------------------------------------------------
    def _build_wandb_settings(self, cfg):
        tags_raw = cfg.get('wandb_tags', None)
        tags = None
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
        elif isinstance(tags_raw, (list, tuple)):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        settings = {
            'project': cfg.get('wandb_project'),
            'entity': cfg.get('wandb_entity'),
            'group': cfg.get('wandb_group'),
            'job_type': cfg.get('wandb_job_type'),
            'notes': cfg.get('wandb_notes'),
            'run_name': cfg.get('wandb_run_name'),
            'tags': tags,
        }
        return settings

    def _effective_config_snapshot(self, stage=None):
        snap = copy.deepcopy(self.config)
        resolved = {
            'stage': stage or 'init',
            'rd_mode_effective': self.rd_mode or 'none',
            'rd_input_type_effective': self.rd_input_type or 'none',
            'rd_mask_dir_effective': self.rd_mask_dir or '',
            'rd_weights_dir_effective': self.rd_weights_dir or '',
            'rd_w_min_effective': self.rd_w_min,
            'resize_mode_effective': str(self.config.get('resize_mode', 'resize')),
            'cache_mode_effective': str(self.config.get('cache_mode', 'mmap')),
            'rd_cache_weights': bool(self.config.get('rd_cache_weights', False)),
            'metrics_use_rd': bool(self.config.get('metrics_use_rd', False)),
            'val_metrics_use_rd': bool(self.config.get('val_metrics_use_rd', self.config.get('metrics_use_rd', False))),
            'test_metrics_use_rd': bool(self.config.get('test_metrics_use_rd', self.config.get('metrics_use_rd', False))),
            'batch_size': int(self.config.get('batchSize', 1)),
            'noise_level': int(self.config.get('noise_level', 0)),
            'ddp_world_size': _get_world_size(),
            'device': str(getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))),
            'train_dataset_len': None,
            'val_dataset_len': None,
            'train_loader_len': None,
            'val_loader_len': None,
        }
        try:
            resolved['train_dataset_len'] = len(self.train_data.dataset)
        except Exception:
            pass
        try:
            resolved['val_dataset_len'] = len(self.val_data.dataset)
        except Exception:
            pass
        try:
            resolved['train_loader_len'] = len(self.train_data)
        except Exception:
            pass
        try:
            resolved['val_loader_len'] = len(self.val_data)
        except Exception:
            pass
        if stage == 'train':
            resolved['save_every_n_epochs'] = int(self.config.get('save_every_n_epochs', 0) or 0)
        if stage == 'test':
            resolved['mc_runs_effective'] = int(self.config.get('mc_runs', 1))
            resolved['mc_mode'] = str(self.config.get('mc_mode', 'dropout'))
        snap['_resolved'] = resolved
        return snap

    def _log_stage_config(self, stage):
        snap = self._effective_config_snapshot(stage=stage)
        self._config_snapshots[stage] = snap
        if stage not in self._config_logged_stages and _is_main_process():
            try:
                print(f"[config/{stage}] Effective configuration:")
                print(json.dumps(snap, indent=2, sort_keys=True))
            except Exception as _e:
                print(f"[config/{stage}] Failed to pretty-print config: {_e}")
                print(str(snap))
            self._config_logged_stages.add(stage)
        return snap


    def train(self, max_epochs: Optional[int] = None):

        ###### Training ######

        # === Distributed setup ===
        ddp_enabled, local_rank = _setup_ddp_if_needed(self.config)
        self.ddp_enabled = ddp_enabled
        self.local_rank = local_rank if ddp_enabled else 0
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda', self.local_rank) if torch.cuda.is_available() else torch.device('cpu')
        else:
            if ddp_enabled:
                self.device = torch.device('cuda', self.local_rank)

        # Helper: rebuild a DataLoader with a given sampler (avoids setting sampler post-init)
        def _rebuild_loader_with_sampler(ld, sampler, shuffle: bool = False):
            if ld is None:
                return ld
            try:
                # Some kwargs may not exist depending on torch version / settings
                kwargs = dict(
                    batch_size=ld.batch_size,
                    shuffle=False,  # sampler and shuffle are mutually exclusive
                    num_workers=ld.num_workers,
                    pin_memory=getattr(ld, 'pin_memory', False),
                    timeout=getattr(ld, 'timeout', 0),
                    drop_last=getattr(ld, 'drop_last', False),
                    sampler=sampler,
                )
                # Only pass persistent_workers/prefetch_factor when num_workers > 0
                if getattr(ld, 'num_workers', 0) and getattr(ld, 'persistent_workers', False):
                    kwargs['persistent_workers'] = ld.persistent_workers
                if getattr(ld, 'num_workers', 0) and hasattr(ld, 'prefetch_factor') and ld.prefetch_factor is not None:
                    kwargs['prefetch_factor'] = ld.prefetch_factor
                return DataLoader(ld.dataset, **kwargs)
            except TypeError:
                # Extremely defensive fallback without optional args that sometimes cause TypeErrors
                return DataLoader(ld.dataset, batch_size=ld.batch_size, shuffle=False,
                                  num_workers=ld.num_workers, pin_memory=getattr(ld, 'pin_memory', False),
                                  drop_last=getattr(ld, 'drop_last', False), sampler=sampler)

        # (Logger initialization moved to after DDP sampler attach)

        # Move & wrap models
        self.netG_A2B = self.netG_A2B.to(self.device)
        if ddp_enabled:
            self.netG_A2B = DDP(self.netG_A2B, device_ids=[self.local_rank], output_device=self.local_rank,
                                find_unused_parameters=False)

        if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
            self.R_A = self.R_A.to(self.device)
            if ddp_enabled:
                self.R_A = DDP(self.R_A, device_ids=[self.local_rank], output_device=self.local_rank,
                               find_unused_parameters=False)

        # Ensure discriminators/generators are on the right device too
        if hasattr(self, 'netD_B') and isinstance(self.netD_B, torch.nn.Module):
            self.netD_B = self.netD_B.to(self.device)
            if ddp_enabled:
                self.netD_B = DDP(self.netD_B, device_ids=[self.local_rank], output_device=self.local_rank,
                                   find_unused_parameters=False)

        if getattr(self.config, 'bidirect', False) or self.config.get('bidirect', False):
            if hasattr(self, 'netG_B2A') and isinstance(self.netG_B2A, torch.nn.Module):
                self.netG_B2A = self.netG_B2A.to(self.device)
                if ddp_enabled:
                    self.netG_B2A = DDP(self.netG_B2A, device_ids=[self.local_rank], output_device=self.local_rank,
                                         find_unused_parameters=False)
            if hasattr(self, 'netD_A') and isinstance(self.netD_A, torch.nn.Module):
                self.netD_A = self.netD_A.to(self.device)
                if ddp_enabled:
                    self.netD_A = DDP(self.netD_A, device_ids=[self.local_rank], output_device=self.local_rank,
                                       find_unused_parameters=False)

        # Attach DistributedSampler by rebuilding the DataLoaders (do not set sampler post-init)
        if ddp_enabled:
            try:
                if hasattr(self, 'train_data') and hasattr(self.train_data, 'dataset'):
                    tr_sampler = DistributedSampler(self.train_data.dataset, shuffle=True, drop_last=False)
                    self.train_data = _rebuild_loader_with_sampler(self.train_data, tr_sampler)
                if hasattr(self, 'val_data') and hasattr(self.val_data, 'dataset'):
                    va_sampler = DistributedSampler(self.val_data.dataset, shuffle=False, drop_last=False)
                    self.val_data = _rebuild_loader_with_sampler(self.val_data, va_sampler)
            except Exception as _e:
                if _is_main_process():
                    print(f"[DDP] Sampler setup warning: {_e}")

        train_cfg_snapshot = self._log_stage_config('train')

        # (Re)create Logger here, AFTER DDP samplers are attached, so batches_epoch reflects per-rank size
        if self.logger is None:
            try:
                batches_epoch = len(self.train_data)
            except Exception:
                batches_epoch = 0
            self.logger = Logger(
                self.config['name'],
                self.config['port'],
                self.config['n_epochs'],
                batches_epoch,
                is_main=_is_main_process(),
                wandb_settings=self._wandb_settings,
                wandb_config=copy.deepcopy(train_cfg_snapshot)
            )

        # Setup rich progress bar for training with two-level progress (epochs and batches)
        total_epochs = int(self.config['n_epochs'])
        start_epoch = int(self.config.get('epoch', 0))
        if max_epochs is None:
            target_epoch = total_epochs
        else:
            target_epoch = min(total_epochs, start_epoch + int(max_epochs))
        if start_epoch >= target_epoch:
            return
        try:
            total_batches = len(self.train_data)
        except Exception:
            total_batches = None

        if _is_main_process():
            try:
                ds_len = len(self.train_data.dataset)
                smp_len = len(self.train_data.sampler) if hasattr(self.train_data, 'sampler') and self.train_data.sampler is not None else ds_len
                print(f"[DDP] world_size={_get_world_size()} dataset={ds_len} per-rank-samples={smp_len} batch_size={self.train_data.batch_size} per-rank-batches={len(self.train_data)}", flush=True)
            except Exception:
                pass

        progress = None
        if not self._progress_disable and _is_main_process():
            progress = Progress(
                TextColumn("[bold green]Train[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
        num_epochs_this_run = target_epoch - start_epoch

        # --- Checkpoint saving cadence ---
        save_every_n = int(self.config.get('save_every_n_epochs', 0))  # 0 disables periodic saving

        def _unique_path(path: str) -> str:
            """Return a non-conflicting path by appending -vK if needed."""
            import os
            if not os.path.exists(path):
                return path
            base, ext = os.path.splitext(path)
            k = 1
            while True:
                cand = f"{base}-v{k}{ext}"
                if not os.path.exists(cand):
                    return cand
                k += 1

        with (progress if progress is not None else nullcontext()):
            # Outer task: epochs
            if progress is not None:
                task_epochs = progress.add_task(f"Epochs", total=num_epochs_this_run)
            else:
                task_epochs = None
            for epoch in range(start_epoch, target_epoch):
                try:
                    if isinstance(getattr(self.train_data, 'sampler', None), DistributedSampler):
                        self.train_data.sampler.set_epoch(epoch)
                except Exception:
                    pass
                # Inner task: batches in this epoch
                if progress is not None:
                    task_batches = progress.add_task(
                        f"Epoch {epoch + 1}/{total_epochs}",
                        total=total_batches if (isinstance(total_batches, int) and total_batches > 0) else None
                    )
                else:
                    task_batches = None
                # === Epoch-level loss aggregators (to log average train loss per epoch) ===
                ep_sums = defaultdict(float)   # name -> summed loss over the epoch
                ep_counts = defaultdict(int)   # name -> number of times this loss was available
                n_batches_this_epoch = 0
                def _ep_acc(name, val):
                    if val is None:
                        return
                    try:
                        v = float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                    except Exception:
                        return
                    ep_sums[name] += v
                    ep_counts[name] += 1
                for i, batch in enumerate(self.train_data):

                    # Initialize loss holders to ensure safe logging even if a branch is skipped
                    SR_loss = None
                    adv_loss = None
                    SM_loss = None
                    loss_D_B = None
                    total_loss = None

                    # Set model input
                    real_A = batch['A'].to(self.device, non_blocking=True)  # Accelerate
                    real_B = batch['B'].to(self.device, non_blocking=True)  # Accelerate

                    # Only retain the single-direction NC+R branch
                    if self.config['regist']:  # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        use_fixed_guidance = self.rd_input_type in ('mask', 'weights')
                        if use_fixed_guidance:
                            # Prefer preloaded per-slice weights from the dataset for alignment and speed
                            if 'rd_weight' in batch and batch['rd_weight'] is not None:
                                w_batch = batch['rd_weight'].to(real_B.device, dtype=real_B.dtype)
                            else:
                                w_batch = load_weights_for_batch(
                                    batch, i, real_B,
                                    self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min
                                )
                            sr_core = masked_l1(SysRegist_A2B, real_B, w_batch)
                        elif 'body_mask' in batch:
                            w_body = batch['body_mask'].to(real_B.device, dtype=real_B.dtype)
                            sr_core = masked_l1(SysRegist_A2B, real_B, w_body)
                        else:
                            sr_core = self.L1_loss(SysRegist_A2B, real_B)
                        SR_loss = self.config['Corr_lamda'] * sr_core  ### masked SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, torch.ones_like(pred_fake0))
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        total_loss = SM_loss + adv_loss + SR_loss
                        total_loss.backward()
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

                    # --- accumulate for epoch averages ---
                    n_batches_this_epoch += 1
                    _ep_acc('total', total_loss)
                    _ep_acc('D_B', loss_D_B)
                    _ep_acc('SR', SR_loss)
                    _ep_acc('adv',   adv_loss)
                    _ep_acc('smooth', SM_loss)
                    # Log losses every 50 batches, including breakdown components
                    if i % 50 == 0:
                        log_losses = {}
                        if total_loss is not None:
                            log_losses['loss/total'] = float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else float(total_loss)
                        if loss_D_B is not None:
                            log_losses['loss/D_B'] = float(loss_D_B.item()) if isinstance(loss_D_B, torch.Tensor) else float(loss_D_B)
                        if SR_loss is not None:
                            log_losses['loss/SR'] = float(SR_loss.item()) if isinstance(SR_loss, torch.Tensor) else float(SR_loss)
                        if adv_loss is not None:
                            log_losses['loss/adv'] = float(adv_loss.item()) if isinstance(adv_loss, torch.Tensor) else float(adv_loss)
                        if SM_loss is not None:
                            log_losses['loss/smooth'] = float(SM_loss.item()) if isinstance(SM_loss, torch.Tensor) else float(SM_loss)
                        if log_losses:
                            self.logger.log_step(log_losses)
                    # advance inner progress
                    try:
                        progress.advance(task_batches, 1)
                    except Exception:
                        pass
                    # also update the epoch task with fractional progress so ETA is meaningful
                    try:
                        if progress is not None and task_epochs is not None and isinstance(total_batches, int) and total_batches > 0:
                            # completed epochs so far + fractional within this epoch
                            frac = (i + 1) / float(total_batches)
                            progress.update(task_epochs, completed=(epoch - start_epoch) + frac)
                    except Exception:
                        pass
                # one epoch finished → advance outer progress and snap to integer boundary
                try:
                    if progress is not None and task_epochs is not None:
                        progress.advance(task_epochs, 1)
                        # ensure completed count is exact integer at epoch end
                        progress.update(task_epochs, completed=(epoch - start_epoch + 1))
                except Exception:
                    pass

                # remove inner batch progress bar to avoid it staying on screen
                try:
                    if progress is not None and task_batches is not None:
                        progress.remove_task(task_batches)
                except Exception:
                    pass

                # === Log epoch-averaged training losses to W&B (main process only) ===
                if _is_main_process():
                    ep_means = {}
                    for k, s in ep_sums.items():
                        c = max(1, ep_counts.get(k, 0))
                        ep_means[k] = s / c
                    self.logger.log_epoch(ep_means, epoch + 1)

                # Save model checkpoints (periodic + non-conflicting filenames)
                if _is_main_process():
                    should_save = (isinstance(save_every_n, int) and save_every_n > 0 and (
                                (epoch + 1) % save_every_n == 0))
                    # Also ensure we save the last epoch even if it doesn't align with the cadence
                    if not should_save and (epoch + 1) == total_epochs:
                        should_save = True
                    if should_save:
                        save_root = self.config.get('save_root', './')
                        os.makedirs(save_root, exist_ok=True)

                        # --- Generator A2B ---
                        g_state = self.netG_A2B.module.state_dict() if isinstance(self.netG_A2B,
                                                                                  DDP) else self.netG_A2B.state_dict()
                        arch_g = os.path.join(save_root, f"netG_A2B_ep{epoch + 1:04d}.pth")
                        arch_g = _unique_path(arch_g)
                        torch.save(g_state, arch_g)
                        # maintain a stable latest file for downstream code (e.g., test())
                        torch.save(g_state, os.path.join(save_root, 'netG_A2B.pth'))

                        # --- Registration network (if present) ---
                        if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
                            r_state = self.R_A.module.state_dict() if isinstance(self.R_A,
                                                                                 DDP) else self.R_A.state_dict()
                            arch_r = os.path.join(save_root, f"R_A_ep{epoch + 1:04d}.pth")
                            arch_r = _unique_path(arch_r)
                            torch.save(r_state, arch_r)
                            torch.save(r_state, os.path.join(save_root, 'R_A.pth'))

                #############val###############
                if epoch % self.config['val_freq'] == 0:
                    # Switch models to eval mode and record previous state
                    was_train_G = self.netG_A2B.training
                    self.netG_A2B.eval()
                    was_train_R = None
                    if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
                        was_train_R = self.R_A.training
                        self.R_A.eval()
                    with torch.no_grad():
                        # --- Optional masked metrics helpers for validation ---
                        # Set up a progress bar for validation
                        try:
                            total_val_batches = len(self.val_data)
                        except Exception:
                            total_val_batches = None
                        vprogress = None
                        if not self._progress_disable and _is_main_process():
                            vprogress = Progress(
                                TextColumn("[cyan]Validation[/]"),
                                BarColumn(),
                                MofNCompleteColumn(),
                                TimeElapsedColumn(),
                                TimeRemainingColumn(),
                            )
                        mae_sum = 0.0
                        psnr_sum = 0.0
                        ssim_sum = 0.0
                        sample_count = 0
                        FID_val = None
                        LPIPS_sum = 0.0
                        LPIPS_count = 0
                        val_metrics_modules = self._prepare_val_metrics_modules()
                        if val_metrics_modules is not None:
                            fid_metric = val_metrics_modules.get('fid', None)
                            if fid_metric is not None:
                                try:
                                    fid_metric.reset()
                                except Exception:
                                    pass
                            lpips_metric = val_metrics_modules.get('lpips', None)
                        else:
                            fid_metric = None
                            lpips_metric = None

                        def _save_val_keep_mask(slice_name: str, mask_arr):
                            if not self.val_keep_masks_dir:
                                return
                            try:
                                keep_png = ((np.asarray(mask_arr) > 0).astype(np.uint8) * 255)
                                cv2.imwrite(os.path.join(self.val_keep_masks_dir, f"{slice_name}.png"), keep_png)
                            except Exception:
                                pass
                        with (vprogress if vprogress is not None else nullcontext()):
                            if vprogress is not None:
                                vtask = vprogress.add_task(
                                    f"Val Epoch {epoch + 1}/{self.config['n_epochs']}",
                                    total=total_val_batches if (
                                            isinstance(total_val_batches, int) and total_val_batches > 0) else None
                                )
                            else:
                                vtask = None
                            for i, batch in enumerate(self.val_data):
                                # 异步拷到 GPU
                                real_A = batch['A'].to(self.device, non_blocking=True)
                                real_Bt = batch['B'].to(self.device, non_blocking=True)

                                # 前向在 GPU 上
                                fake_Bt = self.netG_A2B(real_A)
                                if fid_metric is not None or lpips_metric is not None:
                                    fake_for_metrics = self._prepare_images_for_metrics(fake_Bt)
                                    real_for_metrics = self._prepare_images_for_metrics(real_Bt)
                                    if fid_metric is not None:
                                        try:
                                            fid_metric.update(fake_for_metrics, real=False)
                                            fid_metric.update(real_for_metrics, real=True)
                                        except Exception:
                                            pass
                                    if lpips_metric is not None:
                                        try:
                                            lpips_batch = lpips_metric(fake_for_metrics, real_for_metrics)
                                            if lpips_batch is not None:
                                                LPIPS_sum += float(lpips_batch.mean().item())
                                                LPIPS_count += 1
                                        except Exception:
                                            pass

                                # 转 numpy 评估
                                real_B = real_Bt.detach().cpu().numpy()
                                fake_B = fake_Bt.detach().cpu().numpy()

                                # squeeze to 2D if (B,1,H,W) single-sample
                                real_B = np.squeeze(real_B)
                                fake_B = np.squeeze(fake_B)

                                # 逐 batch 聚合：计算当前批次的平均指标
                                try:
                                    # 支持批量 (B,H,W)
                                    if real_B.ndim == 3 and fake_B.ndim == 3:
                                        Bnow = real_B.shape[0]
                                        for b in range(Bnow):
                                            r = real_B[b]
                                            f = fake_B[b]
                                            # squeeze to (H,W)
                                            r = r[0] if (r.ndim == 3 and r.shape[0] == 1) else np.squeeze(r)
                                            f = f[0] if (f.ndim == 3 and f.shape[0] == 1) else np.squeeze(f)

                                            # === 构造“评分用 keep 权重”：keep = body * rd_weight_keep ===
                                            keep2d = None
                                            if self.val_metrics_use_rd:
                                                body = (r != -1).astype(np.float32)
                                                keep2d = body
                                                try:
                                                    w_batch = batch.get('rd_weight', None)
                                                    if isinstance(w_batch, torch.Tensor):
                                                        w_batch = w_batch.detach().cpu().numpy()
                                                    if w_batch is not None:
                                                        w2d = np.squeeze(w_batch[b])
                                                        if w2d.shape != r.shape:
                                                            try:
                                                                w2d = cv2.resize(w2d.astype(np.float32),
                                                                                 (r.shape[1], r.shape[0]),
                                                                                 interpolation=cv2.INTER_NEAREST)
                                                            except Exception:
                                                                w2d = None
                                                        if w2d is not None:
                                                            w2d = np.clip(w2d.astype(np.float32), 0.0, 1.0)
                                                            # rd_weight 已是不对齐区域的排除权重的补集（保留区）
                                                            keep2d = body * w2d
                                                except Exception:
                                                    pass
                                                if keep2d is not None and keep2d.sum() <= 0:
                                                    keep2d = body  # 回退

                                            if self.val_keep_masks_dir and keep2d is not None:
                                                _save_val_keep_mask(compose_slice_name(batch, i, b), keep2d)

                                            mask_for_metrics = keep2d if (self.val_metrics_use_rd and keep2d is not None) else None
                                            mae_sum += compute_mae(f, r, mask=mask_for_metrics)
                                            psnr_sum += compute_psnr(f, r, mask=mask_for_metrics)
                                            ssim_sum += compute_ssim(f, r, mask=mask_for_metrics)
                                        sample_count += float(Bnow)
                                    else:
                                        # 单图
                                        r = real_B
                                        f = fake_B

                                        keep2d = None
                                        if self.val_metrics_use_rd:
                                            body = (r != -1).astype(np.float32)
                                            keep2d = body
                                            try:
                                                w = batch.get('rd_weight', None)
                                                if isinstance(w, torch.Tensor):
                                                    w = w.detach().cpu().numpy()
                                                if w is not None:
                                                    w2d = np.squeeze(w)
                                                    if w2d.shape != r.shape:
                                                        try:
                                                            w2d = cv2.resize(w2d.astype(np.float32),
                                                                             (r.shape[1], r.shape[0]),
                                                                             interpolation=cv2.INTER_NEAREST)
                                                        except Exception:
                                                            w2d = None
                                                    if w2d is not None:
                                                        w2d = np.clip(w2d.astype(np.float32), 0.0, 1.0)
                                                        keep2d = body * w2d
                                            except Exception:
                                                pass
                                            if keep2d is not None and keep2d.sum() <= 0:
                                                keep2d = body

                                        if self.val_keep_masks_dir and keep2d is not None:
                                            _save_val_keep_mask(compose_slice_name(batch, i, 0), keep2d)

                                        mask_for_metrics = keep2d if (self.val_metrics_use_rd and keep2d is not None) else None
                                        mae_sum += compute_mae(f, r, mask=mask_for_metrics)
                                        psnr_sum += compute_psnr(f, r, mask=mask_for_metrics)
                                        ssim_sum += compute_ssim(f, r, mask=mask_for_metrics)
                                        sample_count += 1.0
                                except Exception:
                                    # 回退仅计算 MAE
                                    mae_sum += float(self.MAE(fake_B, real_B))
                                    sample_count += 1.0

                                # 前端进度
                                try:
                                    vprogress.advance(vtask, 1)
                                except Exception:
                                    pass

                        # 汇总（Sum-Reduce）
                        try:
                            device0 = self.device
                            t = torch.tensor([mae_sum, psnr_sum, ssim_sum, sample_count], dtype=torch.float64, device=device0)
                            t = _reduce_tensor_sum(t)
                            mae_sum, psnr_sum, ssim_sum, sample_count = [float(x) for x in t.tolist()]
                        except Exception:
                            pass

                        val_mae = mae_sum / max(1.0, sample_count)
                        val_psnr = psnr_sum / max(1.0, sample_count)
                        val_ssim = ssim_sum / max(1.0, sample_count)

                        if fid_metric is not None and _is_main_process():
                            try:
                                fid_value = float(fid_metric.compute())
                                fid_metric.reset()
                            except Exception:
                                fid_value = None
                        else:
                            fid_value = None
                        if lpips_metric is not None and LPIPS_count > 0:
                            lpips_value = float(LPIPS_sum / LPIPS_count)
                        else:
                            lpips_value = None

                        if _is_main_process():
                            msg = f"Val MAE: {val_mae:.6f} | PSNR: {val_psnr:.3f} dB | SSIM: {val_ssim:.4f}"
                            if fid_value is not None:
                                msg += f" | FID: {fid_value:.4f}"
                            if lpips_value is not None:
                                msg += f" | LPIPS: {lpips_value:.6f}"
                            print(msg)

                        # === 采样并上传验证图像到 wandb ===
                        try:
                            if self.val_sample_images and self.val_sample_images > 0:
                                # 重新跑一小批来采样（避免在上面循环中缓存大量图像）
                                imgs_input = []
                                imgs_real = []
                                imgs_pred = []
                                imgs_mask_keep = []
                                with torch.no_grad():
                                    for j, batch_val in enumerate(self.val_data):
                                        real_Av = batch_val['A'].to(self.device, non_blocking=True)
                                        real_Bv = batch_val['B'].to(self.device, non_blocking=True)
                                        pred_Bv = self.netG_A2B(real_Av)

                                        Bnow = real_Bv.shape[0]
                                        k = min(self.val_sample_images - len(imgs_real), Bnow)
                                        if k <= 0:
                                            break
                                        idxs = random.sample(range(Bnow), k)

                                        w_batch = batch_val.get('rd_weight', None)
                                        if isinstance(w_batch, torch.Tensor):
                                            w_batch = w_batch.detach().cpu().numpy()

                                        for idx in idxs:
                                            Ab = real_Av[idx].detach().cpu().numpy().squeeze()
                                            rb = real_Bv[idx].detach().cpu().numpy().squeeze()
                                            pb = pred_Bv[idx].detach().cpu().numpy().squeeze()

                                            # keep-mask（用于可视化） = body × rd_weight_keep
                                            try:
                                                body = (rb != -1).astype(np.uint8)
                                                if w_batch is not None:
                                                    w2d = np.squeeze(w_batch[idx]).astype(np.float32)
                                                    if w2d.shape != rb.shape:
                                                        w2d = cv2.resize(w2d, (rb.shape[1], rb.shape[0]), interpolation=cv2.INTER_NEAREST)
                                                    w2d = np.clip(w2d, 0.0, 1.0)
                                                    keep_vis = (w2d * body * 255.0).astype(np.uint8)
                                                else:
                                                    keep_vis = (body * 255).astype(np.uint8)
                                            except Exception:
                                                keep_vis = ((rb != -1).astype(np.uint8) * 255)

                                            def _to_uint8(a):
                                                a = np.squeeze(a)
                                                if a.ndim != 2:
                                                    a = a.astype(np.float32)
                                                    a = a[0] if a.ndim == 3 else a
                                                amin, amax = float(np.min(a)), float(np.max(a))
                                                if amin >= -1.001 and amax <= 1.001:
                                                    a = ((a + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                                                else:
                                                    a = ((a - amin) / (amax - amin + 1e-6) * 255.0).clip(0, 255).astype(
                                                        np.uint8)
                                                return a

                                            imgs_input.append(wandb.Image(_to_uint8(Ab)))
                                            imgs_real.append(wandb.Image(_to_uint8(rb)))
                                            imgs_pred.append(wandb.Image(_to_uint8(pb)))
                                            imgs_mask_keep.append(wandb.Image(keep_vis))

                                        if len(imgs_real) >= self.val_sample_images:
                                            break

                                if imgs_real and imgs_pred and _is_main_process():
                                    wandb.log({
                                        f'images/val_input_A_ep{epoch + 1:04d}': imgs_input,
                                        f'images/val_real_B_ep{epoch + 1:04d}': imgs_real,
                                        f'images/val_pred_B_ep{epoch + 1:04d}': imgs_pred,
                                        f'images/val_mask_keep_ep{epoch + 1:04d}': imgs_mask_keep,
                                        'epoch': epoch + 1,
                                    })
                        except Exception as e:
                            print(f"[wandb] val image upload failed: {e}")

                        # === 日志数值指标 ===
                        if _is_main_process():
                            try:
                                log_dict = {
                                    'val/MAE': val_mae,
                                    'val/PSNR': val_psnr,
                                    'val/SSIM': val_ssim,
                                    'epoch': epoch + 1,
                                }
                                if fid_value is not None:
                                    log_dict['val/FID'] = fid_value
                                if lpips_value is not None:
                                    log_dict['val/LPIPS'] = lpips_value
                                wandb.log(log_dict)
                                # 同步到 summary 方便排行榜
                                wandb.run.summary['best/val_MAE'] = min(
                                    float(wandb.run.summary.get('best/val_MAE', float('inf'))), float(val_mae))
                                wandb.run.summary['latest/val_MAE'] = float(val_mae)
                                if fid_value is not None:
                                    best_fid = float(wandb.run.summary.get('best/val_FID', float('inf')))
                                    wandb.run.summary['best/val_FID'] = float(min(best_fid, fid_value))
                                    wandb.run.summary['latest/val_FID'] = float(fid_value)
                                if lpips_value is not None:
                                    best_lpips = float(wandb.run.summary.get('best/val_LPIPS', float('inf')))
                                    wandb.run.summary['best/val_LPIPS'] = float(min(best_lpips, lpips_value))
                                    wandb.run.summary['latest/val_LPIPS'] = float(lpips_value)
                                try:
                                    wandb.run.summary['val/metrics_use_rd'] = bool(self.val_metrics_use_rd)
                                except Exception:
                                    pass
                            except Exception as e:
                                print(f"[wandb] val log failed: {e}")

                        # === 保存最佳模型（按 val/MAE 最小化） ===
                        try:
                            if val_mae < self.best_val_mae and _is_main_process():
                                self.best_val_mae = float(val_mae)
                                ckpt = {
                                    'epoch': epoch + 1,
                                    'val_mae': float(val_mae),
                                    'netG_A2B': self.netG_A2B.state_dict(),
                                }
                                if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
                                    ckpt['R_A'] = self.R_A.state_dict()
                                os.makedirs(self.config.get('save_root', './'), exist_ok=True)
                                torch.save(ckpt, self.best_ckpt_path)
                                print(f"[val] New best MAE {val_mae:.6f}. Saved checkpoint → {self.best_ckpt_path}")
                                try:
                                    wandb.run.summary['best/val_MAE'] = float(val_mae)
                                    wandb.run.summary['best/epoch'] = epoch + 1
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"[val] saving best checkpoint failed: {e}")

                        # === 固定样本跟踪：每次验证都记录同一批三元组（Input/Truth/Pred） ===
                        try:
                            if self.val_track_enable and isinstance(self.val_track_indices, list) and len(
                                    self.val_track_indices) > 0:
                                tbl = wandb.Table(columns=[
                                    "id",
                                    "input",
                                    "truth",
                                    "pred",
                                    "rd_weight",
                                    "keep_w2d",
                                    "body_mask",
                                    "body_fuzzy",
                                    "rd_source_exclude",
                                ])
                                # eval mode
                                self.netG_A2B.eval()
                                for idx in self.val_track_indices:
                                    try:
                                        sample = self.val_data.dataset[idx]
                                    except Exception:
                                        continue
                                    A = sample.get('A', None)
                                    B = sample.get('B', None)
                                    if A is None or B is None:
                                        continue
                                    # Resolve ID
                                    pid = sample.get('patient_id', None)
                                    sid = sample.get('slice_id', None)
                                    sid_str = f"{pid}_{sid}" if (
                                                pid is not None and sid is not None) else f"idx{idx:05d}"
                                    # To CUDA batch
                                    if isinstance(A, np.ndarray):
                                        A = torch.from_numpy(A)
                                    if isinstance(B, np.ndarray):
                                        B = torch.from_numpy(B)
                                    if A.ndim == 2:
                                        A = A.unsqueeze(0)
                                    if B.ndim == 2:
                                        B = B.unsqueeze(0)
                                    A = A.unsqueeze(0).to(self.device, non_blocking=True)  # (1,1,H,W)
                                    B = B.to(self.device, non_blocking=True)
                                    # Forward
                                    pred = self.netG_A2B(A)
                                    # To numpy for visualization
                                    A_np = A.detach().cpu().numpy().squeeze()
                                    B_np = B.detach().cpu().numpy().squeeze()
                                    P_np = pred.detach().cpu().numpy().squeeze()

                                    # Optional RD weight preview (grayscale 0..255)
                                    W_img = None
                                    keep_array = None
                                    raw_rd_mask = None
                                    try:
                                        # 仅当存在真实文件时才展示 rd 预览，避免把“前景回退权重”当作 mask 展示
                                        if bool(sample.get('rd_has_file', False)):
                                            w = sample.get('rd_weight', None)
                                            if w is not None:
                                                if isinstance(w, torch.Tensor):
                                                    w = w.detach().cpu().numpy()
                                                w = np.squeeze(w)
                                                if w.ndim == 3 and w.shape[0] == 1:
                                                    w = w[0]
                                                keep_array = np.clip(w.astype(np.float32), 0.0, 1.0)
                                                w_min, w_max = float(np.min(keep_array)), float(np.max(keep_array))
                                                if w_max > w_min:
                                                    w_vis = ((keep_array - w_min) / (w_max - w_min + 1e-6) * 255.0).clip(0,
                                                                                                                        255).astype(
                                                        np.uint8)
                                                else:
                                                    w_vis = np.zeros_like(keep_array, dtype=np.uint8)
                                                W_img = wandb.Image(w_vis,
                                                                    caption=f"rd_weight (file) [{w_min:.3f},{w_max:.3f}]")
                                                # 额外：加载原始排除 mask 便于检查
                                                base_mask = None
                                                if self.rd_mask_dir:
                                                    slice_name = sid_str
                                                    npy_path = os.path.join(self.rd_mask_dir, f"{slice_name}.npy")
                                                    png_path = os.path.join(self.rd_mask_dir, f"{slice_name}.png")
                                                    if os.path.exists(npy_path):
                                                        try:
                                                            base_mask = np.load(npy_path)
                                                        except Exception:
                                                            base_mask = None
                                                    elif os.path.exists(png_path):
                                                        try:
                                                            base_mask = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
                                                            if base_mask is not None and base_mask.ndim == 3:
                                                                base_mask = cv2.cvtColor(base_mask, cv2.COLOR_BGR2GRAY)
                                                        except Exception:
                                                            base_mask = None
                                                if base_mask is not None:
                                                    base_mask = np.squeeze(base_mask)
                                                    if base_mask.shape != body_mask.shape:
                                                        try:
                                                            base_mask = cv2.resize(
                                                                base_mask.astype(np.float32),
                                                                (body_mask.shape[1], body_mask.shape[0]),
                                                                interpolation=cv2.INTER_NEAREST,
                                                            )
                                                        except Exception:
                                                            base_mask = None
                                                    if base_mask is not None:
                                                        base_mask = base_mask.astype(np.float32)
                                                        if base_mask.max() > 1.0:
                                                            base_mask = base_mask / 255.0
                                                        raw_rd_mask = wandb.Image(
                                                            (np.clip(base_mask, 0.0, 1.0) * 255.0).astype(np.uint8),
                                                            caption="rd_source_exclude",
                                                        )
                                    except Exception:
                                        W_img = None

                                    # 记录 keep 权重与 body mask 的可视化
                                    keep_img = None
                                    body_img = None
                                    body_fuzzy_img = None
                                    try:
                                        body_arr = np.squeeze(B_np)
                                        if body_arr.ndim == 3 and body_arr.shape[0] == 1:
                                            body_arr = body_arr[0]
                                        body_mask = (body_arr != -1).astype(np.uint8)
                                        body_img = wandb.Image((body_mask * 255).astype(np.uint8), caption="body")
                                        body_fuzzy = (np.abs(body_arr + 1.0) < 5e-3).astype(np.uint8)
                                        body_fuzzy_img = wandb.Image((body_fuzzy * 255).astype(np.uint8), caption="body_fuzzy")
                                        if keep_array is not None:
                                            if keep_array.shape != body_mask.shape:
                                                try:
                                                    keep_array = cv2.resize(
                                                        keep_array.astype(np.float32),
                                                        (body_mask.shape[1], body_mask.shape[0]),
                                                        interpolation=cv2.INTER_NEAREST,
                                                    )
                                                except Exception:
                                                    pass
                                            keep_img = wandb.Image(
                                                (np.clip(keep_array, 0.0, 1.0) * 255.0).astype(np.uint8),
                                                caption="keep_w2d",
                                            )
                                    except Exception:
                                        body_img = None
                                        body_fuzzy_img = None
                                        keep_img = None

                                    # Convert to uint8
                                    def _to_uint8(a):
                                        a = np.squeeze(a)
                                        if a.ndim != 2:
                                            a = a.astype(np.float32)
                                            a = a[0] if a.ndim == 3 else a
                                        amin, amax = float(np.min(a)), float(np.max(a))
                                        if amin >= -1.001 and amax <= 1.001:
                                            a = ((a + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                                        else:
                                            a = ((a - amin) / (amax - amin + 1e-6) * 255.0).clip(0, 255).astype(
                                                np.uint8)
                                        return a

                                    tbl.add_data(
                                        sid_str,
                                        wandb.Image(_to_uint8(A_np), caption=f"{sid_str}: input"),
                                        wandb.Image(_to_uint8(B_np), caption=f"{sid_str}: truth"),
                                        wandb.Image(_to_uint8(P_np), caption=f"{sid_str}: pred"),
                                        W_img,
                                        keep_img,
                                        body_img,
                                        body_fuzzy_img,
                                        raw_rd_mask,
                                    )
                                if _is_main_process():
                                    wandb.log({
                                        f'tables/track_triplets_ep{epoch + 1:04d}': tbl,
                                        'epoch': epoch + 1,
                                    })
                        except Exception as e:
                            print(f"[wandb] fixed val-track table failed: {e}")

                    # Restore models to previous training/eval state
                    if was_train_G:
                        self.netG_A2B.train()
                    if was_train_R:
                        self.R_A.train()

        self.config['epoch'] = target_epoch

    def test(self, ):
        # Ensure device
        if not hasattr(self, 'device'):
            ddp_enabled, local_rank = _setup_ddp_if_needed(self.config)
            self.device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

        # 在分布式环境下只让主进程执行 test，避免重复保存图片/CSV
        if _dist_is_available_and_initialized() and not _is_main_process():
            return

        test_cfg_snapshot = self._log_stage_config('test')
        try:
            if wandb.run is not None:
                wandb.config.update(copy.deepcopy(test_cfg_snapshot), allow_val_change=True)
        except Exception as _e:
            print(f"[wandb] config update failed during test: {_e}")

        # --- Read injection config and quality thresholds ---
        synth_cfg = dict(self.config.get('inject_nc', {})) if isinstance(self.config.get('inject_nc', {}), dict) else {}
        synth_enable_global = bool(synth_cfg.get('enable', False))
        quality_ssim_min = float(synth_cfg.get('quality_ssim_min', 0.85))
        quality_mae_max = float(synth_cfg.get('quality_mae_max', 0.12))

        g_ckpt_path = resolve_model_path(self.config, 'g_model_name', 'netG_A2B.pth')
        self.netG_A2B.load_state_dict(torch.load(g_ckpt_path))
        os.makedirs(self.config['image_save'], exist_ok=True)
        pred_save_root = self.config.get('pred_save', os.path.join(self.config['image_save'], 'pred'))
        os.makedirs(pred_save_root, exist_ok=True)
        # === Create output dirs for MC mean and uncertainty maps ===
        mean_save_root = self.config.get('pred_mean_save', os.path.join(self.config['image_save'], 'pred_mean'))
        os.makedirs(mean_save_root, exist_ok=True)
        uncert_save_root = self.config.get('uncert_save', os.path.join(self.config['image_save'], 'uncert'))
        os.makedirs(uncert_save_root, exist_ok=True)
        residual_save_root = self.config.get('residual_save', os.path.join(self.config['image_save'], 'residual'))
        save_residuals = bool(self.config.get('save_residuals', True))
        if save_residuals:
            os.makedirs(residual_save_root, exist_ok=True)
        # --- Save directories for RD artifacts (for training Scheme B) ---
        rd_seeds_dir = self.config.get('rd_seeds_dir', os.path.join(self.config['image_save'], 'rd_seeds'))
        rd_masks_dir = self.config.get('rd_masks_dir', os.path.join(self.config['image_save'], 'rd_masks'))
        rd_weights_dir = self.config.get('rd_weights_dir', os.path.join(self.config['image_save'], 'rd_weights'))
        os.makedirs(rd_seeds_dir, exist_ok=True)
        os.makedirs(rd_masks_dir, exist_ok=True)
        os.makedirs(rd_weights_dir, exist_ok=True)
        import csv
        # Enable evaluation-after-registration if configured and registration net exists
        use_reg = bool(self.config.get('eval_with_registration', False)) and self.config.get('regist',
                                                                                             False) and hasattr(self,
                                                                                                                'R_A') and hasattr(
            self, 'spatial_transform')
        if use_reg:
            ra_path = resolve_model_path(self.config, 'r_model_name', 'R_A.pth')
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
        plot_every_n = int(self.config.get('plot_every_n', 1))  # save 1 of every N slices (1 = all)
        plot_workers = int(self.config.get('test_plot_workers', 0))

        # Whether to use RD masks/weights for metrics during test (foreground minus misalignment area)
        test_metrics_use_rd = bool(self.config.get('test_metrics_use_rd', self.config.get('val_metrics_use_rd', self.config.get('metrics_use_rd', False))))
        if _is_main_process():
            print(f"[test] metrics_use_rd={bool(test_metrics_use_rd)} (mask = foreground − misalignment)")

        # Where to persist the exact keep mask used for metrics
        metrics_keep_dir = os.path.join(self.config['image_save'], 'metrics_keep')
        os.makedirs(metrics_keep_dir, exist_ok=True)

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

            # Run composites out-of-process to avoid Matplotlib GIL contention
            plot_executor = ProcessPoolExecutor(max_workers=plot_workers) if (plot_workers > 0 and save_composite) else None
            plot_futures = []
            plot_progress = None
            plot_task_id = None
            plot_task_total = 0
            if plot_executor is not None and _is_main_process():
                plot_progress = Progress(
                    TextColumn("[magenta]Plots[/]"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                )

            def _schedule_plot(timer_key: str, **call_kwargs):
                if not save_composite:
                    return
                if plot_executor is None:
                    with _Timer(timer_key, prof):
                        plot_composite(**call_kwargs)
                else:
                    nonlocal plot_task_id, plot_task_total
                    fut = plot_executor.submit(plot_composite, **call_kwargs)
                    plot_futures.append(fut)
                    if plot_progress is not None:
                        if plot_task_id is None:
                            plot_task_id = plot_progress.add_task("plots", total=0)
                            plot_task_total = 0
                        plot_task_total += 1
                        plot_progress.update(plot_task_id, total=plot_task_total)

            def _drain_plot_futures():
                if plot_executor is None:
                    return
                pending = []
                for fut in plot_futures:
                    if fut.done():
                        _await_plot_future(fut)
                    else:
                        pending.append(fut)
                plot_futures[:] = pending

            def _await_plot_future(fut):
                nonlocal plot_task_id
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[test][warn] plot task failed: {exc}")
                if plot_progress is not None and plot_task_id is not None:
                    plot_progress.advance(plot_task_id)

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
            ) if _is_main_process() else None

            total_processed_slices = 0

            context_managers = []
            if progress is not None:
                context_managers.append(progress)
            if plot_progress is not None:
                context_managers.append(plot_progress)

            from contextlib import ExitStack

            with ExitStack() as stack:
                for cm in context_managers:
                    stack.enter_context(cm)

                if progress is not None:
                    task_id = progress.add_task("eval", total=total_slices if (
                            isinstance(total_slices, int) and total_slices > 0) else None)
                else:
                    task_id = None

                for i, batch in enumerate(self.val_data):
                    with _Timer('batch_total', prof):
                        _drain_plot_futures()
                        # Move to GPU
                        real_A = batch['A'].to(self.device, non_blocking=True)
                        real_Bt = batch['B'].to(self.device, non_blocking=True)
                        real_Bnp = real_Bt.detach().cpu().numpy()

                        B = real_Bnp.shape[0]
                        # Try to get patient_id and slice_id from batch, fallback to None
                        patient_ids = batch.get('patient_id', [None] * B)
                        slice_ids = batch.get('slice_id', [None] * B)

                        # Try to fetch batch-level RD weights (B,1,H,W) for masking metrics
                        w_batch = None
                        if test_metrics_use_rd and ('rd_weight' in batch) and (batch['rd_weight'] is not None):
                            w_batch = batch['rd_weight']
                            if isinstance(w_batch, torch.Tensor):
                                w_batch = w_batch.detach().cpu().numpy()

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

                                # —— 新增：保存用于绘图的指标 mask 可视化（3通道 uint8）
                                metrics_keep_vis = None

                                # Metrics per-slice
                                with _Timer('metrics', prof):
                                    if test_metrics_use_rd:
                                        # Build final keep weight directly from rd_weight (already keep mask)
                                        keep = None
                                        try:
                                            body = (r != -1).astype(np.float32)
                                            keep = body
                                            if w_batch is not None:
                                                w2d = np.squeeze(w_batch[b]).astype(np.float32)
                                                if w2d.shape != r.shape:
                                                    try:
                                                        w2d = cv2.resize(w2d, (r.shape[1], r.shape[0]),
                                                                         interpolation=cv2.INTER_NEAREST)
                                                    except Exception:
                                                        w2d = None
                                                if w2d is not None:
                                                    w2d = np.clip(w2d, 0.0, 1.0)
                                                    keep = body * w2d
                                            if keep is not None and np.sum(keep) <= 0:
                                                keep = body
                                        except Exception:
                                            keep = None

                                        if keep is not None:
                                            try:
                                                km = (keep > 0).astype(np.uint8) * 255
                                                # Turn binary keep mask into yellow overlay (R=G=keep, B=0)
                                                metrics_keep_vis = np.stack([km, km, np.zeros_like(km)], axis=-1)
                                            except Exception:
                                                metrics_keep_vis = None
                                        mask_for_metrics = keep if test_metrics_use_rd else None
                                        mae_i = compute_mae(fe, r, mask=mask_for_metrics)
                                        psnr_i = compute_psnr(fe, r, mask=mask_for_metrics)
                                        ssim_i = compute_ssim(fe, r, mask=mask_for_metrics)
                                    else:
                                        mae_i = compute_mae(fe, r)
                                        psnr_i = compute_psnr(fe, r)
                                        ssim_i = compute_ssim(fe, r)
                                counters['metric_points'] += 1
                                # 独立导出“评分用 keep mask”
                                try:
                                    if test_metrics_use_rd and (keep is not None):
                                        keep_png = ((keep > 0).astype(np.uint8) * 255)
                                        cv2.imwrite(os.path.join(metrics_keep_dir, f"{slice_name}.png"), keep_png)
                                except Exception:
                                    pass
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
                                MAE += mae_i
                                PSNR += psnr_i
                                SSIM += ssim_i
                                N += 1

                                # Accumulate per-slice list for mean/std
                                d = per_slice_values.setdefault(slice_name, {'mae': [], 'psnr': [], 'ssim': []})
                                d['mae'].append(mae_i)
                                d['psnr'].append(psnr_i)
                                d['ssim'].append(ssim_i)

                                # Only on first MC run, produce visualization and save pred image
                                if t == 0:
                                    inp = real_A[b].detach().cpu().numpy().squeeze()
                                    gt = r
                                    pred = fe
                                    residual = pred - gt
                                    if save_residuals:
                                        try:
                                            np.save(os.path.join(residual_save_root, f"{slice_name}.npy"),
                                                    residual.astype(np.float32))
                                        except Exception as err:
                                            print(f"[warn] Failed to save residual for {slice_name}: {err}")
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
                                            zmap = robust_zscore(residual, mask=body)
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
                                                pvals = 2.0 * (1.0 - normal_cdf(absz_s))
                                                seeds = bh_fdr_mask(pvals, q=q_fdr, mask=body)

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

                                            nonempty = bool(np.any(seeds))

                                            # 4) Optional fallback: allow empty (preferred) or percentile fallback (legacy)
                                            if not nonempty:
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

                                            nonempty = bool(np.any(seeds))
                                            min_area = max(1, int(area_fr * max(1, body_area)))
                                            # Seed dilation as before
                                            if nonempty and seed_dilate > 0:
                                                from skimage.morphology import binary_dilation, disk
                                                with _Timer('rd_seed_post', prof):
                                                    seeds = binary_dilation(seeds, disk(int(seed_dilate))) & body

                                            final_mask = np.zeros_like(seeds, dtype=bool)
                                            if nonempty:
                                                with _Timer('rd_grow_cc', prof):
                                                    grown = hysteresis_from_seeds(zmap, seeds, Tl=Tl_fix)
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
                                                                final_mask = morph_clean(
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

                                    # === Persist RD artifacts for later training ===
                                    try:
                                        # Seeds (boolean)
                                        if seeds is not None:
                                            seeds_bool = seeds.astype(bool)
                                            np.save(os.path.join(rd_seeds_dir, f"{slice_name}.npy"), seeds_bool)
                                            seeds_png = (seeds_bool.astype(np.uint8) * 255)
                                            cv2.imwrite(os.path.join(rd_seeds_dir, f"{slice_name}.png"), seeds_png)
                                        # Final mask (boolean)
                                        if final_mask is not None:
                                            mask_bool = final_mask.astype(bool)
                                            np.save(os.path.join(rd_masks_dir, f"{slice_name}.npy"), mask_bool)
                                            mask_png = (mask_bool.astype(np.uint8) * 255)
                                            cv2.imwrite(os.path.join(rd_masks_dir, f"{slice_name}.png"), mask_png)
                                        # Weight map (float32 in [0,1], may be zero when empty)
                                        if weight_map is not None:
                                            w = weight_map.astype(np.float32)
                                            # Normalize to [0,1] for visualization (if max>0)
                                            w_vis = w
                                            w_max = float(w_vis.max()) if w_vis.size > 0 else 0.0
                                            if w_max > 0:
                                                w_vis = (w_vis / (w_max + 1e-6))
                                            np.save(os.path.join(rd_weights_dir, f"{slice_name}.npy"), w)
                                            w_png = (w_vis * 255.0).astype(np.uint8)
                                            cv2.imwrite(os.path.join(rd_weights_dir, f"{slice_name}.png"), w_png)
                                    except Exception as _e:
                                        print(f"[test][warn] Failed to save RD artifacts for {slice_name}: {_e}")
                                    # 缓存供“MC 结束后 + 有了std”再统一绘图
                                    composite_cache[slice_name] = {
                                        'inp': inp, 'gt': gt, 'pred': pred, 'residual': residual,
                                        'rd_enable': rd_enable,
                                        'zmap': zmap, 'seeds': seeds, 'final_mask': final_mask,
                                        'weight_map': weight_map,
                                        'metrics_keep_vis': metrics_keep_vis,
                                        'mae_i': mae_i, 'psnr_i': psnr_i, 'ssim_i': ssim_i
                                    }

                                    # Defer plotting to after-MC when mc_runs>1 and enabled; otherwise plot now
                                    # When no MC accumulation, optionally plot immediately
                                    if mc_runs == 1 and save_composite:
                                        # sampling: only plot 1 of every N slices to speed up
                                        plot_ok = (plot_every_n <= 1) or (
                                                (counters.get('plot_enqueued', 0) % max(1, plot_every_n)) == 0)
                                        if plot_ok:
                                            metrics = {'mae': mae_i, 'psnr': psnr_i, 'ssim': ssim_i}
                                            rd_pack = None
                                            if rd_enable:
                                                rd_pack = {'zmap': zmap, 'seeds': seeds, 'final_mask': final_mask,
                                                           'weight_map': weight_map,
                                                           'q_fdr': self.config.get('rd_fdr_q', 0.10)}
                                            save_path = os.path.join(self.config['image_save'], f"{slice_name}.png")
                                            _schedule_plot(
                                                'plot_immediate',
                                                inp=inp,
                                                gt=gt,
                                                pred=pred,
                                                residual=residual,
                                                metrics=metrics,
                                                rd_data=rd_pack,
                                                uncertainty=None,
                                                use_reg=use_reg,
                                                save_path=save_path,
                                                overlay_rgb=metrics_keep_vis,
                                            )
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
                                metrics_keep_vis_cache = c.get('metrics_keep_vis', None)
                                rd_enable = c['rd_enable']
                                rd_pack = None
                                if rd_enable:
                                    rd_pack = {'zmap': c['zmap'], 'seeds': c['seeds'], 'final_mask': c['final_mask'],
                                               'weight_map': c['weight_map'],
                                               'q_fdr': self.config.get('rd_fdr_q', 0.10)}
                                metrics = {'mae': c['mae_i'], 'psnr': c['psnr_i'], 'ssim': c['ssim_i']}
                                save_path = os.path.join(self.config['image_save'], f"{slice_name}.png")
                                plot_ok = (plot_every_n <= 1) or (
                                        (counters.get('plot_enqueued_after', 0) % max(1, plot_every_n)) == 0)
                                if plot_ok:
                                    _schedule_plot(
                                        'plot_after_mc',
                                        inp=inp,
                                        gt=gt,
                                        pred=pred,
                                        residual=residual,
                                        metrics=metrics,
                                        rd_data=rd_pack,
                                        uncertainty=None,
                                        use_reg=use_reg,
                                        save_path=save_path,
                                        overlay_rgb=metrics_keep_vis_cache,
                                    )
                                    counters['plots'] += 1
                                counters['plot_enqueued_after'] = counters.get('plot_enqueued_after', 0) + 1

            if plot_executor is not None:
                for fut in plot_futures:
                    _await_plot_future(fut)
                plot_executor.shutdown(wait=True)

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
            psnr = compute_psnr(fake, real)
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
        return compute_mae(fake, real)

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

    def _prepare_val_metrics_modules(self):
        if not _is_main_process():
            return None
        if self._val_metric_support is None:
            support = {}
            device = getattr(self, 'device', torch.device('cpu'))
            try:
                if FrechetInceptionDistance is not None:
                    support['fid'] = FrechetInceptionDistance(feature=64, reset_real_features=True).to(device)
            except Exception as exc:
                print(f"[val] FID metric init failed: {exc}")
            try:
                if LearnedPerceptualImagePatchSimilarity is not None:
                    support['lpips'] = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
            except Exception as exc:
                print(f"[val] LPIPS metric init failed: {exc}")
            self._val_metric_support = support
        return self._val_metric_support

    @staticmethod
    def _prepare_images_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
        img = tensor.detach()
        if img.ndim != 4:
            raise ValueError(f"Expected BCHW tensor for metrics, got {img.shape}")
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        img = torch.clamp((img + 1.0) * 0.5, 0.0, 1.0)
        return img

    @staticmethod
    def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
        return module.module if isinstance(module, DDP) else module

    def get_epoch(self) -> int:
        return int(self.config.get('epoch', 0))

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'epoch': int(self.config.get('epoch', 0)),
            'best_val_mae': float(getattr(self, 'best_val_mae', float('inf'))),
            'config_snapshot': copy.deepcopy(self.config),
            'netG_A2B': self._unwrap(self.netG_A2B).state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'netD_B': self._unwrap(self.netD_B).state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
        }
        if hasattr(self, 'netG_B2A'):
            state['netG_B2A'] = self._unwrap(self.netG_B2A).state_dict()
        if hasattr(self, 'optimizer_D_A'):
            state['optimizer_D_A'] = self.optimizer_D_A.state_dict()
        if hasattr(self, 'netD_A'):
            state['netD_A'] = self._unwrap(self.netD_A).state_dict()
        if hasattr(self, 'R_A'):
            state['R_A'] = self._unwrap(self.R_A).state_dict()
        if hasattr(self, 'optimizer_R_A'):
            state['optimizer_R_A'] = self.optimizer_R_A.state_dict()
        if extra:
            state['extra'] = extra
        torch.save(state, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        state = torch.load(path, map_location=self.device)
        self._unwrap(self.netG_A2B).load_state_dict(state['netG_A2B'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self._unwrap(self.netD_B).load_state_dict(state['netD_B'])
        self.optimizer_D_B.load_state_dict(state['optimizer_D_B'])
        if hasattr(self, 'netG_B2A') and 'netG_B2A' in state:
            self._unwrap(self.netG_B2A).load_state_dict(state['netG_B2A'])
        if hasattr(self, 'netD_A') and 'netD_A' in state:
            self._unwrap(self.netD_A).load_state_dict(state['netD_A'])
        if hasattr(self, 'optimizer_D_A') and 'optimizer_D_A' in state:
            self.optimizer_D_A.load_state_dict(state['optimizer_D_A'])
        if hasattr(self, 'R_A') and 'R_A' in state:
            self._unwrap(self.R_A).load_state_dict(state['R_A'])
        if hasattr(self, 'optimizer_R_A') and 'optimizer_R_A' in state:
            self.optimizer_R_A.load_state_dict(state['optimizer_R_A'])
        self.config['epoch'] = int(state.get('epoch', self.config.get('epoch', 0)))
        self.best_val_mae = float(state.get('best_val_mae', getattr(self, 'best_val_mae', float('inf'))))
        return state.get('extra')

    def set_rd_weights(self, weights_dir: Optional[str], rd_w_min: Optional[float] = None):
        if rd_w_min is not None:
            self.rd_w_min = float(rd_w_min)
        if weights_dir:
            self.rd_mode = 'weights'
            self.rd_input_type = 'weights'
            self.rd_weights_dir = os.path.abspath(weights_dir)
        else:
            self.rd_mode = None
            self.rd_input_type = None
            self.rd_weights_dir = ''
        if hasattr(self, 'train_data') and hasattr(self.train_data, 'dataset') and hasattr(self.train_data.dataset, 'set_rd_config'):
            self.train_data.dataset.set_rd_config(self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min)
        if hasattr(self, 'val_data') and hasattr(self.val_data, 'dataset') and hasattr(self.val_data.dataset, 'set_rd_config'):
            self.val_data.dataset.set_rd_config(self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min)
