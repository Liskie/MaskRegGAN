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
from pathlib import Path
from typing import Optional, Dict, List, Set
import yaml

from torchvision.transforms import RandomAffine, ToPILImage
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import cv2
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.distributed as dist
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
    compute_mae, compute_psnr, compute_ssim, resolve_model_path, to_uint8_image, smooth_weight_map_tv
)
from .datasets import ImageDataset, ValDataset
from .reg import Reg
from .transformer import Transformer_2D
from models.CycleGan import Generator as BaseGenerator, Discriminator
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
from .confidence_utils import compute_weight_map


class SharedBackboneGenerator(nn.Module):
    """Shared trunk + per-fold head CycleGAN generator."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        n_residual_blocks: int = 9,
        n_heads: int = 3,
        upsample_mode: str = "resize",
        share_body: bool = True,
    ):
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_heads = n_heads
        self.share_body = bool(share_body)

        # Instantiate baseline generator to reuse architecture layout
        base = BaseGenerator(input_nc, output_nc, n_residual_blocks=n_residual_blocks, upsample_mode=upsample_mode)
        self.shared_head = base.model_head
        self.shared_body = base.model_body if self.share_body else None

        # Build per-head branches (deep copies of required components)
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            if self.share_body:
                branch = copy.deepcopy(base.model_tail)
            else:
                branch = nn.Sequential(
                    copy.deepcopy(base.model_body),
                    copy.deepcopy(base.model_tail),
                )
            self.heads.append(branch)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.shared_head(x)
        if self.share_body and self.shared_body is not None:
            feats = self.shared_body(feats)
        return feats

    def decode(self, feats: torch.Tensor, head_idx: int) -> torch.Tensor:
        if head_idx < 0 or head_idx >= self.n_heads:
            raise ValueError(f"Invalid head index {head_idx}")
        return self.heads[head_idx](feats)

    def forward(self, x: torch.Tensor, head_indices: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        if head_indices is None:
            head_indices = list(range(self.n_heads))
        feats = self.encode(x)
        outputs = {}
        for h in head_indices:
            if h < 0 or h >= self.n_heads:
                raise ValueError(f"Invalid head index {h}")
            outputs[h] = self.heads[h](feats)
        return outputs


class CycTrainerFusion:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._config_snapshots = {}
        self._config_logged_stages = set()
        self._wandb_settings = self._build_wandb_settings(config)
        self._progress_disable = bool(config.get('disable_progress', False))
        # --- Fusion-specific configuration ---
        self.n_heads = int(config.get('fusion_heads', 3))
        if self.n_heads <= 0:
            raise ValueError("fusion_heads must be positive")
        if config.get('bidirect', False):
            raise ValueError("CycTrainerFusion does not support bidirectional setup (bidirect must be False)")
        # default head is no longer used; every patient must be mapped explicitly
        self._fusion_default_head = None
        self._cv_root = config.get('cv_root')
        self._fold_assignments: Dict[str, int] = {}
        self._fold_id_lookup: Dict[int, int] = {}
        self._head_to_fold_raw: Dict[int, int] = {}
        self._head_to_fold_dir: Dict[int, Path] = {}
        self._fold_val_patients: Dict[int, Set[str]] = {}

        # Normalize LR to float (config may contain string like "5e-5")
        try:
            self.lr = float(config.get('lr', 0.0001))
        except Exception:
            self.lr = 0.0001

        ## def networks
        self.netG_A2B = SharedBackboneGenerator(
            config['input_nc'],
            config['output_nc'],
            n_residual_blocks=config.get('fusion_residual_blocks', 9),
            n_heads=self.n_heads,
            upsample_mode=str(config.get('generator_upsample_mode', 'resize')).lower(),
            share_body=bool(config.get('fusion_share_body', True)),
        ).cuda()
        self.netD_B = Discriminator(config['output_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # --- Optional registration network ---
        self.use_registration = bool(config.get('regist', False))
        self.R_A = None
        self.optimizer_R_A = None
        self.spatial_transform = Transformer_2D().cuda()
        if self.use_registration:
            reg_fake_nc = int(config.get('reg_fake_nc', config.get('output_nc', config['input_nc'])))
            reg_real_nc = int(config.get('reg_real_nc', config.get('output_nc', config['input_nc'])))
            self.R_A = Reg(config['size'], config['size'], reg_fake_nc, reg_real_nc).cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Optional pretrained weights
        g_pre = config.get('pretrained_g')
        if g_pre:
            try:
                state = torch.load(os.path.expanduser(g_pre), map_location='cpu')
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                if isinstance(state, dict) and 'netG_A2B' in state:
                    state = state['netG_A2B']
                self._unwrap(self.netG_A2B).load_state_dict(state, strict=False)
                print(f"[init] Loaded pretrained generator from {g_pre}")
            except Exception as _e:
                print(f"[init][warn] Failed to load pretrained generator '{g_pre}': {_e}")

        r_pre = config.get('pretrained_r')
        if r_pre and hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
            try:
                r_state = torch.load(os.path.expanduser(r_pre), map_location='cpu')
                if isinstance(r_state, dict) and 'state_dict' in r_state:
                    r_state = r_state['state_dict']
                if isinstance(r_state, dict) and 'R_A' in r_state:
                    r_state = r_state['R_A']
                self._unwrap(self.R_A).load_state_dict(r_state, strict=False)
                print(f"[init] Loaded pretrained registration from {r_pre}")
            except Exception as _e:
                print(f"[init][warn] Failed to load pretrained registration '{r_pre}': {_e}")

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
        # Option to ignore background pixels marked as -1 (useful when inputs are full images)
        self.ignore_neg1_background = bool(config.get('ignore_neg1_background', False))

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
        self.metrics_use_rd = bool(config.get('metrics_use_rd', False)) if not hasattr(self,
                                                                                       'metrics_use_rd') else self.metrics_use_rd
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

        # --- DataLoader runtime controls ---
        level = config['noise_level']
        self._n_workers = int(config.get('n_cpu', 0))
        self._dl_timeout = int(config.get('dataloader_timeout', 0))
        self._dl_persistent = bool(config.get('persistent_workers', True))
        self._dl_prefetch = int(config.get('prefetch_factor', 2))

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
        domain_a_dir = config.get('domain_a_dir')
        domain_b_dir = config.get('domain_b_dir')
        val_domain_a_dir = config.get('val_domain_a_dir', domain_a_dir)
        val_domain_b_dir = config.get('val_domain_b_dir', domain_b_dir)
        domain_a_channels = config.get('domain_a_channels', config.get('input_nc'))
        domain_b_channels = config.get('domain_b_channels', config.get('output_nc'))
        self._cv_train_subdir = config.get('cv_train_subdir', 'train')
        self._cv_val_subdir = config.get('cv_val_subdir', 'val')
        self._cv_domain_a_subdir = config.get('cv_domain_a_subdir', 'A')
        self._cv_domain_b_subdir = config.get('cv_domain_b_subdir', 'B')

        self.rd_fallback_mode = str(config.get('rd_fallback_mode', 'body')).lower()

        train_dataset = ImageDataset(
            config['dataroot'], level,
            transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False,
            rd_input_type=self.rd_input_type,
            rd_mask_dir=self.rd_mask_dir,
            rd_weights_dir=self.rd_weights_dir,
            rd_w_min=self.rd_w_min,
            cache_mode=config.get('cache_mode', 'mmap'),
            rd_cache_weights=config.get('rd_cache_weights', False),
            domain_a_dir=domain_a_dir,
            domain_b_dir=domain_b_dir,
            domain_a_channels=domain_a_channels,
            domain_b_channels=domain_b_channels,
            rd_fallback_mode=self.rd_fallback_mode,
            ignore_neg1_background=self.ignore_neg1_background,
        )
        self.train_data = DataLoader(train_dataset, **dl_kwargs)

        # Build per-fold validation datasets from cv_root
        self.fold_val_loaders: Dict[int, DataLoader] = {}
        self.fold_val_datasets: Dict[int, Dataset] = {}
        cv_root = Path(self.config.get('cv_root', ''))
        if not cv_root.is_dir():
            raise RuntimeError("cv_root must point to the cross-validation root with fold directories")
        val_kwargs = dict(
            batch_size=int(self.config.get('val_batchSize', self.config['batchSize'])),
            shuffle=False,
            num_workers=self._n_workers,
            pin_memory=self.config.get('cuda', False),
            timeout=self._dl_timeout,
        )
        if self._n_workers > 0:
            val_kwargs.update(
                persistent_workers=self._dl_persistent,
                prefetch_factor=max(1, self._dl_prefetch),
            )
        fold_dirs = sorted(p for p in cv_root.glob('fold_*') if p.is_dir())
        for fold_dir in fold_dirs:
            try:
                fold_raw_id = int(fold_dir.name.split('_')[-1])
            except ValueError:
                continue
            if len(self.fold_val_loaders) >= self.n_heads:
                break
            train_dir = fold_dir / self._cv_train_subdir
            val_dir = fold_dir / self._cv_val_subdir
            if not train_dir.is_dir() or not val_dir.is_dir():
                continue
            fold_val_a_dir = str((val_dir / self._cv_domain_a_subdir).resolve())
            fold_val_b_dir = str((val_dir / self._cv_domain_b_subdir).resolve())
            val_dataset = ValDataset(
                str(val_dir),
                transforms_=[ToTensor(), last_tf],
                unaligned=False,
                rd_input_type=self.rd_input_type,
                rd_mask_dir=self.rd_mask_dir,
                rd_weights_dir=self.rd_weights_dir,
                rd_w_min=self.rd_w_min,
                cache_mode=config.get('cache_mode', 'mmap'),
                rd_cache_weights=config.get('rd_cache_weights', False),
                domain_a_dir=fold_val_a_dir,
                domain_b_dir=fold_val_b_dir,
                domain_a_channels=domain_a_channels,
                domain_b_channels=domain_b_channels,
                rd_fallback_mode=self.rd_fallback_mode,
                ignore_neg1_background=self.ignore_neg1_background,
            )
            head_idx = len(self.fold_val_loaders)
            self.fold_val_datasets[head_idx] = val_dataset
            self.fold_val_loaders[head_idx] = DataLoader(val_dataset, **val_kwargs)
            self._head_to_fold_raw[head_idx] = fold_raw_id
            self._head_to_fold_dir[head_idx] = val_dir
            self._fold_id_lookup[fold_raw_id] = head_idx
            val_patients: Set[str] = set()
            val_b_dir = val_dir / self._cv_domain_b_subdir
            if val_b_dir.is_dir():
                for file_path in val_b_dir.iterdir():
                    pid = self._extract_patient_id(file_path.name)
                    if pid:
                        val_patients.add(str(pid))
            if val_patients:
                self._fold_val_patients[head_idx] = val_patients
        if not self.fold_val_loaders:
            raise RuntimeError(f"No fold validation loaders could be built from cv_root='{cv_root}'")
        if len(self.fold_val_loaders) != self.n_heads:
            raise RuntimeError(
                f"fusion_heads={self.n_heads} but found {len(self.fold_val_loaders)} fold validation splits under '{cv_root}'"
            )

        try:
            ordered_datasets = [self.fold_val_datasets[idx] for idx in sorted(self.fold_val_datasets.keys())]
            self.val_data = DataLoader(ConcatDataset(ordered_datasets), **val_kwargs)
        except Exception:
            self.val_data = None

        self._finalize_fold_assignments(cv_root)

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

                tr_cnt = _count_with_progress(self.train_data.dataset, 'train') if hasattr(self,
                                                                                           'train_data') and hasattr(
                    self.train_data, 'dataset') else 0
                print(
                    f"[init] mask/weight presence → train: {tr_cnt} / {len(self.train_data.dataset) if hasattr(self.train_data, 'dataset') else '?'}")
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
        self.val_sample_images = max(0, int(self.config.get('val_sample_images', 50)))  # legacy total-cap semantics
        # Optional explicit per-head override; when >0 we sample this many slices per head
        self.val_sample_images_per_head = max(0, int(self.config.get('val_sample_images_per_head', 0)))

        # Fixed validation tracking (log the same K triplets every validation)
        self.val_track_enable = bool(self.config.get('val_track_enable', True))
        self.val_track_fixed_n = max(0, int(self.config.get('val_track_fixed_n', 50)))
        self.val_track_per_head = max(0, int(self.config.get('val_track_per_head', 0)))
        self.val_track_strategy = str(self.config.get('val_track_strategy', 'first')).strip().lower()
        self.val_track_seed = int(self.config.get('val_track_seed', 0))
        try:
            ds_len = sum(len(ds) for ds in self.fold_val_datasets.values())
        except Exception:
            ds_len = 0
        self.val_track_indices: List = []
        total_tracked = 0
        per_head_limit = 0
        if self.val_track_per_head > 0:
            per_head_limit = self.val_track_per_head
        elif self.val_track_fixed_n > 0:
            per_head_limit = max(1, int(math.ceil(self.val_track_fixed_n / max(1, self.n_heads))))
        self._val_track_per_head_limit = per_head_limit

        if per_head_limit > 0 and self.val_track_enable:
            if self.val_track_strategy in ('random_each', 'random-epoch', 'random_epoch'):
                # Resample indices every validation; we only precompute the per-head limit here.
                total_tracked = per_head_limit * max(1, len(self.fold_val_datasets))
                print(f"[val-track] Tracking random samples per head each eval (up to {per_head_limit} per head).")
            else:
                rng = None
                if self.val_track_strategy in ('random', 'random_fixed', 'rand', 'shuffle'):
                    rng = random.Random(self.val_track_seed)
                for fold_id in sorted(self.fold_val_datasets.keys()):
                    ds = self.fold_val_datasets[fold_id]
                    head_limit = min(per_head_limit, len(ds))
                    if head_limit <= 0:
                        continue
                    if rng is None:
                        picked = list(range(head_limit))  # original behavior: first N
                    else:
                        picked = rng.sample(range(len(ds)), k=head_limit)
                    for idx in picked:
                        self.val_track_indices.append((fold_id, int(idx)))
                    total_tracked += head_limit
                if total_tracked > 0:
                    per_head_msg = f"up to {per_head_limit}"
                    mode = self.val_track_strategy or 'first'
                    print(f"[val-track] Tracking {per_head_msg} samples per head (total {total_tracked}) each eval (mode={mode}).")
        elif self.val_track_enable and self.val_track_fixed_n == 0 and self.val_track_per_head == 0:
            print("[val-track] Tracking disabled (0 samples requested).")

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
    # Fusion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_patient_id(name: str) -> Optional[str]:
        if name is None:
            return None
        base = os.path.basename(str(name))
        stem, _ = os.path.splitext(base)
        # Handle pattern like 1PA054.nii_z0000
        if ".nii_" in stem:
            prefix, suffix = stem.split(".nii_", 1)
            if suffix.startswith("z") and suffix[1:].isdigit():
                return prefix
        parts = stem.split('_', 1)
        if len(parts) == 2:
            return parts[0]
        return stem

    def _finalize_fold_assignments(self, cv_root: Path):
        assignments: Dict[str, int] = {}
        # Base mapping: derive from validation fold directories we actually loaded
        for head_idx, patients in self._fold_val_patients.items():
            for pid in patients:
                assignments[str(pid)] = head_idx

        self._fold_assignments = assignments
        if assignments:
            summary = defaultdict(int)
            for head_idx in assignments.values():
                summary[int(head_idx)] += 1
            try:
                print(f"[fusion] Fold assignment counts: {dict(summary)}")
                head_patients = {head: sorted(list(pids)) for head, pids in self._fold_val_patients.items()}
                max_len = max((len(pids) for pids in head_patients.values()), default=0)
                if max_len > 0:
                    header = "index".ljust(6)
                    for head in range(self.n_heads):
                        header += f"head_{head}".ljust(14)
                    print("[fusion] Fold membership table:")
                    print("[fusion] " + header)
                    for idx in range(max_len):
                        row = f"{idx:<6}"
                        for head in range(self.n_heads):
                            patients = head_patients.get(head, [])
                            pid = patients[idx] if idx < len(patients) else ""
                            row += pid.ljust(14)
                        print("[fusion] " + row)
            except Exception:
                pass

    def _resolve_batch_fold_ids(self, batch) -> List[int]:
        folds: List[int] = []
        B = batch['B'].shape[0]
        pid_list = batch.get('patient_id', None)
        for idx in range(B):
            pid_val = None
            try:
                if isinstance(pid_list, (list, tuple)):
                    pid_val = pid_list[idx] if idx < len(pid_list) else None
                elif torch.is_tensor(pid_list):
                    pid_val = pid_list[idx].item()
                else:
                    pid_val = pid_list
            except Exception:
                pid_val = None
            pid_str = str(pid_val) if pid_val is not None else None
            if pid_str not in self._fold_assignments:
                raise KeyError(f"Patient '{pid_str}' missing fold assignment; check manifests/val directories.")
            fid = self._fold_assignments[pid_str]
            if fid < 0 or fid >= self.n_heads:
                raise ValueError(f"Fold assignment {fid} for patient '{pid_str}' is out of range [0,{self.n_heads}).")
            folds.append(int(fid))
        return folds

    def _confidence_config(self) -> Dict[str, float]:
        cfg = {
            "gap_max": float(self.config.get("confidence_gap_max", 0.05)),
            "thr_low": float(self.config.get("confidence_thr_low", 0.05)),
            "thr_high": float(self.config.get("confidence_thr_high", 0.15)),
            "w_min": float(self.config.get("confidence_w_min", 0.05)),
            "lambda_if": float(self.config.get("confidence_lambda_if", 1.0)),
            "lambda_oof": float(self.config.get("confidence_lambda_oof", 2.0)),
            "lambda_gap": float(self.config.get("confidence_lambda_gap", 1.0)),
        }
        return cfg

    def _build_conf_eval_loader(self):
        dataset = self.train_data.dataset
        batch_size = int(self.config.get('fusion_conf_batch', self.config.get('batchSize', 1)))
        kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._n_workers,
            pin_memory=self.config.get('cuda', False),
            drop_last=False,
        )
        if self._n_workers > 0:
            kwargs.update(
                persistent_workers=self._dl_persistent,
                prefetch_factor=max(1, self._dl_prefetch),
            )
        return DataLoader(dataset, **kwargs)

    def _generate_confidence_weights(
        self,
        epoch: int,
        residual_cache: Optional[dict],
        weights_dir_override: Optional[Path] = None,
        allow_existing: bool = False,
        write_summary: bool = True,
    ) -> Optional[str]:
        if residual_cache is None:
            return None
        weight_root = Path(self.config.get('fusion_confidence_dir', os.path.join(self.save_root, 'fusion_weights')))
        weight_root.mkdir(parents=True, exist_ok=True)
        weights_dir = weights_dir_override or (weight_root / f"epoch_{epoch:04d}")
        if weights_dir.exists() and not (allow_existing or bool(self.config.get('fusion_confidence_overwrite', False))):
            return str(weights_dir)
        os.makedirs(weights_dir, exist_ok=True)
        comp_dir: Optional[Path] = None
        if bool(self.config.get('fusion_confidence_save_components', False)):
            comp_dir = weights_dir / "components"
            comp_dir.mkdir(exist_ok=True)

        cfg = self._confidence_config()
        tv_enable = bool(self.config.get('confidence_tv_enable', False))
        tv_strength = float(self.config.get('confidence_tv_strength', 0.2))
        tv_iters = int(self.config.get('confidence_tv_iters', 15))
        tv_preserve_body = bool(self.config.get('confidence_tv_preserve_body', True))
        stats = []
        for slice_name, entry in residual_cache.items():
            residuals = entry['residuals']
            fold = int(entry['fold'])
            body = entry['body'].astype(np.float32)
            r_oof = residuals.get(fold)
            if r_oof is None:
                r_oof = next(iter(residuals.values()))
            r_if_stack = [residuals[h] for h in residuals.keys() if h != fold]
            if not r_if_stack:
                r_if_stack = [r_oof]
            result = compute_weight_map(
                r_oof=r_oof,
                r_if_stack=r_if_stack,
                gap_max=cfg['gap_max'],
                thr_low=cfg['thr_low'],
                thr_high=cfg['thr_high'],
                w_min=cfg['w_min'],
                lam_if=cfg['lambda_if'],
                lam_oof=cfg['lambda_oof'],
                lam_gap=cfg['lambda_gap'],
            )
            weight_map = (result['weights'] * body).astype(np.float32)
            if tv_enable:
                # Confine smoothing to body if requested
                mask = body if tv_preserve_body else None
                weight_map = smooth_weight_map_tv(weight_map, strength=tv_strength, n_iters=tv_iters, mask=mask)
            np.save(weights_dir / f"{slice_name}.npy", weight_map)
            if comp_dir is not None:
                np.savez_compressed(
                    comp_dir / f"{slice_name}.npz",
                    c_if=result['c_if'],
                    c_oof=result['c_oof'],
                    c_gap=result['c_gap'],
                    gap=result['gap'],
                    r_oof=r_oof,
                    r_if_mean=result['r_if_mean'],
                )
            stats.append(float(weight_map.mean()))

        if write_summary:
            summary = {
                "epoch": epoch,
                "count": len(stats),
                "avg_weight": float(np.mean(stats)) if stats else None,
                "cfg": cfg,
            }
            try:
                with (weights_dir / "summary.json").open("w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=2)
            except Exception:
                pass

        return str(weights_dir)

    def _maybe_update_confidence_weights(self, epoch: int, residual_cache: Optional[dict]):
        if not bool(self.config.get('fusion_confidence', False)):
            return
        weights_path = None
        if self.ddp_enabled and _dist_is_available_and_initialized():
            # Use per-rank residual caches to write weights in parallel; rank0 will broadcast the path.
            weight_root = Path(self.config.get('fusion_confidence_dir', os.path.join(self.save_root, 'fusion_weights')))
            weight_root.mkdir(parents=True, exist_ok=True)
            weights_dir = weight_root / f"epoch_{epoch:04d}"
            weights_path = str(weights_dir)
            try:
                os.makedirs(weights_dir, exist_ok=True)
            except Exception:
                pass
            try:
                if residual_cache:
                    self._generate_confidence_weights(
                        epoch,
                        residual_cache,
                        weights_dir_override=weights_dir,
                        allow_existing=True,
                        write_summary=_is_main_process(),
                    )
            except Exception as exc:
                if _is_main_process():
                    print(f"[fusion] confidence weight generation failed: {exc}")
                weights_path = None
            obj = [weights_path]
            try:
                dist.broadcast_object_list(obj, src=0)
                weights_path = obj[0]
            except Exception as exc:
                if _is_main_process():
                    print(f"[fusion] broadcast weights_path failed: {exc}")
        else:
            if _is_main_process():
                try:
                    weights_path = self._generate_confidence_weights(epoch, residual_cache)
                except Exception as exc:
                    print(f"[fusion] confidence weight generation failed: {exc}")
                    weights_path = None
        if weights_path:
            self.set_rd_weights(weights_path, rd_w_min=self._confidence_config()['w_min'])

    def _collect_validation_metrics(self, epoch: int):
        val_metrics_modules = self._prepare_val_metrics_modules()
        fid_metric = None
        lpips_metric = None
        if val_metrics_modules is not None:
            if self.config.get('val_enable_fid', True):
                fid_metric = val_metrics_modules.get('fid')
                if fid_metric is not None:
                    try:
                        fid_metric.reset()
                    except Exception:
                        fid_metric = None
            if self.config.get('val_enable_lpips', True):
                lpips_metric = val_metrics_modules.get('lpips')

        use_reg = bool(self.config.get('eval_with_registration', False) and hasattr(self, 'R_A') and hasattr(self,
                                                                                                             'spatial_transform'))
        reg_net = self._unwrap(self.R_A) if use_reg else None
        spat = self.spatial_transform if use_reg else None

        mae_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        sample_count = 0.0
        lpips_sum = 0.0
        lpips_count = 0
        # Keep per-rank residual caches when fusion_confidence is on; we will gather them later.
        collect_conf = bool(self.config.get('fusion_confidence', False))
        residual_cache = {} if collect_conf else None

        def _body_mask_from_gt(arr_np):
            body = np.ones_like(np.squeeze(arr_np), dtype=np.float32)
            if not self.ignore_neg1_background:
                body = (np.squeeze(arr_np) != -1).astype(np.float32)
            if body.ndim == 3:
                body = body.all(axis=0).astype(np.float32)
            return body

        total_val_batches = sum(len(loader) for loader in self.fold_val_loaders.values())
        vprogress = None
        if not self._progress_disable and _is_main_process():
            vprogress = Progress(
                TextColumn("[cyan]Validation[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )

        heads_all = sorted(self.fold_val_loaders.keys()) or list(range(self.n_heads))
        processed = 0

        def _save_val_keep_mask(slice_name: str, mask_arr):
            if not self.val_keep_masks_dir:
                return
            try:
                keep_png = ((np.asarray(mask_arr) > 0).astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(self.val_keep_masks_dir, f"{slice_name}.png"), keep_png)
            except Exception:
                pass

        torch.set_grad_enabled(False)
        with (vprogress if vprogress is not None else nullcontext()) as pg:
            if pg is not None:
                task = pg.add_task(f"Val Epoch {epoch + 1}/{self.config['n_epochs']}", total=total_val_batches)
            for fold_id, loader in self.fold_val_loaders.items():
                for batch_idx, batch in enumerate(loader):
                    real_A = batch['A'].to(self.device, non_blocking=True)
                    real_B = batch['B'].to(self.device, non_blocking=True)
                    with torch.no_grad():
                        outputs = self.netG_A2B(real_A, head_indices=heads_all)
                    if use_reg and reg_net is not None and spat is not None:
                        aligned = {}
                        for head in heads_all:
                            with torch.no_grad():
                                Trans = reg_net(outputs[head], real_B)
                                aligned[head] = spat(outputs[head], Trans)
                    else:
                        aligned = outputs

                    pred_fold = aligned[fold_id]
                    real_np = real_B.detach().cpu().numpy()
                    pred_np = pred_fold.detach().cpu().numpy()
                    B = real_np.shape[0]
                    w_batch = batch.get('rd_weight', None)
                    if isinstance(w_batch, torch.Tensor):
                        w_batch = w_batch.detach().cpu().numpy()

                    batch_mae = 0.0
                    batch_psnr = 0.0
                    batch_ssim = 0.0
                    residuals_store = residual_cache is not None

                    for b in range(B):
                        gt = np.squeeze(real_np[b])
                        pred = np.squeeze(pred_np[b])
                        mask = None
                        if self.val_metrics_use_rd:
                            mask = _body_mask_from_gt(gt)
                            if w_batch is not None:
                                w2d = np.squeeze(w_batch[b]).astype(np.float32)
                                if w2d.ndim == 3:
                                    if w2d.shape[0] == 1:
                                        w2d = w2d[0]
                                    else:
                                        w2d = np.mean(w2d, axis=0)
                                if w2d.shape != gt.shape:
                                    w2d = cv2.resize(w2d, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                                w2d = np.clip(w2d, 0.0, 1.0)
                                mask = mask * w2d
                            if mask.sum() <= 0:
                                mask = _body_mask_from_gt(gt)
                        slice_name = compose_slice_name(batch, batch_idx, b)
                        if self.val_keep_masks_dir and mask is not None:
                            _save_val_keep_mask(slice_name, mask)
                        batch_mae += compute_mae(pred, gt, mask=mask)
                        batch_psnr += compute_psnr(pred, gt, mask=mask)
                        batch_ssim += compute_ssim(pred, gt, mask=mask)

                        if residuals_store:
                            body = _body_mask_from_gt(gt)
                            entry = {'fold': fold_id, 'body': body, 'residuals': {}}
                            for head in heads_all:
                                arr = aligned[head][b].detach().cpu().numpy()
                                res = np.abs(np.squeeze(arr) - gt)
                                if res.ndim == 3:
                                    res = res.mean(axis=0)
                                entry['residuals'][head] = res
                            residual_cache[slice_name] = entry

                    mae_sum += batch_mae
                    psnr_sum += batch_psnr
                    ssim_sum += batch_ssim
                    sample_count += float(B)

                    if fid_metric is not None and _is_main_process():
                        fake_for_metrics = self._prepare_images_for_metrics(pred_fold)
                        real_for_metrics = self._prepare_images_for_metrics(real_B)
                        try:
                            fid_metric.update(fake_for_metrics, real=False)
                            fid_metric.update(real_for_metrics, real=True)
                        except Exception:
                            pass
                    if lpips_metric is not None:
                        fake_lp = self._prepare_images_for_metrics(pred_fold)
                        real_lp = self._prepare_images_for_metrics(real_B)
                        try:
                            lpips_batch = lpips_metric(fake_lp, real_lp)
                            if lpips_batch is not None:
                                lpips_sum += float(lpips_batch.mean().item())
                                lpips_count += 1
                        except Exception:
                            pass

                    processed += 1
                    if pg is not None:
                        pg.update(task, completed=processed)
        torch.set_grad_enabled(True)

        try:
            tensor = torch.tensor([mae_sum, psnr_sum, ssim_sum, sample_count], dtype=torch.float64, device=self.device)
            tensor = _reduce_tensor_sum(tensor)
            mae_sum, psnr_sum, ssim_sum, sample_count = [float(x) for x in tensor.tolist()]
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
        if lpips_metric is not None and lpips_count > 0:
            lpips_value = float(lpips_sum / lpips_count)
        else:
            lpips_value = None

        return val_mae, val_psnr, val_ssim, fid_value, lpips_value, residual_cache

    def _generator_module(self) -> SharedBackboneGenerator:
        return self._unwrap(self.netG_A2B)

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
            'test_metrics_use_rd': bool(
                self.config.get('test_metrics_use_rd', self.config.get('metrics_use_rd', False))),
            'fusion_confidence': bool(self.config.get('fusion_confidence', False)),
            'fusion_confidence_dir': self.config.get('fusion_confidence_dir', ''),
            'val_enable_fid': bool(self.config.get('val_enable_fid', True)),
            'val_enable_lpips': bool(self.config.get('val_enable_lpips', True)),
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
            resolved['val_dataset_len'] = sum(len(ds) for ds in self.fold_val_datasets.values())
        except Exception:
            pass
        try:
            resolved['train_loader_len'] = len(self.train_data)
        except Exception:
            pass
        try:
            resolved['val_loader_len'] = sum(len(loader) for loader in self.fold_val_loaders.values())
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
                for fold_id, loader in getattr(self, 'fold_val_loaders', {}).items():
                    if hasattr(loader, 'dataset'):
                        va_sampler = DistributedSampler(loader.dataset, shuffle=False, drop_last=False)
                        self.fold_val_loaders[fold_id] = _rebuild_loader_with_sampler(loader, va_sampler)
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

        # --- Learning-rate schedulers (linear decay after decay_epoch) ---
        decay_epoch = int(self.config.get('decay_epoch', total_epochs))
        self.lr_scheduler_G = None
        self.lr_scheduler_D = None
        self.lr_scheduler_R = None
        if decay_epoch > 0 and decay_epoch < total_epochs:
            lr_lambda = LambdaLR(total_epochs, start_epoch, decay_epoch).step
            try:
                self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
                self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)
                if hasattr(self, 'optimizer_R_A') and self.optimizer_R_A is not None:
                    self.lr_scheduler_R = torch.optim.lr_scheduler.LambdaLR(self.optimizer_R_A, lr_lambda=lr_lambda)
            except Exception as _e:
                if _is_main_process():
                    print(f"[warn] LR scheduler setup failed (decay disabled): {_e}")

        if _is_main_process():
            try:
                ds_len = len(self.train_data.dataset)
                smp_len = len(self.train_data.sampler) if hasattr(self.train_data,
                                                                  'sampler') and self.train_data.sampler is not None else ds_len
                print(
                    f"[DDP] world_size={_get_world_size()} dataset={ds_len} per-rank-samples={smp_len} batch_size={self.train_data.batch_size} per-rank-batches={len(self.train_data)}",
                    flush=True)
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
                ep_sums = defaultdict(float)  # name -> summed loss over the epoch
                ep_counts = defaultdict(int)  # name -> number of times this loss was available
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

                    # === Fusion generator update (shared trunk + active heads) ===
                    batch_folds = self._resolve_batch_fold_ids(batch)
                    self.optimizer_G.zero_grad()
                    if self.optimizer_R_A is not None:
                        self.optimizer_R_A.zero_grad()
                    active_indices: Dict[int, torch.Tensor] = {}
                    for head_id in range(self.n_heads):
                        idxs = [idx for idx, fid in enumerate(batch_folds) if fid != head_id]
                        if idxs:
                            active_indices[head_id] = torch.as_tensor(idxs, device=self.device, dtype=torch.long)
                    use_fixed_guidance = self.rd_input_type in ('mask', 'weights')
                    rd_weights_full = None
                    if use_fixed_guidance:
                        if 'rd_weight' in batch and batch['rd_weight'] is not None:
                            rd_weights_full = batch['rd_weight'].to(real_B.device, dtype=real_B.dtype)
                        else:
                            rd_weights_full = load_weights_for_batch(
                                batch, i, real_B,
                                self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min,
                                ignore_background=self.ignore_neg1_background
                            ).to(real_B.device, dtype=real_B.dtype)
                    body_mask_full = None
                    if 'body_mask' in batch and batch['body_mask'] is not None:
                        body_mask_full = batch['body_mask'].to(real_B.device, dtype=real_B.dtype)

                    sr_tensors: List[torch.Tensor] = []
                    adv_tensors: List[torch.Tensor] = []
                    smooth_tensors: List[torch.Tensor] = []
                    total_tensors: List[torch.Tensor] = []

                    heads_to_run = list(range(self.n_heads))
                    outputs = self.netG_A2B(real_A, head_indices=heads_to_run)

                    for head_id in heads_to_run:
                        idx_tensor = active_indices.get(head_id)
                        if idx_tensor is None:
                            zero = outputs[head_id].sum() * 0.0
                            sr_tensors.append(zero)
                            adv_tensors.append(zero)
                            smooth_tensors.append(zero)
                            total_tensors.append(zero)
                            continue
                        fake_full = outputs[head_id]
                        fake_sel = fake_full.index_select(0, idx_tensor)
                        real_B_sel = real_B.index_select(0, idx_tensor)
                        Trans = None
                        SysRegist_A2B = fake_sel
                        if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
                            Trans = self.R_A(fake_sel, real_B_sel)
                            SysRegist_A2B = self.spatial_transform(fake_sel, Trans)
                        if use_fixed_guidance:
                            w_sel = rd_weights_full.index_select(0, idx_tensor)
                            sr_core = masked_l1(SysRegist_A2B, real_B_sel, w_sel)
                        elif body_mask_full is not None:
                            w_body = body_mask_full.index_select(0, idx_tensor)
                            sr_core = masked_l1(SysRegist_A2B, real_B_sel, w_body)
                        else:
                            sr_core = self.L1_loss(SysRegist_A2B, real_B_sel)
                        SR_head = self.config['Corr_lamda'] * sr_core
                        pred_fake = self.netD_B(fake_sel)
                        adv_head = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))
                        SM_head = torch.tensor(0.0, device=fake_sel.device)
                        if Trans is not None:
                            SM_head = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        total_head = SR_head + adv_head + SM_head
                        sr_tensors.append(SR_head)
                        adv_tensors.append(adv_head)
                        smooth_tensors.append(SM_head)
                        total_tensors.append(total_head)

                    if total_tensors:
                        combined_loss = torch.stack(total_tensors).mean()
                        combined_loss.backward()
                        if self.optimizer_R_A is not None:
                            self.optimizer_R_A.step()
                        self.optimizer_G.step()
                    else:
                        # No gradients were produced; reset just in case.
                        if self.optimizer_R_A is not None:
                            self.optimizer_R_A.zero_grad(set_to_none=True)
                        self.optimizer_G.zero_grad(set_to_none=True)

                    def _mean_or_none(tensors: List[torch.Tensor]):
                        if not tensors:
                            return None
                        return torch.stack([t.float() for t in tensors]).mean()

                    SR_loss = _mean_or_none([t.detach() for t in sr_tensors])
                    adv_loss = _mean_or_none([t.detach() for t in adv_tensors])
                    SM_loss = _mean_or_none([t.detach() for t in smooth_tensors])
                    total_loss = _mean_or_none([t.detach() for t in total_tensors])

                    # === Discriminator update ===
                    self.optimizer_D_B.zero_grad()
                    with torch.no_grad():
                        fresh_outputs = self.netG_A2B(real_A, head_indices=heads_to_run)
                    loss_D_components: List[torch.Tensor] = []
                    loss_D_logs: List[torch.Tensor] = []
                    for head_id in heads_to_run:
                        idx_tensor = active_indices.get(head_id)
                        if idx_tensor is None or idx_tensor.numel() == 0:
                            loss_head = fresh_outputs[head_id].sum() * 0.0
                        else:
                            fake_sel = fresh_outputs[head_id].index_select(0, idx_tensor)
                            real_B_sel = real_B.index_select(0, idx_tensor)
                            head_pred_fake = self.netD_B(fake_sel)
                            head_pred_real = self.netD_B(real_B_sel)
                            loss_head = self.config['Adv_lamda'] * (
                                    self.MSE_loss(head_pred_fake, torch.zeros_like(head_pred_fake)) +
                                    self.MSE_loss(head_pred_real, torch.ones_like(head_pred_real))
                            )
                        loss_D_components.append(loss_head)
                        loss_D_logs.append(loss_head.detach())
                    if loss_D_components:
                        loss_D_total = torch.stack(loss_D_components).mean()
                        loss_D_total.backward()
                        self.optimizer_D_B.step()
                        loss_D_B = torch.stack(loss_D_logs).mean()
                    else:
                        self.optimizer_D_B.zero_grad(set_to_none=True)
                        loss_D_B = None

                    # --- accumulate for epoch averages ---
                    n_batches_this_epoch += 1
                    _ep_acc('total', total_loss)
                    _ep_acc('D_B', loss_D_B)
                    _ep_acc('SR', SR_loss)
                    _ep_acc('adv', adv_loss)
                    _ep_acc('smooth', SM_loss)
                    # Log losses every 50 batches, including breakdown components
                    if i % 50 == 0:
                        log_losses = {}
                        if total_loss is not None:
                            log_losses['loss/total'] = float(total_loss.item()) if isinstance(total_loss,
                                                                                              torch.Tensor) else float(
                                total_loss)
                        if loss_D_B is not None:
                            log_losses['loss/D_B'] = float(loss_D_B.item()) if isinstance(loss_D_B,
                                                                                          torch.Tensor) else float(
                                loss_D_B)
                        if SR_loss is not None:
                            log_losses['loss/SR'] = float(SR_loss.item()) if isinstance(SR_loss,
                                                                                        torch.Tensor) else float(
                                SR_loss)
                        if adv_loss is not None:
                            log_losses['loss/adv'] = float(adv_loss.item()) if isinstance(adv_loss,
                                                                                          torch.Tensor) else float(
                                adv_loss)
                        if SM_loss is not None:
                            log_losses['loss/smooth'] = float(SM_loss.item()) if isinstance(SM_loss,
                                                                                            torch.Tensor) else float(
                                SM_loss)
                        if log_losses:
                            self.logger.log_step(log_losses)
                    # advance inner progress
                    try:
                        progress.advance(task_batches, 1)
                    except Exception:
                        pass
                    # also update the epoch task with fractional progress so ETA is meaningful
                    try:
                        if progress is not None and task_epochs is not None and isinstance(total_batches,
                                                                                           int) and total_batches > 0:
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
                if (epoch + 1) % self.config['val_freq'] == 0:
                    if _is_main_process() and self.config.get('val_debug_log', False):
                        print("[val-debug] validation start", flush=True)
                    was_train_G = self.netG_A2B.training
                    self.netG_A2B.eval()
                    was_train_R = None
                    if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
                        was_train_R = self.R_A.training
                        self.R_A.eval()

                    val_mae, val_psnr, val_ssim, fid_value, lpips_value, residual_cache = self._collect_validation_metrics(
                        epoch)

                    if was_train_G:
                        self.netG_A2B.train()
                    if was_train_R and hasattr(self, 'R_A'):
                        self.R_A.train()

                    if _is_main_process():
                        msg = f"Val MAE: {val_mae:.6f} | PSNR: {val_psnr:.3f} dB | SSIM: {val_ssim:.4f}"
                        if fid_value is not None:
                            msg += f" | FID: {fid_value:.4f}"
                        if lpips_value is not None:
                            msg += f" | LPIPS: {lpips_value:.6f}"
                        print(msg)

                        # === 采样并上传验证图像到 wandb ===
                        try:
                            sample_cap_total = self.val_sample_images
                            per_head_limit = 0
                            if self.val_sample_images_per_head > 0:
                                per_head_limit = self.val_sample_images_per_head
                            elif sample_cap_total > 0:
                                per_head_limit = max(1, int(math.ceil(sample_cap_total / max(1, self.n_heads))))
                            if per_head_limit > 0:
                                imgs_input: List[wandb.Image] = []
                                imgs_real: List[wandb.Image] = []
                                imgs_pred: List[wandb.Image] = []
                                imgs_pred_pre: List[wandb.Image] = []
                                per_head_counts: Dict[int, int] = {}
                                with torch.no_grad():
                                    for fold_id, loader in self.fold_val_loaders.items():
                                        collected = 0
                                        for batch_val in loader:
                                            real_Av = batch_val['A'].to(self.device, non_blocking=True)
                                            real_Bv = batch_val['B'].to(self.device, non_blocking=True)
                                            preds = self.netG_A2B(real_Av, head_indices=[fold_id])
                                            pred_Bv = preds[fold_id]
                                            pred_pre = pred_Bv
                                            if bool(self.config.get('eval_with_registration', False)) and hasattr(self,
                                                                                                                  'R_A') and hasattr(
                                                    self, 'spatial_transform'):
                                                flow = self.R_A(pred_Bv, real_Bv)
                                                pred_Bv = self.spatial_transform(pred_Bv, flow)
                                            Bnow = real_Bv.shape[0]
                                            if Bnow <= 0:
                                                continue
                                            remaining = max(0, per_head_limit - collected)
                                            if remaining <= 0:
                                                break
                                            k = min(remaining, Bnow)
                                            idxs = random.sample(range(Bnow), k) if Bnow > 1 and k < Bnow else list(
                                                range(k))
                                            for idx in idxs:
                                                Ab = real_Av[idx].detach().cpu().numpy().squeeze()
                                                rb = real_Bv[idx].detach().cpu().numpy().squeeze()
                                                pb = pred_Bv[idx].detach().cpu().numpy().squeeze()

                                                imgs_input.append(wandb.Image(to_uint8_image(Ab)))
                                                imgs_real.append(wandb.Image(to_uint8_image(rb)))
                                                imgs_pred.append(
                                                    wandb.Image(to_uint8_image(pb), caption=f"head {fold_id} reg"))
                                                imgs_pred_pre.append(wandb.Image(
                                                    to_uint8_image(pred_pre[idx].detach().cpu().numpy().squeeze()),
                                                    caption=f"head {fold_id} pre"))
                                                collected += 1
                                            if collected >= per_head_limit:
                                                break
                                        per_head_counts[fold_id] = collected
                                        if collected == 0:
                                            print(
                                                f"[wandb] warning: head {fold_id} yielded 0 samples for val image logging",
                                                flush=True)
                                if imgs_real and imgs_pred and _is_main_process():
                                    log_payload = {
                                        f'images/val_input_A_ep{epoch + 1:04d}': imgs_input,
                                        f'images/val_real_B_ep{epoch + 1:04d}': imgs_real,
                                        f'images/val_pred_B_ep{epoch + 1:04d}': imgs_pred,
                                        f'images/val_pred_B_preReg_ep{epoch + 1:04d}': imgs_pred_pre,
                                        'epoch': epoch + 1,
                                    }
                                    for head_idx, count in per_head_counts.items():
                                        log_payload[f'val/image_samples/head_{head_idx}'] = int(count)
                                    wandb.log(log_payload)
                                    if _is_main_process() and self.config.get('val_debug_log', False):
                                        print("[val-debug] wandb image upload complete", flush=True)
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
                                if _is_main_process() and self.config.get('val_debug_log', False):
                                    print("[val-debug] wandb scalar log complete", flush=True)
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
                                if _is_main_process() and self.config.get('val_debug_log', False):
                                    print("[val-debug] wandb summary updated", flush=True)
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
                                if _is_main_process() and self.config.get('val_debug_log', False):
                                    print("[val-debug] checkpoint saved", flush=True)
                        except Exception as e:
                            print(f"[val] saving best checkpoint failed: {e}")

                        # === 固定样本跟踪：每次验证都记录同一批三元组（Input/Truth/Pred） ===
                        try:
                            # Build tracking indices either fixed (precomputed) or randomly per validation.
                            track_indices = []
                            if self.val_track_enable:
                                if self.val_track_strategy in ('random_each', 'random-epoch', 'random_epoch'):
                                    per_head_limit = int(getattr(self, '_val_track_per_head_limit', 0) or 0)
                                    if per_head_limit > 0:
                                        rng = random.Random(int(self.val_track_seed) + int(epoch + 1))
                                        for fold_id in sorted(self.fold_val_datasets.keys()):
                                            ds = self.fold_val_datasets.get(fold_id)
                                            if ds is None:
                                                continue
                                            head_limit = min(per_head_limit, len(ds))
                                            if head_limit <= 0:
                                                continue
                                            picked = rng.sample(range(len(ds)), k=head_limit)
                                            for idx in picked:
                                                track_indices.append((fold_id, int(idx)))
                                else:
                                    if isinstance(self.val_track_indices, list) and len(self.val_track_indices) > 0:
                                        track_indices = list(self.val_track_indices)

                            if self.val_track_enable and track_indices:
                                tbl = wandb.Table(columns=[
                                    "id",
                                    "fold",
                                    "input",
                                    "truth",
                                    "pred",
                                    "norm_loss_weight",
                                    "loss_weight_white1",
                                    "body_mask",
                                    "rd_source_exclude",
                                ])
                                # eval mode
                                self.netG_A2B.eval()
                                for fold_id, sample_idx in track_indices:
                                    ds = self.fold_val_datasets.get(fold_id)
                                    if ds is None or sample_idx >= len(ds):
                                        continue
                                    try:
                                        sample = ds[sample_idx]
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
                                            pid is not None and sid is not None) else f"fold{fold_id}_idx{sample_idx:05d}"
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
                                    head_id = fold_id
                                    head_outputs_single = self.netG_A2B(A, head_indices=[head_id])
                                    pred = head_outputs_single[head_id]
                                    # To numpy for visualization
                                    A_np = A.detach().cpu().numpy().squeeze()
                                    B_np = B.detach().cpu().numpy().squeeze()
                                    P_np = pred.detach().cpu().numpy().squeeze()

                                    body_arr = np.squeeze(B_np)
                                    if self.ignore_neg1_background:
                                        body_mask = np.ones(body_arr.shape[-2:], dtype=np.uint8)
                                    else:
                                        if body_arr.ndim == 3 and body_arr.shape[0] == 1:
                                            body_arr = body_arr[0]
                                        if body_arr.ndim == 3 and body_arr.shape[0] > 1:
                                            body_mask = (body_arr != -1).all(axis=0).astype(np.uint8)
                                        else:
                                            body_mask = (body_arr != -1).astype(np.uint8)

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
                                                if w.ndim == 3:
                                                    if w.shape[0] == 1:
                                                        w = w[0]
                                                    else:
                                                        w = np.mean(w, axis=0)
                                                keep_array = np.clip(w.astype(np.float32), 0.0, 1.0)
                                                w_min, w_max = float(np.min(keep_array)), float(np.max(keep_array))
                                                if w_max > w_min:
                                                    w_vis = ((keep_array - w_min) / (
                                                                w_max - w_min + 1e-6) * 255.0).clip(0,
                                                                                                    255).astype(
                                                        np.uint8)
                                                else:
                                                    w_vis = np.zeros_like(keep_array, dtype=np.uint8)
                                                W_img = wandb.Image(
                                                    w_vis,
                                                    caption=f"norm loss weight (min-max) [{w_min:.3f},{w_max:.3f}]"
                                                )
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
                                    try:
                                        body_img = wandb.Image((body_mask * 255).astype(np.uint8), caption="body")
                                        if keep_array is not None:
                                            if keep_array.ndim == 3:
                                                if keep_array.shape[0] == 1:
                                                    keep_array = keep_array[0]
                                                else:
                                                    keep_array = np.mean(keep_array, axis=0)
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
                                                caption="loss weight (white=1)",
                                            )
                                    except Exception:
                                        body_img = None
                                        keep_img = None

                                    # Convert to uint8
                                    tbl.add_data(
                                        sid_str,
                                        fold_id,
                                        wandb.Image(to_uint8_image(A_np), caption=f"{sid_str}: input"),
                                        wandb.Image(to_uint8_image(B_np), caption=f"{sid_str}: truth"),
                                        wandb.Image(to_uint8_image(P_np), caption=f"{sid_str}: head {fold_id} pred"),
                                        W_img,
                                        keep_img,
                                        body_img,
                                        raw_rd_mask,
                                    )
                                if _is_main_process():
                                    wandb.log({
                                        f'tables/track_triplets_ep{epoch + 1:04d}': tbl,
                                        'epoch': epoch + 1,
                                    })
                                    if self.config.get('val_debug_log', False):
                                        print("[val-debug] wandb table logged", flush=True)
                        except Exception as e:
                            print(f"[wandb] fixed val-track table failed: {e}")

                        if _is_main_process() and self.config.get('val_debug_log', False):
                            print("[val-debug] validation post-processing complete", flush=True)

                    # Update dynamic confidence weights after validation (main rank generates)
                    if bool(self.config.get('fusion_confidence', False)):
                        if _is_main_process() and self.config.get('val_debug_log', False):
                            print("[val-debug] updating confidence weights", flush=True)
                        self._maybe_update_confidence_weights(epoch + 1, residual_cache)
                        if _is_main_process() and self.config.get('val_debug_log', False):
                            print("[val-debug] confidence weights ready", flush=True)
                        if self.ddp_enabled:
                            dist.barrier()

                    # Restore models to previous training/eval state
                    if was_train_G:
                        self.netG_A2B.train()
                    if was_train_R:
                        self.R_A.train()

                    # Step LR schedulers (if enabled)
                    try:
                        if self.lr_scheduler_G is not None:
                            self.lr_scheduler_G.step()
                        if self.lr_scheduler_D is not None:
                            self.lr_scheduler_D.step()
                        if self.lr_scheduler_R is not None:
                            self.lr_scheduler_R.step()
                        if _is_main_process():
                            try:
                                lr_g = self.optimizer_G.param_groups[0]['lr']
                                lr_d = self.optimizer_D_B.param_groups[0]['lr']
                                log_lr = {'lr/G': lr_g, 'lr/D_B': lr_d, 'epoch': epoch + 1}
                                if hasattr(self, 'optimizer_R_A') and self.optimizer_R_A is not None:
                                    log_lr['lr/R_A'] = self.optimizer_R_A.param_groups[0]['lr']
                                wandb.log(log_lr)
                            except Exception:
                                pass
                    except Exception as _e:
                        if _is_main_process() and self.config.get('val_debug_log', False):
                            print(f"[warn] LR scheduler step failed: {_e}")

        self.config['epoch'] = target_epoch

    def test(self, ):
        raise NotImplementedError('Please use experiment05-fusion-training-test.py for evaluation.')

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
            if self.config.get('val_enable_fid', True):
                try:
                    if FrechetInceptionDistance is not None:
                        support['fid'] = FrechetInceptionDistance(feature=64, reset_real_features=True).to(device)
                except Exception as exc:
                    print(f"[val] FID metric init failed: {exc}")
            if self.config.get('val_enable_lpips', True):
                try:
                    if LearnedPerceptualImagePatchSimilarity is not None:
                        support['lpips'] = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(
                            device)
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
        if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module):
            state['R_A'] = self._unwrap(self.R_A).state_dict()
        if hasattr(self, 'optimizer_R_A') and self.optimizer_R_A is not None:
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
        if hasattr(self, 'R_A') and isinstance(self.R_A, torch.nn.Module) and 'R_A' in state:
            self._unwrap(self.R_A).load_state_dict(state['R_A'])
        if hasattr(self, 'optimizer_R_A') and self.optimizer_R_A is not None and 'optimizer_R_A' in state:
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
        if hasattr(self, 'train_data') and hasattr(self.train_data, 'dataset') and hasattr(self.train_data.dataset,
                                                                                           'set_rd_config'):
            self.train_data.dataset.set_rd_config(self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir,
                                                  self.rd_w_min, self.rd_fallback_mode,
                                                  ignore_neg1_background=self.ignore_neg1_background)
        if hasattr(self, 'fold_val_datasets'):
            for ds in self.fold_val_datasets.values():
                if hasattr(ds, 'set_rd_config'):
                    ds.set_rd_config(self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min,
                                     self.rd_fallback_mode,
                                     ignore_neg1_background=self.ignore_neg1_background)
