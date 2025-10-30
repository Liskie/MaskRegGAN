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
    compute_mae, compute_psnr, compute_ssim, resolve_model_path
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

        ## def networks
        self.netG_A2B = SharedBackboneGenerator(
            config['input_nc'],
            config['output_nc'],
            n_residual_blocks=config.get('fusion_residual_blocks', 9),
            n_heads=self.n_heads,
            upsample_mode=str(config.get('generator_upsample_mode', 'resize')).lower(),
            share_body=bool(config.get('fusion_share_body', True)),
        ).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        if not config.get('regist', False):
            raise ValueError("CycTrainerFusion requires regist=True to operate with shared registration.")
        self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
        self.spatial_transform = Transformer_2D().cuda()
        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

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
        train_dataset = ImageDataset(
            config['dataroot'], level,
            transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False,
            rd_input_type=self.rd_input_type,
            rd_mask_dir=self.rd_mask_dir,
            rd_weights_dir=self.rd_weights_dir,
            rd_w_min=self.rd_w_min,
            cache_mode=config.get('cache_mode', 'mmap'),
            rd_cache_weights=config.get('rd_cache_weights', False),
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
            val_dir = fold_dir / 'val'
            if not val_dir.is_dir():
                continue
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
            )
            head_idx = len(self.fold_val_loaders)
            self.fold_val_datasets[head_idx] = val_dataset
            self.fold_val_loaders[head_idx] = DataLoader(val_dataset, **val_kwargs)
            self._head_to_fold_raw[head_idx] = fold_raw_id
            self._head_to_fold_dir[head_idx] = val_dir
            self._fold_id_lookup[fold_raw_id] = head_idx
            val_patients: Set[str] = set()
            val_b_dir = val_dir / 'B'
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

        if per_head_limit > 0 and self.val_track_enable:
            for fold_id in sorted(self.fold_val_datasets.keys()):
                ds = self.fold_val_datasets[fold_id]
                head_limit = min(per_head_limit, len(ds))
                for idx in range(head_limit):
                    self.val_track_indices.append((fold_id, idx))
                total_tracked += head_limit
            if total_tracked > 0:
                per_head_msg = f"up to {per_head_limit}"
                print(f"[val-track] Tracking {per_head_msg} samples per head (total {total_tracked}) each eval.")
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

    def _generate_confidence_weights(self, epoch: int, residual_cache: Optional[dict]) -> Optional[str]:
        if residual_cache is None:
            return None
        weight_root = Path(self.config.get('fusion_confidence_dir', os.path.join(self.save_root, 'fusion_weights')))
        weight_root.mkdir(parents=True, exist_ok=True)
        weights_dir = weight_root / f"epoch_{epoch:04d}"
        if weights_dir.exists() and not bool(self.config.get('fusion_confidence_overwrite', False)):
            return str(weights_dir)
        os.makedirs(weights_dir, exist_ok=True)
        comp_dir: Optional[Path] = None
        if bool(self.config.get('fusion_confidence_save_components', False)):
            comp_dir = weights_dir / "components"
            comp_dir.mkdir(exist_ok=True)

        cfg = self._confidence_config()
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
        if _is_main_process():
            try:
                weights_path = self._generate_confidence_weights(epoch, residual_cache)
            except Exception as exc:
                print(f"[fusion] confidence weight generation failed: {exc}")
                weights_path = None
        if self.ddp_enabled:
            obj = [weights_path]
            dist.broadcast_object_list(obj, src=0)
            weights_path = obj[0]
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
        residual_cache = {} if _is_main_process() else None

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
                            mask = (gt != -1).astype(np.float32)
                            if w_batch is not None:
                                w2d = np.squeeze(w_batch[b]).astype(np.float32)
                                if w2d.shape != gt.shape:
                                    w2d = cv2.resize(w2d, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                                w2d = np.clip(w2d, 0.0, 1.0)
                                mask = mask * w2d
                            if mask.sum() <= 0:
                                mask = (gt != -1).astype(np.float32)
                        slice_name = compose_slice_name(batch, batch_idx, b)
                        if self.val_keep_masks_dir and mask is not None:
                            _save_val_keep_mask(slice_name, mask)
                        batch_mae += compute_mae(pred, gt, mask=mask)
                        batch_psnr += compute_psnr(pred, gt, mask=mask)
                        batch_ssim += compute_ssim(pred, gt, mask=mask)

                        if residuals_store:
                            body = (gt != -1).astype(np.float32)
                            entry = {'fold': fold_id, 'body': body, 'residuals': {}}
                            for head in heads_all:
                                arr = aligned[head][b].detach().cpu().numpy()
                                entry['residuals'][head] = np.abs(np.squeeze(arr) - gt)
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
                                self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min
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
                        SM_head = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        total_head = SR_head + adv_head + SM_head
                        sr_tensors.append(SR_head)
                        adv_tensors.append(adv_head)
                        smooth_tensors.append(SM_head)
                        total_tensors.append(total_head)

                    if total_tensors:
                        combined_loss = torch.stack(total_tensors).mean()
                        combined_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                    else:
                        # No gradients were produced; reset just in case.
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
                if epoch % self.config['val_freq'] == 0:
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

                                                def _to_uint8(a):
                                                    a = np.squeeze(a)
                                                    if a.ndim != 2:
                                                        a = a.astype(np.float32)
                                                        a = a[0] if a.ndim == 3 else a
                                                    amin, amax = float(np.min(a)), float(np.max(a))
                                                    if amin >= -1.001 and amax <= 1.001:
                                                        a = ((a + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                                                    else:
                                                        a = ((a - amin) / (amax - amin + 1e-6) * 255.0).clip(0,
                                                                                                             255).astype(
                                                            np.uint8)
                                                    return a

                                                imgs_input.append(wandb.Image(_to_uint8(Ab)))
                                                imgs_real.append(wandb.Image(_to_uint8(rb)))
                                                imgs_pred.append(
                                                    wandb.Image(_to_uint8(pb), caption=f"head {fold_id} reg"))
                                                imgs_pred_pre.append(wandb.Image(
                                                    _to_uint8(pred_pre[idx].detach().cpu().numpy().squeeze()),
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
                                    "rd_source_exclude",
                                ])
                                # eval mode
                                self.netG_A2B.eval()
                                for fold_id, sample_idx in self.val_track_indices:
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
                                    if body_arr.ndim == 3 and body_arr.shape[0] == 1:
                                        body_arr = body_arr[0]
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
                                                if w.ndim == 3 and w.shape[0] == 1:
                                                    w = w[0]
                                                keep_array = np.clip(w.astype(np.float32), 0.0, 1.0)
                                                w_min, w_max = float(np.min(keep_array)), float(np.max(keep_array))
                                                if w_max > w_min:
                                                    w_vis = ((keep_array - w_min) / (
                                                                w_max - w_min + 1e-6) * 255.0).clip(0,
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
                                    try:
                                        body_img = wandb.Image((body_mask * 255).astype(np.uint8), caption="body")
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
                                        wandb.Image(_to_uint8(P_np), caption=f"{sid_str}: head {fold_id} pred"),
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
        test_metrics_use_rd = bool(self.config.get('test_metrics_use_rd', self.config.get('val_metrics_use_rd',
                                                                                          self.config.get(
                                                                                              'metrics_use_rd',
                                                                                              False))))
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
            plot_executor = ProcessPoolExecutor(max_workers=plot_workers) if (
                        plot_workers > 0 and save_composite) else None
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

                        batch_folds = self._resolve_batch_fold_ids(batch)
                        heads_in_batch = sorted(set(batch_folds))

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
                                head_outputs = self.netG_A2B(real_A_in, head_indices=heads_in_batch)
                                fake_Bt = torch.zeros_like(real_A_in)
                                for head_id in heads_in_batch:
                                    sel_idx = [idx for idx, fid in enumerate(batch_folds) if fid == head_id]
                                    if not sel_idx:
                                        continue
                                    idx_tensor = torch.as_tensor(sel_idx, dtype=torch.long, device=self.device)
                                    fake_head = head_outputs[head_id].index_select(0, idx_tensor)
                                    fake_Bt.index_copy_(0, idx_tensor, fake_head)
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
        if hasattr(self, 'train_data') and hasattr(self.train_data, 'dataset') and hasattr(self.train_data.dataset,
                                                                                           'set_rd_config'):
            self.train_data.dataset.set_rd_config(self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir,
                                                  self.rd_w_min)
        if hasattr(self, 'fold_val_datasets'):
            for ds in self.fold_val_datasets.values():
                if hasattr(ds, 'set_rd_config'):
                    ds.set_rd_config(self.rd_input_type, self.rd_mask_dir, self.rd_weights_dir, self.rd_w_min)
