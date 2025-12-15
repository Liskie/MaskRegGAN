import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import re
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image as PILImage

from .utils import load_weight_or_mask_for_slice, rd_file_exists_for_slice


class ImageDataset(Dataset):
    def __init__(self, root, noise_level, count=None, transforms_1=None, transforms_2=None, unaligned=False,
                 rd_input_type=None, rd_mask_dir='', rd_weights_dir='', rd_w_min=0.0,
                 cache_mode='none', rd_cache_weights=False,
                 domain_a_dir=None, domain_b_dir=None,
                 domain_a_channels=None, domain_b_channels=None,
                 rd_fallback_mode='body', ignore_neg1_background=False):
        transforms_1 = transforms_1 or []
        transforms_2 = transforms_2 or []
        self.domain_a_dir = domain_a_dir
        self.domain_b_dir = domain_b_dir
        self.domain_a_channels = int(domain_a_channels) if domain_a_channels is not None else None
        self.domain_b_channels = int(domain_b_channels) if domain_b_channels is not None else None
        self.files_A = self._resolve_files(root, domain_a_dir, 'A')
        self.files_B = self._resolve_files(root, domain_b_dir, 'B')
        self.unaligned = unaligned
        self.noise_level = noise_level
        # residual-detection guidance settings (optional)
        self.rd_input_type = rd_input_type
        self.rd_mask_dir = rd_mask_dir
        self.rd_weights_dir = rd_weights_dir
        self.rd_w_min = rd_w_min
        self.cache_mode = str(cache_mode).lower()
        self.rd_cache_weights = bool(rd_cache_weights)
        self.rd_fallback_mode = str(rd_fallback_mode or 'body').lower()
        self.ignore_neg1_background = bool(ignore_neg1_background)
        # Setup caches for A/B arrays and weight maps
        self._A_cache = [None] * len(self.files_A)
        self._B_cache = [None] * len(self.files_B)
        self._W_cache = {}  # slice_name -> np.ndarray (cached rd_weight)

        # --- Keep references to affine + resize components so we can reuse params for masks ---
        self._resize_op = None
        if transforms_1:
            # assume last transform is resize/padding op
            self._resize_op = transforms_1[-1]
        elif transforms_2:
            self._resize_op = transforms_2[-1]
        if self._resize_op is None:
            raise ValueError("ImageDataset expects a resize transform as the last element of transforms_1/2")

        self._affine_conf_1 = self._extract_affine_conf(transforms_1)
        self._affine_conf_2 = self._extract_affine_conf(transforms_2)
        # Identity fallback when no affine is configured
        self._identity_params = (0.0, (0.0, 0.0), 1.0, (0.0, 0.0))

    def _resolve_files(self, root, override_dir, default_subdir):
        if override_dir:
            pattern = os.path.join(os.path.expanduser(override_dir), '*')
        else:
            pattern = os.path.join(root, default_subdir, '*')
        return sorted(glob.glob(pattern))

    def _load_np(self, path):
        if self.cache_mode == 'mmap':
            return np.load(path, mmap_mode='r')
        return np.load(path).astype(np.float32)

    def _load_image_file(self, path, expect_channels=None):
        with PILImage.open(path) as img:
            if expect_channels == 1:
                img = img.convert('L')
            elif expect_channels == 3:
                img = img.convert('RGB')
            else:
                img = img.convert('RGB')
            arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        else:
            arr = arr.transpose(2, 0, 1)
        arr = (arr / 255.0 * 2.0) - 1.0
        return arr.astype(np.float32)

    def _standardize_array(self, arr, expect_channels=None):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = arr.transpose(2, 0, 1)
        if expect_channels is not None and arr.shape[0] != expect_channels:
            if arr.shape[0] == 1 and expect_channels == 3:
                arr = np.repeat(arr, 3, axis=0)
            elif arr.shape[0] == 3 and expect_channels == 1:
                arr = arr[:1, ...]
            else:
                raise ValueError(f"Cannot match channels for {arr.shape} -> {expect_channels}")
        return arr

    def _load_path(self, path, expect_channels=None):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            arr = self._load_np(path)
        elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            arr = self._load_image_file(path, expect_channels)
        else:
            raise ValueError(f"Unsupported file extension '{ext}' for path '{path}'")
        return self._standardize_array(arr, expect_channels)

    def _get_array(self, which, index):
        if which == 'A':
            files = self.files_A; cache = self._A_cache; expect_channels = self.domain_a_channels
        else:
            files = self.files_B; cache = self._B_cache; expect_channels = self.domain_b_channels
        path = files[index % len(files)]
        if self.cache_mode == 'none':
            return self._load_path(path, expect_channels)
        cached = cache[index % len(files)]
        if cached is None:
            cached = self._load_path(path, expect_channels)
            cache[index % len(files)] = cached
        return cached

    def __getitem__(self, index):
        file_A = self.files_A[index % len(self.files_A)]
        file_B = self.files_B[index % len(self.files_B)]
        arr_A = np.asarray(self._get_array('A', index), dtype=np.float32)
        arr_B = np.asarray(self._get_array('B', index), dtype=np.float32)

        if self.noise_level == 0:
            # Noise == 0 → use the "small" affine (transforms_2) for both domains
            params_shared = self._sample_affine_params(self._affine_conf_2, arr_B.shape)
            item_A = self._apply_img_transforms(arr_A, params_shared, self._affine_conf_2)
            item_B = self._apply_img_transforms(arr_B, params_shared, self._affine_conf_2)
            affine_params_B = params_shared
            affine_conf_B = self._affine_conf_2
        else:
            # Otherwise use the wider-range affine (transforms_1) independently per domain
            params_A = self._sample_affine_params(self._affine_conf_1, arr_A.shape)
            params_B = self._sample_affine_params(self._affine_conf_1, arr_B.shape)
            item_A = self._apply_img_transforms(arr_A, params_A, self._affine_conf_1)
            item_B = self._apply_img_transforms(arr_B, params_B, self._affine_conf_1)
            affine_params_B = params_B
            affine_conf_B = self._affine_conf_1

        rd_weight = None

        def _parse_ids(path):
            base = os.path.basename(path)
            m = re.match(r'^(?P<pid>.+?)\.nii_z(?P<sid>\d+)\.npy$', base)
            if m:
                return m.group('pid'), m.group('sid')
            # fallback: use name without extension; split at first underscore
            name, _ = os.path.splitext(base)
            parts = name.split('_', 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            # final fallback: treat filename stem as patient id
            return name, '0'

        pid, sid = _parse_ids(file_B)
        # Compose a stable slice_name and optionally load RD weight/mask
        slice_name = f"{pid}_{sid}" if (pid is not None and sid is not None) else os.path.splitext(os.path.basename(file_B))[0]
        rd_has_file = False
        body_mask = self._build_body_mask(item_B)
        if self.rd_input_type is not None:
            rd_has_file = rd_file_exists_for_slice(slice_name, self.rd_input_type, self.rd_mask_dir,
                                                   self.rd_weights_dir)
            tgt = item_B.detach().cpu().numpy().squeeze()
            if self.rd_cache_weights:
                w_cached = self._W_cache.get(slice_name, None)
                if w_cached is None:
                    w2d = load_weight_or_mask_for_slice(slice_name, tgt, self.rd_input_type, self.rd_mask_dir,
                                                        self.rd_weights_dir, float(self.rd_w_min),
                                                        ignore_background=self.ignore_neg1_background)
                    self._W_cache[slice_name] = w2d
                else:
                    w2d = w_cached
            else:
                w2d = load_weight_or_mask_for_slice(slice_name, tgt, self.rd_input_type, self.rd_mask_dir,
                                                    self.rd_weights_dir, float(self.rd_w_min),
                                                    ignore_background=self.ignore_neg1_background)
            rd_weight = self._apply_mask_transforms(w2d, affine_params_B, affine_conf_B)
        else:
            if self.rd_fallback_mode == 'full':
                rd_weight = torch.ones_like(body_mask)
            elif self.rd_fallback_mode == 'none':
                rd_weight = None
            else:
                rd_weight = body_mask

        sample = {'A': item_A, 'B': item_B, 'patient_id': pid, 'slice_id': sid}
        if rd_weight is not None:
            sample['rd_weight'] = rd_weight
        if body_mask is not None:
            sample['body_mask'] = body_mask
        sample['rd_has_file'] = bool(rd_has_file)
        return sample

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    # ---- helpers ----
    def _extract_affine_conf(self, transform_list):
        for t in transform_list:
            if isinstance(t, transforms.RandomAffine):
                return {
                    'degrees': t.degrees,
                    'translate': t.translate,
                    'scale': t.scale,
                    'shear': t.shear,
                    'fill': t.fill if isinstance(t.fill, (int, float)) else (t.fill[0] if t.fill else 0.0),
                    'interpolation': getattr(t, 'interpolation', InterpolationMode.NEAREST),
                }
        return None

    def _sample_affine_params(self, conf, shape):
        if conf is None:
            return self._identity_params
        H, W = int(shape[-2]), int(shape[-1])
        return transforms.RandomAffine.get_params(conf['degrees'], conf['translate'], conf['scale'], conf['shear'],
                                                  [H, W])

    def _ensure_tensor(self, arr):
        arr_np = np.asarray(arr, dtype=np.float32)
        if not arr_np.flags.writeable:
            arr_np = np.array(arr_np, copy=True)
        ten = torch.from_numpy(arr_np)
        if ten.ndim == 2:
            ten = ten.unsqueeze(0)
        elif ten.ndim == 3 and ten.shape[0] not in (1, 3):
            # assume HWC
            ten = ten.permute(2, 0, 1)
        return ten.float()

    def _build_body_mask(self, tensor, background_val=-1.0):
        if self.ignore_neg1_background:
            mask = torch.ones_like(tensor, dtype=torch.float32)
        else:
            mask = (tensor != background_val).float()
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask, _ = torch.min(mask, dim=0, keepdim=True)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask

    def _apply_img_transforms(self, arr, params, conf):
        tensor = self._ensure_tensor(arr)
        tensor = self._apply_affine(tensor, params, conf, is_mask=False)
        tensor = self._resize_op(tensor)
        return tensor

    def _apply_mask_transforms(self, w2d, params, conf):
        tensor = self._ensure_tensor(w2d)
        tensor = self._apply_affine(tensor, params, conf, is_mask=True)
        tensor = self._resize_op(tensor)
        return torch.clamp(tensor, 0.0, 1.0)

    def _apply_affine(self, tensor, params, conf, is_mask=False):
        angle, translations, scale, shear = params
        interp = InterpolationMode.NEAREST
        fill = 0.0 if is_mask else -1.0
        if conf is not None:
            interp = conf.get('interpolation', InterpolationMode.NEAREST)
            fill_val = conf.get('fill', -1.0)
            if isinstance(fill_val, (tuple, list)):
                fill_val = float(fill_val[0])
            fill = 0.0 if is_mask else float(fill_val)
        # torchvision expects Python numbers
        trans = (float(translations[0]), float(translations[1]))
        shear = tuple(float(s) for s in (shear if isinstance(shear, (tuple, list)) else (0.0, 0.0)))
        return TF.affine(tensor, angle=float(angle), translate=trans, scale=float(scale), shear=shear,
                         interpolation=interp if not is_mask else InterpolationMode.NEAREST, fill=float(fill))

    def set_rd_config(self, rd_input_type=None, rd_mask_dir='', rd_weights_dir='', rd_w_min=0.0,
                      rd_fallback_mode=None, ignore_neg1_background=None):
        self.rd_input_type = rd_input_type
        self.rd_mask_dir = rd_mask_dir or ''
        self.rd_weights_dir = rd_weights_dir or ''
        self.rd_w_min = float(rd_w_min)
        if rd_fallback_mode is not None:
            self.rd_fallback_mode = str(rd_fallback_mode).lower()
        if ignore_neg1_background is not None:
            self.ignore_neg1_background = bool(ignore_neg1_background)
        self._W_cache = {}



class ValDataset(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False, rd_input_type=None, rd_mask_dir='', rd_weights_dir='', rd_w_min=0.0,
                 cache_mode='none', rd_cache_weights=False, domain_a_dir=None, domain_b_dir=None,
                 domain_a_channels=None, domain_b_channels=None, rd_fallback_mode='body', ignore_neg1_background=False):
        transforms_ = transforms_ or []
        self._resize_op = transforms_[-1] if transforms_ else None
        if self._resize_op is None:
            raise ValueError("ValDataset expects a resize transform as the last element of transforms_")
        self.unaligned = unaligned
        self.domain_a_dir = domain_a_dir
        self.domain_b_dir = domain_b_dir
        self.domain_a_channels = int(domain_a_channels) if domain_a_channels is not None else None
        self.domain_b_channels = int(domain_b_channels) if domain_b_channels is not None else None
        self.files_A = self._resolve_files(root, domain_a_dir, 'A')
        self.files_B = self._resolve_files(root, domain_b_dir, 'B')
        # residual-detection guidance settings (optional)
        self.rd_input_type = rd_input_type
        self.rd_mask_dir = rd_mask_dir
        self.rd_weights_dir = rd_weights_dir
        self.rd_w_min = rd_w_min
        self.cache_mode = str(cache_mode).lower()
        self.rd_cache_weights = bool(rd_cache_weights)
        self.rd_fallback_mode = str(rd_fallback_mode or 'body').lower()
        self.ignore_neg1_background = bool(ignore_neg1_background)
        self._A_cache = [None] * len(self.files_A)
        self._B_cache = [None] * len(self.files_B)
        self._W_cache = {}

    def _resolve_files(self, root, override_dir, default_subdir):
        if override_dir:
            pattern = os.path.join(os.path.expanduser(override_dir), '*')
        else:
            pattern = os.path.join(root, default_subdir, '*')
        return sorted(glob.glob(pattern))

    def _load_np(self, path):
        if self.cache_mode == 'mmap':
            return np.load(path, mmap_mode='r')
        return np.load(path).astype(np.float32)

    def _load_image_file(self, path, expect_channels=None):
        with PILImage.open(path) as img:
            if expect_channels == 1:
                img = img.convert('L')
            elif expect_channels == 3:
                img = img.convert('RGB')
            else:
                img = img.convert('RGB')
            arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        else:
            arr = arr.transpose(2, 0, 1)
        arr = (arr / 255.0 * 2.0) - 1.0
        return arr.astype(np.float32)

    def _standardize_array(self, arr, expect_channels=None):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = arr.transpose(2, 0, 1)
        if expect_channels is not None and arr.shape[0] != expect_channels:
            if arr.shape[0] == 1 and expect_channels == 3:
                arr = np.repeat(arr, 3, axis=0)
            elif arr.shape[0] == 3 and expect_channels == 1:
                arr = arr[:1, ...]
            else:
                raise ValueError(f"Cannot match channels for {arr.shape} -> {expect_channels}")
        return arr

    def _load_path(self, path, expect_channels=None):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            arr = self._load_np(path)
        elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            arr = self._load_image_file(path, expect_channels)
        else:
            raise ValueError(f"Unsupported file extension '{ext}' for path '{path}'")
        return self._standardize_array(arr, expect_channels)

    def _get_array(self, which, index):
        if which == 'A':
            files = self.files_A; cache = self._A_cache; expect_channels = self.domain_a_channels
        else:
            files = self.files_B; cache = self._B_cache; expect_channels = self.domain_b_channels
        path = files[index % len(files)]
        if self.cache_mode == 'none':
            return self._load_path(path, expect_channels)
        cached = cache[index % len(files)]
        if cached is None:
            cached = self._load_path(path, expect_channels)
            cache[index % len(files)] = cached
        return cached

    def __getitem__(self, index):
        arr_A = np.asarray(self._get_array('A', index), dtype=np.float32)
        item_A = self._resize_op(self._ensure_tensor(arr_A))
        if self.unaligned:
            bidx = random.randint(0, len(self.files_B) - 1)
            arr_B = np.asarray(self._get_array('B', bidx), dtype=np.float32)
            file_B = self.files_B[bidx]
        else:
            arr_B = np.asarray(self._get_array('B', index), dtype=np.float32)
            file_B = self.files_B[index % len(self.files_B)]
        item_B = self._resize_op(self._ensure_tensor(arr_B))

        rd_weight = None

        def _parse_ids(path):
            base = os.path.basename(path)
            m = re.match(r'^(?P<pid>.+?)\.nii_z(?P<sid>\d+)\.npy$', base)
            if m:
                return m.group('pid'), m.group('sid')
            # fallback: use name without extension; split at first underscore
            name, _ = os.path.splitext(base)
            parts = name.split('_', 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            # final fallback
            return name, '0'

        pid, sid = _parse_ids(file_B)
        # Compose a stable slice_name and optionally load RD weight/mask
        slice_name = f"{pid}_{sid}" if (pid is not None and sid is not None) else os.path.splitext(os.path.basename(file_B))[0]
        rd_has_file = False
        if self.rd_input_type is not None:
            rd_has_file = rd_file_exists_for_slice(slice_name, self.rd_input_type, self.rd_mask_dir,
                                                   self.rd_weights_dir)
            tgt = item_B.detach().cpu().numpy().squeeze()
            if self.rd_cache_weights:
                w_cached = self._W_cache.get(slice_name, None)
                if w_cached is None:
                    w2d = load_weight_or_mask_for_slice(slice_name, tgt, self.rd_input_type, self.rd_mask_dir,
                                                        self.rd_weights_dir, float(self.rd_w_min),
                                                        ignore_background=self.ignore_neg1_background)
                    self._W_cache[slice_name] = w2d
                else:
                    w2d = w_cached
            else:
                w2d = load_weight_or_mask_for_slice(slice_name, tgt, self.rd_input_type, self.rd_mask_dir,
                                                    self.rd_weights_dir, float(self.rd_w_min),
                                                    ignore_background=self.ignore_neg1_background)
            rd_weight = torch.clamp(self._resize_op(self._ensure_tensor(w2d)), 0.0, 1.0)

        sample = {'A': item_A, 'B': item_B, 'patient_id': pid, 'slice_id': sid}
        if rd_weight is not None:
            sample['rd_weight'] = rd_weight
        sample['rd_has_file'] = bool(rd_has_file)
        return sample

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def _ensure_tensor(self, arr):
        arr_np = np.asarray(arr, dtype=np.float32)
        if not arr_np.flags.writeable:
            arr_np = np.array(arr_np, copy=True)
        ten = torch.from_numpy(arr_np)
        if ten.ndim == 2:
            ten = ten.unsqueeze(0)
        elif ten.ndim == 3 and ten.shape[0] not in (1, 3):
            ten = ten.permute(2, 0, 1)
        return ten.float()

    def set_rd_config(self, rd_input_type=None, rd_mask_dir='', rd_weights_dir='', rd_w_min=0.0,
                      rd_fallback_mode=None, ignore_neg1_background=None):
        self.rd_input_type = rd_input_type
        self.rd_mask_dir = rd_mask_dir or ''
        self.rd_weights_dir = rd_weights_dir or ''
        self.rd_w_min = float(rd_w_min)
        if rd_fallback_mode is not None:
            self.rd_fallback_mode = str(rd_fallback_mode).lower()
        if ignore_neg1_background is not None:
            self.ignore_neg1_background = bool(ignore_neg1_background)
        self._W_cache = {}
