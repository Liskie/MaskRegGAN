import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import re


class ImageDataset(Dataset):
    def __init__(self, root, noise_level, count=None, transforms_1=None, transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level = noise_level

    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            file_A = self.files_A[index % len(self.files_A)]
            item_A = self.transform2(np.load(file_A).astype(np.float32))

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            file_B = self.files_B[index % len(self.files_B)]
            item_B = self.transform2(np.load(file_B).astype(np.float32))
        else:
            # if noise !=0, A and B make different transform
            file_A = self.files_A[index % len(self.files_A)]
            file_B = self.files_B[index % len(self.files_B)]
            item_A = self.transform1(np.load(file_A).astype(np.float32))
            item_B = self.transform1(np.load(file_B).astype(np.float32))

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
            # final fallback: return None, None
            return None, None

        pid, sid = _parse_ids(file_B)
        return {'A': item_A, 'B': item_B, 'patient_id': pid, 'slice_id': sid}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        file_A = self.files_A[index % len(self.files_A)]
        item_A = self.transform(np.load(file_A).astype(np.float32))
        if self.unaligned:
            bidx = random.randint(0, len(self.files_B) - 1)
            file_B = self.files_B[bidx]
            item_B = self.transform(np.load(file_B).astype(np.float32))
        else:
            file_B = self.files_B[index % len(self.files_B)]
            item_B = self.transform(np.load(file_B).astype(np.float32))

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
            # final fallback: return None, None
            return None, None

        pid, sid = _parse_ids(file_B)
        return {'A': item_A, 'B': item_B, 'patient_id': pid, 'slice_id': sid}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
