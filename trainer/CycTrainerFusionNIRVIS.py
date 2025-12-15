import copy
import os

from .CycTrainerFusion import CycTrainerFusion


class CycTrainerFusionNIRVIS(CycTrainerFusion):
    """Fusion trainer specialization for the VIPL-MumoFace NIR↔VIS dataset."""

    def __init__(self, config):
        cfg = copy.deepcopy(config)
        root = os.path.expanduser(cfg.get('nirvis_root', cfg.get('dataroot', '')))
        if not root:
            raise ValueError("nirvis_root or dataroot must be provided for CycTrainerFusionNIRVIS.")
        train_split = cfg.get('nirvis_train_split', 'train')
        val_split = cfg.get('nirvis_val_split', 'test')
        nir_dir_name = cfg.get('nirvis_nir_dir', 'NIR')
        vis_dir_name = cfg.get('nirvis_vis_dir', 'RGB')
        direction = str(cfg.get('nirvis_direction', 'nir2vis')).lower()

        train_root = os.path.join(root, train_split)
        val_root = os.path.join(root, val_split)
        nir_train_dir = os.path.join(train_root, nir_dir_name)
        vis_train_dir = os.path.join(train_root, vis_dir_name)
        nir_val_dir = os.path.join(val_root, nir_dir_name)
        vis_val_dir = os.path.join(val_root, vis_dir_name)

        if not os.path.isdir(nir_train_dir) or not os.path.isdir(vis_train_dir):
            raise FileNotFoundError(
                f"NIR/VIS train directories not found: nir='{nir_train_dir}', vis='{vis_train_dir}'"
            )

        cfg.setdefault('rd_mode', 'none')
        cfg.setdefault('dataroot', train_root)
        cfg.setdefault('val_dataroot', val_root)

        if direction == 'nir2vis':
            cfg.setdefault('input_nc', 1)
            cfg.setdefault('output_nc', 3)
            cfg.setdefault('domain_a_dir', nir_train_dir)
            cfg.setdefault('domain_b_dir', vis_train_dir)
            cfg.setdefault('val_domain_a_dir', nir_val_dir)
            cfg.setdefault('val_domain_b_dir', vis_val_dir)
            cfg.setdefault('domain_a_channels', 1)
            cfg.setdefault('domain_b_channels', 3)
            cfg.setdefault('cv_domain_a_subdir', nir_dir_name)
            cfg.setdefault('cv_domain_b_subdir', vis_dir_name)
        elif direction == 'vis2nir':
            cfg.setdefault('input_nc', 3)
            cfg.setdefault('output_nc', 1)
            cfg.setdefault('domain_a_dir', vis_train_dir)
            cfg.setdefault('domain_b_dir', nir_train_dir)
            cfg.setdefault('val_domain_a_dir', vis_val_dir)
            cfg.setdefault('val_domain_b_dir', nir_val_dir)
            cfg.setdefault('domain_a_channels', 3)
            cfg.setdefault('domain_b_channels', 1)
            cfg.setdefault('cv_domain_a_subdir', vis_dir_name)
            cfg.setdefault('cv_domain_b_subdir', nir_dir_name)
        else:
            raise ValueError("nirvis_direction must be 'nir2vis' or 'vis2nir'")

        cfg.setdefault('cv_train_subdir', 'train')
        cfg.setdefault('cv_val_subdir', 'val')
        cfg.setdefault('cv_root', cfg.get('nirvis_cv_root', os.path.join(root, 'cv_folds')))

        super().__init__(cfg)
