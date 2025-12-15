import copy
import os

from .CycTrainerFusion import CycTrainerFusion


class CycTrainerFusionSAROPT(CycTrainerFusion):
    """
    Specialization of CycTrainerFusion for the SAROPT RGB↔SAR dataset.
    Auto-configures domain directories/channels for PNG inputs so we can train RGB→SAR or SAR→RGB
    without duplicating data into A/B subfolders.
    """

    def __init__(self, config):
        cfg = copy.deepcopy(config)
        root = os.path.expanduser(cfg.get('saropt_root', cfg.get('dataroot', '')))
        if not root:
            raise ValueError("saropt_root or dataroot must be provided for CycTrainerFusionSAROPT.")
        opt_dir_name = cfg.get('saropt_opt_dir', 'opt_256_oc_0.2')
        sar_dir_name = cfg.get('saropt_sar_dir', 'sar_256_oc_0.2')
        direction = str(cfg.get('saropt_direction', 'rgb2sar')).lower()

        opt_dir = os.path.join(root, opt_dir_name)
        sar_dir = os.path.join(root, sar_dir_name)
        if not os.path.isdir(opt_dir) or not os.path.isdir(sar_dir):
            raise FileNotFoundError(f"SAROPT directories not found: opt='{opt_dir}', sar='{sar_dir}'")

        cfg.setdefault('rd_mode', 'none')
        cfg.setdefault('dataroot', root)
        if 'val_dataroot' not in cfg or not cfg['val_dataroot']:
            cfg['val_dataroot'] = root

        if direction == 'rgb2sar':
            cfg.setdefault('input_nc', 3)
            cfg.setdefault('output_nc', 1)
            cfg.setdefault('domain_a_dir', opt_dir)
            cfg.setdefault('domain_b_dir', sar_dir)
        elif direction == 'sar2rgb':
            cfg.setdefault('input_nc', 1)
            cfg.setdefault('output_nc', 3)
            cfg.setdefault('domain_a_dir', sar_dir)
            cfg.setdefault('domain_b_dir', opt_dir)
        else:
            raise ValueError("saropt_direction must be 'rgb2sar' or 'sar2rgb'")

        cfg.setdefault('domain_a_channels', cfg.get('input_nc'))
        cfg.setdefault('domain_b_channels', cfg.get('output_nc'))
        cfg.setdefault('cv_train_subdir', 'train')
        cfg.setdefault('cv_val_subdir', 'val')
        if direction == 'rgb2sar':
            cfg.setdefault('cv_domain_a_subdir', 'opt')
            cfg.setdefault('cv_domain_b_subdir', 'sar')
        else:
            cfg.setdefault('cv_domain_a_subdir', 'sar')
            cfg.setdefault('cv_domain_b_subdir', 'opt')

        val_root = os.path.expanduser(cfg.get('saropt_val_root', cfg.get('val_dataroot', root)))
        val_opt_dir_name = cfg.get('saropt_val_opt_dir', opt_dir_name)
        val_sar_dir_name = cfg.get('saropt_val_sar_dir', sar_dir_name)
        val_opt_dir = os.path.join(val_root, val_opt_dir_name)
        val_sar_dir = os.path.join(val_root, val_sar_dir_name)

        if direction == 'rgb2sar':
            cfg.setdefault('val_domain_a_dir', val_opt_dir)
            cfg.setdefault('val_domain_b_dir', val_sar_dir)
        else:
            cfg.setdefault('val_domain_a_dir', val_sar_dir)
            cfg.setdefault('val_domain_b_dir', val_opt_dir)

        super().__init__(cfg)
