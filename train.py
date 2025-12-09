import argparse
from trainer import (
    Cyc_Trainer,
    CycTrainerCross,
    CycTrainerFusion,
    CycTrainerFusionSAROPT,
    CycTrainerFusionNIRVIS,
    Nice_Trainer,
    P2p_Trainer,
    Munit_Trainer,
    Unit_Trainer,
)
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    trainer_kind = config.get('trainer')
    trainer = None
    if trainer_kind == 'cross':
        trainer = CycTrainerCross(config)
    elif trainer_kind == 'fusion':
        trainer = CycTrainerFusion(config)
    elif trainer_kind == 'fusion_saropt':
        trainer = CycTrainerFusionSAROPT(config)
    elif trainer_kind == 'fusion_nirvis':
        trainer = CycTrainerFusionNIRVIS(config)
    elif config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'Munit':
        trainer = Munit_Trainer(config)
    elif config['name'] == 'Unit':
        trainer = Unit_Trainer(config)
    elif config['name'] == 'NiceGAN':
        trainer = Nice_Trainer(config)
    elif config['name'] == 'U-gat':
        trainer = Ugat_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)
    else:
        raise ValueError(f"No trainer available for trainer='{trainer_kind}' name='{config.get('name')}'")

    trainer.train()


###################################
if __name__ == '__main__':
    main()
