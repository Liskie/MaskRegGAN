#!/usr/bin/python3

import argparse
from trainer import Cyc_Trainer, P2p_Trainer, Nice_Trainer
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yaml/P2p.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'Munit':
        trainer = Munit_Trainer(config)
    elif config['name'] == 'Unit':
        trainer = Unit_Trainer(config)
    elif config['name'] == 'Nice':
        trainer = Nice_Trainer(config)
    elif config['name'] == 'U-gat':
        trainer = Ugat_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)

    trainer.test()


###################################
if __name__ == '__main__':
    main()
