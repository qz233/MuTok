import os
import argparse

from trainer import Trainer
from data import get_drum_dataloader
from model import DrumRFDiT, XCodecWrapper
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--show-implements', action='store_true')
parser.add_argument('--config', type=str)
args = parser.parse_args()


def train(config):
    train_dataset = get_drum_dataloader(config, "train")
    valid_dataset = get_drum_dataloader(config, "valid")
    print('Initializing models')
    model = DrumRFDiT(config)
    codec = XCodecWrapper(config).requires_grad_(False)
    trainer = Trainer(model, codec, train_dataset, valid_dataset, config)
    trainer.train()


if __name__ == "__main__":    
    config = OmegaConf.load(args.config)
    if not os.path.exists(os.path.join(config.model_cache, config.launch_name)):
        os.makedirs(os.path.join(config.model_cache, config.launch_name))

    train(config)
