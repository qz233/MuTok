import os
import warnings
import argparse

from trainer import Trainer
from data import get_dataloader, available_dataset
from models import Model, available_models
from utils import is_main_process
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--show-implements', action='store_true')
parser.add_argument('--config', type=str)
args = parser.parse_args()


def train(config):
    train_dataset = get_dataloader(config, "train")
    valid_dataset = get_dataloader(config, "valid")
    print('Initializing models')
    model = Model(config)
    trainer = Trainer(model, train_dataset, valid_dataset, config)
    trainer.train()

def test(config):
    test_dataset = get_dataloader(config, "test")
    print('Initializing models')
    model = Model(config)
    trainer = Trainer(model, [], test_dataset, config)
    print(trainer.evaluate())

if __name__ == "__main__":
    if args.show_implements:
        print(f"Available datasets: {available_dataset}")
        print(f"Available models: {available_models}")
        exit()
    
    config = OmegaConf.load(args.config)
    config.launch_name = f"{config.model_name}_{config.dataset_name}_{config.launch_suffix}"
    if is_main_process() and not os.path.exists(os.path.join(config.model_cache, config.launch_name)):
        os.makedirs(os.path.join(config.model_cache, config.launch_name))

    if config.type == "train":
        train(config)
    elif config.type == "test":
        test(config)
    else:
        raise ValueError()