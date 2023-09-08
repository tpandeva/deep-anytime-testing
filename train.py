import torch
import numpy as np
import hydra
from trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

@hydra.main(config_path='configs', config_name='config.yaml')
def train_pipeline(cfg: DictConfig):
    print(cfg)
    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.project, config=wandb.config)
    # initialize data
    datagen = instantiate(cfg.data)

    # initialize operator
    operator = instantiate(cfg.operator)

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize network
    net = instantiate(cfg.model)
    print(net)
    wandb.watch(net)

    # initialize the trainer object and fit the network to the task
    trainer = Trainer(cfg.train, net, operator, datagen, device, cfg.seed)
    trainer.train()


if __name__ == "__main__":
    train_pipeline()