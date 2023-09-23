import torch
import numpy as np
import hydra
from trainer import TrainerC2ST
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate
from torchvision import transforms

@hydra.main(config_path='configs', config_name='config.yaml')
def train_pipeline(cfg: DictConfig):
    print(cfg)


    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.project, config=wandb.config)
    # initialize data
    datagen = instantiate(cfg.data)

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize operator
    tau1_list, tau2_list = [], []
    if cfg.tau1 is not None:
        for key in cfg.tau1.keys():
            operator = instantiate(cfg.tau1[key])
            tau1_list.append(operator)
    if cfg.tau2 is not None:
        for key in cfg.tau2.keys():
            operator = instantiate(cfg.tau2[key])
            tau2_list.append(operator)

    tau1 = transforms.Compose(tau1_list)
    tau2 = transforms.Compose(tau2_list)

    # initialize network
    net = instantiate(cfg.model).to(device)
    print(net)
    wandb.watch(net)

    # initialize the trainer object and fit the network to the task
    trainer = TrainerC2ST(cfg.train, net, tau1, tau2, datagen, device, cfg.data.data_seed)
    trainer.train()


if __name__ == "__main__":
    train_pipeline()