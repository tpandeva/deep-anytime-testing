import torch
import numpy as np
import hydra
from trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import wandb
from operators import RotateOperator
from data import BlobDataGen
from models import MMDEMLP

@hydra.main(config_path='configs', config_name='blob.yaml')
def train_pipeline(cfg: DictConfig):
    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project="MMDE-seq", config=wandb.config)
    # initialize data
    datagen = BlobDataGen(cfg.data)

    # initialize operator
    if cfg.operator.type == "rotation":
        operator = RotateOperator(cfg.operator.p, cfg.operator.d)
    else:
        raise NotImplementedError
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initialize network
    net = MMDEMLP(cfg.model).to(device)

    trainer = Trainer(cfg.train, net, operator, datagen,device,cfg.seed)
    trainer.train()


if __name__ == "__main__":
    train_pipeline()