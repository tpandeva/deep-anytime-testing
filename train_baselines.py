import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader
from baselines.mmd_test import mmd_test_rbf
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

    if cfg.train.name=="deep":
        net = instantiate(cfg.model).to(device)
        print(net)
        wandb.watch(net)
        # initialize the trainer object and fit the network to the task
        trainer = TrainerC2ST(cfg.train, net, tau1, tau2, datagen, device, cfg.data.data_seed)
        trainer.train()
    elif cfg.train.name=="mmd":
        # load model
        x_all, y_all = None, None
        for r in range(cfg.train.seqs):
            data = datagen.generate(r+1, tau1, tau2)
            data_loader = DataLoader(data, batch_size = len(data), shuffle=True)
            for i, (x, y) in enumerate(data_loader):
                x_all = x if x_all is None else torch.cat((x_all, x), dim=0)
                y_all = y if y_all is None else torch.cat((y_all, y), dim=0)
                p_val = mmd_test_rbf(x_all, y_all, int(np.sqrt(len(x_all))))
                wandb.log({"p_val": p_val, "running_seed": 100*(cfg.data.data_seed+1)+r+1}) # (self.data_seed+1)*100+seed
                print(f"Batch {i}, p_val: {p_val}")



if __name__ == "__main__":
    train_pipeline()