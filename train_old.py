import torch
import numpy as np
import hydra
import logging
import pickle
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import wandb
from operators import RotateOperator

from data import BlobData
from models import MMDEMLP, EarlyStopper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path='configs', config_name='blob.yaml')
def train(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project="MMDE-seq", config = wandb.config)
    type = cfg.data.type
    samples = cfg.data.samples
    input_size = cfg.model.input_size
    hidden_layer_size = cfg.model.hidden_layer_size
    output_size = cfg.model.output_size
    batch_norm = cfg.model.batch_norm
    drop_out = cfg.model.drop_out.flag
    p = cfg.model.drop_out.p
    lr = cfg.train.lr
    epochs = cfg.train.epochs
    seqs = cfg.train.seqs
    bootstraps = cfg.train.bootstraps
    stats = np.zeros((bootstraps, seqs))
    operator = RotateOperator(p=cfg.operator.p)
    for seed in range(bootstraps):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = MMDEMLP(input_size, hidden_layer_size, output_size, batch_norm, drop_out, p).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        val_data = BlobData(type, samples, seed + 123456789)
        train_data = BlobData(type, samples, seed * bootstraps + 123456789)
        mmdes = []
        for k in range(seqs):
            early_stopper = EarlyStopper(patience=cfg.train.patience, min_delta=cfg.train.delta)
            val_loader = DataLoader(val_data, batch_size=samples, shuffle=True)
            train_loader = DataLoader(train_data, batch_size=samples, shuffle=True)

            for t in range(epochs):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    z = torch.stack([x,y], dim=2)
                    tau_z = operator.compute(z)
                    z = torch.transpose(z, 1, 2).reshape(-1, 2 * operator.p)
                    tau_z = torch.transpose(tau_z, 1, 2).reshape(-1, 2 * operator.p)
                    out = net(z, tau_z)
                    loss = -out.mean()
                    mmde_train = torch.exp(out.sum()/2)
                    wandb.log({"train_eval": mmde_train})
                    wandb.log({"train_loss": loss})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    z = torch.concat((x, y), dim=1)
                    tau_z = torch.concat((y, x), dim=1)
                    out = net(z, tau_z).detach()

                    loss_val = -out.mean()
                    mmde_val = torch.exp(out.sum()/2)
                    logging.info(f'Batch: {k} Interation: {t},  train_loss: {loss.item()}, train_mmde: {mmde_train.item()}, '
                                 f'val_loss {loss_val.item()}, val_mmde {mmde_val.item()}')
                    wandb.log({"val_eval": mmde_val})
                    wandb.log({"val_loss": loss_val})
                if early_stopper.early_stop(loss_val.detach()) or (t+1)==epochs:
                    test_data = BlobData(type, samples, seed * bootstraps + k + 2 + 123456789)
                    test_loader = DataLoader(test_data, batch_size=samples, shuffle=True)
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        z = torch.concat((x, y), dim=1)
                        tau_z = torch.concat((y, x), dim=1)
                        out = net(z, tau_z).detach()
                        mmde_conditional = torch.exp(out.sum()/2)
                    mmdes.append(mmde_conditional.item())
                    stats[seed, k] = mmde_conditional.item()

                    mmde = np.prod(np.array(mmdes[cfg.T:])) if k>cfg.T else 1
                    logging.info(f"Batch: {k} Interation: {t} test conditional mmde {mmde_conditional.item()}, test_eval,{mmde}")
                    wandb.log({"test_eval": mmde_conditional})
                    wandb.log({"batch": k})
                    train_data = torch.utils.data.ConcatDataset([train_data, val_data])
                    val_data = test_data
                    break

            if mmde > (1./cfg.alpha):
                logging.info("Reject null at %f", mmde)
                wandb.log({"steps": k})
                break

    with open('results_blob.pickle', "wb") as open_file:
        pickle.dump(stats, open_file)


if __name__ == "__main__":
    train()
