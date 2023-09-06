import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from models import EarlyStopper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self,cfg, net, operator, datagen,device, seed):
        self.seed = seed*10000
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.seqs = cfg.seqs
        self.patience = cfg.patience
        self.delta = cfg.delta
        self.alpha = cfg.alpha
        self.T = cfg.T
        self.operator = operator
        self.net = net
        self.datagen = datagen
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)
        self.bs = cfg.batch_size


    def log(self, logs):
        for key, value in logs.items():
            wandb.log({key: value})
            logging.info(f"Progress {key}: {value}")

    def train_batch(self,  train_loader):
        for z in train_loader:
            z = z.to(self.device)
            tau_z = self.operator.compute(z)
            z = torch.transpose(z, 1, 2).reshape(-1, 2 * self.operator.p)
            tau_z = torch.transpose(tau_z, 1, 2).reshape(-1, 2 * self.operator.p)

            out = self.net(z, tau_z)
            loss = -out.mean()
            mmde_train = torch.exp(out.sum() / 2)
            self.log({"train_eval": mmde_train.item(), "train_loss": loss.item()})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, mmde_train

    def evaluate(self, loader, mode='val'):
        for z in loader:
            z = z.to(self.device)
            tau_z = self.operator.compute(z)
            z = torch.transpose(z, 1, 2).reshape(-1, 2 * self.operator.p)
            tau_z = torch.transpose(tau_z, 1, 2).reshape(-1, 2 * self.operator.p)
            out = self.net(z, tau_z).detach()
            loss = -out.mean()
            mmde = torch.exp(out.sum() / 2)
            self.log({f"{mode}_eval": mmde.item(), f"{mode}_loss": loss.item()})
        return mmde, loss

    def load_data(self, seed):
        data = self.datagen.generate(seed)
        data_loader = DataLoader(data, batch_size=self.bs, shuffle=True)
        return data, data_loader

    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_data, train_loader = self.load_data(self.seed)
        val_data, val_loader = self.load_data(self.seed + 1)
        mmdes = []
        for k in range(self.seqs):
            train_loader = DataLoader(train_data, batch_size=self.bs, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.bs, shuffle=True)
            for t in range(self.epochs):
                self.train_batch(train_loader)

                _, loss_val = self.evaluate(val_loader, mode='val')

                if self.early_stopper.early_stop(loss_val.detach()) or (t + 1) == self.epochs:
                    test_data, test_loader = self.load_data(self.seed + k + 2)
                    mmde_conditional,_ = self.evaluate(test_loader, mode='test')
                    mmdes.append(mmde_conditional.item())
                    mmde = np.prod(np.array(mmdes[self.T:])) if k > self.T else 1
                    self.log({"aggregated_test_eval":mmde})
                    train_data = ConcatDataset([train_data, val_data])
                    val_data = test_data
                    self.log({"iterations":t})
                    break

            self.early_stopper.reset()

            if mmde > (1. / self.alpha):
                logging.info("Reject null at %f", mmde)
                self.log({"steps": k})
                return k




