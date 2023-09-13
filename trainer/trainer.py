import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from models import EarlyStopper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Trainer class encapsulates the training, evaluation, and logging processes for a neural network.

    Attributes:
        Several attributes are initialized from the configuration object, `cfg`, such as learning rate (`lr`),
        number of epochs (`epochs`), patience for early stopping (`patience`), and others.
        net: Neural network model to be trained.
        operator: Operator used to compute transformations on input data.
        datagen: Data generator used to produce training, validation, and test datasets.
        device: Device (CPU/GPU) on which computations will be performed.
    """

    def __init__(self, cfg, net, operator, datagen, device, seed):
        """Initializes the Trainer object with the provided configurations and parameters."""
        self.seed = seed * 10000
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.seqs = cfg.seqs
        self.patience = cfg.earlystopping.patience
        self.delta = cfg.earlystopping.delta
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
        """Log metrics for visualization and monitoring."""
        for key, value in logs.items():
            wandb.log({key: value})
            logging.info(f"Progress {key}: {value}")

    def train_evaluate_epoch(self, loader, mode="train"):
        """Train/Evaluate the model for one epocj and log the results."""
        for z in loader:
            z = z.to(self.device)
            tau_z = self.operator.compute(z)
            z = torch.transpose(z, 1, 2).reshape(-1, 2 * self.operator.p)
            tau_z = torch.transpose(tau_z, 1, 2).reshape(-1, 2 * self.operator.p)
            if mode == "train":
                out = self.net(z, tau_z)
            else:
                out = self.net(z, tau_z).detach()
            loss = -out.mean()
            mmde = torch.exp(out.sum())
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.log({f"{mode}_eval": mmde.item(), f"{mode}_loss": loss.item()})
        return loss, mmde

    def load_data(self, seed):
        """Load data using the datagen object and return a DataLoader object."""
        data = self.datagen.generate(seed)
        data_loader = DataLoader(data, batch_size=self.bs, shuffle=True)
        return data, data_loader

    def train(self):
        """Train the model for a specified number of sequences, epochs, and apply early stopping if required."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_data, train_loader = self.load_data(self.seed)
        val_data, val_loader = self.load_data(self.seed + 1)
        mmdes = []
        for k in range(self.seqs):
            train_loader = DataLoader(train_data, batch_size=self.bs, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.bs, shuffle=True)
            for t in range(self.epochs):
                self.train_evaluate_epoch(train_loader)
                loss_val, _ = self.train_evaluate_epoch(val_loader, mode='val')
                if self.early_stopper.early_stop(loss_val.detach()) or (t + 1) == self.epochs:
                    test_data, test_loader = self.load_data(self.seed + k + 2)
                    _, mmde_conditional = self.train_evaluate_epoch(test_loader, mode='test')
                    mmdes.append(mmde_conditional.item())
                    mmde = np.prod(np.array(mmdes[self.T:])) if k >= self.T else 1
                    self.log({"aggregated_test_eval": mmde})
                    train_data = ConcatDataset([train_data, val_data])
                    val_data = test_data
                    self.log({"iterations": t})
                    break
            self.early_stopper.reset()
            if mmde > (1. / self.alpha):
                logging.info("Reject null at %f", mmde)
                self.log({"steps": k})
                return k



