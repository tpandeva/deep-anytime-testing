from abc import abstractmethod
from abc import ABC
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import pickle
from models import EarlyStopper



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

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """Initializes the Trainer object with the provided configurations and parameters."""
        self.data_seed = data_seed
        self.seed = cfg.seed
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.seqs = cfg.seqs
        self.patience = cfg.earlystopping.patience
        self.delta = cfg.earlystopping.delta
        self.alpha = cfg.alpha
        self.T = cfg.T
        self.tau1 = tau1
        self.tau2 = tau2
        self.net = net
        self.datagen = datagen
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)
        self.bs = cfg.batch_size
        self.save = cfg.save
        self.save_dir = cfg.save_dir

    def log(self, logs):
        """Log metrics for visualization and monitoring."""
        for key, value in logs.items():
            wandb.log({key: value})
            logging.info(f"Progress {key}: {value}")

    def train_evaluate_epoch(self, loader, mode="train"):
        """Train/Evaluate the model for one epocj and log the results."""
        aggregated_loss = 0
        mmde_mult, mmde_mean = 1,0
        num_samples = len(loader.dataset)
        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            tau_z = tau_z.to(self.device)
            if mode == "train":
                out = self.net(z,tau_z)
            else:
                out = self.net(z, tau_z).detach()
            loss = -out.mean()
            aggregated_loss +=-out.sum()
            mmde_mult *= torch.exp(out.sum())
            mmde_mean += torch.exp(out).sum()/num_samples
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.log({f"{mode}_eval_mult": mmde_mult.item(),f"{mode}_eval_log": -aggregated_loss.item(),f"{mode}_eval_mean": mmde_mean.item(), f"{mode}_loss": aggregated_loss.item()/num_samples})
        return aggregated_loss/num_samples, mmde_mult, mmde_mean, -aggregated_loss

    def load_data(self, seed, mode= "train"):
        """Load data using the datagen object and return a DataLoader object."""
        data = self.datagen.generate(seed, self.tau1, self.tau2)
        if mode in ["train", "val"]:
            data_loader = DataLoader(data, batch_size=self.bs, shuffle=True)
        else: data_loader = DataLoader(data, batch_size=len(data), shuffle=True)
        return data, data_loader

    def train(self):
        """Train the model for a specified number of sequences, epochs, and apply early stopping if required."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_data, train_loader = self.load_data(self.seed, mode = "train")
        val_data, val_loader = self.load_data(self.seed + 1, mode = "val")
        mmdes_mult, mmdes_mean, log_mmdes = [], [], []
        for k in range(self.seqs):

            for t in range(self.epochs):
                self.train_evaluate_epoch(train_loader)
                loss_val, _, _,_ = self.train_evaluate_epoch(val_loader, mode='val')
                if self.early_stopper.early_stop(loss_val.detach()) or (t + 1) == self.epochs:
                    test_data, test_loader = self.load_data(self.seed + k + 2, mode = "test")
                    _, mmde_mult_conditional, mmde_mean_conditional, log_mmde = self.train_evaluate_epoch(test_loader, mode='test')
                    mmdes_mult.append(mmde_mult_conditional.item())
                    mmdes_mean.append(mmde_mean_conditional.item())
                    log_mmdes.append(log_mmde.item())
                    mmde_mult = np.prod(np.array(mmdes_mult[self.T:])) if k >= self.T else 1
                    mmde_mean = np.prod(np.array(mmdes_mean[self.T:])) if k >= self.T else 1
                    log_mmdes_a = np.sum(np.array(log_mmdes[self.T:])) if k >= self.T else 0
                    self.log({"aggregated_test_eval_mult": mmde_mult, "aggregated_test_eval_mean": mmde_mean, "aggregated_test_eval_log": log_mmdes_a})
                    train_data = ConcatDataset([train_data, val_data])
                    val_data = test_data
                    train_loader = DataLoader(train_data, batch_size=self.bs, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=self.bs, shuffle=True)
                    self.log({"iterations": t})
                    break
            self.early_stopper.reset()
            if mmde_mult > (1. / self.alpha):
                logging.info("Reject null at %f", mmde_mult)
                self.log({"steps_mult": k})
            if mmde_mean > (1. / self.alpha):
                logging.info("Reject null at %f", mmde_mean)
                self.log({"steps_mean": k})
            if log_mmdes_a > np.log(1. / self.alpha):
                logging.info("Reject null at %f", log_mmdes_a)
                self.log({"steps_log": k})

        if self.save:
            import os
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            pickle.dump([mmde_mult, mmde_mean], open(self.save_dir + f"mmdes_{self.data_seed}.pkl", "wb"))



