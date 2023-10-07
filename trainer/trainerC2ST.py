from trainer import Trainer
import numpy as np
import torch
import time

class TrainerC2ST(Trainer):
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
        super().__init__(cfg, net, tau1, tau2, datagen, device, data_seed)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.opt_lmbd = 0
        self.run_mean = 0
        self.grad_sq_sum = 1
        self.truncation_level = 0.5

    def testing_by_betting(self, y, logits):
        w = 2 * y - 1
        f = torch.nn.Softmax()
        ft = 2 * f(logits)[:, 1] - 1
        e_val = torch.exp(torch.sum(torch.log(1 + w * ft)))
        n_samples = y.shape[0]
        payoffs = w * ft
        e_val_ons = 1
        # for i in range(n_samples):

        grad = self.run_mean / (1 + self.run_mean * self.opt_lmbd)
        self.grad_sq_sum += grad ** 2
        self.opt_lmbd = max(0, min(
            self.truncation_level, self.opt_lmbd + 2 / (2 - np.log(3)) * grad / self.grad_sq_sum))
        e_val_ons = torch.exp(torch.log(1 + self.opt_lmbd * payoffs.sum()))
        self.run_mean = payoffs.mean()
        return e_val, e_val_ons

    def e_c2st(self, y, logits):

        # H0
        # p(y|x) of MLE under H0: p(y|x) = p(y), is just the empirical frequency of y in the test data.
        emp_freq_class0 = 1 - (y[y == 1]).sum() / y.shape[0]
        emp_freq_class1 = (y[y == 1]).sum() / y.shape[0]

        # H1
        # Probabilities under empirical model (using train data)
        f = torch.nn.Softmax()
        prob = f(logits)
        pred_prob_class0 = prob[:, 0]
        pred_prob_class1 = prob[:, 1]
        log_eval = torch.sum(y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(
            pred_prob_class0 / emp_freq_class0)).double()
        eval = torch.exp(log_eval)
        # E-value
        return eval

    def first_k_unique_permutations(self, n, k):
        if np.log(k) > n * (np.log(n) - 1) + 0.5 * (np.log(2 * np.pi * n)): k = n
        unique_perms = set()
        while len(unique_perms) < k:
            unique_perms.add(tuple(np.random.choice(n, n, replace=False)))
        return list(unique_perms), k

    def s_c2st(self, y, logits, n_per=100):
        y_hat = torch.argmax(logits, dim=1)
        n = y.shape[0]
        accuracy = torch.sum(y == y_hat) / n
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            # divide into new X, Y
            y_perm = y.clone()[ind]
            # compute accuracy
            stats[r] = torch.sum(y_perm == y_hat) / y.shape[0]
        sorted_stats = np.sort(stats)
        p_val = np.sum(sorted_stats > accuracy.item()) / n_per
        return p_val, accuracy

    def train_evaluate_epoch(self, loader, mode="train"):
        """Train/Evaluate the model for one epocj and log the results."""
        aggregated_loss = 0
        e_val, tb_val, tb_val_ons = 1, 1, 1
        num_samples = len(loader.dataset)
        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            # z = z.transpose(2, 1).flatten(0).view(2*samples,-1)[...,:-1]
            tau_z = tau_z.to(self.device)
            # tau_z = tau_z.transpose(2, 1).flatten(0).view(2*samples,-1)[...,-1]
            if mode == "train":
                self.net = self.net.train()
                out1 = self.net(z)
                out2 = self.net(tau_z)
            else:
                self.net = self.net.eval()
                out1 = self.net(z)
                out2 = self.net(tau_z)
            out = torch.concat((out1, out2))
            labels = torch.concat((torch.ones((z.shape[0], 1)), torch.zeros((z.shape[0], 1)))).squeeze(1).long().to(
                self.device)
            loss = self.loss(out, labels)
            aggregated_loss += loss
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # compute e-c2st and s-c2st
            if mode == "test":
                start = time.time()
                e_val *= self.e_c2st(labels, out.detach())
                end = time.time()
                ec2st_time = end - start

                start = time.time()
                p_val, acc = self.s_c2st(labels, out.detach(), n_per=100)
                end = time.time()
                sc2st_time = end - start

                start = time.time()
                results_tb = self.testing_by_betting(labels, out.detach())
                tb_val *= results_tb[0]
                tb_val_ons *= results_tb[1]
                end = time.time()
                tbons_time = end - start
                self.log({"ec2st_time": ec2st_time, "sc2st_time": sc2st_time, "tbons_time": tbons_time})
                self.log(
                    {f"{mode}_e-value": e_val.item(), f"{mode}_p-value": p_val.item(),
                     f"{mode}_tb-value": tb_val.item(), f"{mode}_tb-ons-value": tb_val_ons.item(),
                    })
            self.log({ f"{mode}_loss": aggregated_loss.item() / num_samples, f"{mode}_accuracy": acc.item() })
        return aggregated_loss / num_samples, tb_val_ons

class TrainerSC2ST(TrainerC2ST):
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
        super().__init__(cfg, net[0], tau1, tau2, datagen, device, data_seed)
        param = []
        self.net = net
        for i in range(cfg.ps):
            param = param + list(self.net[i].parameters())
        self.optimizer = torch.optim.Adam(param, lr=cfg.train.lr)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def s_c2st(self, y, logits, n_per=100):
        y_hat = torch.argmax(logits, dim=1)
        n = y.shape[0]
        accuracy = torch.sum(y == y_hat) / n
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            # divide into new X, Y
            y_perm = y.clone()[ind]
            # compute accuracy
            stats[r] = torch.sum(y_perm == y_hat) / y.shape[0]
        sorted_stats = np.sort(stats)
        p_val = np.sum(sorted_stats > accuracy.item()) / n_per
        return p_val, accuracy

    def train_evaluate_epoch(self, loader, mode="train"):
        """Train/Evaluate the model for one epocj and log the results."""
        aggregated_loss = 0
        e_val, tb_val, tb_val_ons = 1, 1, 1
        num_samples = len(loader.dataset)
        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            # z = z.transpose(2, 1).flatten(0).view(2*samples,-1)[...,:-1]
            tau_z = tau_z.to(self.device)
            # tau_z = tau_z.transpose(2, 1).flatten(0).view(2*samples,-1)[...,-1]
            num_pseudo_samples = tau_z.shape[0]
            loss = 0
            for i in range(num_pseudo_samples):
                self.net[i] = self.net[i].train() if mode == "train" else self.net[i].eval()
                out1 = self.net[i](z)
                out2 = self.net[i](tau_z[...,i])
                out = torch.concat((out1, out2))
                labels = torch.concat((torch.ones((z.shape[0], 1)), torch.zeros((z.shape[0], 1)))).squeeze(1).long().to(
                self.device)
                loss += self.loss(out.clone(), labels)
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # compute e-c2st and s-c2st
                aggregated_loss += loss
                if mode == "test":
                    p_val, acc = self.s_c2st(labels, out.detach(), n_per=100)
                    self.log(
                        {f"{mode}_e-value": e_val.item(), f"{mode}_p-value_{i}": p_val.item(),
                         f"{mode}_accuracy": acc.item()
                        })
            self.log({ f"{mode}_loss": aggregated_loss.item() / num_samples })
        return aggregated_loss / (num_samples*num_pseudo_samples), tb_val_ons


