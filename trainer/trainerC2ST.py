from trainer import Trainer
import numpy as np
import torch

class TrainerC2ST(Trainer):

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """
        Initializes the TrainerC2ST object by extending the Trainer class.

        Args:
        (same as Trainer class)
        """
        super().__init__(cfg, net, tau1, tau2, datagen, device, data_seed)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.opt_lmbd = 0  # betting strategy parameter
        self.run_mean = 0  # Running mean of payoffs
        self.grad_sq_sum = 1  # Sum of squares of gradients
        self.truncation_level = 0.5  # Level for truncating optimization

    def testing_by_betting(self, y, logits):
        """
        Implement the ONS betting strategy.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.

        Returns:
        - tuple: E-value and E-value using ONS betting strategy.
        """
        # Calculate the weights and transformed probabilities using softmax
        w = 2 * y - 1
        f = torch.nn.Softmax(dim=1)
        ft = 2 * f(logits)[:, 1] - 1
        e_val = torch.exp(torch.sum(torch.log(1 + w * ft)))
        payoffs = w * ft

        # Update optimization parameters using gradient information
        grad = self.run_mean / (1 + self.run_mean * self.opt_lmbd)
        self.grad_sq_sum += grad ** 2
        self.opt_lmbd = max(0, min(
            self.truncation_level, self.opt_lmbd + 2 / (2 - np.log(3)) * grad / self.grad_sq_sum))
        e_val_ons = torch.exp(torch.log(1 + self.opt_lmbd * payoffs.sum()))
        self.run_mean = payoffs.mean()

        return e_val, e_val_ons

    def e_c2st(self, y, logits):
        """
        Evaluate the E-C2ST for given targets and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.

        Returns:
        - torch.Tensor: Evaluated E-value.
        """
        # H0: Empirical frequencies
        emp_freq_class0 = 1 - (y[y == 1]).sum() / y.shape[0]
        emp_freq_class1 = (y[y == 1]).sum() / y.shape[0]

        # H1: Probabilities under empirical model (using train data)
        f = torch.nn.Softmax(dim=1)
        prob = f(logits)
        pred_prob_class0 = prob[:, 0]
        pred_prob_class1 = prob[:, 1]
        log_eval = torch.sum(y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(
            pred_prob_class0 / emp_freq_class0)).double()
        eval = torch.exp(log_eval)

        return eval

    def first_k_unique_permutations(self, n, k):
        """
        Generate the first k unique permutations of range(n).

        Args:
        - n (int): Size of the set.
        - k (int): Number of unique permutations required.

        Returns:
        - list: List of k unique permutations.
        """
        if np.log(k) > n * (np.log(n) - 1) + 0.5 * (np.log(2 * np.pi * n)):
            k = n
        unique_perms = set()
        while len(unique_perms) < k:
            unique_perms.add(tuple(np.random.choice(n, n, replace=False)))
        return list(unique_perms), k

    def s_c2st(self, y, logits, n_per=100):
        """
        Evaluate the permutation-based Two-Sample Test (TST) for given labels and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.
        - n_per (int, optional): Number of permutations. Default is 100.

        Returns:
        - tuple: p-value and accuracy.
        """
        y_hat = torch.argmax(logits, dim=1)
        n = y.shape[0]
        accuracy = torch.sum(y == y_hat) / n
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            y_perm = y.clone()[ind]
            stats[r] = torch.sum(y_perm == y_hat) / y.shape[0]
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= accuracy.item())+1) / (n_per+1)

        return p_val, accuracy
    def l_c2st(self, y, logits, n_per=100):
        """
        Evaluate the permutation-based Two-Sample Test (TST) for given labels and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.
        - n_per (int, optional): Number of permutations. Default is 100.

        Returns:
        - tuple: p-value and accuracy.
        """
        y_hat = torch.argmax(logits, dim=1)
        logit = logits[:,1] - logits[:,0]
        n= y.shape[0]
        true_stat = logit[y == 1].mean() - logit[y == 0].mean()
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            logit_perm = logit.clone()[ind]
            stats[r] = logit_perm[y == 1].mean() - logit_perm[y == 0].mean()
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= true_stat.item())+1) / (n_per+1)

        return p_val

    def train_evaluate_epoch(self, loader, mode="train"):
        """
        Train/Evaluate the model for one epoch using the C2ST approach.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train" or "test". Determines how to run the model.

        Returns:
        - tuple: Aggregated loss and E-value using ONS betting strategy for the current epoch.
        """
        aggregated_loss = 0
        e_val, tb_val_ons = 1, 1
        num_samples = len(loader.dataset)
        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            tau_z = tau_z.to(self.device)
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

            # Compute the C2ST and other evaluation metrics
            if mode == "test":
                e_val *= self.e_c2st(labels, out.detach())
                p_val, acc = self.s_c2st(labels, out.detach())
                l_val = self.l_c2st(labels, out.detach())
                results_tb = self.testing_by_betting(labels, out.detach())
                tb_val_ons *= results_tb[1]

                self.log(
                    {f"{mode}_e-value": e_val.item(),
                     f"{mode}_p-value-lc2st": l_val.item(),
                     f"{mode}_p-value-sc2st": p_val.item(),
                     f"{mode}_tb-ons-value": tb_val_ons.item(),
                     })

            self.log({
                f"{mode}_loss": aggregated_loss.item() / num_samples
            })

        return aggregated_loss / num_samples, tb_val_ons

# Only for the mnist experiments
class TrainerSC2ST(TrainerC2ST):

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """
        Initializes the TrainerSC2ST object by extending the TrainerC2ST class.

        This variant is specific for MNIST experiments with multiple network models.

        Args:
        (same as TrainerC2ST class)
        """
        super().__init__(cfg, net[0], tau1, tau2, datagen, device, data_seed)
        param = []
        self.net = net
        for i in range(cfg.ps):
            param = param + list(self.net[i].parameters())
        self.optimizer = torch.optim.Adam(param, lr=cfg.lr)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def train_evaluate_epoch(self, loader, mode="train"):
        """
        Train/Evaluate the model for one epoch using the C2ST approach,
        but specifically designed for MNIST experiments with multiple network models.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train" or "test". Determines how to run the model.

        Returns:
        - tuple: Aggregated loss and E-value using ONS betting strategy for the current epoch.
        """
        aggregated_loss = 0
        e_val, tb_val, tb_val_ons = 1, 1, 1
        num_samples = len(loader.dataset)
        num_pseudo_samples = None  # Placeholder for the number of pseudo samples. Will be initialized later.

        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            tau_z = tau_z.to(self.device)

            if num_pseudo_samples is None:
                num_pseudo_samples = tau_z.shape[-1]

            for i in range(num_pseudo_samples):
                self.net[i] = self.net[i].train() if mode == "train" else self.net[i].eval()

                out1 = self.net[i](z)
                out2 = self.net[i](tau_z[..., i])

                out = torch.concat((out1.clone(), out2.clone()))

                labels = torch.concat((torch.ones((z.shape[0], 1)), torch.zeros((z.shape[0], 1)))).squeeze(1).long().to(
                    self.device)

                loss = self.loss(out, labels)
                aggregated_loss += loss

                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Compute the C2ST and other evaluation metrics
                if mode == "test":
                    p_val, acc = self.s_c2st(labels, out.detach())
                    results_tb = self.testing_by_betting(labels, out.detach())
                    tb_val_ons = results_tb[1]

                    self.log({
                        f"{mode}_p-value_{i}": p_val,
                        f"{mode}_accuracy": acc
                    })

            self.log({
                f"{mode}_loss": aggregated_loss.item() / num_samples
            })

        return aggregated_loss / (num_samples * num_pseudo_samples), tb_val_ons



