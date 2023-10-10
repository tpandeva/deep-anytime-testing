from trainer import Trainer
import numpy as np
import torch


class TrainerECRT(Trainer):
    """Trainer class for the ECRT method."""

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """
        Initializes the TrainerECRT object by extending the Trainer class.

        Args:
        (same as Trainer class)
        """
        super().__init__(cfg, net, tau1, tau2, datagen, device, data_seed)
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.integral_vector = torch.tensor(np.linspace(0, 1, 1001, endpoint=False)[1:]).to(device)
        self.stb = [torch.ones(1000, device=device) for _ in range(3)]

    def ecrt(self, y, x, x_tilde, batches, mode):
        """
        Implement the ECRT method.
        Code adapted from https://github.com/shaersh/ecrt/

        Args:
        - y (torch.Tensor): Ground truth labels.
        - x (torch.Tensor): Input samples.
        - x_tilde (torch.Tensor): Perturbed samples.
        - batches (list): List of batch sizes.
        - mode (str): Either "train" or "test".

        Returns:
        - torch.Tensor: Test statistic.
        """
        self.net = self.net.eval()
        total_samples = y.shape[0]
        test_stat = torch.nn.MSELoss(reduction='mean')
        st, stb = [], []
        i = 0

        for b in batches:
            # split data into batches of size b
            num_chunks = int(total_samples / b)
            index_sets_seq = np.array_split(range(total_samples), num_chunks)
            stb.append(torch.ones(1000))
            for ind in index_sets_seq:
                y_tb, x_tb, x_tilde_tb = y[ind], x[ind], x_tilde[ind]
                pred_tilde_tb = self.net(x_tilde_tb)
                pred_tb = self.net(x_tb).detach()

                q = test_stat(pred_tb.squeeze(), y_tb)
                q_tilde = test_stat(pred_tilde_tb.squeeze(), y_tb)
                wealth = torch.nn.Tanh()(q_tilde - q)

                if mode == "test":
                    self.stb[i] = self.stb[i].to(self.device) * (1 + self.integral_vector * wealth)
                    stb[i] = self.stb[i].clone()
                else:
                    stb[i] = stb[i].to(self.device) * (1 + self.integral_vector * wealth)

            st.append(stb[i].mean())
            i += 1
        st = torch.stack(st).mean()
        return st

    def train_evaluate_epoch(self, loader, mode="train"):
        """
        Train/Evaluate the model for one epoch using the ECRT approach.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train" or "test".

        Returns:
        - tuple: Aggregated loss and E-value for the current epoch.
        """
        aggregated_loss, loss_tilde = 0, 0
        num_samples = len(loader.dataset)

        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            features = z[:, :-1]
            target = z[:, -1]
            tau_z = tau_z.to(self.device)

            self.net = self.net.train() if mode == "train" else self.net.eval()

            out = self.net(features)
            loss = self.loss(out.squeeze(), target)
            aggregated_loss += loss

            y_tilde = self.net(tau_z[:, :-1]).detach()
            loss_tilde += self.loss(y_tilde.squeeze(), target)

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            e_val = self.ecrt(target, features, tau_z[:, :-1], [2, 5, 10], mode)

        self.log({
            f"{mode}_loss": aggregated_loss.item() / (i + 1),
            f"{mode}_e-val": e_val.item(),
            f"{mode}_loss_tilde": loss_tilde.item() / (i + 1)
        })
        return aggregated_loss / num_samples, e_val
