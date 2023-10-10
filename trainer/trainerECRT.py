from trainer import Trainer
import numpy as np
import torch

class TrainerECRT(Trainer):
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
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.integral_vector = torch.tensor(np.linspace(0, 1, 1001, endpoint=False)[1:]).to(device)
        self.stb = [torch.ones(1000, device=device), torch.ones(1000, device=device), torch.ones(1000, device=device)]

    def ecrt(self,y,x, x_tilde, batches, mode):
        # iterate over batch sizes
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
                y_tb = y[ind]
                x_tb = x[ind]
                x_tilde_tb = x_tilde[ind]
                pred_tilde_tb = self.net(x_tilde_tb)
                pred_tb = self.net(x_tb).detach()
                q = test_stat(pred_tb.squeeze(), y_tb)
                q_tilde = test_stat(pred_tilde_tb.squeeze(), y_tb)
                wealth = torch.nn.Tanh()(q_tilde - q)
                if mode=="test":
                    self.stb[i] = self.stb[i].to(self.device)*(1 + self.integral_vector * wealth )
                    stb[i] =  self.stb[i].clone()
                else:
                    stb[i] = stb[i].to(self.device)*(1 + self.integral_vector * wealth )

            st.append(stb[i].mean())
            i += 1
        st = torch.stack(st).mean()
        return st

    def train_evaluate_epoch(self, loader, mode="train"):
        """Train/Evaluate the model for one epocj and log the results."""
        aggregated_loss, loss_tilde = 0,0
        num_samples = len(loader.dataset)
        for i, (z, tau_z) in enumerate(loader):
            # BATCH SIZE X
            z = z.to(self.device)
            features = z[:, :-1]
            target = z[:, -1]
            # z = z.transpose(2, 1).flatten(0).view(2*samples,-1)[...,:-1]
            tau_z = tau_z.to(self.device)
            # tau_z = tau_z.transpose(2, 1).flatten(0).view(2*samples,-1)[...,-1]
            if mode == "train":
                self.net = self.net.train()
                out = self.net(features)
            else:
                self.net = self.net.eval()
                out = self.net(features)
            loss =  self.loss(out.squeeze(), target)
            aggregated_loss +=loss
            y_tilde = self.net(tau_z[:,:-1]).detach()
            loss_tilde += self.loss(y_tilde.squeeze(), target)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # compute e-c2st and s-c2st

            e_val = self.ecrt(target, features, tau_z[:,:-1], [2, 5, 10], mode)
        self.log({ f"{mode}_loss": aggregated_loss.item()/(i+1),  f"{mode}_e-val": e_val.item(), f"{mode}_loss_tilde": loss_tilde.item()/(i+1)})
        return aggregated_loss/num_samples, e_val
