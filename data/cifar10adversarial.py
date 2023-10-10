import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator, DatasetOperator
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
class Cifar10Adversarial(DatasetOperator):

    def __init__(self, z, tau1, tau2, epsilon,model):
        if len(tau2.transforms) ==0: tau2 = tau1
        super().__init__(tau1, tau2)
        self.images, labels = z
        self.z = self.images
        self.images_adv = self.fgsm_attack(model, self.images.clone(), labels, epsilon)
        self.tau1_z = self.tau1(self.images)
        self.tau2_z = self.tau2(self.images_adv)

    def __getitem__(self, idx):
        tau1_z, tau2_z = self.tau1_z[idx], self.tau2_z[idx]
        # if self.tau1 is not None:
        #     tau1_z = self.tau1(tau1_z[None,...]).squeeze()
        # if self.tau2 is not None:
        #     tau2_z = self.tau2(tau2_z[None,...]).squeeze()
        return tau1_z, tau2_z

    def fgsm_attack(self, model, images, labels, epsilon):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.CrossEntropyLoss()
        images.requires_grad_(True)
        outputs = model(images)
        loss = criterion(outputs, labels).to(device)
        model.zero_grad()
        loss.backward()

        gradient = images.grad.data
        perturbations = epsilon * torch.sign(gradient)
        adversarial_images = images + perturbations
        adversarial_images = torch.clamp(adversarial_images, 0, 1)

        return adversarial_images

class Cifar10AdversarialDataGen(DataGenerator):
    def __init__(self, samples,  data_seed, file, epsilon, model, file_to_model, download):
        super().__init__("type2", samples, data_seed)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.z = torchvision.datasets.CIFAR10(root=file, train=False,
                                                download=download, transform=transform_test)
        total_samples = len(self.z)
        num_chunks = int(total_samples / samples)
        self.index_sets_seq = np.array_split(np.random.permutation(total_samples), num_chunks)
        self.epsilon = epsilon
        self.model = model
        checkpoint = torch.load(file_to_model, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def generate(self,seed, tau1, tau2) -> Dataset:
        ind = self.index_sets_seq[seed]
        subset = Subset(self.z, ind)
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=1)
        z = next(iter(loader))
        return Cifar10Adversarial(z, tau1, tau2, self.epsilon, self.model)