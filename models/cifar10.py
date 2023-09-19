import torch.nn as nn
import torchvision.models as models

class Cifar10Net(nn.Module):
    def __init__(self, num_classes):
        super(Cifar10Net, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Modify conv1 to suit CIFAR-10
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Modify the final fully connected layer according to the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    def forward(self,x):
        return self.model(x)