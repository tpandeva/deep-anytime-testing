import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from cifar10_classifier_train import test
from models import Cifar10Net

def fgsm_attack(model, criterion, images, labels, device, epsilon):
    images.requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels).to(device)
    model.zero_grad()
    loss.backward()

    gradient = images.grad.data
    perturbations = epsilon * torch.sign(gradient)
    adversarial_images = images + perturbations
    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    return adversarial_images, perturbations
def test_adversarial(model, testloader, criterion, device, epsilon):
    adversarial_correct = 0
    attack_success = 0
    total = 0

    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        adversarial_images, _ = fgsm_attack(model, criterion, images, labels, device, epsilon)

        adversarial_outputs = model(adversarial_images)

        _, adversarial_predicted = torch.max(adversarial_outputs.data, 1)

        adversarial_correct += (adversarial_predicted == labels).sum().item()
        attack_success += (adversarial_predicted != labels).sum().item()
        total += labels.size(0)

    adversarial_accuracy = 100.0 * adversarial_correct / total
    attack_success_rate = 100.0 * attack_success / total
    print(f'Epsilon = {epsilon}:')
    print(f'Adversarial Accuracy: {adversarial_accuracy:.2f}%')
    print(f'Attack Success Rate: {attack_success_rate:.2f}%')
    print('------------------------------------------------------')
    return adversarial_accuracy, attack_success_rate

if __name__ == '__main__':
    epsilon_values = [0.01, 0.03, 0.07, 0.1, 0.3, 0.5]
    criterion = nn.CrossEntropyLoss()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    batch_size = 64

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    # Load the best model
    # best_model = models.resnet50(pretrained=True)
    # # Modify conv1 to suit CIFAR-10
    # num_features = best_model.fc.in_features
    # best_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # best_model.fc = nn.Linear(num_features, 10)
    best_model = Cifar10Net(10)
    # Load checkpoints
    checkpoint = torch.load('best_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    test_accuracy = checkpoint['test_accuracy']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = best_model.to(device)
    print("Best Trained Model Loaded!")
    print(f"Checkpoint at Epoch {epoch + 1} with accuracy of {test_accuracy}%")

    # Test the best model on adversarial examples

    # Evaluate adversarial attacks for each epsilon value
    adversarial_accuracies = []
    attack_success_rates = []
    print("Testing with clean data again to compare with checkpoint accuracy...")
    _, clean_test_accuracy = test(best_model, testloader, criterion, device)
    print(
        f"Clean Adv Accuracy: {clean_test_accuracy:.2f}%\nClean Attack Success Rate: {100 - clean_test_accuracy:.2f}%")
    if (clean_test_accuracy == test_accuracy):
        print("Matches with the Checkpoint Accuracy!")
    print('-----------------------------')
    print("Testing with adversarial examples...")
    for epsilon in epsilon_values:
        adversarial_accuracy, attack_success_rate = test_adversarial(best_model, testloader, criterion, device, epsilon)
        adversarial_accuracies.append(adversarial_accuracy)
        attack_success_rates.append(attack_success_rate)

