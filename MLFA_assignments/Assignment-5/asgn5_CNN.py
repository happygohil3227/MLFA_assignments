# -*- coding: utf-8 -*-
#Gohil Happy
#21IM30006

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a basic CNN block (Vanilla)
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out

# Define the CNN-Vanilla model
class CNNVanilla(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNVanilla, self).__init__()
        self.cnn_block1 = CNNBlock(3, 16)
        self.cnn_block2 = CNNBlock(16, 32)
        self.cnn_block3 = CNNBlock(32, 64)
        self.fc = nn.Linear(64, num_classes)


    def forward(self, x):
        out = self.cnn_block1(x)
        out = self.cnn_block2(out)
        out = self.cnn_block3(out)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Define the Residual block (CNN-Resnet)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the CNN-Resnet model
class CNNResnet(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNResnet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(32, 3, stride=1)
        self.layer2 = self.make_layer(64, 3, stride=2)
        self.layer3 = self.make_layer(128, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# Experiment 1 (with data normalization)
# Define transformation and data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Training parameters
num_epochs = 50

# Lists to store training accuracy
train_accuracy_vanilla = []
train_accuracy_resnet = []

vanilla_model = CNNVanilla().to(device)
resnet_model = CNNResnet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_vanilla = torch.optim.Adam(vanilla_model.parameters(), lr=0.001)
optimizer_resnet = torch.optim.Adam(resnet_model.parameters(), lr=0.001)

# Main training loop
for epoch in range(num_epochs):
    # Set models to training mode
    vanilla_model.train()
    resnet_model.train()

    running_loss_vanilla = 0.0
    running_loss_resnet = 0.0
    correct_vanilla = 0
    correct_resnet = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer_vanilla.zero_grad()
        optimizer_resnet.zero_grad()

        # Forward pass for vanilla model
        outputs_vanilla = vanilla_model(inputs)
        loss_vanilla = criterion(outputs_vanilla, labels)
        loss_vanilla.backward()
        optimizer_vanilla.step()

        # Forward pass for resnet model
        outputs_resnet = resnet_model(inputs)
        loss_resnet = criterion(outputs_resnet, labels)
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update statistics
        running_loss_vanilla += loss_vanilla.item()
        running_loss_resnet += loss_resnet.item()

        _, predicted_vanilla = outputs_vanilla.max(1)
        correct_vanilla += predicted_vanilla.eq(labels).sum().item()

        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)

    accuracy_vanilla = 100 * correct_vanilla / total
    accuracy_resnet = 100 * correct_resnet / total

    # Store the training accuracy for each epoch
    train_accuracy_vanilla.append(accuracy_vanilla)
    train_accuracy_resnet.append(accuracy_resnet)

    # Print training accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]:')
    print(f'CNN-Vanilla Training Accuracy: {accuracy_vanilla:.2f}%')
    print(f'CNN-Resnet Training Accuracy: {accuracy_resnet:.2f}%')
    print()

# Plot training accuracy vs. epochs
plt.figure()
plt.plot(train_accuracy_vanilla, label='CNN-Vanilla')
plt.plot(train_accuracy_resnet, label='CNN-Resnet')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()

# Compare and determine the better model based on the training accuracy
if max(train_accuracy_vanilla) > max(train_accuracy_resnet):
    print("CNN-Vanilla performs better.")
else:
    print("CNN-Resnet performs better.")

# Evaluation on the test set
vanilla_model.eval()  # Set the model to evaluation mode
resnet_model.eval()

correct_vanilla = 0
correct_resnet = 0
total = 0

with torch.no_grad():  # Disable gradient tracking during evaluation
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass for vanilla model
        outputs_vanilla = vanilla_model(inputs)
        _, predicted_vanilla = outputs_vanilla.max(1)
        correct_vanilla += predicted_vanilla.eq(labels).sum().item()

        # Forward pass for resnet model
        outputs_resnet = resnet_model(inputs)
        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)

# Calculate accuracy on the test set
test_accuracy_vanilla = 100 * correct_vanilla / total
test_accuracy_resnet = 100 * correct_resnet / total

# Compare and determine which model performs better on the test set
if test_accuracy_vanilla > test_accuracy_resnet:
    print("CNN-Vanilla performs better on the test set.")
else:
    print("CNN-Resnet performs better on the test set.")
print(f'CNN-vanilla with normalization{test_accuracy_vanilla}')
print('')
print(f'CNN-resnet with normalization{test_accuracy_resnet}')

# Experiment 2(with out data normalization)
# Define transformation and data loaders
transform = transforms.Compose([transforms.ToTensor()])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer_resnet = torch.optim.Adam(CNNResnet().parameters(), lr=0.001)

# Training parameters
num_epochs = 50

# Lists to store training accuracy

train_accuracy_resnet_2 = []


resnet_model_2 = CNNResnet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer_resnet = torch.optim.Adam(resnet_model_2.parameters(), lr=0.001)

# Main training loop
for epoch in range(num_epochs):
    # Set models to training mode
    vanilla_model.train()
    resnet_model.train()

    running_loss_vanilla_2 = 0.0
    running_loss_resnet_2 = 0.0
    correct_vanilla_2 = 0
    correct_resnet_2 = 0
    total_2 = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients

        optimizer_resnet.zero_grad()


        # Forward pass for resnet model
        outputs_resnet = resnet_model_2(inputs)
        loss_resnet = criterion(outputs_resnet, labels)
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update statistics

        running_loss_resnet += loss_resnet.item()


        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)

    accuracy_resnet = 100 * correct_resnet / total

    # Store the training accuracy for each epoch

    train_accuracy_resnet_2.append(accuracy_resnet)

    # Print training accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]:')
    print(f'CNN-Resnet Training Accuracy: {accuracy_resnet:.2f}%')
    print()

# Plot training accuracy vs. epochs
plt.figure()
plt.plot(train_accuracy_resnet, label='with_norm')
plt.plot(train_accuracy_resnet_2, label='with_out_norm')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()

# Evaluation on the test set

resnet_model_2.eval()


correct_resnet = 0
total = 0

with torch.no_grad():  # Disable gradient tracking during evaluation
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)


        # Forward pass for resnet model
        outputs_resnet = resnet_model_2(inputs)
        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)

# Calculate accuracy on the test set
test_accuracy_resnet_2 = 100 * correct_resnet / total
print(f'CNN-resnet without normalization{test_accuracy_resnet_2}')

# parameter calculation of model
total_params_vanilla = 0
total_params_resnet = 0
for param in vanilla_model.parameters():
    total_params_vanilla += param.numel()
for param in resnet_model.parameters():
    total_params_resnet += param.numel()

print(f'no. of parameter in vanilla:{total_params_vanilla}')
print(f'no. of parameter in resnet:{total_params_resnet}')

#experiment 3
#_____________________________________________________SGD___________________________________________#
# a)SGD

# Define transformation and data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



# Training parameters
num_epochs = 20

# Lists to store training accuracy

train_accuracy_resnet_SGD = []


resnet_model_sgd = CNNResnet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_resnet = torch.optim.SGD(resnet_model_sgd.parameters(), lr=0.001)

# Main training loop
for epoch in range(num_epochs):
    # Set models to training mode

    resnet_model_sgd.train()


    running_loss_resnet = 0.0

    correct_resnet = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients

        optimizer_resnet.zero_grad()



        # Forward pass for resnet model
        outputs_resnet = resnet_model_sgd(inputs)
        loss_resnet = criterion(outputs_resnet, labels)
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update statistics

        running_loss_resnet += loss_resnet.item()

        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)


    accuracy_resnet = 100 * correct_resnet / total

    # Store the training accuracy for each epoch

    train_accuracy_resnet_SGD.append(accuracy_resnet)

    # Print training accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]:')

    print(f'CNN-Resnet Training Accuracy: {accuracy_resnet:.2f}%')
    print()


#_______________________________________mini-batch with 0 mometum__________________#
# b)SGD

# Define transformation and data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# Lists to store training accuracy

train_accuracy_resnet_mini_0 = []


resnet_model_mini_0 = CNNResnet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_resnet = torch.optim.SGD(resnet_model_mini_0.parameters(), lr=0.001, momentum = 0)

# Main training loop
for epoch in range(num_epochs):
    # Set models to training mode

    resnet_model_mini_0.train()


    running_loss_resnet = 0.0

    correct_resnet = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients

        optimizer_resnet.zero_grad()



        # Forward pass for resnet model
        outputs_resnet = resnet_model_mini_0(inputs)
        loss_resnet = criterion(outputs_resnet, labels)
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update statistics

        running_loss_resnet += loss_resnet.item()

        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)


    accuracy_resnet = 100 * correct_resnet / total

    # Store the training accuracy for each epoch

    train_accuracy_resnet_mini_0.append(accuracy_resnet)

    # Print training accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]:')

    print(f'CNN-Resnet Training Accuracy: {accuracy_resnet:.2f}%')
    print()

#_______________________________________mini-batch with 0.9 mometum__________________#
# b)SGD

# Define transformation and data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Training parameters


# Lists to store training accuracy

train_accuracy_resnet_mini_09 = []


resnet_model_mini_09 = CNNResnet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_resnet = torch.optim.SGD(resnet_model_mini_09.parameters(), lr=0.001, momentum=0.9)

# Main training loop
for epoch in range(num_epochs):
    # Set models to training mode

    resnet_model_mini_09.train()


    running_loss_resnet = 0.0

    correct_resnet = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients

        optimizer_resnet.zero_grad()



        # Forward pass for resnet model
        outputs_resnet = resnet_model_mini_09(inputs)
        loss_resnet = criterion(outputs_resnet, labels)
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update statistics

        running_loss_resnet += loss_resnet.item()

        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)


    accuracy_resnet = 100 * correct_resnet / total

    # Store the training accuracy for each epoch

    train_accuracy_resnet_mini_09.append(accuracy_resnet)

    # Print training accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]:')

    print(f'CNN-Resnet Training Accuracy: {accuracy_resnet:.2f}%')
    print()

# Plot training accuracy vs. epochs
plt.figure()
plt.plot(train_accuracy_resnet, label='ADAM')
plt.plot(train_accuracy_resnet_SGD, label='True SGD')
plt.plot(train_accuracy_resnet_mini_0, label='mini_batch_momentum_0')
plt.plot(train_accuracy_resnet_mini_09, label='mini_batch_momentum_0.9')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()

print(train_accuracy_resnet)

# experiment 4 (changing the model depth )

# Define the Residual block (CNN-Resnet)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CNNResnet(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNResnet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(16, 3, stride=1)
        self.layer2 = self.make_layer(32, 3, stride=2)
        self.layer3 = self.make_layer(64, 3, stride=2)
        self.layer4 = self.make_layer(128, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



# Training parameters
num_epochs = 50
batch_size = 256


# Define transformation and data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# Lists to store training accuracy

train_accuracy_resnet = []


resnet_model = CNNResnet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_resnet = torch.optim.Adam(resnet_model.parameters(), lr=0.001)

# Main training loop
for epoch in range(num_epochs):
    # Set models to training mode

    resnet_model.train()


    running_loss_resnet = 0.0

    correct_resnet = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients

        optimizer_resnet.zero_grad()



        # Forward pass for resnet model
        outputs_resnet = resnet_model(inputs)
        loss_resnet = criterion(outputs_resnet, labels)
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update statistics

        running_loss_resnet += loss_resnet.item()

        _, predicted_resnet = outputs_resnet.max(1)
        correct_resnet += predicted_resnet.eq(labels).sum().item()

        total += labels.size(0)


    accuracy_resnet = 100 * correct_resnet / total

    # Store the training accuracy for each epoch

    train_accuracy_resnet.append(accuracy_resnet)

    # Print training accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]:')

    print(f'CNN-Resnet Training Accuracy: {accuracy_resnet:.2f}%')
    print()
