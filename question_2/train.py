import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import CNN
from src.evaluate import evaluate

def train():
    wandb.init(project="inat-hyper-sweep")
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Data
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_len = int(0.2 * len(train_set))
    train_len = len(train_set) - val_len
    train_data, val_data = random_split(train_set, [train_len, val_len])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # Model
    model = CNN(config.filters, config.dropout, config.activation, config.use_batchnorm, config.padding).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_accuracy = 100 * correct / total
        val_accuracy = evaluate(model, val_loader, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })
