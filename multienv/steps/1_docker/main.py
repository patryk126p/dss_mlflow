import argparse
import json
import logging

import mlflow
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class ConvNet(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        self.conv1 = nn.Conv2d(
            self.model_config["conv1"]["in_channels"],
            self.model_config["conv1"]["out_channels"],
            kernel_size=self.model_config["conv1"]["kernel_size"],
            padding=self.model_config["conv1"]["padding"],
        )
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(self.model_config["max_pool"])
        self.conv2 = nn.Conv2d(
            self.model_config["conv2"]["in_channels"],
            self.model_config["conv2"]["out_channels"],
            kernel_size=self.model_config["conv2"]["kernel_size"],
            padding=self.model_config["conv2"]["padding"],
        )
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(self.model_config["max_pool"])
        self.fc = nn.Linear(
            self.model_config["fc"]["in_features"],
            self.model_config["fc"]["out_features"],
        )

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.reshape(-1, self.model_config["fc"]["in_features"])
        out = self.fc(out)
        return out


def run_experiment(args: argparse.Namespace):
    logger.info(f"Downloading {args.dataset}")
    dataset_class = getattr(torchvision.datasets, args.dataset)
    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_set = dataset_class(
        root="data", train=True, download=True, transform=transforms
    )
    test_set = dataset_class(
        root="data", train=False, download=True, transform=transforms
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Reading model config from {args.model_config}")
    mlflow.log_artifact(args.model_config)
    with open(args.model_config) as fp:
        model_config = json.load(fp)

    learning_rate = model_config["learning_rate"]
    n_epochs = model_config["epochs"]

    model = ConvNet(model_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    logger.info("Training model")
    training_loop(n_epochs, optimizer, model, criterion, train_loader)

    logger.info(f"Saving model to {args.model_name}")
    torch.save(model.state_dict(), args.model_name)
    logger.info(f"Uploading {args.model_name} to artifact store")
    mlflow.log_artifact(args.model_name)

    logger.info("Calculating test accuracy")
    acc = get_accuracy(model, test_loader)
    logger.info(f"Accuracy on test data: {acc:.2f}")
    mlflow.log_metric("test_acc", acc)


def training_loop(n_epochs, optimizer, model, criterion, train_loader):
    for epoch in range(n_epochs):
        losses = []

        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch: {epoch}; loss: {sum(losses) / len(losses)}")
        acc = get_accuracy(model, train_loader)
        logger.info(f"Accuracy: {acc:.2f}")
        mlflow.log_metric("train_acc", acc, epoch)


def get_accuracy(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for xs, ys in data_loader:
            scores = model(xs)
            _, predictions = torch.max(scores, dim=1)
            correct += (predictions == ys).sum()
            total += ys.shape[0]

        acc = float(correct) / float(total) * 100
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "dataset", type=str, help="Name of torchvision class for downloading dataset"
    )
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument(
        "model_config",
        type=str,
        help="Path to json with model configuration",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of model artifact",
    )
    args = parser.parse_args()

    run_experiment(args)
