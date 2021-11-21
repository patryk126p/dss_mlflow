import argparse
import json
import logging
import os

import mlflow
import torch
import torchvision
from src.model import ConvNet
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


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

    model_path = os.path.join("model", args.model_name)
    logger.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Uploading {args.model_name} to artifact store")
    mlflow.log_artifact(model_path)

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
