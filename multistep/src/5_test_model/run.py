"""
Test model on test data
"""
import argparse
import logging

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace):
    logger.info(f"Reading test data: {args.test_data}")
    x = pd.read_csv(args.test_data)
    y = x.pop(args.target)

    logger.info(f"Loading model from {args.model_path}")
    model = joblib.load(args.model_path)

    logger.info("Evaluating model")
    preds = model.predict(x)
    acc = accuracy_score(y, preds)
    logger.info(f"Accuracy on test data: {acc:.2f}")
    mlflow.log_metric("accuracy", acc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test model against test dataset")
    parser.add_argument("test_data", type=str, help="Path to test dataset")
    parser.add_argument("target", type=str, help="Name of target variable")
    parser.add_argument("model_path", type=str, help="Path to model")
    arguments = parser.parse_args()

    go(arguments)
