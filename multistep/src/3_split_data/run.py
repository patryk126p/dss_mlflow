"""
Split dataset into train and test
"""
import argparse
import logging
import os

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace):
    logger.info(f"Reading clean data: {args.clean_data}")
    df = pd.read_csv(args.clean_data)

    logger.info("Splitting data")
    train, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    file_names = args.file_names.split(",")
    for part, name in zip([train, test], file_names):
        file_path = os.path.join("data", name)
        logger.info(f"Saving {file_path}")
        part.to_csv(file_path, index=False)
        logger.info(f"Uploading {name} to artifact store")
        mlflow.log_artifact(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data to train and test sets")
    parser.add_argument("clean_data", type=str, help="Location of clean data to split")
    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items",
    )
    parser.add_argument(
        "random_seed", type=int, help="Seed for random number generator"
    )
    parser.add_argument(
        "file_names",
        type=str,
        help="Comma separated list of names for train and test datasets",
    )
    arguments = parser.parse_args()

    go(arguments)
