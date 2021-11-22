"""
Clean dataset
"""
import argparse
import logging
import os

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace):
    col_names = args.col_names.split(",")
    logger.info(f"Reading raw data: {args.raw_data}")
    df = pd.read_csv(args.raw_data, header=None, names=col_names)

    logger.info(f"Dropping rows with empty values")
    df.dropna(axis=0, how="any", inplace=True)

    file_path = os.path.join("data", args.file_name)
    logger.info(f"Saving output file to {file_path}")
    df.to_csv(file_path, index=False)
    logger.info(f"Uploading {args.file_name} to artifact store")
    mlflow.log_artifact(file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data clean_data")
    parser.add_argument(
        "--raw_data", type=str, help="Location of raw data to clean", required=True
    )
    parser.add_argument(
        "--file_name", type=str, help="Output artifact name", required=True
    )
    parser.add_argument(
        "--col_names",
        type=str,
        help="Comma separated list of column names",
        required=True,
    )
    arguments = parser.parse_args()

    go(arguments)
