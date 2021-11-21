"""
Download dataset
"""
import argparse
import logging
import os

import mlflow
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace):
    logger.info(f"Downloading {args.uri}")
    data = requests.get(args.uri).content.decode("utf-8")
    file_path = os.path.join("data", args.file_name)
    with open(file_path, "w") as fh:
        fh.write(data)
    logger.info(f"Uploading {args.file_name} to artifact store")
    mlflow.log_artifact(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")
    parser.add_argument("uri", type=str, help="URI of file to download")
    parser.add_argument("file_name", type=str, help="Name of the downloaded file")
    arguments = parser.parse_args()

    go(arguments)
