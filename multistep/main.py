import argparse
import os

import mlflow
import yaml

STEPS = [
    "download",
    "clean",
    "split",
    "train",
    "test",
]


def go(args: argparse.Namespace):

    mlflow.log_artifact(args.config_path)
    with open(args.config_path, "r") as fh:
        config = yaml.safe_load(fh)

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else STEPS

    if "download" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "1_get_data"),
            "main",
            parameters={
                "uri": config["download"]["uri"],
                "file_name": config["download"]["file_name"],
            },
        )

    if "clean" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "2_clean_data"),
            "main",
            parameters={
                "raw_data": config["clean"]["raw_data"],
                "file_name": config["clean"]["file_name"],
                "col_names": config["clean"]["col_names"],
            },
        )

    if "split" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "3_split_data"),
            "main",
            parameters={
                "clean_data": config["split"]["clean_data"],
                "test_size": config["split"]["test_size"],
                "random_seed": config["split"]["random_seed"],
                "file_names": config["split"]["file_names"],
            },
        )

    if "train" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "4_train_model"),
            "main",
            parameters={
                "train_data": config["train"]["train_data"],
                "target": config["train"]["target"],
                "model_config": config["train"]["model_config"],
                "model_name": config["train"]["model_name"],
            },
        )

    if "test" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "5_test_model"),
            "main",
            parameters={
                "test_data": config["test"]["test_data"],
                "target": config["test"]["target"],
                "model_path": config["test"]["model_path"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("config_path", type=str, help="Path to run configuration")
    arguments = parser.parse_args()
    go(arguments)
