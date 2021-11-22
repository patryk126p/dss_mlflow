import os

import mlflow


def go():

    _ = mlflow.run(
        os.path.join("steps", "1_docker"),
        "main",
        parameters={
            "dataset": "MNIST",
            "batch_size": 64,
            "model_config": "model_config.json",
            "model_name": "dl.model",
        },
    )

    _ = mlflow.run(
        os.path.join("steps", "2_r"),
        "main",
    )

    _ = mlflow.run(
        os.path.join("steps", "3_bash"),
        "main",
    )


if __name__ == "__main__":
    go()
