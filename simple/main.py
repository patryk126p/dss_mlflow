import argparse

from src.utils import run_experiment

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
