# Data Science Summit
## MLflow Projects

Repo showing two approaches (one in two flavours) to structuring ML projects/pipelines using MLflow Projects. Additionally, MLflow Tracking used for tracking params, metrics and artifacts

## Setup

1. Install requirements `pip install -r requirements.txt`

## How to use this repo

Move to repo root directory
```
cd <REPO_ROOT>
```

Execute to run simple project:
```
mlflow run simple --experiment-name simple
```

Execute to run multi-step project:
```
mlflow run multistep --experiment-name multistep
```

Execute to run multi-env project:<br>
(requires docker - `cd multienv/steps/1_docker` & `docker build -t mlflow_docker .` and R)
```
mlflow run multienv --experiment-name multienv
```

Execute to start local MLflow server to see tracking UI:
```
mlflow server
```
