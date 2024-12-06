from typing import Tuple, Union
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import pandas as pd
import numpy as np

OurModels = Union[linear_model.SGDRegressor, linear_model.Ridge, RandomForestRegressor]


def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    model: OurModels,
    params: dict,
    cv: int = 5,
    verbose: int = 2,
) -> GridSearchCV:
    grid_search = GridSearchCV(
        model,
        params,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        return_train_score=True,
        verbose=verbose,
    )

    grid_search.fit(X, y)
    return grid_search

def random_search(
    X: np.ndarray,
    y: np.ndarray,
    model: OurModels,
    params: dict,
    cv: int = 5,
    n_iters: int = 100,
    verbose: int = 2,
) -> GridSearchCV:
    rand_search = RandomizedSearchCV(
        model,
        params,
        n_iter=100,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        return_train_score=True,
        verbose=verbose,
    )

    rand_search.fit(X, y)
    return rand_search

def save_model(model, output_dir: str = "results/"):

    # Get the model configuration
    config = model.get_params()
    class_name = model.__class__.__name__

    # Save the configuration to a JSON file
    with open(f"{output_dir}/{class_name}_config.json", "w") as f:
        json.dump(config, f)

def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
) -> Tuple[OurModels, list]:

    model.fit(X_train, y_train)
    y_hat = model.predict(X_dev)

    mae = mean_absolute_error(y_dev, y_hat)
    rmse = root_mean_squared_error(y_dev, y_hat)
    r2 = root_mean_squared_error(y_dev, y_hat)

    return model, [mae, rmse, r2]


def train_in_batches(
    model: OurModels,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    batch_size: int = 32,
    epochs: int = 10,
    dev_epoch_rate: int = 1,
) -> Tuple[OurModels, list]:

    n_samples = X_train.shape[0]

    metrics_results = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Loop over the data in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            model.partial_fit(X_batch, y_batch)

        if (epoch + 1) % dev_epoch_rate == 0:

            y_hat = model.predict(X_dev)

            mae = mean_absolute_error(y_dev, y_hat)
            rmse = root_mean_squared_error(y_dev, y_hat)
            r2 = root_mean_squared_error(y_dev, y_hat)

            metrics_results.append([epoch + 1, mae, rmse, r2])

    return model, metrics_results
