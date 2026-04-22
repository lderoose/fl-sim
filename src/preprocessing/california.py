import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_california(test_size: float = 0.2) -> tuple[torch.Tensor, ...]:
    """Load and preprocess the California housing dataset (regression).

    Dataset properties:
        Samples total: 20640, Dimensionality: 8, Target: real in [0.15, 5].

    Applies StandardScaler on the features (fit on train, transform on test).
    Assumes each worker applies the same preprocessing.

    Args:
        test_size (float): Fraction of data allocated to the test set.

    Returns:
        tuple: (x_train, y_train, x_test, y_test).
    """
    X, y = fetch_california_housing(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)  # train on x_train not on x_test

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    return x_train, y_train, x_test, y_test
