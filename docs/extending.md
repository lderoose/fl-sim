# Extending the framework

This page describes how to add new components without modifying the core training logic.

---

## Adding a preprocessing function

A preprocessor is Python function that loads and preprocesses a dataset into PyTorch tensors. It is called once before training starts.

### Interface

```python
def load_and_preprocess_my_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
    return x_train, y_train, x_test, y_test
```

All four tensors must be `torch.Tensor`. Labels (`y_train`, `y_test`) should use `torch.long` for classification tasks and `torch.float` for regression.

### Steps

**1.** Create `src/preprocessing/my_dataset.py` and implement the function:

```python
import torch
from conf.paths import DATA_DIR

def load_and_preprocess_my_dataset(path_data=DATA_DIR):
    # load raw data, apply normalization/scaling, convert to tensors
    ...
    return x_train, y_train, x_test, y_test
```

**2.** Import and assign it in `conf/application.py`:

```python
from src.preprocessing.my_dataset import load_and_preprocess_my_dataset

PREPROCESSOR = load_and_preprocess_my_dataset
```

> The preprocessor is called with no arguments at runtime. If it requires a `path_data` argument, use a default value as shown above.

---

## Adding a dataset

Adding a full dataset involves registering it in two configuration files in addition to writing the preprocessor.

### Steps

**1.** Write the preprocessing function as described above.

**2.** Register the loss function in `conf/criterion.py`:

```python
CONF_DATASET_CRITERION: dict = {
    ...
    "my_dataset": nn.MSELoss,   # or nn.CrossEntropyLoss for classification
}
```

**3.** If the dataset is a classification task, add its name to `CLASSIFICATION_DATASET` in `conf/convention.py`. This allows the framework to validate that accuracy metrics are only enabled for classification datasets.

```python
CLASSIFICATION_DATASET = [
    "mnist",
    "cifar10",
    "femnist",
    "my_dataset",   # add here
]
```

**4.** Define a model for the dataset (see [below](#defining-a-new-model)).

**5.** Update `conf/application.py` with all relevant parameters:

```python
DATASET_NAME = "my_dataset"
PREPROCESSOR = load_and_preprocess_my_dataset
MODEL = MyDatasetModel
```

---

## Defining a new model

A model is a standard PyTorch `nn.Module` subclass. The convention in this codebase is to define a base class holding dataset constants (`INPUT_SIZE`, `OUTPUT_SIZE`, etc.), then derive concrete architectures from it.

### Steps

**1.** Create `src/models/my_dataset_models.py`:

```python
import torch
import torch.nn as nn


class MyDatasetModel(nn.Module):
    INPUT_SIZE = ...   # number of input features
    OUTPUT_SIZE = ...  # number of output units


class MyDatasetModelSimple(MyDatasetModel):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(self.INPUT_SIZE, self.OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
```

**2.** Import and assign it in `conf/application.py`:

```python
from src.models.my_dataset_models import MyDatasetModelSimple

MODEL = MyDatasetModelSimple
```

Note that `MODEL` expects the class itself, not an instance.
