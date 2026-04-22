# Configuration

All experiment parameters are defined in `conf/template_application.py`.

> **Note:** not all parameter combinations have been tested. While they are expected to work, some edge cases may still contain bugs.

## Why a Python configuration file?

Unlike YAML or JSON, a Python file allows parameters to be expressed as native objects — optimizer classes, model architectures, loss functions — without having to encode every possible option as a string or integer and then resolve it at runtime. This keeps the configuration concise and fully type-aware.

## Convention

For optional parameters, setting the value to `None` disables the corresponding feature.

---

### General

| Parameter | Type | Description |
|---|---|---|
| `DEVICE` | `str` | Computation device. `"cpu"` or `"cuda:0"` (replace `0` with the target GPU index). |
| `FEDERATED` | `bool` | If `True`, runs a federated learning simulation. If `False`, runs in [classical](terminology.md) mode. |
| `SEED` | `int` | Random seed for reproducibility. |

---

### Federation

| Parameter | Type | Description |
|---|---|---|
| `N_WORKERS` | `int` | Number of workers participating in the federation. Only used when `FEDERATED=True`. |

---

### Dataset

| Parameter | Type | Description |
|---|---|---|
| `DATASET_NAME` | `str` | Name of the dataset. Supported values: `"mnist"`, `"cifar10"`, `"femnist"`, `"california"`. |
| `PREPROCESSOR` | `callable` | Preprocessing function class (not an instance). Must be compatible with the chosen dataset. |
| `LIMIT_N_DATA_TRAIN` | `int \| float \| None` | Limits the total training set size. An `int` sets an absolute number of samples (shared across all workers). A `float` in `(0, 1]` sets a fraction. `None` uses the full dataset. |
| `LIMIT_N_DATA_TEST` | `int \| float \| None` | Same as `LIMIT_N_DATA_TRAIN` but applied to the test set. |
| `VALIDATION_SIZE` | `float \| None` | Fraction of the training set to use as a validation set. |
| `BATCH_SIZE` | `int \| None` | See [N_STEPS and BATCH_SIZE](#n_steps-and-batch_size). |

---

### Model

| Parameter | Type | Description |
|---|---|---|
| `MODEL` | `nn.Module` subclass | The model architecture class (not an instance). Must be compatible with the chosen dataset. |
| `FLOAT_PRECISION` | `str` | Floating-point precision for model parameters. One of `"float16"`, `"float32"`, `"float64"`. |

---

### Training

| Parameter | Type | Description |
|---|---|---|
| `N_EPOCHS` | `int` | Number of training [epochs](terminology.md). One epoch is a full cycle during which all workers collectively process their entire training data once. |
| `N_STEPS` | `int \| None` | See [N_STEPS and BATCH_SIZE](#n_steps-and-batch_size). |
| `CRITERION` | `nn.Module` | Loss function. Automatically derived (see `conf/criterion.py`). |

#### `N_STEPS` and `BATCH_SIZE`

These two parameters control how data is distributed across workers and gradient steps. They are mutually exclusive in federated mode: exactly one must be set, the other must be `None`.

**Strategy A — fixed number of steps (`N_STEPS` is set, `BATCH_SIZE=None`):**
The training data is divided into exactly `N_STEPS` equally-sized global batches. At each [step](terminology.md), each worker receives an equal share of that batch. The batch size per worker is derived automatically from the data size.

**Strategy B — fixed batch size (`BATCH_SIZE` is set, `N_STEPS=None`):**
The global batch size is fixed to `BATCH_SIZE`. The number of steps per epoch is then derived as `n_train_samples / BATCH_SIZE`. Each worker receives an equal share of each global batch.

In **classical** mode, `N_STEPS` is not used. `BATCH_SIZE` is passed directly to PyTorch's `DataLoader`.

---

### Optimizer

```python
OPTIMIZER = {
    "name": optim.SGD,        # any torch.optim optimizer class
    "kwargs": {
        "lr": 0.1,            # learning rate
        "weight_decay": 0,    # L2 regularization strength (0 or None to disable)
    },
}
```

The `"name"` field accepts any `torch.optim` optimizer class. All keyword arguments supported by that class can be passed through `"kwargs"`.

---

### Compression *(federated only)*

```python
COMPRESSOR = {
    "name": compressor.TopKCompressor,  # TopKCompressor | RandomKCompressor | None
    "kwargs": {
        "k": 0.1,             # compression ratio or absolute count (see below)
        "k_as_factor": True,  # True: k is a fraction; False: k is an integer count
    },
}
```

| Compressor | Description |
|---|---|
| `TopKCompressor` | Keeps the `k` gradient coordinates with the largest absolute values. |
| `RandomKCompressor` | Keeps `k` randomly selected coordinates. The estimator is unbiased (selected values are rescaled by `n/k`). |
| `None` | No compression; the full gradient is transmitted. |

**Parameter `k`:**
- If `k_as_factor=True`: `k` must be a `float` in `(0, 0.5)`. It represents the fraction of coordinates to keep. The upper bound of `0.5` ensures that the compressed representation (stored as index-value pairs) is always smaller than the original tensor.
- If `k_as_factor=False`: `k` must be a positive `int` specifying the absolute number of coordinates to keep.

| Parameter | Type | Description |
|---|---|---|
| `SPARSE_WEIGHT_UPDATES` | `bool` | When `True`, the master sends only the updated weight indices to workers instead of the full model state, reducing communication overhead. This flag is only effective when a compressor is active; it is silently ignored otherwise. |

---

### Variant *(federated only)*

```python
VARIANT = {
    "name": None,   # ErrorFeedback | ProjFL | ErrorFeedback21 | Diana | None
    "kwargs": {
        "alpha": 0.75,  # scaling factor for the error buffer  (ErrorFeedback, Diana)
        "beta": 0.9,    # momentum coefficient for the master  (Diana only)
        "K": 1,         # sliding window size                  (ProjFL only)
        "EF": False,    # enable error feedback                (ProjFL only)
    },
}
```

Variants are algorithmic enhancements applied on top of compressed communication. Each variant is described in detail in the associated paper. Only kwargs relevant to the chosen variant are used; the others are silently ignored.

---

### Learning Rate Scheduler

```python
LR_SCHEDULER = {
    "name": lr_scheduler.ReduceLROnPlateau,  # any torch lr_scheduler class | None
    "kwargs": {
        "patience": 2,
        "factor": 0.1,
        "min_lr": 1e-3,
    },
}
```

Any `torch.optim.lr_scheduler` class is supported; pass its arguments through `"kwargs"`.

---

### Stopping Criterion

```python
STOPPING_CRITERION = {
    "name": EarlyStopping,  # EarlyStopping | None
    "kwargs": {
        "patience": 10,                             # epochs to wait before stopping
        "mode": "min",                              # "min" or "max"
        "delta": 1e-4,                              # minimum improvement threshold
        "verbose": True,                            # print messages on improvement
        "path": "model_checkpoints/checkpoint.pt", # checkpoint save path
    },
}
```

| kwarg | Description |
|---|---|
| `patience` | Number of epochs without improvement after which training stops. |
| `mode` | `"min"` stops when the metric stops decreasing (e.g. loss). `"max"` stops when it stops increasing (e.g. accuracy). |
| `delta` | Minimum change to qualify as an improvement. |
| `verbose` | If `True`, logs a message each time the best metric is updated and the model is checkpointed. |
| `path` | File path where the best model checkpoint is saved. The parent directory is created automatically. |

---

### Report

```python
REPORT_PARAMS = {
    "path_save": ...,              # output path for the JSON report
    # --- metrics to record ---
    "loss_train": True,
    "loss_validation": True,
    "loss_test": True,
    "accuracy_train": True,
    "accuracy_validation": True,
    "accuracy_test": True,
    "var_loss_train": True,
    "var_loss_validation": True,
    "var_loss_test": True,
    "time_exec": True,
    "communication_cost": True,
    "gradients_l1_norm": False,
    "gradients_l2_norm": True,
    "weights_l1_norm": False,
    "weights_l2_norm": False,
    "weights": False,              # caution: produces very large files
}
```

Each boolean flag controls whether the corresponding metric is recorded during training and saved to the report file. Set `"weights": True` only when strictly necessary, as it can produce very large output files.
