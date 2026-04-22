# fl-sim

A PyTorch-based local simulation framework for federated learning experiments, easily configurable through a single file.

fl-sim supports both classical (non-federated) and federated training, with optional gradient compression. Compression was the focus of the [associated paper](#related-paper), but it is disabled by default and the repository is designed for broader use beyond that context.

## Overview

All experiment parameters are declared in a single Python configuration file. Running `main.py` reads that file, preprocesses the dataset, and launches either a **[classical](docs/terminology.md#classical)** or a federated training loop.

In **[classical](docs/terminology.md#classical)** mode, a single model is trained on the full dataset with standard mini-batch gradient descent.

In **federated** mode, a master-worker architecture is simulated locally: the master holds the global model and coordinates a configurable number of workers, each training on its own data shard. At each step, workers compute gradients and send them to the master, which aggregates them and broadcasts the updated model back. Gradient **compression** (top-k or random-k) can be enabled to reduce the volume of data exchanged, and several compression-based **variants** (Error Feedback, EF21, ProjFL, DIANA) are available — see the [associated paper](#related-paper) for details.

Across both modes, the framework tracks training, validation, and test metrics at each epoch and saves them to a JSON report. Early stopping and learning rate scheduling are also supported.

> **Note — classical vs. federated equivalence.**
> Federated and classical training are mathematically equivalent when no compression is used and workers collectively cover the full dataset. However, due to floating-point rounding, strict numerical equality is not achievable when `n_workers > 1`: the two modes split the gradient computation differently, which leads to tiny differences (~1e-7 in float32). This is not a bug. The consistency test suite (`tests/consistency/`) verifies equivalence up to a tolerance of `1e-5`.

## Installation

**Requirements:** Python >= 3.14, [uv](https://docs.astral.sh/uv/)

Clone the repository, create the virtual environment, and install the dependencies:

```bash
git clone <repository_url>
cd fl-sim
uv sync
```

To activate the virtual environment:

```bash
source .venv/bin/activate
```

### Verifying the installation

The test suite can be run with:

```bash
pytest
```

A set of ready-to-run example configurations is also provided in `examples/conf/`. To run all of them in sequence:

```bash
bash examples/run_ex.sh
```

This runs the following experiments and saves each report to `report/`:

| Configuration | Description |
|---|---|
| `classical_california` | Classical training on the California housing dataset |
| `federated_mnist_no_compression` | Federated training on MNIST without compression |
| `federated_mnist_random` | Federated training on MNIST with random-k compression |
| `federated_mnist_top` | Federated training on MNIST with top-k compression |
| `federated_mnist_top_ef` | Top-k with error feedback (EF) |
| `federated_mnist_top_projfl` | Top-k with ProjFL variant |
| `federated_mnist_top_projfl_ef` | Top-k with ProjFL and error feedback |

### Running an experiment

Edit `conf/application.py` to configure your experiment (see [configuration reference](docs/configuration.md)), then run:

```bash
python main.py
```

By default, the report is saved to `report/`. This can be changed in `conf/application.py` via the `REPORT_PARAMS` parameter. It can also be overridden at runtime:

```bash
python main.py --path_report path/to/output.json
```

The configuration file path can also be overridden, which is useful for running multiple experiments without touching `conf/application.py`:

```bash
python main.py --path_config path/to/my_config.py --path_report path/to/output.json
```

### Plotting results *(compression experiments only)*

`create_plots.py` generates plots showing the evolution of the loss against the communication cost for a set of experiments simultaneously. It was used to produce part of the figures in the [associated paper](#related-paper). Edit the `LS_REPORTS` list at the top of the file to select which reports to include, then run:

```bash
python create_plots.py
```

Plots are saved to `plots/`. This script is only meaningful when comparing compression-based experiments, as it tracks communication cost alongside loss.

---

## Related paper

This repository was used to run the experiments of the following paper (under submission):

> [ProjFL: Projection-Based Compression for Federated Learning](https://arxiv.org/abs/2511.05593)
