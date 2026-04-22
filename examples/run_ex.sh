#!/bin/bash

configs=(
    classical_california
    federated_mnist_no_compression
    federated_mnist_random
    federated_mnist_top
    federated_mnist_top_ef
    federated_mnist_top_projfl
    federated_mnist_top_projfl_ef
)
for config in "${configs[@]}"; do
    uv run main.py --path_config "examples/conf/${config}.py" --path_report "report/${config}.json"
done
