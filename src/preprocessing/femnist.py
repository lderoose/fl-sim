import json
from glob import glob
from pathlib import Path

import torch

from conf.paths import DATA_DIR


def load_and_preprocess_femnist(
    path_data: Path = DATA_DIR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    """Load and preprocess the FEMNIST dataset and build the per-writer index mapping.

    NB:
    - Part of this function should be in DataManager, but FEMNIST has a different
      structure than other datasets (federated by design).
    - To download the data:
      1. git clone https://github.com/TalwalkarLab/leaf.git
      2. Fix a bug in leaf/data/femnist/preprocess/data_to_json.py l64 by using
         Image.Resampling.LANCZOS
      3. Run (for 10% of the full dataset):
         cd leaf/data/femnist/ && ./preprocess.sh -s niid --sf 0.1 -k 0 -t sample
      4. Move the data to DATA_DIR/femnist/

    Args:
        path_data (Path): Root directory for dataset storage. Defaults to DATA_DIR.

    Returns:
        tuple: (x_train, y_train, x_test, y_test, dic_idx) where dic_idx maps
            each writer id to its list of indices in x_train.
    """
    x_train, y_train, x_test, y_test = [], [], [], []

    paths_train = glob(path_data / "femnist/sf_01/train/*.json")
    paths_test = glob(path_data / "femnist/sf_01/test/*.json")

    # sort path
    paths_train.sort(key=lambda x: int(x.split("/")[-1].split("_")[2]))
    paths_test.sort(key=lambda x: int(x.split("/")[-1].split("_")[2]))

    # init variables to catch indexes of every writer
    dic_idx = {}
    row_cpt, writer_cpt = 0, 0
    for path in paths_train:
        with open(path, "r") as f:
            data = json.load(f)
        for add_wid, user in enumerate(data["users"]):
            x_train.extend(data["user_data"][user]["x"])
            y_train.extend(data["user_data"][user]["y"])

            current_writer = writer_cpt + add_wid
            last_idx = row_cpt + data["num_samples"][add_wid]
            dic_idx[f"{current_writer}"] = [*range(row_cpt, last_idx)]
            row_cpt = last_idx
        writer_cpt += len(data["users"])

    for path in paths_test:
        with open(path, "r") as f:
            data = json.load(f)
        for user in data["users"]:
            x_test.extend(data["user_data"][user]["x"])
            y_test.extend(data["user_data"][user]["y"])

    x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, 1, 28, 28)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, 1, 28, 28)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, y_train, x_test, y_test, dic_idx
