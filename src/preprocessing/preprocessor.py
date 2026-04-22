from typing import Optional, TypeAlias

import torch

from src import utils
from src.preprocessing.data_manager import DataManager


class Preprocessor:
    """Applies the dataset-agnostic preprocessing pipeline.

    Calls conf["preprocessor"]() to load and preprocess data, applies data
    limits, converts to the configured float precision, and optionally creates
    a validation split.
    """

    # x_train, y_train, x_test, y_test, x_val, y_val
    TypeOut: TypeAlias = tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]

    # TypeOut + dic of indexes of every writer/worker for FEMNIST dataset
    ExTypeOut: TypeAlias = tuple[*TypeOut, dict[str, list[int]]]

    def __call__(self, conf: dict) -> TypeOut | ExTypeOut:
        """Load, preprocess, limit, and optionally split the dataset described in conf.

        Calls conf["preprocessor"](), applies data limits, converts to the configured
        float precision, and creates a validation split if requested. Preprocessors that
        return 5 values (x_train, y_train, x_test, y_test, dic_idx) are treated as
        FEMNIST-style datasets; those returning 4 values are treated as standard datasets.

        Args:
            conf (dict): Configuration dictionary produced by ConfParser.

        Returns:
            TypeOut | ExTypeOut: (x_train, y_train, x_test, y_test, x_val, y_val,
                dic_idx). x_val and y_val are None when no validation split is used.
                dic_idx is None for preprocessors that do not return it.
        """
        dic_idx = None

        result = conf["preprocessor"]()
        if len(result) == 5:
            x_train, y_train, x_test, y_test, dic_idx = result
        else:
            x_train, y_train, x_test, y_test = result

        # limit the number of data if needed
        x_train, y_train = DataManager.limit_data(
            limit=conf["limit_n_data_train"], x=x_train, y=y_train
        )
        x_test, y_test = DataManager.limit_data(
            limit=conf["limit_n_data_test"], x=x_test, y=y_test
        )

        # put in the selected precision (torch.float32 by default)
        if (float_precision := conf["params_master"]["float_precision"]) != "float32":
            torch_precision = utils.get_torch_precision(float_precision)
            x_train = x_train.to(torch_precision)
            x_test = x_test.to(torch_precision)

        if conf["validation_size"] is not None:
            if dic_idx is not None:
                x_train, y_train, x_val, y_val, dic_idx = (
                    DataManager.create_validation_femnist(
                        x_train=x_train,
                        y_train=y_train,
                        val_size=conf["validation_size"],
                        random_state=conf["seed"],
                        dic_idx=dic_idx,
                    )
                )
            else:
                x_train, y_train, x_val, y_val = DataManager.create_validation(
                    x_train=x_train,
                    y_train=y_train,
                    val_size=conf["validation_size"],
                    random_state=conf["seed"],
                )
        else:
            x_val = None
            y_val = None

        return x_train, y_train, x_test, y_test, x_val, y_val, dic_idx
