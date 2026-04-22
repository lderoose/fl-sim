import importlib.util
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from conf.convention import DATE_FORMAT
from conf.paths import LOG_DIR


class AllStaticMethods(type):
    """class ClassName(metaclass=AllStaticMethods) involves all methods of ClassName become static"""

    def __new__(cls, name, bases, class_dict):
        for attr_name, attr_value in class_dict.items():
            if callable(attr_value):  # checks whether attr is a function
                class_dict[attr_name] = staticmethod(attr_value)
        return super().__new__(cls, name, bases, class_dict)


class FakeTensor(torch.Tensor):
    """the class compressors need to deal with object having an attribute 'grad' filled"""

    def __init__(self, weight: torch.Tensor) -> None:
        """Initialize FakeTensor by assigning the given tensor as its grad attribute.

        Args:
            weight (torch.Tensor): Tensor to expose as the grad attribute.
        """
        self.grad = weight
        return None



def clean_log_folder(log_dir: Path = LOG_DIR) -> None:
    """delete the log folder with its content and create a new one."""
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    # create, in both cases, the log folder
    os.mkdir(log_dir)
    return None


def compute_nsteps(dic_idx_workers: dict) -> int:
    """n steps is the maximal number of steps"""
    n_steps = 0
    for ls_idx in dic_idx_workers.values():
        n_steps = max(len(ls_idx), n_steps)
    return n_steps


def compute_weights(
    dic_idx_workers: dict, n_steps: int, precision: str
) -> torch.Tensor:
    """compute the weights for each worker for each step

    NB: there is a weight for each worker at each step, so weights it's a 2D matrix
    Args:
        dic_idx_workers (dict): dictionnary that indicate how many data the workers used for each step
        n_steps (int): the number of step (max)

    Returns:
        torch.Tensor: 2D matrix containing the weights for each worker and for each step
    """
    torch_precision = get_torch_precision(precision)
    weights = torch.zeros(size=(len(dic_idx_workers), n_steps), dtype=torch_precision)

    for step in range(n_steps):

        nb_datas = 0

        for worker, ls_ls_idx in dic_idx_workers.items():

            # some worker can be not involved in all steps
            try:
                n = len(ls_ls_idx[step])
                weights[int(worker), step] = n
                nb_datas += n

            except IndexError:
                # the 0 value already exist in weights due to init
                pass

        weights[:, step] = torch.div(weights[:, step], nb_datas)
    return weights


def load_model(conf: dict) -> nn.Module:
    """check if the model is relevant according dataset name and if yes load it. Otherwise, raises an error."""
    m = conf["params_master"]["model"]
    preproc_dataset_name = conf["dataset_name"]
    preproc_dataset_name = re.sub(r"_", "", preproc_dataset_name)
    if (ds := preproc_dataset_name) not in (m.__name__.lower()):
        raise ValueError(f"the model {m} is not compatible with the dataset {ds}")

    # put the relevant weight precision
    torch_precision = get_torch_precision(conf["params_master"]["float_precision"])
    return m().to(torch_precision)


def compute_lnorm(weights: list[torch.Tensor], is_grad: bool, p: int) -> list[float]:
    """compute the L norm of the weights or the L norm of the gradients.
    NB: the gradients must be an attribute of the weights.
    """
    ls_lnorm = []
    for weight in weights:
        if not is_grad:
            l1norm = weight.norm(p=p).item()
        else:
            l1norm = weight.grad.norm(p=p).item()
        ls_lnorm.append(l1norm)
    return ls_lnorm


def compute_exec_time(start: float) -> float:
    """Compute elapsed time in seconds from a given start timestamp to now."""
    end_time_epoch = time.time()
    return end_time_epoch - start


def compute_n_layers(model: nn.Module) -> int:
    """compute the number of layers of the model"""
    return len(list(model.parameters()))


def get_shape_gradients(model: nn.Module) -> dict[int, tuple]:
    """given the model, get the shape of gradients for each layer"""
    return {i: tuple(e.shape) for i, e in enumerate(model.parameters())}


def get_torch_precision(input: str) -> torch.dtype:
    """given the precision with a string return the relevant pytorch precision"""
    if input == "float32":
        torch_precision = torch.float32
    elif input == "float64":
        torch_precision = torch.float64
    elif input == "float16":
        torch_precision = torch.float16
    else:
        raise NotImplementedError(f"the precision '{input}' is not supported.")
    return torch_precision


def get_report_name(cpr: dict, federated: bool) -> str:
    """Build a timestamped filename for an experiment report.

    Prepends "nocpr_" when no compressor is used and "classical_" when training
    is non-federated.

    Args:
        cpr (dict): Compressor configuration dict with at least a "name" key.
        federated (bool): Whether the experiment uses federated training.

    Returns:
        str: Report filename ending in ".json".
    """
    dnow = datetime.now().strftime(format=DATE_FORMAT)
    if not federated:
        report_name = f"classical_{dnow}.json"
    else:
        report_name = f"{dnow}.json"
    if cpr["name"] is None:
        report_name = f"nocpr_{report_name}"
    return report_name


def load_config(path_config: str) -> ModuleType:
    """imports the relevant configuration file dynamically."""

    if not os.path.exists(path_config):
        raise ValueError(f"The configuration file '{path_config}' doesn't exist")

    module_name = os.path.splitext(os.path.basename(path_config))[0]  # rm .py

    spec = importlib.util.spec_from_file_location(module_name, path_config)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the module from '{path_config}'")

    conf = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = conf
    spec.loader.exec_module(conf)

    return conf


def extract_trainable_name_layers(model: nn.Module) -> list[str]:
    """given the model, extract a the name of the layer where the parameters are trainable"""
    LS_EXCLUDED = [".running_mean", ".running_var", ".num_batches_tracked"]  # batchnorm
    return [
        x
        for x in model.state_dict().keys()
        if not any(substr in x for substr in LS_EXCLUDED)
    ]
