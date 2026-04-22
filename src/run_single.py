#!/usr/bin/env python3


import copy
import os
import pprint
from typing import Optional

import numpy as np
import torch

from src import utils
from src.preprocessing.conf_parser import ConfParser
from src.preprocessing.preprocessor import Preprocessor
from src.report.report import Report
from src.trainer.classical_trainer import ClassicalTrainer
from src.trainer.federated_trainer import FederatedTrainer


def main(path_config: str, path_report: Optional[str] = None) -> None:
    """Run a single experiment from a configuration file.

    Loads the configuration, preprocesses the dataset, initializes the model and
    report, then dispatches to either FederatedTrainer or ClassicalTrainer depending
    on the configuration.

    Args:
        path_config (str): Path to the Python configuration file to load.
        path_report (Optional[str]): Override for the report save path defined in the
            configuration. If None, the path from the configuration file is used.
    """
    app = utils.load_config(path_config)
    CONF = ConfParser()(app)
    # update the path_save if needed
    if path_report is not None:
        CONF["params_report"]["path_save"] = path_report
    pprint.pprint(CONF)

    # fix seed
    seed = CONF.get("seed")
    np.random.seed(seed)
    torch.manual_seed(CONF.get("seed"))
    if "cuda" in CONF["device"]:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # delete old logs files
    utils.clean_log_folder()

    # create required folder
    path_folder_report = os.path.dirname(CONF["params_report"]["path_save"])
    if not os.path.exists(path_folder_report):
        os.mkdir(path_folder_report)

    # loading dataset & define model
    x_train, y_train, x_test, y_test, x_val, y_val, dic_idx = Preprocessor()(CONF)

    # load model
    model = utils.load_model(CONF)

    # create empty report for saving data during fitting
    report = Report(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        params=CONF,
    )

    # run
    federated = CONF.get("federated")
    if federated:
        trainer = FederatedTrainer(
            conf=CONF,
            master_model=copy.deepcopy(model),
            x_train=x_train,
            y_train=y_train,
            report=report,
            dic_indexes=dic_idx
        )

    else:
        trainer = ClassicalTrainer(
            conf=CONF,
            model=copy.deepcopy(model),
            x_train=x_train,
            y_train=y_train,
            report=report,
        )

    trainer()
