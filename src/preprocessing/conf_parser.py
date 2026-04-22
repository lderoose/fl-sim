from types import ModuleType

from conf.convention import CLASSIFICATION_DATASET


class ConfParser:
    """Parses a Python configuration module into a structured dictionary.

    Reads configuration variables from a Python module (e.g. application.py),
    assembles them into a nested dict, and validates parameter consistency before
    returning.
    """

    @staticmethod
    def check_parameters_consistency(conf: dict) -> None:
        """Validate the experiment configuration for internal consistency.

        Centralizes pre-training checks. Some additional checks may still occur
        at other locations when they cannot be performed without loading data.

        Args:
            conf (dict): Configuration dictionary produced by __call__.

        Raises:
            TypeError: If a parameter has an unexpected type.
            ValueError: If a parameter combination is invalid (e.g. variant without
                compressor, or accuracy metric enabled for a non-classification dataset).
        """
        # check device type
        if not isinstance(conf["device"], str):
            raise TypeError(
                f"expected 'str' type for parameter 'device', get {conf['device']} instead."
            )

        # check parameters limiting the number of data during the training and eval process
        for p_value in [conf["limit_n_data_train"], conf["limit_n_data_test"]]:
            if p_value is not None:
                if isinstance(p_value, int):
                    if p_value < conf["n_workers"]:
                        raise ValueError(
                            f"expected at least 1 data per worker, get a total of {p_value} data for {conf['n_workers']} instead."
                        )
                    # it's difficult to check 'p_value > n_data' without loading the datas so checking is directly performed in the limiting function
                elif isinstance(p_value, float):
                    if not 0 < p_value <= 1:
                        raise ValueError(
                            f"expected a value belonging to [0:1] if a limit is a float, get {p_value} instead."
                        )
                else:
                    raise TypeError(
                        f"expected type int | float | None for parameters limiting the number of data, get {type(p_value)} instead."
                    )

        # check validation consistency
        if conf["validation_size"] is None and conf["params_report"]["loss_validation"]:
            raise ValueError(
                f"in the configuration file you set 'loss_validation': True with 'VALIDATION_SIZE'=0.0"
            )

        # check whether a compressor is defined when variant is enable
        if conf["params_master"]["variant"]["name"] is not None:
            if conf["params_master"]["compressor"]["name"] is None:
                raise ValueError(
                    f"in the configuration file, you have defined a variant but no compressor has been found."
                )

        # check whether we report relevant metric in the report according to the selected dataset
        ls_keys = ["accuracy_train", "accuracy_validation", "accuracy_test"]
        for k in ls_keys:
            if k in conf["params_report"].keys():
                if conf["params_report"][k]:
                    if (ds := conf["dataset_name"]) not in CLASSIFICATION_DATASET:
                        raise ValueError(
                            f"the dataset '{ds}' is not listed in the classification dataset (cf. conf.convention.py) but you set '{k}=True' in the conf."
                        )
        return None

    def __call__(self, pyconffile: ModuleType) -> dict:
        """Parse a Python configuration module and return a validated configuration dict.

        Args:
            pyconffile (ModuleType): Loaded Python module containing configuration
                variables (e.g. the output of utils.load_config).

        Returns:
            dict: Structured configuration dictionary ready for use by trainers and
                preprocessors.
        """
        conf = {
            "device": pyconffile.DEVICE,
            "federated": pyconffile.FEDERATED,
            "seed": pyconffile.SEED,
            "n_workers": pyconffile.N_WORKERS,
            "dataset_name": pyconffile.DATASET_NAME,
            "preprocessor": pyconffile.PREPROCESSOR,
            "limit_n_data_train": pyconffile.LIMIT_N_DATA_TRAIN,
            "limit_n_data_test": pyconffile.LIMIT_N_DATA_TEST,
            "validation_size": pyconffile.VALIDATION_SIZE,
            "params_master": {
                "n_epochs": pyconffile.N_EPOCHS,
                "n_steps": pyconffile.N_STEPS,
                "batch_size": pyconffile.BATCH_SIZE,
                "model": pyconffile.MODEL,
                "optimizer": pyconffile.OPTIMIZER,
                "criterion": pyconffile.CRITERION,
                "compressor": pyconffile.COMPRESSOR,
                "variant": pyconffile.VARIANT,
                "float_precision": pyconffile.FLOAT_PRECISION,
                "sparse_weight_updates": pyconffile.SPARSE_WEIGHT_UPDATES,
                "stopping_criterion": pyconffile.STOPPING_CRITERION,
                "lr_scheduler": pyconffile.LR_SCHEDULER,
            },
            "params_report": pyconffile.REPORT_PARAMS,
        }
        __class__.check_parameters_consistency(conf)
        return conf
