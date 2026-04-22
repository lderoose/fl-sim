import copy
import json
from typing import Optional

import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

from src import utils


class Report:
    """Stores and serializes training metrics collected during an experiment.

    At each epoch, the report can record losses (train/test/validation), accuracy,
    gradient and weight norms, execution time, communication costs, and the learning
    rate. At the end of training the report is saved to a JSON file.
    """

    def __init__(
        self,
        params: dict,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the report with dataset tensors and experiment parameters.

        Args:
            params (dict): Full configuration dictionary produced by ConfParser.
            x_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training labels.
            x_test (torch.Tensor): Test input data.
            y_test (torch.Tensor): Test labels.
            x_val (Optional[torch.Tensor]): Validation input data. If None, validation
                metrics are not computed.
            y_val (Optional[torch.Tensor]): Validation labels.
        """
        self.device = params.get("device")
        self.params_report = params.get("params_report")
        self.path = self.params_report["path_save"]
        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.x_test = x_test.to(self.device)
        self.y_test = y_test.to(self.device)

        if (x_val is not None) and (y_val is not None):
            self.use_validation = True
            self.x_val = x_val.to(self.device)
            self.y_val = y_val.to(self.device)
        else:
            self.use_validation = False
            self.x_val = None
            self.y_val = None

        self.loss_function = params.get("params_master")["criterion"]()
        params_serialized = self._make_serializable(p=params)
        self.report = {
            "params": params_serialized,
            "data_epoch": {},
        }
        return None

    def _make_serializable(self, p: dict) -> dict:
        """Convert non-JSON-serializable values in the parameter dict to strings.

        Replaces class objects (model, optimizer, criterion, compressor, lr_scheduler,
        variant, stopping_criterion) with their __name__ string. Other values are
        left unchanged.

        Args:
            p (dict): Original parameter dictionary (not modified in place).

        Returns:
            dict: Deep copy of p with non-serializable values replaced by strings.
        """
        p_ser = copy.deepcopy(p)
        p_ser["params_master"]["model"] = p["params_master"]["model"].__name__
        p_ser["params_master"]["optimizer"]["name"] = p["params_master"]["optimizer"][
            "name"
        ].__name__
        p_ser["params_master"]["criterion"] = p["params_master"]["criterion"].__name__
        if p_ser["params_master"]["compressor"]["name"] is not None:
            p_ser["params_master"]["compressor"]["name"] = p["params_master"][
                "compressor"
            ]["name"].__name__
        if p_ser["params_master"]["lr_scheduler"]["name"] is not None:
            p_ser["params_master"]["lr_scheduler"]["name"] = p["params_master"][
                "lr_scheduler"
            ]["name"].__name__
        if p_ser["params_master"]["variant"]["name"] is not None:
            p_ser["params_master"]["variant"]["name"] = p["params_master"]["variant"][
                "name"
            ].__name__
        if p_ser["params_master"]["stopping_criterion"]["name"] is not None:
            p_ser["params_master"]["stopping_criterion"]["name"] = p["params_master"][
                "stopping_criterion"
            ]["name"].__name__
        p_ser["preprocessor"] = p["preprocessor"].__qualname__
        return p_ser

    def _fill_cc(self, epoch: int, **kwargs: dict) -> None:
        """Store the communication cost summary for a given epoch if requested.

        Requires "dic_cc" in kwargs and "communication_cost" enabled in params_report.

        Args:
            epoch (int): Index of the current epoch.
            **kwargs: Must contain "dic_cc" (dict with cc summary) when applicable.

        Raises:
            ValueError: If the cc entry for this epoch has already been filled.
        """
        if self.params_report["communication_cost"]:
            if "dic_cc" in kwargs:
                if "cc" not in self.report["data_epoch"][epoch].keys():
                    self.report["data_epoch"][epoch]["cc"] = kwargs["dic_cc"]
                else:
                    raise ValueError(
                        f"the dictionary of cc has already been filled. You are trying to override it. {self.report['data_epoch'][epoch]['cc']}"
                    )
        return None

    def _fill_loss_and_metrics(self, epoch: int, dataset: str, **kwargs: dict) -> None:
        """Compute and store the loss (and optionally accuracy) for a given dataset split.

        The model must be in evaluation mode. Loss and accuracy are only computed
        if the corresponding keys are enabled in params_report and have not yet been
        filled for this epoch.

        Args:
            epoch (int): Index of the current epoch.
            dataset (str): One of "train", "test", or "validation".
            **kwargs: Must contain "model" (nn.Module in eval mode) when a loss
                needs to be computed.

        Raises:
            ValueError: If an unsupported dataset split is requested, or if the
                model is in training mode when a loss must be computed.
        """
        # to compute some metrics when the predictions are done
        ls_metrics = ["accuracy"]
        ls_key_metrics = [f"{m}_{dataset}" for m in ls_metrics]

        if dataset == "train":
            key = "loss_train"
            x, y = self.x_train, self.y_train
        elif dataset == "test":
            key = "loss_test"
            x, y = self.x_test, self.y_test
        elif dataset == "validation":
            key = "loss_validation"
            x, y = self.x_val, self.y_val
            if self.x_val is None and self.y_val is None:
                # exit
                return None
        else:
            raise ValueError(
                f"expected 'train', 'test' or 'validation' for parameter dataset, get {dataset} instead."
            )

        # make the prediction to compute the loss
        if key in self.params_report.keys():
            if self.params_report[key]:
                if key not in self.report["data_epoch"][epoch]:
                    if "model" in kwargs:
                        if not kwargs["model"].training:
                            with torch.no_grad():
                                pred = kwargs["model"](x)
                        else:
                            raise ValueError(
                                f"the master model shouldn't be in training mode."
                            )
                        loss = self.loss_function(pred, y)
                        self.report["data_epoch"][epoch][key] = loss.item()

                        # compute the variance of the pred
                        key_var = f"var_{key}"
                        if key_var in self.params_report.keys():
                            if self.params_report[key_var]:
                                if key_var not in self.report["data_epoch"][epoch]:
                                    if isinstance(
                                        self.loss_function, torch.nn.CrossEntropyLoss
                                    ):
                                        loss_indiv = F.cross_entropy(
                                            pred, y, reduction="none"
                                        )

                                    elif isinstance(
                                        self.loss_function, torch.nn.BCELoss
                                    ):
                                        loss_indiv = F.binary_cross_entropy(
                                            pred, y, reduction="none"
                                        )
                                    else:
                                        m = f"the calculcation of the variance of the loss for '{self.loss_function}' is not yet supported"
                                        raise NotImplementedError(m)
                                    self.report["data_epoch"][epoch][key_var] = (
                                        torch.var(loss_indiv).detach().tolist()
                                    )
                                else:
                                    ValueError(
                                        f"the accuracy for the epoch {epoch} has been already written in the report. You are trying to overwrite it."
                                    )
                    else:
                        raise ValueError(
                            f"the argument 'model' is required if the {key} need to be calculate."
                        )

                else:
                    ValueError(
                        f"the loss for the epoch {epoch} has been already written in the report. You are trying to overwrite it."
                    )

        # given the predictions, compute some metrics as asked in the conf
        for key_metric in ls_key_metrics:
            if key_metric in self.params_report.keys():
                if self.params_report[key_metric]:
                    if key_metric not in self.report["data_epoch"][epoch]:
                        metric = key_metric.split("_")[0]
                        if metric == "accuracy":
                            y_pred = pred.argmax(dim=1)
                            # TODO: it's better to stay on the GPU to compute acc
                            acc = accuracy_score(y_pred.to("cpu"), y.to("cpu"))
                            self.report["data_epoch"][epoch][key_metric] = acc
                        else:
                            raise NotImplementedError(f"{metric=}")
                    else:
                        ValueError(
                            f"the accuracy for the epoch {epoch} has been already written in the report. You are trying to overwrite it."
                        )
        return None

    def _fill_time_exec(self, epoch: int, **kwargs: dict) -> None:
        """Store the epoch execution time if requested and not yet filled.

        Args:
            epoch (int): Index of the current epoch.
            **kwargs: Must contain "time_exec" (float) when applicable.

        Raises:
            ValueError: If time_exec is requested but not provided in kwargs.
        """
        if self.params_report["time_exec"]:
            if "time_exec" not in self.report["data_epoch"][epoch]:
                if "time_exec" in kwargs:
                    self.report["data_epoch"][epoch]["time_exec"] = kwargs["time_exec"]
                else:
                    raise ValueError(
                        f"add the 'time_exec' argument if the execution time is to be stored in the history."
                    )
        return None

    def _fill_weights(self, epoch: int, **kwargs: dict) -> None:
        """Store the model weights for a given epoch if requested.

        Args:
            epoch (int): Index of the current epoch.
            **kwargs: Must contain "model" (nn.Module) when applicable.

        Raises:
            ValueError: If weights are requested but no model is provided.
        """
        if self.params_report["weights"]:
            if "weights" not in self.report["data_epoch"][epoch]:
                if "model" in kwargs:
                    self.report["data_epoch"][epoch]["weights"] = [
                        w.detach().tolist() for w in kwargs["model"].parameters()
                    ]  # torch tensor is not serializable
                else:
                    raise ValueError(
                        f"the argument 'model' is required if the weights of the model is to be store in the history."
                    )
        return None

    def _fill_lnorm(self, epoch: int, **kwargs: dict) -> None:
        """Store L-norm values of weights or gradients for a given epoch if requested.

        Supports "gradients_l1_norm", "gradients_l2_norm", "weights_l1_norm",
        and "weights_l2_norm" keys in params_report.

        Args:
            epoch (int): Index of the current epoch.
            **kwargs: Must contain "model" (nn.Module) when a norm is requested.

        Raises:
            ValueError: If a norm is requested but no model is provided.
        """
        LS_ALLOWED_KEY = [
            "gradients_l1_norm",
            "gradients_l2_norm",
            "weights_l1_norm",
            "weights_l2_norm",
        ]
        for allowed_key in LS_ALLOWED_KEY:
            if allowed_key in self.params_report.keys():
                if self.params_report[allowed_key]:
                    if allowed_key not in self.report["data_epoch"][epoch]:
                        if "model" in kwargs:
                            # extract the selected norm
                            k_splitted = allowed_key.split("_")
                            norm_p = int(k_splitted[1][1])
                            is_grad = True if k_splitted[0] == "gradients" else False
                            self.report["data_epoch"][epoch][allowed_key] = (
                                utils.compute_lnorm(
                                    weights=kwargs["model"].parameters(),
                                    is_grad=is_grad,
                                    p=norm_p,
                                )
                            )
                        else:
                            raise ValueError(
                                f"the argument 'model' is required to compute the norm."
                            )
        return None

    def _fill_learning_rate(self, epoch: int, **kwargs: dict) -> None:
        """Store the learning rate for a given epoch if provided.

        Args:
            epoch (int): Index of the current epoch.
            **kwargs: May contain "learning_rate" (float).
        """
        if "learning_rate" in kwargs:
            if "learning_rate" not in self.report["data_epoch"][epoch]:
                self.report["data_epoch"][epoch]["lr"] = kwargs["learning_rate"]
        return None

    def fill(self, epoch: int, **kwargs) -> None:
        """Record all requested metrics for a given epoch.

        Creates an empty dict for the epoch on the first call, then delegates to
        the individual _fill_* methods. Accepted keyword arguments depend on which
        metrics are enabled in params_report.

        Args:
            epoch (int): Index of the current epoch.
            **kwargs: Optional keyword arguments forwarded to the _fill_* methods
                (e.g. model, time_exec, dic_cc, learning_rate).
        """
        # the first call of this function will create an empty dict for this specific epoch
        if epoch not in self.report["data_epoch"]:
            self.report["data_epoch"][epoch] = {}

        for dataset in ["train", "test", "validation"]:
            self._fill_loss_and_metrics(epoch, dataset=dataset, **kwargs)
        self._fill_cc(epoch, **kwargs)
        self._fill_time_exec(epoch, **kwargs)
        self._fill_weights(epoch, **kwargs)
        self._fill_lnorm(epoch, **kwargs)
        self._fill_learning_rate(epoch, **kwargs)
        return None

    def save(self) -> None:
        """Serialize the report to a JSON file at the configured path.

        If path is None, nothing is saved.
        """
        if self.path is not None:
            with open(self.path, "w") as f:
                json.dump(self.report, f)
        return None
