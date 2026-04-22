import json
import re
from pathlib import Path

import pandas as pd


class ReportReader:
    """Reads and parses a JSON experiment report produced by the Report class.

    Provides accessor methods to extract losses, accuracy, gradient norms,
    execution times, and communication costs, with support for the x-axis
    expressed as cumulative communication cost.
    """

    def __init__(self, path_report: Path) -> None:
        """Load and parse a report from a JSON file.

        Args:
            path_report (Path): Path to the JSON report file.
        """
        self.path_report = path_report
        self.report_name = path_report.name

        # read the report
        self.dic_report = __class__.read_report(self.path_report)

        # useful general info
        self.method = self.get_method()
        self.dataset_name = self.get_params()["dataset_name"]
        self.n_epochs = self.get_params_master()["n_epochs_up"]
        self.loss_name = self.get_loss_name()

        # if early stopping enables only
        self.epoch_model_used = self.get_epoch_model_used()

        # parse the data linked to each epoch
        self.df_data_epoch = self._preprocess_data_epoch()
        return None

    def _preprocess_data_epoch(self) -> pd.DataFrame:
        """Build the per-epoch DataFrame from the raw report dict.

        Computes cumulative and per-epoch communication costs for each
        direction (master_to_workers, workers_to_master, total).

        Returns:
            pd.DataFrame: DataFrame indexed by epoch with one column per metric.
        """
        ls_keys = ["master_to_workers", "workers_to_master", "total"]

        df = pd.DataFrame.from_dict(self.dic_report["data_epoch"]).T

        for k in ls_keys:

            df[f"cum_cc_{k}_{self.dataset_name}_{self.method}"] = df["cc"].apply(lambda x: x[k])

            df[f"cc_{k}_{self.dataset_name}_{self.method}"] = df[
                f"cum_cc_{k}_{self.dataset_name}_{self.method}"
            ].diff(1)
            df.loc["0", f"cc_{k}_{self.dataset_name}_{self.method}"] = df.loc[
                "0", f"cum_cc_{k}_{self.dataset_name}_{self.method}"
            ]
            df[f"cc_{k}_{self.dataset_name}_{self.method}"] = df[
                f"cc_{k}_{self.dataset_name}_{self.method}"
            ].astype(int)

        df.drop(columns=["cc"], inplace=True)
        return df

    def get_epoch_model_used(self) -> int:
        """Return the epoch index of the model that should be used for evaluation.

        When early stopping is enabled, the best model is saved patience epochs
        before the last one, so the effective epoch is n_epochs - 1 - patience.

        Returns:
            int: Index of the epoch whose saved checkpoint is the best model.
        """
        epoch_model_used = self.n_epochs - 1
        params_master = self.get_params_master()
        if "stopping_criterion" in params_master.keys():
            if params_master["stopping_criterion"]["name"] == "EarlyStopping":
                patience = params_master["stopping_criterion"]["kwargs"]["patience"]
                epoch_model_used -= patience
        return epoch_model_used

    def get_method(self) -> str:
        """Return a short string identifying the compression method used in this report.

        The string is "nocpr" when no compression is used, otherwise it encodes the
        compressor type, k value, and variant (if any).

        Returns:
            str: Method identifier string (e.g. "Top0.1factor_EF0.9").
        """
        dic_params_master = self.get_params_master()

        # compressor
        cpr = dic_params_master["compressor"]
        cpr_name = cpr["name"]

        if cpr_name is None:
            method = "nocpr"

        if cpr_name == "TopKCompressor":
            method = f"Top{cpr['kwargs']['k']}"
            if cpr["kwargs"]["k_as_factor"]:
                method += "factor"

        elif cpr["name"] == "RandomKCompressor":
            method = f"Random{cpr['kwargs']['k']}"
            if cpr["kwargs"]["k_as_factor"]:
                method += "factor"

        # variant
        vrt = dic_params_master["variant"]

        if vrt["name"] is not None:
            if vrt["name"] == "ErrorFeedback":
                method += f"_EF{vrt['kwargs']['alpha']}"

            elif vrt["name"] == "ProjFL":
                method += f"_ProjFL{vrt['kwargs']['K']}"

                if vrt["kwargs"]["EF"]:
                    method += "EF"

            elif vrt["name"] == "ErrorFeedback21":
                batch_size = self.get_batch_size()
                if batch_size is not None:
                    method += f"EF21_B{batch_size}"
                else:
                    method += f"EF21"

            elif vrt["name"] == "Diana":
                batch_size = self.get_batch_size()
                alpha = self.get_params_master()["variant"]["kwargs"]["alpha"]
                beta = self.get_params_master()["variant"]["kwargs"]["beta"]
                if batch_size is not None:
                    method += f"Diana_a{alpha}_b{beta}_B{batch_size}"
                else:
                    method += f"Diana_a{alpha}_b{beta}"

            else:
                raise ValueError(
                    f"define how this variant : {vrt['name']} should be displayed in the legend of the plots."
                )

        return method

    def get_loss_name(self) -> str:
        """Return the name of the loss function used in this experiment.

        Returns:
            str: Loss class name (e.g. "CrossEntropyLoss").
        """
        loss_name = str(self.get_params_master()["criterion"]).split(".")[-1]
        return re.sub(r"\'>", "", loss_name)

    @staticmethod
    def read_report(path_report: Path) -> dict:
        """Load a JSON report file and return it as a Python dict.

        Args:
            path_report (Path): Path to the JSON report file.

        Returns:
            dict: Parsed report dictionary.
        """
        with open(path_report, "rb") as file:
            dic_report = json.load(file)
        return dic_report

    def get_params(self) -> dict:
        """Return the top-level parameters dict from the report.

        Returns:
            dict: Parameters as stored in the report (cf. ConfParser output).
        """
        return self.dic_report["params"]

    def get_params_master(self) -> dict:
        """Return the master-specific parameters dict from the report.

        Returns:
            dict: Parameters under the "params_master" key.
        """
        return self.get_params()["params_master"]

    def get_batch_size(self) -> int:
        """Return the global batch size used by the master.

        Returns:
            int: Batch size value from the master parameters.
        """
        return self.get_params_master()["batch_size"]

    def get_loss(self, is_train: bool = True) -> pd.DataFrame:
        """Return the loss evolution over epochs.

        Args:
            is_train (bool): If True, return the training loss; otherwise the test loss.

        Returns:
            pd.DataFrame: Single-column DataFrame indexed by epoch.
        """
        col_loss = "loss_train" if is_train else "loss_test"
        df = self.df_data_epoch[col_loss].to_frame()
        df.rename(
            columns={col_loss: f"{self.dataset_name}_{col_loss}_{self.method}"},
            inplace=True,
        )
        return df

    def get_loss_cc(
        self, cc_metric: str = "total", is_train: bool = True
    ) -> pd.DataFrame:
        """Return the loss evolution with cumulative communication cost as the x-axis.

        Args:
            cc_metric (str): Communication cost direction to use as x-axis.
                One of "total", "master_to_workers", or "workers_to_master".
            is_train (bool): If True, use training loss; otherwise test loss.

        Returns:
            pd.DataFrame: DataFrame indexed by cumulative communication cost.
        """
        col_cc = f"cum_cc_{cc_metric}_{self.dataset_name}_{self.method}"
        df_loss = self.get_loss(is_train)
        df_loss.index = self.df_data_epoch[col_cc]
        df_loss.index.name = "cum_cc"
        return df_loss

    def get_accuracy(self, is_train: bool = True) -> pd.DataFrame:
        """Return the accuracy evolution over epochs.

        Args:
            is_train (bool): If True, return training accuracy; otherwise test accuracy.

        Returns:
            pd.DataFrame: Single-column DataFrame indexed by epoch.
        """
        col_acc = "accuracy_train" if is_train else "accuracy_test"
        df = self.df_data_epoch[col_acc].to_frame()
        df.rename(
            columns={col_acc: f"{self.dataset_name}_{col_acc}_{self.method}"},
            inplace=True,
        )
        return df

    def get_accuracy_cc(
        self, cc_metric: str = "total", is_train: bool = True
    ) -> pd.DataFrame:
        """Return the accuracy evolution with cumulative communication cost as the x-axis.

        Args:
            cc_metric (str): Communication cost direction to use as x-axis.
            is_train (bool): If True, use training accuracy; otherwise test accuracy.

        Returns:
            pd.DataFrame: DataFrame indexed by cumulative communication cost.
        """
        col_cc = f"cum_cc_{cc_metric}_{self.dataset_name}_{self.method}"
        df_loss = self.get_accuracy(is_train)
        df_loss.index = self.df_data_epoch[col_cc]
        df_loss.index.name = "cum_cc"
        return df_loss

    def get_gradient_norm(self, p: int = 2) -> pd.DataFrame:
        """Return the gradient L-norm evolution over epochs (summed across layers).

        Args:
            p (int): Order of the norm (1 or 2).

        Returns:
            pd.DataFrame: Single-column DataFrame indexed by epoch with summed norms.
        """
        col_norm = f"gradients_l{p}_norm"
        df = self.df_data_epoch[col_norm].to_frame()
        df.rename(
            columns={col_norm: f"{self.dataset_name}_{col_norm}_{self.method}"},
            inplace=True,
        )
        df = df.applymap(sum)
        return df

    def get_gradient_norm_cc(
        self, cc_metric: str = "total", p: int = 2
    ) -> pd.DataFrame:
        """Return the gradient L-norm evolution with cumulative communication cost as the x-axis.

        Args:
            cc_metric (str): Communication cost direction to use as x-axis.
            p (int): Order of the norm (1 or 2).

        Returns:
            pd.DataFrame: DataFrame indexed by cumulative communication cost.
        """
        col_cc = f"cum_cc_{cc_metric}_{self.dataset_name}_{self.method}"
        df_loss = self.get_gradient_norm(p)
        df_loss.index = self.df_data_epoch[col_cc]
        df_loss.index.name = "cum_cc"
        return df_loss

    def get_timexec(self) -> pd.DataFrame:
        """Return the per-epoch execution time series.

        Returns:
            pd.DataFrame: Series of execution times indexed by epoch.
        """
        return self.df_data_epoch["time_exec"]

    def get_cum_timexec(self) -> pd.DataFrame:
        """Return the cumulative execution time over epochs.

        Returns:
            pd.DataFrame: Single-column DataFrame with cumulative execution time.
        """
        df_time_exec = self.get_timexec()
        df_cum = pd.DataFrame(df_time_exec.cumsum())
        df_cum.columns = [f"{self.dataset_name}_cumulative_timexec_{self.method}"]
        return df_cum

    def get_l1_norm(self, is_weights: bool = True) -> pd.DataFrame:
        """Return the L1-norm evolution layer by layer over epochs.

        Args:
            is_weights (bool): If True, return the weights L1 norm; otherwise
                the gradients L1 norm.

        Returns:
            pd.DataFrame: DataFrame with one column per layer, indexed by epoch.
        """
        c = "weights_l1_norm" if is_weights else "gradients_l1_norm"
        df_l1_norm = self.df_data_epoch[c]
        df_l1_norm = pd.DataFrame(df_l1_norm.tolist())
        df_l1_norm.columns = [f"{c}_{i}" for i in range(df_l1_norm.shape[1])]
        return df_l1_norm

    def get_weights_L1_norm(self) -> pd.DataFrame:
        """Return the L1-norm evolution of the model weights, layer by layer.

        Returns:
            pd.DataFrame: DataFrame with one column per layer, indexed by epoch.
        """
        return self.get_l1_norm(is_weights=True)

    def get_gradients_L1_norm(self) -> pd.DataFrame:
        """Return the L1-norm evolution of the gradients, layer by layer.

        Returns:
            pd.DataFrame: DataFrame with one column per layer, indexed by epoch.
        """
        return self.get_l1_norm(is_weights=False)
