import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import Unpack

from conf.convention import DATE_FORMAT
from conf.paths import PLOT_DIR
from src.report.report_reader import ReportReader

warnings.filterwarnings("ignore", category=FutureWarning)


class ReportAnalyzer:
    """Aggregates multiple experiment reports and produces comparison plots.

    Expects exactly one report without compression (the reference) and any number
    of reports with compression. Validates that all reports share the same dataset
    and loss function before generating plots.
    """

    LS_HANDLED_KWARGS = ["ref_metric", "to_plot", "dataset", "title"]

    def __init__(self, ls_path_reports: list[Path]) -> None:
        """Load and validate all experiment reports.

        Args:
            ls_path_reports (list[Path]): Paths to the JSON report files to compare.

        Raises:
            ValueError: If there is not exactly one report without compression, or if
                the reports use different datasets or loss functions.
        """
        self.reports: list[ReportReader] = []

        # read every report
        for path_report in ls_path_reports:
            self.reports.append(ReportReader(path_report=path_report))

        # find the index of the report without compression
        self.idx_nocpr = self.get_idx_nocpr_report()

        self.report_name_ref = self.reports[self.idx_nocpr].report_name
        self.loss_name_ref = self.reports[self.idx_nocpr].loss_name
        self.dataset_name_ref = self.reports[self.idx_nocpr].dataset_name
        self.n_epochs_ref = self.reports[
            self.idx_nocpr
        ].n_epochs  # all reports must be compare according this number of epochs

        # some checking
        self._check_identical_dataset()
        self._check_identical_loss()
        return None

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[ReportReader]:
        return iter(self.reports)

    def __repr__(self) -> str:
        ls_reports_name = [r.report_name for r in self.reports]
        return f"{ls_reports_name}"

    def __getitem__(self, k) -> ReportReader:
        return self.reports[k]

    def _check_kwargs(self, kwargs: dict) -> None:
        """Raise TypeError if any keyword argument is not in LS_HANDLED_KWARGS.

        Args:
            kwargs (dict): Keyword arguments to validate.

        Raises:
            TypeError: If an unexpected keyword argument is found.
        """
        for kw in kwargs:
            if kw not in self.LS_HANDLED_KWARGS:
                raise TypeError(f"Unexpected keyword argument '{kw}'")
        return None

    def _check_identical_dataset(self) -> None:
        """Raise ValueError if the reports do not all refer to the same dataset.

        Raises:
            ValueError: If any report uses a different dataset than the reference.
        """
        for report in self:
            dataset_name_tmp = report.dic_report["params"]["dataset_name"]
            if self.dataset_name_ref != dataset_name_tmp:
                raise ValueError(
                    f"report {self.report_name_ref} and {report.report_name} doesn't refer to the same dataset: {self.dataset_name_ref} vs {dataset_name_tmp}"
                )
        return None

    def _check_identical_loss(self) -> None:
        """Raise ValueError if the reports do not all use the same loss function.

        Raises:
            ValueError: If any report uses a different loss than the reference.
        """
        for report in self.reports:
            loss_name_tmp = report.dic_report["params"]["params_master"]["criterion"]
            if self.loss_name_ref != loss_name_tmp:
                raise ValueError(
                    f"report {self.report_name_ref} and {report.report_name} doesn't refer to the same loss: {self.loss_name_ref} vs {report.loss_name}"
                )
        return None

    def get_idx_nocpr_report(self) -> int:
        """Return the index of the unique report without compression in the list.

        Raises:
            ValueError: If there are zero or more than one such reports.

        Returns:
            int: Index of the no-compression reference report.
        """
        idx_report_ref = None
        for i, report in enumerate(self.reports):
            if report.method == "nocpr":
                if idx_report_ref is not None:
                    raise ValueError(
                        f"find at least 2 reports without compression: {idx_report_ref} & {i}."
                    )
                else:
                    idx_report_ref = i

        # checks whether a report without compression has been found
        if idx_report_ref is None:
            raise ValueError(
                f"0 report without compression found, add exactly one in the list."
            )
        return idx_report_ref

    def summarize_dataframe(
        self, cc_metric: str = "total", is_train: bool = True
    ) -> tuple[pd.DataFrame, dict]:
        """Build a merged DataFrame with the ref_metric for all reports, indexed by CC.

        Merges all reports on a common cumulative communication cost index and also
        returns a dict with the (x, y) coordinates of the epoch_model_used marker
        for each report.

        Args:
            cc_metric (str): Communication cost direction to use as x-axis.
                One of "total", "master_to_workers", or "workers_to_master".
            is_train (bool): If True, use training split metrics; otherwise test split.

        Returns:
            tuple[pd.DataFrame, dict]: Merged DataFrame and a dict of marker positions
                keyed by column name.
        """
        # init
        dic_idx_nan = {}
        df_cc_ev = pd.DataFrame()

        # to be displayed in the plots (model used)
        dic_epoch_model_used = {}

        for report in self:
            if self.ref_metric == "loss":
                df_tmp = report.get_loss_cc(cc_metric=cc_metric, is_train=is_train)
            elif self.ref_metric == "accuracy":
                df_tmp = report.get_accuracy_cc(cc_metric=cc_metric, is_train=is_train)
            elif self.ref_metric == "gradients_l2_norm":
                df_tmp = report.get_gradient_norm_cc(
                    cc_metric=cc_metric, is_train=is_train, p=2
                )
            else:
                raise NotImplementedError(
                    f"This metric '{self.ref_metric}' is not yet supported."
                )

            # define some columns names
            col_epoch = f"{report.dataset_name}_epoch_{report.method}"

            # create the column 'col_epoch'
            df_tmp[col_epoch] = [*range(len(df_tmp))]

            # update the dictionary to know how to fill the NaN values for each column after the merge
            for col in df_tmp.columns:
                if col not in dic_idx_nan.keys():
                    dic_idx_nan[col] = {
                        "idx_start": df_tmp.index[0],
                        "idx_end": df_tmp.index[-1],
                    }
                else:
                    raise SystemError(
                        f"unable to do the merge, at least 2 reports have similar columns: {col}"
                    )

            # merge
            df_cc_ev = pd.merge(
                df_cc_ev, df_tmp, how="outer", left_index=True, right_index=True
            )

            col_loss = df_tmp.columns.tolist()
            col_loss.remove(col_epoch)
            col_loss = col_loss[0]

            tab = df_cc_ev[df_cc_ev[col_epoch] == report.epoch_model_used]
            loss_epoch_model_used = tab[col_loss].values[0]
            cc_epoch_model_used = tab.index[0]

            dic_epoch_model_used[col_loss] = {
                "epoch": report.epoch_model_used,
                "x": cc_epoch_model_used,
                "y": loss_epoch_model_used,
            }

        return df_cc_ev, dic_epoch_model_used

    def _plot_summary_dataframe(
        self,
        df_summary: pd.DataFrame,
        path_save_folder: Optional[Path] = None,
        cc_metric: str = "total",
        is_train: bool = True,
    ) -> None:
        """Plot the merged summary DataFrame and save it to disk.

        Draws one line per method with a log-scale x-axis (communication cost).
        The no-compression reference is drawn in black; compressed methods use
        colored markers.

        Args:
            df_summary (pd.DataFrame): Merged DataFrame produced by summarize_dataframe.
            path_save_folder (Optional[Path]): Directory where the plot file is saved.
            cc_metric (str): Communication cost metric used as the x-axis label.
            is_train (bool): If True, uses the training split label; otherwise test.
        """
        # define filename
        suffixe = "trainset" if is_train else "testset"
        filename = f"plot_{cc_metric}CC_{suffixe}.pdf"
        path_file = path_save_folder / filename

        # plot
        ls_cols = [c for c in df_summary if self.ref_metric in c]

        plt.figure()

        markers = ["D", "*", "P", "x", "v", "1", ".", "|", "o", "", ">"]
        colors = [
            "teal",
            "darkviolet",
            "darkblue",
            "darkgreen",
            "darkorange",
            "darkred",
            "dodgerblue",
            "teal",
            "green",
            "violet",
        ]

        for i, col in enumerate(ls_cols):
            col_epoch = col.replace(self.ref_metric, "epoch")
            col_epoch = col_epoch.replace("_train", "").replace(
                "_test", ""
            )  # delete train | test
            max_epoch = df_summary[col_epoch].max() + 1  # +1, it starts at 0

            # clean label
            if is_train:
                lab = re.sub(
                    rf"{self.dataset_name_ref}_{self.ref_metric}_train_", "", col
                )
            else:
                lab = re.sub(
                    rf"{self.dataset_name_ref}_{self.ref_metric}_test_", "", col
                )
            lab = re.sub(r"factor", "", lab)

            if "nocpr" in col_epoch:
                (line,) = plt.plot(
                    # df_summary.index,
                    df_summary[col].dropna(),
                    color="black",
                    label=f"{lab} -- {int(max_epoch)} epochs",
                )

            else:
                n_pts = len(df_summary[col].dropna())
                indices = np.linspace(0, n_pts - 1, 5, dtype=int)

                (line,) = plt.plot(
                    # df_summary.index,
                    df_summary[col].dropna(),
                    marker=markers[i],
                    color=colors[i],
                    label=f"{lab} -- {int(max_epoch)} epochs",
                    markevery=indices,
                    markersize=10,
                )

        ylab = f"test {self.ref_metric}" if not is_train else f"train {self.ref_metric}"
        plt.ylabel(ylab)
        plt.xscale("log")
        plt.xlabel("communication costs (bytes)")
        plt.grid()
        plt.title(self.title)
        plt.legend()
        plt.savefig(path_file)
        return None

    def __call__(
        self, path_save_folder: Optional[str] = None, **kwargs: Unpack[dict]
    ) -> None:
        """Generate and save comparison plots for all configured metrics and datasets.

        Creates a timestamped subdirectory under path_save_folder and writes one
        PDF file per (dataset, cc_metric) combination.

        Args:
            path_save_folder (Optional[str]): Root directory for saving plots.
                Defaults to the PLOT_DIR path defined in conf/paths.py.
            **kwargs: Optional keyword arguments:
                - ref_metric (str): Metric to plot ("loss", "accuracy",
                  "gradients_l2_norm"). Defaults to "loss".
                - to_plot (list[str]): List of cc_metric values to plot.
                  Defaults to ["total", "master_to_workers", "workers_to_master"].
                - dataset (list[str]): Dataset splits to plot.
                  Defaults to ["train", "test"].
                - title (str): Title for all generated plots. Defaults to "TBD".
        """
        # unpack kwargs
        self._check_kwargs(kwargs)
        self.ref_metric = kwargs.get("ref_metric", "loss")
        ls_cc_metric = kwargs.get(
            "to_plot", ["total", "master_to_workers", "workers_to_master"]
        )
        ls_dataset = kwargs.get("dataset", ["train", "test"])
        self.title = kwargs.get("title", "TBD")

        # os - there is a main folder which contain a second folder (name is the current date) and all plots will be inserted inside
        if path_save_folder is None:
            path_save_folder = PLOT_DIR  # default folder

        dnow = datetime.now().strftime(format=DATE_FORMAT)
        path_save_folder = path_save_folder / f"{dnow}"

        # checks whether this folder already exist
        if not os.path.exists(path_save_folder):
            os.makedirs(path_save_folder)

        # make the plots
        pbar = tqdm(total=len(ls_dataset) * len(ls_cc_metric))
        for dataset in ls_dataset:
            is_train = True if dataset == "train" else False
            for cc_metric in ls_cc_metric:
                df_summary, _ = self.summarize_dataframe(
                    cc_metric=cc_metric,
                    is_train=is_train,
                )

                self._plot_summary_dataframe(
                    df_summary=df_summary,
                    cc_metric=cc_metric,
                    is_train=is_train,
                    path_save_folder=path_save_folder
                )

                pbar.update(1)
        return None
