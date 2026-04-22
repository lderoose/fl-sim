import os
import json
import shutil
import subprocess

from conf.paths import ROOT, REPORT_DIR


def test_swu_consistent(path_exec) -> None:

    # settings
    PATH_CONF_ENABLE = ROOT / "tests/consistency/swu/conf_swu_enable.py"
    PATH_CONF_DISABLE = ROOT / "tests/consistency/swu/conf_swu_disable.py"
    PATH_REPORT_ENABLE = ROOT / "tmp/swu_enable.json"
    PATH_REPORT_DISABLE = ROOT / "tmp/swu_disable.json"

    # create a tmp directory
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for path_conf, path_report in zip(
        [PATH_CONF_ENABLE, PATH_CONF_DISABLE], [PATH_REPORT_ENABLE, PATH_REPORT_DISABLE]
    ):
        subprocess.run(
            [
                "python",
                path_exec,
                "--path_config",
                path_conf,
                "--path_report",
                path_report,
            ]
        )

    # read the generated reports
    with open(PATH_REPORT_ENABLE, "r") as f:
        enable_report = json.load(f)
    with open(PATH_REPORT_DISABLE, "r") as f:
        disable_report = json.load(f)

    # check losses from the first report equal (not perfectly equal) losses from the second one
    assert (
        enable_report["params"]["params_master"]["n_epochs"]
        == disable_report["params"]["params_master"]["n_epochs"]
    ), "the # of epochs is not the same between both reports."

    for epoch in range(enable_report["params"]["params_master"]["n_epochs"]):
        loss_train_disable = disable_report["data_epoch"][f"{epoch}"]["loss_train"]
        loss_test_disable = disable_report["data_epoch"][f"{epoch}"]["loss_test"]
        loss_train_enable = enable_report["data_epoch"][f"{epoch}"]["loss_train"]
        loss_test_enable = enable_report["data_epoch"][f"{epoch}"]["loss_test"]

        assert (
            loss_train_disable == loss_train_enable
        ), f"expected same loss for epoch '{epoch}' on train set, \
            get {loss_train_disable=} vs {loss_train_enable=} instead."
        assert (
            loss_test_disable == loss_test_enable
        ), f"expected same loss for epoch '{epoch}' on train set, \
            get {loss_test_disable=} vs {loss_test_enable=} instead."

    # delete the useless reports
    shutil.rmtree("tmp")

    return None
