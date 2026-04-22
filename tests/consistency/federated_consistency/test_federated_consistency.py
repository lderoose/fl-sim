import os
import json
import shutil
import subprocess

import numpy as np

from conf.paths import ROOT


def test_federated_consistency(path_exec) -> None:
    # settings
    TOL_EQUALITY = 1e-5
    PATH_CONF_CLASSICAL = (
        ROOT / "tests/consistency/federated_consistency/conf_classical.py"
    )
    PATH_CONF_FEDERATED = (
        ROOT / "tests/consistency/federated_consistency/conf_federated.py"
    )
    PATH_REPORT_CLASSICAL = ROOT / "tmp/classical.json"
    PATH_REPORT_FEDERATED = ROOT / "tmp/federated.json"

    # create a tmp directory
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for path_conf, path_report in zip(
        [PATH_CONF_CLASSICAL, PATH_CONF_FEDERATED],
        [PATH_REPORT_CLASSICAL, PATH_REPORT_FEDERATED],
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
    with open(PATH_REPORT_CLASSICAL, "r") as f:
        classical_report = json.load(f)
    with open(PATH_REPORT_FEDERATED, "r") as f:
        federated_report = json.load(f)

    # check losses from the first report equal (not perfectly equal) losses from the second one
    assert (
        classical_report["params"]["params_master"]["n_epochs"]
        == federated_report["params"]["params_master"]["n_epochs"]
    ), "the # of epochs is not the same between both reports."

    for epoch in range(classical_report["params"]["params_master"]["n_epochs"]):
        loss_train_classical = classical_report["data_epoch"][f"{epoch}"]["loss_train"]
        loss_test_classical = classical_report["data_epoch"][f"{epoch}"]["loss_test"]
        loss_train_federated = federated_report["data_epoch"][f"{epoch}"]["loss_train"]
        loss_test_federated = federated_report["data_epoch"][f"{epoch}"]["loss_test"]

        assert np.allclose(
            loss_train_classical, loss_train_federated, rtol=TOL_EQUALITY
        ), f"expected same loss for epoch '{epoch}' on train set, \
            get {loss_train_classical=} vs {loss_train_federated=} instead."
        assert np.allclose(
            loss_test_classical, loss_test_federated, rtol=TOL_EQUALITY
        ), f"expected same loss for epoch '{epoch}' on test set, \
            get {loss_test_classical=} vs {loss_test_federated=} instead."

    # NB: federated and classical training are mathematically equivalent, but strict numerical
    # equality is not achievable when n_workers > 1. Due to floating-point rounding, splitting
    # the gradient computation across workers produces tiny differences (~1e-7 in float32)
    # compared to computing it on the full batch at once. This is not a bug.
    #
    # Concretely, the two approaches compute the same sum in different orders:
    #   Classical: acc = grad_0 + grad_1 + ... + grad_999        (one chain of additions)
    #   Federated: S_0 = grad_0 + ... + grad_{n-1}               (partial sum, worker 0)
    #              S_1 = grad_n + ... + grad_999                  (partial sum, worker 1)
    #              acc = S_0 + S_1
    # Since floating-point addition is not associative, (a + b) + c != a + (b + c),
    # the two paths produce slightly different results.
    #
    # TOL_EQUALITY = 1e-5 is set well above this noise floor to catch real algorithmic
    # regressions while tolerating unavoidable rounding differences.

    # delete the useless reports
    shutil.rmtree("tmp")
    return None
