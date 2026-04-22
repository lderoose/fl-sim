#!/usr/bin/env python3

from src.report.report_analyzer import ReportAnalyzer

from conf.paths import REPORT_DIR

if __name__ == "__main__":

    LS_REPORTS = [
        'federated_mnist_top.json', 
        'federated_mnist_top_ef.json', 
        'federated_mnist_top_projfl.json', 
        'federated_mnist_top_projfl_ef.json', 
        'federated_mnist_no_compression.json', 
    ]
    ls_path_reports = [REPORT_DIR / e for e in LS_REPORTS]

    repa = ReportAnalyzer(ls_path_reports)
    repa(
        ref_metric="loss",
        to_plot=[
            "total", 
            "master_to_workers", 
            "workers_to_master"
        ],
        dataset=[
            "train", 
            "test"
        ],
        title="Change in loss based on communication costs"
    )
