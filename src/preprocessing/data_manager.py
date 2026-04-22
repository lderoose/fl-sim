import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class DataManager:
    """Utility class for distributing data indices across workers and steps.

    Supports two distribution strategies:
    - Strategy A (get_idx_workers_A): fixed number of steps, data split evenly.
    - Strategy B (get_idx_workers_B): fixed batch size, steps derived from data size.
    """

    def _check_n_datas(dic: dict, n_datas: int) -> None:
        """Assert that the index distribution covers every data point exactly once.

        Args:
            dic (dict): Worker index dictionary produced by get_idx_workers_A/B.
            n_datas (int): Expected total number of data points.

        Raises:
            AssertionError: If any index is missing or duplicated.
        """
        ls_th_idx = []
        s = 0
        for step in dic.values():
            for e in step:
                s += len(e)
                ls_th_idx.extend(e)

        assert (
            s == n_datas
        ), f"there is either more data or less than required: {n_datas=} != {s}"
        ls_real_idx = [*range(n_datas)]
        ls_th_idx.sort()

        set_diff = set(ls_real_idx).difference(set(ls_th_idx))
        assert (
            ls_real_idx == ls_th_idx
        ), f"these index has not been attributed: {set_diff}."
        return None

    def _split_sum(a: int, b: int) -> list[int]:
        """Split an integer a into b integer parts whose sum equals a.

        The parts are as evenly distributed as possible; extra units are assigned
        to the first partitions.

        Args:
            a (int): Integer to split.
            b (int): Number of parts.

        Returns:
            list[int]: List of b integers summing to a.

        Example:
            >>> DataManager._split_sum(10, 3)
            [4, 3, 3]
        """
        base = a // b
        remainder = a % b
        return [base + 1 if i < remainder else base for i in range(b)]

    def _check_inputs(n_workers: int, n_steps: int, n_datas: int) -> None:
        """Validate the basic feasibility constraints for data distribution.

        Args:
            n_workers (int): Number of workers.
            n_steps (int): Number of steps.
            n_datas (int): Total number of data points.

        Raises:
            ValueError: If there are fewer data points than workers, or if the
                number of steps exceeds the number of data points.
        """
        if n_datas < n_workers:
            raise ValueError(
                f"the number of workers is greater than the number of instances."
            )
        if (n_datas // n_steps) == 0:
            raise ValueError(
                f"the number of steps is greater than the number of instances."
            )
        return None

    @staticmethod
    def split_indices_in_steps(indices: list[int], n_steps: int) -> list[list[int]]:
        """Split a list of indices into n_steps contiguous blocks of near-equal size.

        Args:
            indices (list[int]): List of indices to split.
            n_steps (int): Number of blocks to produce.

        Returns:
            list[list[int]]: List of n_steps sub-lists covering all indices.
        """
        n = len(indices)
        step_size = n // n_steps
        remainder = n % n_steps

        steps = []
        start = 0

        for s in range(n_steps):
            extra = 1 if s < remainder else 0
            end = start + step_size + extra
            steps.append(indices[start:end])
            start = end

        return steps

    @staticmethod
    def get_idx_workers_A(
        n_workers: int, n_steps: int, n_datas: int
    ) -> tuple[dict[int, list[int]], float]:
        """Distribute data indices across workers and steps with a fixed step count.

        The global batch at each step is split as evenly as possible across workers.
        Indices are assigned sequentially (no shuffling).

        Args:
            n_workers (int): Number of workers.
            n_steps (int): Number of training steps.
            n_datas (int): Total number of data instances.

        Returns:
            tuple[dict, float]: A tuple (dic_data_index_workers, mean_gbs_step) where:
                - dic_data_index_workers maps each worker id (str) to its list of index
                  batches, one per step.
                - mean_gbs_step is the average global batch size per step.

        Example:
            >>> dic, mean_gbs = DataManager.get_idx_workers_A(n_workers=2, n_steps=2, n_datas=10)
            >>> dic
            {'0': [[0, 1, 2], [5, 6, 7]], '1': [[3, 4], [8, 9]]}
            >>> mean_gbs
            np.float64(5.0)
        """
        __class__._check_inputs(n_workers, n_steps, n_datas)

        dic_data_index_workers = {f"{i}": [] for i in range(n_workers)}

        # gbs : global batch size: the number of instances processed by all the workers during a step
        ls_gbs_step = __class__._split_sum(a=n_datas, b=n_steps)
        mean_gbs_step = np.mean(ls_gbs_step)
        start = 0
        for gbs_step in ls_gbs_step:
            ls_bs_worker = __class__._split_sum(a=gbs_step, b=n_workers)

            for worker, bs_worker in enumerate(ls_bs_worker):
                end = bs_worker + start
                dic_data_index_workers[f"{worker}"].append([*range(start, end)])
                start += bs_worker

        __class__._check_n_datas(dic=dic_data_index_workers, n_datas=n_datas)
        return dic_data_index_workers, mean_gbs_step

    @staticmethod
    def get_idx_workers_B(
        n_datas: int, n_workers: int, batch_size: int
    ) -> dict[int, list]:
        """Distribute data indices across workers using a fixed global batch size.

        Data is split into batches of size batch_size, then each batch is divided
        as evenly as possible across workers. When batch_size is not divisible by
        n_workers, the remainder indices are given one by one to the first workers.

        Args:
            n_datas (int): Total number of data instances.
            n_workers (int): Number of workers.
            batch_size (int): Number of instances per global batch.

        Returns:
            dict[str, list]: Mapping from worker id (str) to its list of index
                batches, one per global batch.

        Example:
            >>> dic = DataManager.get_idx_workers_B(n_datas=10, n_workers=3, batch_size=6)
            >>> dic
            {'0': [[0, 1], [6, 7]], '1': [[2, 3], [8, 9]], '2': [[4, 5]]}
        """
        dic_data_index_workers = {f"{i}": [] for i in range(n_workers)}

        remainder_batch_size = batch_size % n_workers
        n_datas_per_worker = batch_size // n_workers

        # compute n_exceed: the number of datas remaining to fill the exact batch_size
        if remainder_batch_size != 0:
            n_exceed = batch_size - n_workers * (batch_size // n_workers)
        else:
            n_exceed = 0

        # iterate over each batch
        for batch_start in range(0, n_datas, batch_size):

            batch_start_worker = batch_start
            cpt_n_exceed = n_exceed

            for worker_id in range(n_workers):

                # case n_workers is a multiple of the batch_size
                if remainder_batch_size == 0:
                    batch_end_worker = min(
                        batch_start_worker + n_datas_per_worker, n_datas
                    )

                # case n_workers is not a multiple, so workers has not the same amount of datas
                else:

                    # 2 subcases:
                    if cpt_n_exceed > 0:
                        batch_end_worker = min(
                            batch_start_worker + n_datas_per_worker + 1,
                            batch_start + batch_size,
                            n_datas,
                        )
                        cpt_n_exceed -= 1

                    else:
                        batch_end_worker = min(
                            batch_start_worker + n_datas_per_worker,
                            batch_start + batch_size,
                            n_datas,
                        )

                if batch_start_worker != batch_end_worker:
                    dic_data_index_workers[f"{worker_id}"].append(
                        [*range(batch_start_worker, batch_end_worker, 1)]
                    )
                    batch_start_worker = batch_end_worker  # update start position

        __class__._check_n_datas(dic=dic_data_index_workers, n_datas=n_datas)
        return dic_data_index_workers

    @staticmethod
    def limit_data(
        limit: int | float | None, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract a random subset of x and y according to the limit parameter.

        Args:
            limit (int | float | None): If None or 1, all data is returned.
                If a float in (0, 1), a fraction of the data is sampled. If an int,
                exactly that many samples are taken (must not exceed len(x)).
            x (torch.Tensor): Input features tensor.
            y (torch.Tensor): Labels tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Subsampled (x, y) tensors.

        Raises:
            ValueError: If limit is an int larger than the number of available samples.
        """
        if limit is not None:
            if limit != 1:
                n = len(x)
                if isinstance(limit, float):
                    limit = int(n * limit)
                elif isinstance(limit, int):
                    if n < limit:
                        raise ValueError(
                            f"there are {n} data in the dataset but you want to limit at {limit} data"
                        )
                rand_idx = np.random.choice(n, limit, replace=False)
                x = x[rand_idx]
                y = y[rand_idx]
        return x, y

    @staticmethod
    def create_validation(
        x_train: torch.Tensor, y_train: torch.Tensor, val_size: float, random_state: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the training set into a training subset and a validation set.

        Args:
            x_train (torch.Tensor): Training input features.
            y_train (torch.Tensor): Training labels.
            val_size (float): Fraction of data to allocate to validation; must be
                in (0, 1).
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (x_train_up, y_train_up, x_val, y_val).
        """
        assert isinstance(
            val_size, float
        ), f"expected float type for parameter 'percent_validation', get {val_size} instead."
        assert (
            0.0 < val_size < 1.0
        ), f"expected parameter percent_validation belongs to ]0,1[, get {val_size} instead."
        x_train_up, x_val, y_train_up, y_val = train_test_split(
            x_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
            shuffle=True,
        )
        return x_train_up, y_train_up, x_val, y_val

    @staticmethod
    def create_validation_femnist(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        val_size: float,
        random_state: int,
        dic_idx: dict[str, list[int]],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[int]]
    ]:
        """Create a stratified validation set for FEMNIST in a federated learning setting.

        Samples validation data proportionally per class (global stratification),
        removes the selected samples from the training set, and updates dic_idx so
        that the per-worker index lists remain consistent with the new x_train_up.

        Args:
            x_train (torch.Tensor): Training data (N x ...).
            y_train (torch.Tensor): Training labels (N,).
            val_size (float): Fraction of total data to allocate to validation.
            random_state (int): Random seed for reproducibility.
            dic_idx (dict[str, list[int]]): Per-worker index mapping (indices into
                x_train).

        Returns:
            tuple: (x_train_up, y_train_up, x_val, y_val, dic_train_up) where
                dic_train_up has indices relative to x_train_up.
        """
        # Random generator for reproducibility
        rnd = random.Random(random_state)

        # Compute number of validation samples per class (global stratification)
        bincount_global = y_train.bincount()
        n_val_per_class = (bincount_global * val_size).long()

        val_indices = []

        # Stratified sampling global
        for class_, n_val in enumerate(n_val_per_class):
            if n_val.item() == 0:
                continue

            idx_class_global = torch.where(y_train == class_)[0].tolist()
            idx_val_class = rnd.sample(idx_class_global, k=n_val.item())
            val_indices.extend(idx_val_class)

        # Build mask for remaining training data
        mask = torch.ones(len(x_train), dtype=torch.bool)
        mask[val_indices] = False
        x_train_up = x_train[mask]
        y_train_up = y_train[mask]

        # Build validation tensors
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]

        # Create mapping from old indices to new indices in x_train_up
        old_to_new = {}
        new_idx = 0
        for old_idx, keep in enumerate(mask):
            if keep:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        # Update dic_train_up with new indices relative to x_train_up
        dic_train_up = {}
        for worker_id, worker_indices in dic_idx.items():
            dic_train_up[worker_id] = [
                old_to_new[idx] for idx in worker_indices if mask[idx]
            ]

        return x_train_up, y_train_up, x_val, y_val, dic_train_up
