from collections import deque
from typing import Tuple

import torch


class ErrorFeedback:
    """Error feedback (EF) variance-reduction variant for biased compressors.

    Maintains a per-layer error buffer that accumulates the compression error
    from previous steps and adds it to the current gradient before compression.
    """

    def __init__(self, n_layers: int, alpha: float) -> None:
        """Initialize error feedback buffers.

        Args:
            n_layers (int): Number of trainable layers.
            alpha (float): Scaling factor applied to the error before adding it
                to the current gradient.
        """
        self.n_layers = n_layers
        self.alpha = alpha

        self.errors = [0] * n_layers
        return None

    def compute_gtild(self, grad: torch.Tensor, layer_number: int) -> torch.Tensor:
        """Compute the error-corrected gradient for a given layer.

        Args:
            grad (torch.Tensor): Current gradient for this layer.
            layer_number (int): Index of the layer.
        """
        return grad + self.alpha * self.errors[layer_number]

    def update_error(
        self, gtild: torch.Tensor, gtild_sparse: torch.Tensor, layer_number: int
    ) -> None:
        """Update the error buffer as the difference between the corrected and compressed gradients.

        Args:
            gtild (torch.Tensor): Error-corrected gradient before compression.
            gtild_sparse (torch.Tensor): Decompressed (sparse) approximation of gtild.
            layer_number (int): Index of the layer.
        """
        self.errors[layer_number] = gtild - gtild_sparse
        return None


class ProjFL:
    """ProjFL variant based on gradient projection.

    Maintains a sliding window of past K descent directions per layer and projects
    the current gradient onto the subspace orthogonal to the average direction.
    Optionally supports error feedback (EF).
    """

    def __init__(
        self, n_layers: int, layer_gradient_shapes: dict, K: int, EF: bool = False
    ) -> None:
        """Initialize ProjFL state.

        Args:
            n_layers (int): Number of trainable layers.
            layer_gradient_shapes (dict): Mapping from layer index to gradient shape.
            K (int): Window size for the sliding buffer of past descent directions.
            EF (bool): Whether to apply error feedback.
        """
        self.n_layers = n_layers
        self.layer_gradient_shapes = layer_gradient_shapes
        self.K = K
        self.EF = EF
        self.desc_dir = [deque([None], maxlen=self.K) for _ in range(self.n_layers)]
        self.dbar = [
            0 for _ in range(n_layers)
        ]  # 0 is for init, it should be array/matrix
        if EF:
            self.errors = [0] * n_layers
        return None

    def compute_proj(
        self, grad: torch.Tensor, layer_number: int
    ) -> Tuple[float, torch.Tensor]:
        """Project the gradient onto the subspace orthogonal to the average descent direction.

        On the first step (before any direction is stored), the full gradient is
        returned unchanged with alpha=0.

        Args:
            grad (torch.Tensor): Current gradient for this layer.
            layer_number (int): Index of the layer.

        Returns:
            Tuple[float, torch.Tensor]: Projection coefficient alpha and the orthogonal
                gradient component grad_orth.
        """
        if self.desc_dir[layer_number][0] is None:  # init
            alpha = 0
            grad_orth = grad
        else:
            norm = torch.norm(self.dbar[layer_number], p=2) ** 2
            alpha = torch.dot(self.dbar[layer_number].flatten(), grad.flatten()) / norm

            grad_orth = grad - alpha * self.dbar[layer_number]
        return alpha, grad_orth

    def update_desc_dir(
        self, sparse_grad_orth: torch.tensor, alpha: float, layer_number: int
    ) -> None:
        """Append the new descent direction to the sliding window.

        On the first step, the None placeholder is removed before appending.

        Args:
            sparse_grad_orth (torch.Tensor): orthogonal gradient component.
            alpha (float): Projection coefficient from the current step.
            layer_number (int): Index of the layer.
        """
        if self.desc_dir[layer_number][0] is None:
            # remove the None
            self.desc_dir[layer_number].pop()

            # add the relevant value
            self.desc_dir[layer_number].append(sparse_grad_orth)
        else:
            self.desc_dir[layer_number].append(
                alpha * self.dbar[layer_number] + sparse_grad_orth
            )
        return None

    def update_dbar(self, layer_number: int) -> None:
        """Update the average descent direction for a given layer.

        On the first step, dbar is initialized to a zero tensor.
        Subsequently, it is computed as the mean of the K most recent directions.

        Args:
            layer_number (int): Index of the layer.
        """
        if self.desc_dir[layer_number][0] is None:
            self.dbar[layer_number] = torch.zeros(
                size=self.layer_gradient_shapes[layer_number]
            )  # 0 according to the size of the grad
        else:
            self.dbar[layer_number] = torch.mean(
                torch.stack(list(self.desc_dir[layer_number])), dim=0
            )
        return None

    def update_error(
        self,
        sparse_gradient: torch.Tensor,
        gradorth_plus_error: torch.Tensor,
        layer_number: int,
    ) -> None:
        """Update the error feedback buffer for the orthogonal component.

        Args:
            sparse_gradient (torch.Tensor): orthogonal gradient.
            gradorth_plus_error (torch.Tensor): Orthogonal gradient plus previous error.
            layer_number (int): Index of the layer.
        """
        self.errors[layer_number] = gradorth_plus_error - sparse_gradient
        return None


class ErrorFeedback21:
    """EF21 variant: error-feedback using the difference between consecutive compressed gradients.
    
    Maintains a per-layer descent direction updated as an exponential moving
    average of compressed gradient differences.
    
    Ref:
    Richtárik, P., Sokolov, I., & Fatkhullin, I. (2021). EF21: 
    A new, simpler, theoretically better, and practically faster error feedback. 
    Advances in Neural Information Processing Systems, 34, 4384-4396.
    """

    def __init__(self, n_layers: int, gamma: float = 0.75) -> None:
        """Initialize EF21 state.

        Args:
            n_layers (int): Number of trainable layers.
            gamma (float): Momentum coefficient.
        """
        self.n_layers = n_layers
        self.desc_dir = [None] * n_layers
        self.gamma = gamma
        return None

    def update_desc_dir(
        self, sparse_grad_diff: torch.tensor, layer_number: int
    ) -> None:
        """Update the descent direction for a given layer.

        On the first call the direction is initialized to the sparse gradient.

        Args:
            sparse_grad_diff (torch.Tensor): Sparse gradient difference.
            layer_number (int): Index of the layer.
        """
        # sparse_grad_diff = M
        if self.desc_dir[layer_number] is None:
            self.desc_dir[layer_number] = sparse_grad_diff
        else:
            self.desc_dir[layer_number] = (
                self.gamma * self.desc_dir[layer_number] + sparse_grad_diff
            )
        return None



class Diana:
    """DIANA variant: variance-reduction via shift vectors updated with momentum.

    Each worker maintains a shift vector h per layer. The descent direction
    is computed as grad - gamma * h. The master additionally maintains a
    global descent direction updated with a beta coefficient.
    
    Ref:
    Mishchenko, K., Gorbunov, E., Takáč, M., & Richtárik, P. (2025). 
    Distributed learning with compressed gradient differences. 
    Optimization Methods and Software, 40(5), 1181-1196.
    """

    def __init__(
        self, n_layers: int, alpha: float, beta: float = None, gamma: float = 0.75
    ) -> None:
        """Initialize DIANA state.

        Args:
            n_layers (int): Number of trainable layers.
            alpha (float): Step size for updating h.
            beta (float, optional): Momentum coefficient used by the master only.
            gamma (float): Momentum coefficient for the h update.
        """
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta  # master only
        self.gamma = gamma

        self.h = [None] * n_layers
        self.desc_dir = [None] * n_layers  # master only
        return None

    def update_h(self, m: torch.Tensor, layer_number: int) -> None:
        """Update the shift vector h for a given layer.

        Args:
            m (torch.Tensor): gradient in sparse format.
            layer_number (int): Index of the layer.
        """
        if self.h[layer_number] is None:
            self.h[layer_number] = self.alpha * m
        else:
            self.h[layer_number] = self.gamma * self.h[layer_number] + self.alpha * m
        return None

    def update_desc_dir(self, sparse_grad: torch.Tensor, layer_number: int) -> None:
        """Update the global descent direction for a given layer (master only).

        Args:
            sparse_grad (torch.Tensor): gradient in sparse format.
            layer_number (int): Index of the layer.
        """
        if self.desc_dir[layer_number] is None:
            self.desc_dir[layer_number] = sparse_grad
        else:
            self.desc_dir[layer_number] = (
                self.beta * self.desc_dir[layer_number]
                + self.gamma * self.h[layer_number]
                + sparse_grad
            )
        return None
