"""
Base Neural Network Components.

This module provides foundational network building blocks used across all
policy gradient architectures.

Core Idea:
    Reusable network components enable consistent initialization, architecture
    patterns, and gradient flow properties across different algorithm
    implementations.

Mathematical Background:
    Neural networks approximate functions:
        f_θ: S → A  (policy)
        f_φ: S → R  (value)

    Weight initialization affects:
        - Gradient magnitude at initialization
        - Signal propagation through layers
        - Training stability and convergence

    Orthogonal Initialization (Saxe et al., 2014):
        W ~ Orthogonal distribution with gain g
        Preserves gradient norms: ||∇W L|| ≈ ||∇(W+1) L||

References:
    [1] Saxe et al. (2014). Exact solutions to the nonlinear dynamics
        of learning in deep linear neural networks. ICLR.
    [2] He et al. (2015). Delving deep into rectifiers. ICCV.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    Apply orthogonal initialization to network weights.

    Orthogonal initialization preserves gradient magnitude during backpropagation,
    leading to more stable training especially for deep networks.

    Core Idea:
        Initialize weight matrices to be orthogonal (W^T W = I scaled by gain).
        This ensures that gradients neither explode nor vanish as they propagate
        through layers.

    Mathematical Theory:
        For orthogonal matrix Q with ||Q|| = gain:
            ||Qx|| = gain * ||x||  (norm preservation)

        For backpropagation:
            ||∂L/∂W_l|| ≈ ||∂L/∂W_{l+1}||  (gradient stability)

    Parameters
    ----------
    module : nn.Module
        Network module to initialize.
    gain : float, default=1.0
        Scaling factor for weight initialization.

        Recommended values:
            - ReLU activations: gain = sqrt(2)
            - Tanh activations: gain = 5/3
            - Linear/output layers: gain = 1.0
            - Policy output: gain = 0.01 (small initial variance)

    Examples
    --------
    >>> layer = nn.Linear(128, 64)
    >>> init_weights(layer, gain=np.sqrt(2))  # For ReLU
    >>> # Weight matrix is now orthogonal with appropriate scaling

    >>> # Apply to entire network
    >>> network = nn.Sequential(
    ...     nn.Linear(4, 128),
    ...     nn.ReLU(),
    ...     nn.Linear(128, 2)
    ... )
    >>> network.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    Notes
    -----
    Complexity Analysis:
        - Time: O(min(in, out)^2 * max(in, out)) for SVD in orthogonal init
        - Space: O(in * out) for weight matrix

    Algorithm Comparison:
        | Method      | Gradient Flow | Suited For         |
        |-------------|---------------|---------------------|
        | Xavier      | Moderate      | Tanh, Sigmoid       |
        | He          | Good          | ReLU                |
        | Orthogonal  | Excellent     | Deep nets, RNNs, RL |
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron base module.

    Provides a configurable feedforward neural network with consistent
    initialization and activation patterns.

    Core Idea:
        A flexible MLP building block that can be configured for different
        purposes (policy networks, value networks, feature extractors) while
        maintaining consistent initialization and architecture patterns.

    Mathematical Theory:
        Forward pass:
            h_0 = x
            h_i = σ(W_i h_{i-1} + b_i)  for i = 1, ..., L-1
            y = W_L h_{L-1} + b_L

        Gradient flow (with ReLU σ):
            ∂h_i/∂h_{i-1} = diag(h_{i-1} > 0) W_i

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    output_dim : int
        Dimension of output.
    hidden_dims : List[int], default=[128, 128]
        Dimensions of hidden layers.
    activation : nn.Module, default=nn.ReLU()
        Activation function for hidden layers.
    output_activation : Optional[nn.Module], default=None
        Activation function for output layer. None for linear output.

    Attributes
    ----------
    net : nn.Sequential
        The sequential network container.

    Examples
    --------
    >>> # Policy network with softmax output
    >>> policy_net = MLP(4, 2, [128, 128], nn.ReLU(), nn.Softmax(dim=-1))
    >>> x = torch.randn(32, 4)
    >>> probs = policy_net(x)  # shape: (32, 2), sums to 1

    >>> # Value network with linear output
    >>> value_net = MLP(4, 1, [128, 128], nn.ReLU(), None)
    >>> values = value_net(x)  # shape: (32, 1)

    >>> # Feature extractor
    >>> features = MLP(84*84, 512, [256, 256], nn.ReLU(), nn.ReLU())

    Notes
    -----
    Complexity Analysis:
        - Parameters: Σ(d_i * d_{i+1}) for layer dimensions d_i
        - Forward pass: O(batch_size * Σ(d_i * d_{i+1}))
        - Memory: O(batch_size * max(d_i)) for activations

    Architecture Design:
        - Two hidden layers [128, 128] or [256, 256] work for most RL tasks
        - Deeper networks may help for complex observation spaces
        - Width should scale with input complexity
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: nn.Module = None,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]
        if activation is None:
            activation = nn.ReLU()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(activation)
            elif output_activation is not None:
                layers.append(output_activation)

        self.net = nn.Sequential(*layers)
        self._initialize_weights(output_activation)

    def _initialize_weights(self, output_activation: Optional[nn.Module]) -> None:
        """
        Apply orthogonal initialization with appropriate gains.

        Hidden layers use gain=sqrt(2) for ReLU.
        Output layer uses gain=0.01 for small initial policy variance.
        """
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

        # Output layer with smaller initialization
        if output_activation is None:
            # Linear output - find last Linear layer
            for module in reversed(list(self.net.modules())):
                if isinstance(module, nn.Linear):
                    init_weights(module, gain=0.01)
                    break
        else:
            # Has output activation - find second to last Linear layer
            linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) >= 1:
                init_weights(linear_layers[-1], gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)

    def get_num_params(self) -> int:
        """
        Count total number of trainable parameters.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
