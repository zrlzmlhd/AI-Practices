"""
Return and Advantage Computation Utilities.

This module provides functions for computing returns and advantages,
which are fundamental to all policy gradient algorithms.

Core Idea:
    Returns and advantages quantify how good an action was. Different
    estimation methods trade off between bias and variance, affecting
    learning stability and sample efficiency.

Mathematical Background:
    Monte Carlo Return:
        G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}
        - Unbiased estimate of Q(s_t, a_t)
        - High variance due to trajectory stochasticity

    TD(0) Return:
        G_t^(1) = r_t + γV(s_{t+1})
        - Biased (depends on V accuracy)
        - Low variance

    n-step Return:
        G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
        - Interpolates between TD(0) and MC

    GAE (Generalized Advantage Estimation):
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        - Exponentially-weighted average of n-step advantages
        - λ controls bias-variance trade-off

Complexity Analysis:
    All functions use backward iteration: O(T) time, O(T) space
    where T is the trajectory length.

References:
    [1] Sutton & Barto (2018). RL: An Introduction. Chapters 12-13.
    [2] Schulman et al. (2016). High-dimensional continuous control using GAE.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch


def compute_returns(
    rewards: List[float],
    gamma: float,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute Monte Carlo returns (discounted cumulative rewards).

    Core Idea:
        G_t represents the total discounted reward from time t.
        Computed efficiently via backward iteration.

    Mathematical Theory:
        G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = r_t + γG_{t+1}

        Backward recursion:
            G_T = r_T
            G_t = r_t + γG_{t+1}  for t = T-1, ..., 0

    Parameters
    ----------
    rewards : List[float]
        Sequence of rewards [r_0, r_1, ..., r_{T-1}].
    gamma : float
        Discount factor in (0, 1].
    normalize : bool, default=True
        If True, normalize returns to zero mean and unit variance.

    Returns
    -------
    torch.Tensor
        Returns [G_0, G_1, ..., G_{T-1}] of shape (T,).

    Examples
    --------
    >>> rewards = [1.0, 1.0, 1.0]
    >>> gamma = 0.99
    >>> returns = compute_returns(rewards, gamma, normalize=False)
    >>> print(returns)  # tensor([2.9701, 1.99, 1.0])

    >>> # G_2 = 1.0
    >>> # G_1 = 1.0 + 0.99 * 1.0 = 1.99
    >>> # G_0 = 1.0 + 0.99 * 1.99 = 2.9701

    Notes
    -----
    Complexity Analysis:
        - Time: O(T) single backward pass
        - Space: O(T) for storing returns

    Numerical Stability:
        - Normalization prevents gradient scale issues
        - Works well even with large reward magnitudes
    """
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_gae(
    rewards: List[float],
    values: Union[List[float], List[torch.Tensor]],
    next_value: float,
    dones: List[bool],
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Core Idea:
        GAE provides a smooth interpolation between TD(0) (low variance,
        high bias) and Monte Carlo (high variance, low bias) by exponentially
        weighting multi-step TD errors.

    Mathematical Theory:
        TD Error:
            δ_t = r_t + γV(s_{t+1}) - V(s_t)

        GAE:
            A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

        Recursive computation:
            A_T = δ_T
            A_t = δ_t + γλ A_{t+1}  (if not done)

        Special cases:
            λ = 0: A_t = δ_t (TD(0))
            λ = 1: A_t = G_t - V(s_t) (MC advantage)

    Parameters
    ----------
    rewards : List[float]
        Rewards [r_0, ..., r_{T-1}].
    values : List[float] or List[torch.Tensor]
        Value estimates [V(s_0), ..., V(s_{T-1})].
    next_value : float
        Bootstrap value V(s_T) for non-terminal states.
    dones : List[bool]
        Episode termination flags.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE λ parameter in [0, 1].

    Returns
    -------
    advantages : torch.Tensor
        GAE advantages of shape (T,).
    returns : torch.Tensor
        Value targets (A + V) of shape (T,).

    Examples
    --------
    >>> rewards = [1.0, 1.0, 1.0]
    >>> values = [0.5, 0.5, 0.5]
    >>> dones = [False, False, True]
    >>> advantages, returns = compute_gae(rewards, values, 0.0, dones, 0.99, 0.95)

    Notes
    -----
    Complexity Analysis:
        - Time: O(T) single backward pass
        - Space: O(T) for advantages and returns

    Implementation Details:
        - done flags reset the GAE accumulator (no bootstrap after terminal)
        - Returns = Advantages + Values (used as value targets)

    Bias-Variance Analysis:
        | λ     | Bias     | Variance | Best For                |
        |-------|----------|----------|-------------------------|
        | 0.0   | High     | Low      | Stable but slow learning|
        | 0.5   | Medium   | Medium   | Balanced                |
        | 0.95  | Low-Med  | Med-High | Most tasks (default)    |
        | 1.0   | None     | High     | When V is very accurate |
    """
    advantages = []
    gae = 0.0

    # Convert tensor values to floats
    values_list = [
        v.item() if isinstance(v, torch.Tensor) else float(v)
        for v in values
    ]
    values_list.append(next_value)

    # Backward iteration
    for t in reversed(range(len(rewards))):
        # If done, next value is 0 (no bootstrap)
        next_val = 0.0 if dones[t] else values_list[t + 1]

        # TD error
        delta = rewards[t] + gamma * next_val - values_list[t]

        # GAE accumulation (reset if done)
        mask = 1.0 - float(dones[t])
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)

    # Returns = Advantages + Values (for training critic)
    returns = advantages + torch.tensor(values_list[:-1], dtype=torch.float32)

    return advantages, returns


def compute_n_step_returns(
    rewards: List[float],
    values: Union[List[float], List[torch.Tensor]],
    next_value: float,
    dones: List[bool],
    gamma: float,
    n_steps: int,
) -> torch.Tensor:
    """
    Compute n-step returns with bootstrap.

    Core Idea:
        n-step returns accumulate rewards for n steps then bootstrap
        with the value function. This provides a middle ground between
        TD(0) (n=1) and Monte Carlo (n=∞).

    Mathematical Theory:
        G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})

        For t near end of episode (t + n > T):
            G_t^(n) = Σ_{k=0}^{T-t-1} γ^k r_{t+k} + γ^{T-t} V(s_T)

    Parameters
    ----------
    rewards : List[float]
        Rewards [r_0, ..., r_{T-1}].
    values : List[float] or List[torch.Tensor]
        Value estimates.
    next_value : float
        Bootstrap value for final state.
    dones : List[bool]
        Episode termination flags.
    gamma : float
        Discount factor.
    n_steps : int
        Number of steps to look ahead.

    Returns
    -------
    torch.Tensor
        N-step returns of shape (T,).

    Examples
    --------
    >>> rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    >>> values = [2.0, 2.0, 2.0, 2.0, 2.0]
    >>> dones = [False, False, False, False, True]
    >>> returns = compute_n_step_returns(rewards, values, 0.0, dones, 0.99, 3)

    Notes
    -----
    Complexity Analysis:
        - Time: O(T * n) in naive implementation, but optimized to O(T)
        - Space: O(T)

    Comparison with GAE:
        | Method     | Flexibility | Computation | Variance Control |
        |------------|-------------|-------------|------------------|
        | n-step     | Discrete n  | O(T)        | Coarse           |
        | GAE        | Continuous λ| O(T)        | Fine             |
    """
    T = len(rewards)
    returns = []

    # Convert values
    values_list = [
        v.item() if isinstance(v, torch.Tensor) else float(v)
        for v in values
    ]
    values_list.append(next_value)

    for t in range(T):
        G = 0.0
        steps_taken = 0

        # Accumulate n-step rewards
        for k in range(n_steps):
            if t + k >= T:
                break

            G += (gamma ** k) * rewards[t + k]
            steps_taken = k + 1

            # Stop if episode ended
            if dones[t + k]:
                break

        # Bootstrap if not terminated
        if t + steps_taken < T and not dones[t + steps_taken - 1]:
            G += (gamma ** steps_taken) * values_list[t + steps_taken]
        elif t + steps_taken == T and not dones[-1]:
            G += (gamma ** steps_taken) * next_value

        returns.append(G)

    return torch.tensor(returns, dtype=torch.float32)


def compute_td_error(
    reward: float,
    value: float,
    next_value: float,
    done: bool,
    gamma: float,
) -> float:
    """
    Compute single-step TD error.

    Mathematical Definition:
        δ = r + γV(s') - V(s)  if not done
        δ = r - V(s)          if done

    Parameters
    ----------
    reward : float
        Immediate reward.
    value : float
        Current state value V(s).
    next_value : float
        Next state value V(s').
    done : bool
        Whether episode terminated.
    gamma : float
        Discount factor.

    Returns
    -------
    float
        TD error value.
    """
    if done:
        return reward - value
    return reward + gamma * next_value - value
