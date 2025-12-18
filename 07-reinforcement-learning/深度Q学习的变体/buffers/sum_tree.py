"""
Sum Tree Data Structure for Prioritized Experience Replay.

This module implements a binary sum tree for O(log N) prioritized sampling.

Core Idea (核心思想)
====================
SumTree是一种特殊的二叉树数据结构，用于高效实现优先经验回放(PER)。

Tree Structure (树结构)
-----------------------
- **叶节点**: 存储每个样本的优先级值
- **内部节点**: 存储子节点优先级之和
- **根节点**: 存储所有优先级的总和

Mathematical Foundation (数学基础)
==================================
Tree structure for N leaves:
    - Total nodes: 2N - 1
    - Leaf indices: [N-1, 2N-2]
    - Parent of node i: (i - 1) // 2
    - Children of node i: 2i + 1 (left), 2i + 2 (right)

Proportional sampling uses cumulative sum:
    sample(u) → i such that Σ_{j<i} p_j < u ≤ Σ_{j≤i} p_j

Complexity Analysis (复杂度分析)
================================
+------------------+------------+----------------------------------+
| Operation        | Complexity | Notes                            |
+==================+============+==================================+
| add()            | O(log N)   | Insert + update ancestors        |
+------------------+------------+----------------------------------+
| update_priority()| O(log N)   | Update leaf + propagate          |
+------------------+------------+----------------------------------+
| get()            | O(log N)   | Binary search by cumsum          |
+------------------+------------+----------------------------------+
| total_priority   | O(1)       | Root node value                  |
+------------------+------------+----------------------------------+
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


class SumTree:
    """
    Binary sum tree for O(log N) prioritized sampling.

    Core Idea (核心思想)
    --------------------
    SumTree是一种特殊的二叉树数据结构，用于高效实现优先经验回放(PER)。
    其核心思想是：

    1. **叶节点**: 存储每个样本的优先级值
    2. **内部节点**: 存储子节点优先级之和
    3. **根节点**: 存储所有优先级的总和

    Mathematical Foundation (数学基础)
    ----------------------------------
    Tree structure for N leaves:

    - Total nodes: 2N - 1
    - Leaf indices: [N-1, 2N-2]
    - Parent of node i: (i - 1) // 2
    - Children of node i: 2i + 1 (left), 2i + 2 (right)

    Proportional sampling uses cumulative sum:

    .. math::
        \\text{sample}(u) \\to i \\text{ such that } \\sum_{j<i} p_j < u \\leq \\sum_{j \\leq i} p_j

    where u ~ Uniform(0, total_priority).

    Parameters
    ----------
    capacity : int
        Maximum number of elements (leaf nodes)

    Attributes
    ----------
    total_priority : float
        Sum of all priorities (root node value)
    capacity : int
        Maximum number of elements

    Notes
    -----
    - Array-based implementation avoids pointer overhead
    - Priority updates automatically propagate to root
    - Enables efficient stratified sampling for PER

    Examples
    --------
    >>> tree = SumTree(capacity=100)
    >>> tree.add(priority=1.0, data="transition_1")
    >>> tree.total_priority
    1.0
    >>> idx, priority, data = tree.get(cumsum=0.5)
    """

    __slots__ = ("_capacity", "_tree", "_data", "_write_idx", "_size")

    def __init__(self, capacity: int) -> None:
        """
        Initialize sum tree with given capacity.

        Parameters
        ----------
        capacity : int
            Maximum number of leaf nodes (data elements)

        Raises
        ------
        ValueError
            If capacity is not a positive integer
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"capacity must be a positive integer, got {capacity!r}"
            )

        self._capacity = capacity
        # Tree array: internal nodes [0, capacity-2], leaves [capacity-1, 2*capacity-2]
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Optional[Any]] = [None] * capacity
        self._write_idx = 0
        self._size = 0

    @property
    def total_priority(self) -> float:
        """
        Total priority sum (root node value).

        Time complexity: O(1)

        Returns
        -------
        float
            Sum of all stored priorities
        """
        return float(self._tree[0])

    @property
    def capacity(self) -> int:
        """
        Maximum number of elements.

        Returns
        -------
        int
            Maximum leaf node count
        """
        return self._capacity

    def __len__(self) -> int:
        """
        Current number of stored elements.

        Returns
        -------
        int
            Number of elements in tree
        """
        return self._size

    def add(self, priority: float, data: Any) -> None:
        """
        Add element with specified priority.

        Parameters
        ----------
        priority : float
            Priority value (must be non-negative)
        data : Any
            Data to store (typically a Transition)

        Raises
        ------
        ValueError
            If priority is negative

        Notes
        -----
        - O(log N) due to priority propagation
        - Overwrites oldest element when at capacity (FIFO)

        Examples
        --------
        >>> tree = SumTree(capacity=10)
        >>> tree.add(1.0, "data_1")
        >>> tree.add(2.0, "data_2")
        >>> tree.total_priority
        3.0
        """
        if priority < 0:
            raise ValueError(f"priority must be non-negative, got {priority}")

        tree_idx = self._write_idx + self._capacity - 1
        self._data[self._write_idx] = data
        self._update(tree_idx, priority)
        self._write_idx = (self._write_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        """
        Update priority at tree_idx and propagate delta to root.

        Parameters
        ----------
        tree_idx : int
            Index in tree array
        priority : float
            New priority value
        """
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority

        # Propagate change up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta

    def update_priority(self, tree_idx: int, priority: float) -> None:
        """
        Update priority of existing element.

        Parameters
        ----------
        tree_idx : int
            Tree index of leaf node (returned by get())
        priority : float
            New priority value

        Raises
        ------
        ValueError
            If priority is negative

        Examples
        --------
        >>> tree = SumTree(capacity=10)
        >>> tree.add(1.0, "data")
        >>> idx, _, _ = tree.get(0.5)
        >>> tree.update_priority(idx, 2.0)
        >>> tree.total_priority
        2.0
        """
        if priority < 0:
            raise ValueError(f"priority must be non-negative, got {priority}")
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> Tuple[int, float, Any]:
        """
        Sample element by cumulative sum (proportional sampling).

        Parameters
        ----------
        cumsum : float
            Target cumulative sum in [0, total_priority)

        Returns
        -------
        tree_idx : int
            Index in tree array (for priority updates)
        priority : float
            Priority of sampled element
        data : Any
            Stored data element

        Notes
        -----
        O(log N) binary search from root to leaf.

        Examples
        --------
        >>> tree = SumTree(capacity=10)
        >>> tree.add(1.0, "first")
        >>> tree.add(3.0, "second")
        >>> idx, priority, data = tree.get(2.0)
        >>> data
        'second'
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1

            if left >= len(self._tree):
                # Reached leaf
                break

            if cumsum <= self._tree[left]:
                parent = left
            else:
                cumsum -= self._tree[left]
                parent = right

        data_idx = parent - self._capacity + 1
        return parent, float(self._tree[parent]), self._data[data_idx]

    def min_priority(self) -> float:
        """
        Get minimum non-zero priority among stored elements.

        Returns
        -------
        float
            Minimum priority, or 0.0 if empty

        Notes
        -----
        O(N) scan through leaf nodes. For PER, this is called once
        per batch sampling, so overall complexity remains O(B log N).

        Examples
        --------
        >>> tree = SumTree(capacity=10)
        >>> tree.add(1.0, "a")
        >>> tree.add(3.0, "b")
        >>> tree.min_priority()
        1.0
        """
        if self._size == 0:
            return 0.0

        start = self._capacity - 1
        priorities = self._tree[start:start + self._size]
        positive_priorities = priorities[priorities > 0]

        if len(positive_priorities) == 0:
            return 0.0

        return float(np.min(positive_priorities))
