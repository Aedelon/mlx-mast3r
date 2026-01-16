"""Learning rate schedules for optimization.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import math


def linear_schedule(
    alpha: float,
    lr_base: float,
    lr_end: float = 0.0,
) -> float:
    """Linear learning rate schedule.

    Args:
        alpha: Progress value in [0, 1]
        lr_base: Starting learning rate
        lr_end: Ending learning rate

    Returns:
        Current learning rate
    """
    return lr_base + alpha * (lr_end - lr_base)


def cosine_schedule(
    alpha: float,
    lr_base: float,
    lr_end: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule.

    Follows the formula: lr = lr_end + (lr_base - lr_end) * (1 + cos(pi * alpha)) / 2

    Args:
        alpha: Progress value in [0, 1]
        lr_base: Starting learning rate
        lr_end: Ending learning rate

    Returns:
        Current learning rate
    """
    return lr_end + (lr_base - lr_end) * (1 + math.cos(math.pi * alpha)) / 2


def exponential_schedule(
    alpha: float,
    lr_base: float,
    lr_end: float = 0.001,
) -> float:
    """Exponential decay learning rate schedule.

    Args:
        alpha: Progress value in [0, 1]
        lr_base: Starting learning rate
        lr_end: Ending learning rate

    Returns:
        Current learning rate
    """
    if lr_end <= 0:
        lr_end = 1e-8
    log_ratio = math.log(lr_end / lr_base)
    return lr_base * math.exp(alpha * log_ratio)


def warmup_schedule(
    alpha: float,
    lr_base: float,
    warmup_fraction: float = 0.1,
) -> float:
    """Linear warmup followed by constant learning rate.

    Args:
        alpha: Progress value in [0, 1]
        lr_base: Target learning rate
        warmup_fraction: Fraction of training for warmup

    Returns:
        Current learning rate
    """
    if alpha < warmup_fraction:
        return lr_base * (alpha / warmup_fraction)
    return lr_base


def warmup_cosine_schedule(
    alpha: float,
    lr_base: float,
    lr_end: float = 0.0,
    warmup_fraction: float = 0.1,
) -> float:
    """Linear warmup followed by cosine decay.

    Args:
        alpha: Progress value in [0, 1]
        lr_base: Peak learning rate (after warmup)
        lr_end: Final learning rate
        warmup_fraction: Fraction of training for warmup

    Returns:
        Current learning rate
    """
    if alpha < warmup_fraction:
        return lr_base * (alpha / warmup_fraction)

    # Rescale alpha for cosine phase
    alpha_cosine = (alpha - warmup_fraction) / (1 - warmup_fraction)
    return cosine_schedule(alpha_cosine, lr_base, lr_end)


class LRScheduler:
    """Learning rate scheduler for optimization loops."""

    def __init__(
        self,
        lr_base: float,
        lr_end: float = 0.0,
        schedule_type: str = "cosine",
        warmup_fraction: float = 0.0,
    ):
        """Initialize scheduler.

        Args:
            lr_base: Starting/peak learning rate
            lr_end: Final learning rate
            schedule_type: One of 'linear', 'cosine', 'exponential', 'constant'
            warmup_fraction: Fraction of iterations for linear warmup
        """
        self.lr_base = lr_base
        self.lr_end = lr_end
        self.schedule_type = schedule_type
        self.warmup_fraction = warmup_fraction

    def get_lr(self, step: int, total_steps: int) -> float:
        """Get learning rate for current step.

        Args:
            step: Current iteration (0-indexed)
            total_steps: Total number of iterations

        Returns:
            Current learning rate
        """
        if total_steps <= 1:
            return self.lr_base

        alpha = step / (total_steps - 1)

        # Apply warmup if needed
        if self.warmup_fraction > 0 and alpha < self.warmup_fraction:
            return self.lr_base * (alpha / self.warmup_fraction)

        # Rescale alpha for main schedule
        if self.warmup_fraction > 0:
            alpha = (alpha - self.warmup_fraction) / (1 - self.warmup_fraction)

        if self.schedule_type == "constant":
            return self.lr_base
        elif self.schedule_type == "linear":
            return linear_schedule(alpha, self.lr_base, self.lr_end)
        elif self.schedule_type == "cosine":
            return cosine_schedule(alpha, self.lr_base, self.lr_end)
        elif self.schedule_type == "exponential":
            return exponential_schedule(alpha, self.lr_base, self.lr_end)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def __call__(self, step: int, total_steps: int) -> float:
        """Alias for get_lr."""
        return self.get_lr(step, total_steps)
