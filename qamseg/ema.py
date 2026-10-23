"""EMA teacher update. ``φ ← α φ + (1-α) θ``.

``alpha`` near 0.99 is a placeholder, not a tuned value. The teacher is
not a second backward.
"""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, alpha: float = 0.99) -> None:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"EMA alpha must lie in [0, 1], got {alpha}")
    alpha = float(alpha)
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
    for teacher_buf, student_buf in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(teacher_buf):
            teacher_buf.data.mul_(alpha).add_(student_buf.data, alpha=1.0 - alpha)
        else:
            teacher_buf.data.copy_(student_buf.data)
