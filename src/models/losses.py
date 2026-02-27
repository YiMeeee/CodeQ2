from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    CORN 风格简化序回归损失。
    对于K类，学习K-1个阈值二分类任务：y > k ?
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, ordinal_logits: torch.Tensor, target_grade: torch.Tensor) -> torch.Tensor:
        # ordinal_logits: [B, K-1]
        num_thresh = ordinal_logits.shape[1]
        target = torch.zeros_like(ordinal_logits)
        for k in range(num_thresh):
            target[:, k] = (target_grade > k).float()
        return F.binary_cross_entropy_with_logits(ordinal_logits, target)


def expected_grade_from_ordinal(ordinal_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(ordinal_logits)
    return probs.sum(dim=1)


class MultiTaskMeniscusLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        lambda_ord: float = 0.35,
        lambda_surgery: float = 0.25,
        lambda_consistency: float = 0.10,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ord = OrdinalRegressionLoss()
        self.lambda_ord = lambda_ord
        self.lambda_surgery = lambda_surgery
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        grade_logits: torch.Tensor,
        ordinal_logits: torch.Tensor,
        surgery_logit: torch.Tensor,
        grade_target: torch.Tensor,
        surgery_target: torch.Tensor,
    ) -> torch.Tensor:
        loss_grade = self.ce(grade_logits, grade_target)
        loss_ord = self.ord(ordinal_logits, grade_target)
        loss_surgery = F.binary_cross_entropy_with_logits(surgery_logit, surgery_target)

        with torch.no_grad():
            grade_prob = F.softmax(grade_logits, dim=1)
        expected_grade_cls = (grade_prob * torch.arange(grade_logits.shape[1], device=grade_logits.device).float()).sum(dim=1)
        expected_grade_ord = expected_grade_from_ordinal(ordinal_logits)
        loss_consistency = F.l1_loss(expected_grade_cls, expected_grade_ord)

        total = (
            loss_grade
            + self.lambda_ord * loss_ord
            + self.lambda_surgery * loss_surgery
            + self.lambda_consistency * loss_consistency
        )
        return total
