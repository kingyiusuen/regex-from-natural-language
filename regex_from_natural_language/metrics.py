from typing import List, Set

import torch
from torch import Tensor, tensor
from torchmetrics import Metric


def _edit_distance(prediction: List[int], reference: List[int]) -> int:
    dp = [[0] * (len(reference) + 1) for _ in range(len(prediction) + 1)]
    for i in range(len(prediction) + 1):
        dp[i][0] = i
    for j in range(len(reference) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction) + 1):
        for j in range(1, len(reference) + 1):
            if prediction[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


class CER(Metric):
    error: Tensor
    total: Tensor

    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, references: Tensor) -> None:  # type: ignore
        errors = tensor(0, dtype=torch.float)
        total = tensor(0, dtype=torch.float)
        for prediction, reference in zip(predictions, references):
            prediction = [token for token in prediction.tolist() if token not in self.ignore_indices]
            reference = [token for token in reference.tolist() if token not in self.ignore_indices]
            errors += _edit_distance(prediction, reference)
            total += len(reference)
        self.errors += errors
        self.total += total

    def compute(self) -> Tensor:
        """Calculate the word error rate.

        Returns:
            (Tensor) Word error rate
        """
        return self.errors / self.total
