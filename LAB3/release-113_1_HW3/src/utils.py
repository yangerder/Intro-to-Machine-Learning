import typing as t
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    return df


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        ...

    def forward(self, x):
        ...


def accuracy_score(y_trues, y_preds) -> float:
    raise NotImplementedError


def entropy_loss(outputs, targets):
    raise NotImplementedError


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    raise NotImplementedError
