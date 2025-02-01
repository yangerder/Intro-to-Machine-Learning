import typing as t
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    """
    Preprocess the input DataFrame by mapping categorical values,
    filling missing values, and normalizing numeric columns.
    """
    mappings = {
        'person_gender': {'male': 0, 'female': 1},
        'person_home_ownership': {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2},
        'loan_intent': {
            'VENTURE': 0,
            'MEDICAL': 1,
            'PERSONAL': 2,
            'DEBTCONSOLIDATION': 3,
            'HOMEIMPROVEMENT': 4,
            'EDUCATION': 5,
        },
        'previous_loan_defaults_on_file': {'No': 0, 'Yes': 1},
        'person_education': {
            'High School': 0,
            'Associate': 1,
            'Bachelor': 2,
            'Master': 3,
        },
    }
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)

    df.fillna(df.mean(), inplace=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    return df


class WeakClassifier(nn.Module):
    """
    Implement a simple weak classifier using PyTorch.

    Examples:
        - Single-layer model: Linear(input_dim, 1)
        - Two-layer model: Linear(input_dim, k) -> Linear(k, 1)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.layer(x)


def accuracy_score(y_trues, y_preds) -> float:
    """Compute the accuracy score."""
    raise NotImplementedError


def entropy_loss(outputs, targets):
    """
    Compute the entropy loss between outputs and targets.

    Args:
        outputs: Model outputs or predicted probabilities.
        targets: Ground-truth labels.

    Returns:
        A scalar tensor representing the loss.
    """
    outputs = torch.tensor(outputs, dtype=torch.float32) if not isinstance(outputs, torch.Tensor) else outputs
    targets = torch.tensor(targets, dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets

    outputs = torch.clamp(outputs, min=1e-9)

    loss = -torch.sum(targets * torch.log(outputs)) / outputs.shape[0]
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    """
    Plot ROC curves for a list of weak classifiers.

    Args:
        y_preds: List of predicted probabilities for each weak classifier.
        y_trues: Ground-truth labels.
        fpath: File path to save the ROC plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))

    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'Weak Classifier {i + 1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Weak Classifiers')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.savefig(fpath)
    plt.close()
