import typing as t
import numpy as np
import torch
import torch.optim as optim
from sklearn.utils import resample
from .utils import WeakClassifier


class BaggingClassifier:
    """Bagging Classifier using multiple weak classifiers."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the bagging classifier.

        Args:
            input_dim (int): Number of input features.
        """
        # Create 10 learners.
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(10)]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """
        Train the bagging classifier.

        Args:
            X_train (DataFrame): Training features.
            y_train (Series or array-like): Training labels.
            num_epochs (int): Number of epochs for each learner.
            learning_rate (float): Learning rate for training.

        Returns:
            List[float]: Final loss of each weak classifier.
        """
        losses_of_models = []
        for i, model in enumerate(self.learners):
            # Resample the training data
            X_resampled, y_resampled = resample(X_train, y_train, replace=True)

            # Convert to tensors
            X_tensor = torch.tensor(X_resampled.values, dtype=torch.float32)
            y_tensor = torch.tensor(y_resampled, dtype=torch.float32).view(-1, 1)

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                raw_output = model(X_tensor)
                predictions = torch.sigmoid(raw_output)
                loss = -torch.mean(
                    y_tensor * torch.log(predictions + 1e-10) + (1 - y_tensor) * torch.log(1 - predictions + 1e-10)
                )
                loss.backward()
                optimizer.step()

            losses_of_models.append(loss.item())
        return losses_of_models

    def predict_learners(self, X) -> t.List[np.ndarray]:
        """
        Generate predictions from all weak classifiers.

        Args:
            X (DataFrame): Input features.

        Returns:
            List[np.ndarray]: List of predictions from all weak classifiers.
        """
        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        learner_predictions = []
        for learner in self.learners:
            raw_output = learner(X_tensor)
            preds = torch.sigmoid(raw_output).detach().numpy()
            learner_predictions.append(preds)
        return learner_predictions

    def compute_feature_importance(self) -> t.List[float]:
        """
        Compute feature importance based on weight magnitudes.

        Returns:
            List[float]: Importance score for each feature.
        """
        feature_importances = np.zeros(self.learners[0].layer.in_features)

        for learner in self.learners:
            feature_weights = learner.layer.weight.detach().numpy().flatten()
            feature_importances += np.abs(feature_weights)

        return feature_importances.tolist()

    def predict(self, X) -> np.ndarray:
        """
        Generate final predictions by averaging weak learners' outputs.

        Args:
            X (DataFrame): Input features.

        Returns:
            np.ndarray: Final binary predictions.
        """
        learner_predictions = self.predict_learners(X)
        all_preds = np.mean(learner_predictions, axis=0)
        final_predictions = (all_preds >= 0.5).astype(int)
        return final_predictions
