import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier


class AdaBoostClassifier:
    """Implementation of AdaBoost classifier using PyTorch."""

    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        """
        Initialize AdaBoost classifier.

        Args:
            input_dim (int): Dimension of input features.
            num_learners (int): Number of weak classifiers to create.
        """
        self.sample_weights = None
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """
        Train the AdaBoost classifier.

        Args:
            X_train (DataFrame): Training features.
            y_train (array-like): Training labels.
            num_epochs (int): Number of training epochs for each weak learner.
            learning_rate (float): Learning rate for SGD optimizer.
        """
        n_samples = len(X_train)
        self.sample_weights = np.ones(n_samples) / n_samples
        y_train = y_train.reshape(-1, 1)

        for idx, learner in enumerate(self.learners):
            optimizer = optim.SGD(learner.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.float32)

                optimizer.zero_grad()
                raw_output = learner(X_tensor)
                predictions = torch.sigmoid(raw_output)
                sample_weights_tensor = torch.tensor(
                    self.sample_weights, dtype=torch.float32
                ).reshape(-1, 1)
                loss = -torch.sum(
                    sample_weights_tensor * (
                        y_tensor * torch.log(predictions + 1e-10) + (1 - y_tensor) * torch.log(1 - predictions + 1e-10)
                    )
                )
                loss.backward()
                optimizer.step()

                # if epoch % 100 == 0:
                #     print(f"Weak Classifier {idx + 1}, Epoch {epoch}, "
                #           f"Loss: {loss.item():.4f}")

            with torch.no_grad():
                raw_output = learner(X_tensor)
                probabilities = torch.sigmoid(raw_output).numpy()
                predictions = (probabilities >= 0.5).astype(int).flatten()
                incorrect = (predictions != y_train.flatten()).astype(int)
                error = np.sum(self.sample_weights * incorrect) / np.sum(self.sample_weights)
                # print(f"error: {error}")

            error = min(max(error, 1e-10), 1 - 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)

            self.sample_weights *= np.exp(alpha * incorrect)
            self.sample_weights /= np.sum(self.sample_weights)

    def predict_learners(self, X_test) -> t.List[np.ndarray]:
        """
        Get predictions from all weak learners.

        Args:
            X_test (DataFrame): Test features.

        Returns:
            List[np.ndarray]: List of predictions from each weak learner.
        """
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        learner_predictions = []
        for learner in self.learners:
            raw_output = learner(X_tensor)
            preds = torch.sigmoid(raw_output).detach().numpy()
            learner_predictions.append(preds)
        return learner_predictions

    def compute_feature_importance(self) -> t.List[float]:
        """
        Compute feature importance based on weak learner weights.

        Returns:
            List[float]: Importance scores for each feature.
        """
        feature_importances = np.zeros(self.learners[0].layer.in_features)

        for alpha, learner in zip(self.alphas, self.learners):
            feature_weights = learner.layer.weight.detach().numpy().flatten()
            feature_importances += np.abs(feature_weights) * alpha

        return feature_importances.tolist()

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the ensemble of weak learners.

        Args:
            X (DataFrame): Test features.

        Returns:
            np.ndarray: Final binary predictions.
        """
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        learner_preds = [
            alpha * torch.sigmoid(learner(X_tensor)).detach().numpy()
            for alpha, learner in zip(self.alphas, self.learners)
        ]
        weighted_sum = np.sum(learner_preds, axis=0) / np.sum(self.alphas)
        final_prediction = (weighted_sum >= 0.5).astype(int)
        return final_prediction
