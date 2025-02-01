"""
You don't have to follow the structure of the sample code.
However, you should check if your class/function meets the requirements.
"""
import numpy as np
import pandas as pd


class DecisionTree:
    """A basic implementation of a decision tree classifier."""
    def __init__(self, max_depth=0):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the dataset.

        Args:
            X (array-like or DataFrame): Features.
            y (array-like or Series): Target labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Args:
            X (array): Features.
            y (array): Target labels.
            depth (int): Current depth of the tree.

        Returns:
            dict or int: The tree structure or a leaf value.
        """
        if depth == self.max_depth or len(np.unique(y)) == 1:
            leaf_value = int(np.bincount(y).argmax())
            return leaf_value

        feature_index, threshold = find_best_split(X, y)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_subtree = self._grow_tree(
            X[left_indices], y[left_indices], depth + 1
        )
        right_subtree = self._grow_tree(
            X[right_indices], y[right_indices], depth + 1
        )

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def predict(self, X):
        """
        Predict labels for the given features.

        Args:
            X (array-like or DataFrame): Features.

        Returns:
            array: Predicted labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            x (array): Feature values for a single sample.
            tree_node (dict or int): Current tree node.

        Returns:
            int: Predicted label.
        """
        if isinstance(tree_node, int):
            return tree_node

        if isinstance(tree_node, dict):
            if x[tree_node["feature_index"]] <= tree_node["threshold"]:
                return self._predict_tree(x, tree_node["left"])
            return self._predict_tree(x, tree_node["right"])

        raise ValueError("Invalid tree structure")

    def compute_feature_importance(self):
        """
        Compute the importance of each feature based on split counts.

        Returns:
            list: Importance scores for each feature.
        """
        feature_importances = np.zeros(self.tree["feature_index"] + 1)

        def dfs(node):
            if isinstance(node, int) or node is None:
                return
            feature_importances[node["feature_index"]] += 1
            dfs(node["left"])
            dfs(node["right"])

        dfs(self.tree)
        return feature_importances.tolist()


def split_dataset(X, y, feature_index, threshold):
    """
    Split the dataset based on a feature and threshold.

    Args:
        X (array): Features.
        y (array): Target labels.
        feature_index (int): Feature index to split on.
        threshold (float): Threshold value.

    Returns:
        tuple: Left and right splits of features and labels.
    """
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return (X[left_indices], y[left_indices]), (X[right_indices], y[right_indices])


def find_best_split(X, y):
    """
    Find the best feature and threshold to split the data.

    Args:
        X (array-like): Features.
        y (array-like): Target labels.

    Returns:
        tuple: Best feature index and threshold for the split.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    best_feature = None
    best_threshold = None
    best_impurity = float("inf")

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            (X_left, y_left), (X_right, y_right) = split_dataset(
                X, y, feature_index, threshold
            )

            impurity = (len(y_left) / len(y)) * entropy(y_left) + \
                       (len(y_right) / len(y)) * entropy(y_right)

            if impurity < best_impurity:
                best_feature = feature_index
                best_threshold = threshold
                best_impurity = impurity

    return best_feature, best_threshold


def entropy(y):
    """
    Calculate the entropy of the labels.

    Args:
        y (array-like): Labels.

    Returns:
        float: Entropy value.
    """
    if len(y) == 0:
        return 0

    prob = np.bincount(y) / len(y)
    return -np.sum(prob * np.log2(prob + 1e-10))


def gini(y):
    """
    Calculate the Gini impurity of the labels.

    Args:
        y (array-like): Labels.

    Returns:
        float: Gini impurity value.
    """
    if len(y) == 0:
        return 0

    prob = np.bincount(y) / len(y)
    return 1 - np.sum(np.square(prob))
