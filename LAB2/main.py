import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0
        for _ in range(self.num_iterations):
            linear_model = np.dot(inputs, self.weights) + self.intercept
            y_pred = self.sigmoid(linear_model)
            dw = (2 / n_samples) * np.dot(inputs.T, (y_pred - targets))
            db = (2 / n_samples) * np.sum(y_pred - targets)
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, inputs: np.ndarray):
        linear_model = np.dot(inputs, self.weights) + self.intercept
        y_pred_probs = self.sigmoid(linear_model)
        y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred_probs]
        return y_pred_probs, y_pred_classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD."""
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        class_0 = inputs[targets == 0]
        class_1 = inputs[targets == 1]
        self.m0 = np.mean(class_0, axis=0)
        self.m1 = np.mean(class_1, axis=0)
        s0 = np.dot((class_0 - self.m0).T, (class_0 - self.m0))
        s1 = np.dot((class_1 - self.m1).T, (class_1 - self.m1))
        self.sw = s0 + s1
        self.sb = np.outer((self.m0 - self.m1), (self.m0 - self.m1))
        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))
        self.w = self.w / np.linalg.norm(self.w)
        self.slope = self.w[1] / self.w[0]

    def predict(self, inputs: np.ndarray):
        projections = np.dot(inputs, self.w)
        m0 = np.dot(self.m0, self.w)
        m1 = np.dot(self.m1, self.w)
        threshold = (m0 + m1) / 2
        projections[projections <= threshold] = 0
        projections[projections > threshold] = 1
        return projections

    def plot_projection(self, inputs: np.ndarray, targets: np.ndarray):
        plt.figure(figsize=(10, 10))
        plt.axes().set_aspect('equal')
        plt.xlim(np.min(inputs[:, 0]) - 0.2, (np.max(inputs[:, 0]) + 0.2))
        plt.ylim(np.min(inputs[:, 1]) - 0.2, (np.max(inputs[:, 1]) + 0.2))
        plt.scatter(inputs[targets == 0][:, 0], inputs[targets == 0][:, 1],
                    c="r", label='Class 0')
        plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1],
                    c="b", label='Class 1')
        b = np.mean(inputs[:, 1])
        x = np.linspace(np.min(inputs[:, 0]) - 0.2, np.max(inputs[:, 0]) + 0.2, 100)
        plt.plot(x, self.slope * x + b, c="black", linewidth=2, label='Projection Line')
        u = np.array([0, b])
        for i in range(inputs.shape[0]):
            projection = np.dot(inputs[i] - u, self.w) * self.w + u
            plt.scatter(projection[0], projection[1],
                        c=("r" if targets[i] == 0 else "b"), alpha=0.7)
            plt.plot([inputs[i][0], projection[0]], [inputs[i][1], projection[1]],
                     c="lightblue", alpha=0.3)
        plt.title(f"Projection Line: w={self.slope:.4f}, b={b:.2f}")
        plt.legend()
        plt.show()


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    return np.sum(y_trues == y_preds) / len(y_trues)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.1,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Don't modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_pred_fld = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_fld)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    FLD_.plot_projection(x_test, y_test)
    FLD_.plot_projection(x_train, y_train)


if __name__ == '__main__':
    main()
