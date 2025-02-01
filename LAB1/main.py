import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        n_train = np.size(X, 0)
        X_input_train = np.concatenate((X, np.ones([n_train, 1])), axis=1)
        tmp1 = np.matmul(np.transpose(X_input_train), X_input_train)
        tmp1 = np.linalg.pinv(tmp1)
        tmp2 = np.matmul(np.transpose(X_input_train), y)
        self.weights = np.matmul(tmp1, tmp2)
        self.intercept = self.weights[-1]
        self.weights = self.weights[:-1]

    def predict(self, X):
        n_test = np.size(X, 0)
        X_input_test = np.concatenate((X, np.ones([n_test, 1])), axis=1)
        y_pre = np.matmul(X_input_test, np.r_[self.weights, self.intercept])
        return y_pre


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate=1e-4, epochs=70000):
        n_train, n_features = X.shape
        y = y.flatten()
        self.weights = np.zeros(n_features)
        self.intercept = 0
        losses = []

        for i in range(0, epochs):
            y_pred = np.dot(X, self.weights) + self.intercept
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            dw = (2 / n_train) * np.dot(X.T, (y_pred - y))
            db = (2 / n_train) * np.sum(y_pred - y)
            self.weights -= learning_rate * dw
            self.intercept -= learning_rate * db

        return losses

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(losses)
        plt.title("Traing loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.show()


def compute_mse(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=1145000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
