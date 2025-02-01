import pandas as pd
from loguru import logger
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc
from src.decision_tree import entropy, gini


def main():
    """
    Note:
    1. Part of the line should not be modified.
    2. You should implement the algorithm by yourself.
    3. You can change the I/O data type as you need.
    4. You can change the hyperparameters as you want.
    5. You can add/modify/remove args in the function,
       but you need to fit the requirements.
    6. When plotting the feature importance,
       the tick labels of one of the axes should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # Preprocess data
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # AdaBoost Classifier
    clf_adaboost = AdaBoostClassifier(input_dim=X_train.shape[1])
    _ = clf_adaboost.fit(
        X_train, y_train, num_epochs=500, learning_rate=0.007
    )
    final_predictions = clf_adaboost.predict(X_test)
    accuracy_ = np.mean(final_predictions.flatten() == y_test)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')

    y_pred_probs = clf_adaboost.predict_learners(X_test)
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./AdaBoost_learners_roc.png',
    )
    feature_importance = clf_adaboost.compute_feature_importance()

    # Plot feature importance
    feature_names = X_train.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel('AdaBoost_Feature_Importance')
    plt.title('Feature Importance of Each Feature in AdaBoost')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('./AdaBoost_feature_importance.png')

    # Bagging Classifier
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(
        X_train, y_train, num_epochs=500, learning_rate=0.007
    )
    final_predictions = clf_bagging.predict(X_test)
    y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = np.mean(final_predictions.flatten() == y_test)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./bagging_learners_roc.png',
    )
    feature_importance = clf_bagging.compute_feature_importance()

    # Plot feature importance for Bagging
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel('Bagging_Feature_Importance')
    plt.title('Feature Importance of Each Feature in Bagging')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('./Bagging_feature_importance.png')

    # Gini and Entropy Calculation
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")

    # Decision Tree Classifier
    sample_fraction = 0.8
    X_train_sampled = X_train.sample(frac=sample_fraction, random_state=45)
    y_train_sampled = y_train[X_train_sampled.index]
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train_sampled, y_train_sampled)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    feature_importance = clf_tree.compute_feature_importance()

    # Plot feature importance for Decision Tree
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel('DecisionTree_Feature_Importance')
    plt.title('Feature Importance of Each Feature in DecisionTree')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('./DecisionTree_feature_importance.png')


if __name__ == '__main__':
    main()
