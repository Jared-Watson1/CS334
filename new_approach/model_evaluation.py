import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt


def load_data(filepath):
    return pd.read_csv(filepath)


def load_best_params(model_name):
    with open(f"{model_name}_best_params.json", "r") as f:
        best_params = json.load(f)
    return best_params["best_params"]


def preprocess_data(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled


def train_model(X_train, y_train, model_type, best_params):
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(**best_params)
    elif model_type == "knn":
        model = KNeighborsClassifier(**best_params)
    elif model_type == "logistic_regression":
        model = LogisticRegression(**best_params)
    model.fit(X_train, y_train)
    return model


def plot_roc_curves(models, X_test, y_test, classes):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    # Plot linewidth
    lw = 2

    # Compute ROC curve and ROC area for each class
    plt.figure()

    for name, model in models.items():
        classifier = OneVsRestClassifier(model)
        y_score = classifier.fit(X_test, y_test_bin).predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'ROC curve of {name} (area = {roc_auc["micro"]:.2f})',
            lw=lw,
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic to multi-class")
    plt.legend(loc="lower right")
    plt.show()


def main():
    training_path = "data/training.csv"
    testing_path = "data/testing.csv"

    # Load and prepare data
    train_data = load_data(training_path)
    test_data = load_data(testing_path)
    X_train, y_train = train_data.drop("loan_status", axis=1), train_data["loan_status"]
    X_test, y_test = test_data.drop("loan_status", axis=1), test_data["loan_status"]
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    models = {}
    for model_name in ["decision_tree", "knn", "logistic_regression"]:
        best_params = load_best_params(model_name)
        model = train_model(X_train, y_train, model_name, best_params)
        models[model_name] = model

    plot_roc_curves(models, X_test, y_test, classes=np.unique(y_train))


if __name__ == "__main__":
    main()