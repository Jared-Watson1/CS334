import time
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data(training_path, testing_path):
    train_data = pd.read_csv(training_path)
    test_data = pd.read_csv(testing_path)
    return train_data, test_data


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    results = {}
    C_values = [0.01, 0.1, 1, 10, 100]
    for C in C_values:
        print(f"Training Logistic Regression (C={C}, penalty='l2')...")
        model = LogisticRegression(
            C=C, penalty="l2", solver="lbfgs", max_iter=1000
        )  # Increased max_iter
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
        results[str((C, "l2"))] = {
            "accuracy": accuracy,
            "classification_report": report,
        }
        print(f"Accuracy: {accuracy:.2f}")
    return results


def main():
    print("Starting logistic regression training...")
    # training_path = "data/training.csv"
    # testing_path = "data/testing.csv"
    training_path = "data/train_bias_cleaned.csv"
    testing_path = "data/test_bias_cleaned.csv"

    train_data, test_data = load_data(training_path, testing_path)

    print("Preparing feature matrices and target vectors...")
    X_train = train_data.drop("loan_status", axis=1)
    y_train = train_data["loan_status"]
    X_test = test_data.drop("loan_status", axis=1)
    y_test = test_data["loan_status"]

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    results = train_and_evaluate_logistic_regression(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    with open("data/logistic_regression.json", "w") as f:
        json.dump(results, f, indent=4)

    print(
        "Finished logistic regression training and evaluation. Results saved to logistic_regression.json."
    )


if __name__ == "__main__":
    main()
