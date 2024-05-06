import time
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k_values, metrics):
    results = {}
    for metric in metrics:
        for k in k_values:
            print(f"Training KNN (k={k}, metric={metric})...")
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )
            results[(k, metric)] = {
                "accuracy": accuracy,
                "classification_report": report,
            }
            print(f"Accuracy: {accuracy:.2f}")
    return results


def main():
    print("Starting KNN classifier training...")
    # training_path = "data/training_cleaned.csv"
    # testing_path = "data/testing_cleaned.csv"
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

    k_values = [3, 5, 7, 9]  # Example: Test these k values
    metrics = ["euclidean", "manhattan"]
    results = train_and_evaluate_knn(
        X_train_scaled, y_train, X_test_scaled, y_test, k_values, metrics
    )

    # Save the results to a JSON file
    with open("knn_classifier.json", "w") as f:
        json.dump(results, f, indent=4)

    print(
        "Finished KNN classifier training and evaluation. Results saved to knn_classifier.json."
    )


if __name__ == "__main__":
    main()
