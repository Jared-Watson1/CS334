import time
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np


def load_data(filepath):
    return pd.read_csv(filepath)


def load_feature_importance(filepath):
    with open(filepath, "r") as json_file:
        feature_stats = json.load(json_file)
    return feature_stats["feature_importance"]


def select_features(data, feature_importance, correlation_threshold=0.75):
    # Create a mask for highly correlated features
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # get index of feature columns with high correlation
    to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > correlation_threshold)
    ]

    # select features based on importance and low correlation
    selected_features = [
        feature for feature in feature_importance if feature not in to_drop
    ]

    selected_features.append("loan_status")

    return data[selected_features]


def split_and_save_data(data, target_column, test_size, training_file, testing_file):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(training_file, index=False)
    test_data.to_csv(testing_file, index=False)


def main():
    start_time = time.time()
    print("Starting split_data.py")
    filepath = "data/processed_loan_data.csv"
    json_filepath = "data/feature_stats.json"
    training_file = "data/training.csv"
    testing_file = "data/testing.csv"

    data = load_data(filepath)
    print("Loading feature importance")
    feature_importance = load_feature_importance(json_filepath)
    print(f"Feature importance loaded: {time.time() - start_time}")
    # Select the top features for prediction based on importance and correlation
    print("Selecting top features for prediction")
    data_selected_features = select_features(data, feature_importance)
    print(f"Finished selected top features: {time.time() - start_time}")

    # Split the data into training and testing sets and save them
    print("Splitting data...")
    split_and_save_data(
        data_selected_features,
        "loan_status",
        test_size=0.25,
        training_file=training_file,
        testing_file=testing_file,
    )
    print(f"Finished splitting data: {time.time() - start_time}")

    print(
        f"Data has been split into training and testing sets and saved to {training_file} and {testing_file}."
    )


if __name__ == "__main__":
    main()
