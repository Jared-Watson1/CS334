import time
import pandas as pd
import json
import numpy as np


def load_data(filepath):
    return pd.read_csv(filepath)


def load_feature_importance(filepath):
    with open(filepath, "r") as json_file:
        feature_stats = json.load(json_file)
    return feature_stats["feature_importance"]


def select_features(data, feature_importance, correlation_threshold=0.75):
    """Select features based on feature importance and correlation"""
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > correlation_threshold)
    ]

    selected_features = [
        feature for feature in feature_importance if feature not in to_drop
    ]

    selected_features.append("loan_status")

    return data[selected_features]


def drop_unwanted_features(data_file, feature_stats_file, output_file):
    """Drop features not in the selected feature list based on training data"""
    data = load_data(data_file)
    feature_importance = load_feature_importance(feature_stats_file)
    data_selected_features = select_features(data, feature_importance)
    data_selected_features.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")


def main():
    start_time = time.time()
    print("Starting feature selection and data cleanup")

    train_data_file = "data/processed_train_data.csv"
    test_data_file = "data/processed_test_data.csv"
    train_stats_file = "data/train_feature_stats.json"
    train_output_file = "data/training_cleaned.csv"
    test_output_file = "data/testing_cleaned.csv"

    print("Processing training data...")
    drop_unwanted_features(train_data_file, train_stats_file, train_output_file)

    print("Processing test data...")
    drop_unwanted_features(test_data_file, train_stats_file, test_output_file)

    print(f"Finished processing data: {time.time() - start_time}")


if __name__ == "__main__":
    main()
