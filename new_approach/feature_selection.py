import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import json


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def plot_correlation_matrix(data):
    """Plot the correlation matrix for numerical features only."""
    numerical_features = data.select_dtypes(include=["float64", "int64"])
    plt.figure(figsize=(15, 10))
    corr = numerical_features.corr().abs()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Numerical Feature Correlation Matrix")
    plt.show()


def feature_importance(data, target_column):
    """Compute feature importance using a Random Forest classifier"""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_imputed, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    return importance.to_dict()


def select_features(data, target_column):
    """Select features using univariate statistical tests"""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_imputed, y)
    scores = pd.Series(selector.scores_, index=X.columns)
    scores = scores.sort_values(ascending=False)
    return scores.to_dict()


def write_feature_stats_to_json(filepath, importance, scores):
    """Write feature importance and scores to a JSON file."""
    feature_stats = {"feature_importance": importance, "feature_scores": scores}
    with open(filepath, "w") as json_file:
        json.dump(feature_stats, json_file, indent=4)


def main():
    start_time = time.time()
    print("Starting feature selection")
    filepath = "data/processed_loan_data.csv"
    data = load_data(filepath)
    print("Data loaded, plotting correlation matrix")
    plot_correlation_matrix(data)
    print(
        f"Created correlation matrix: {time.time() - start_time}\nComputing feature importance"
    )
    importance = feature_importance(data, "loan_status")
    print(
        f"Feature importance complete: {time.time() - start_time}\nSelecting feature importance"
    )
    scores = select_features(data, "loan_status")
    print(f"Feature importance complete: {time.time() - start_time}")
    feature_stats_path = "data/feature_stats.json"
    write_feature_stats_to_json(feature_stats_path, importance, scores)
    print(f"Feature statistics written to {feature_stats_path}")

    print("Feature Importance:\n", json.dumps(importance, indent=4))
    print("Feature Scores from Univariate Selection:\n", json.dumps(scores, indent=4))


if __name__ == "__main__":
    main()
