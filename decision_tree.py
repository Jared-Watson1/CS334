import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import numpy as np


def load_data(training_path, testing_path):
    # Loads the training and testing data from CSV files.
    train_data = pd.read_csv(training_path)
    test_data = pd.read_csv(testing_path)
    return train_data, test_data


def train_decision_tree(X_train, y_train):
    # Initializes and trains the Decision Tree classifier.
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier


def evaluate_model(classifier, X_test, y_test):
    # Makes predictions and evaluates the model.
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, class_report


def main():
    start_time = time.time()
    print("Starting decision tree training...")
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

    print(f"Finished preparing data: {time.time() - start_time}")

    print("Starting training...")
    classifier = train_decision_tree(X_train, y_train)
    print(f"Finished training: {time.time() - start_time}")

    print("Evaluating model...")
    accuracy, class_report = evaluate_model(classifier, X_test, y_test)
    print(f"Finished evaluating model: {time.time() - start_time}")

    print(f"Accuracy of the Decision Tree model: {accuracy:.2f}")
    print("Classification Report:\n", class_report)


if __name__ == "__main__":
    main()
