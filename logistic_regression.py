import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
import joblib


def load_data():
    train_data = pd.read_csv("data/training_set.csv")
    test_data = pd.read_csv("data/testing_set.csv")
    return train_data, test_data


def prepare_data(df):
    # Filter out the loan status columns as the targets
    target_columns = [col for col in df.columns if col.startswith("loan_status_")]
    y = df[target_columns]
    X = df.drop(target_columns, axis=1)
    return X, y


def train_logistic_regression(X_train, y_train):
    # Initialize the Logistic Regression inside OneVsRestClassifier for multi-label classification
    model = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)
    return acc, h_loss


def main():
    # Load the data
    train_data, test_data = load_data()

    # Prepare the data
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Train the model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate the model
    accuracy, h_loss = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Hamming Loss: {h_loss}")

    # joblib.dump(model, "logistic_regression_model.pkl")


if __name__ == "__main__":
    main()
