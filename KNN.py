import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def train_knn(X_train, y_train):
    # Initialize the KNeighborsClassifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Scale features for KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    # Scale the test features
    X_test_scaled = scaler.transform(X_test)
    # Predicting the Test set results
    y_pred = model.predict(X_test_scaled)
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
    model, scaler = train_knn(X_train, y_train)

    # Evaluate the model
    accuracy, h_loss = evaluate_model(model, scaler, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Hamming Loss: {h_loss}")

    # Optionally save the model and scaler
    joblib.dump(model, "multi_label_knn_model.pkl")
    joblib.dump(scaler, "scaler_knn.pkl")

if __name__ == "__main__":
    main()
