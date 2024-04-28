import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

def load_data():
    print("Loading data...")
    train_data = pd.read_csv("data/training_set.csv")
    test_data = pd.read_csv("data/testing_set.csv")
    print("Data loaded successfully.")
    return train_data, test_data

def prepare_data(df):
    print("Preparing data...")
    # Filter out the loan status columns as the targets
    target_columns = [col for col in df.columns if col.startswith("loan_status_")]
    y = df[target_columns]
    X = df.drop(target_columns, axis=1)
    
    # Randomly sample 5% of the data to manage memory and computation time
    # X, _, y, _ = train_test_split(X, y, test_size=0.95, random_state=42)
    # print(f"Data prepared with {X.shape[0]} samples.")
    return X, y

def reduce_dimensions(X_train, X_test):
    print("Reducing dimensions...")
    # Reduce dimensionality using PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print("Dimensions reduced.")
    return X_train, X_test

def train_knn(X_train, y_train):
    print("Training KNN model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Scale features for KNN
    model = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("KNN model trained successfully.")
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    print("Evaluating model...")
    # Scale the test features
    X_test_scaled = scaler.transform(X_test)
    # Predicting the Test set results using batch processing
    y_pred = batch_predict(model, X_test_scaled, batch_size=100)
    acc = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)
    print(f"Model evaluated. Accuracy: {acc}, Hamming Loss: {h_loss}")
    return acc, h_loss

def batch_predict(model, X, batch_size):
    print("Starting batch prediction...")
    predictions = []
    total_batches = (len(X) + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, len(X), batch_size)):
        end = min(start + batch_size, len(X))
        predictions.extend(model.predict(X[start:end]))
        print(f"Processed batch {i+1}/{total_batches}.")
    print("Batch prediction completed.")
    return predictions

def main():
    # Load the data
    train_data, test_data = load_data()

    # Prepare the data
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Reduce dimensions
    X_train, X_test = reduce_dimensions(X_train, X_test)

    # Train the model
    model, scaler = train_knn(X_train, y_train)

    # Evaluate the model
    accuracy, h_loss = evaluate_model(model, scaler, X_test, y_test)

    # Optionally save the model and scaler
    joblib.dump(model, "optimized_knn_model.pkl")
    joblib.dump(scaler, "optimized_scaler.pkl")

if __name__ == "__main__":
    main()
