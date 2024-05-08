import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_sample_data(file_path="loan.csv", output_file_path="data/raw_loan_data.csv"):
    """Load the dataset and take a random 50% sample because the dataset is too large."""
    data = pd.read_csv(file_path)
    sampled_data = data.sample(frac=0.5, random_state=1)
    sampled_data.to_csv(output_file_path, index=False)
    return sampled_data


def normalize_numerical_features(train_data, test_data):
    """Normalize numerical features in the training and test datasets."""
    numerical_columns = train_data.select_dtypes(include=["int64", "float64"]).columns
    scaler = MinMaxScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
    return train_data, test_data


def remove_columns_with_many_missing_values(data, threshold=0.5):
    """Remove columns with more than a specified threshold of missing values."""
    limit = len(data) * threshold
    data = data.dropna(thresh=limit, axis=1)
    return data


def encode_categorical_variables(train_data, test_data, target_column="loan_status", mappings=None):
    """Encode categorical variables in the training and test datasets."""
    if target_column in train_data.columns:
        train_data[target_column] = train_data[target_column].apply(
            lambda x: "Late" if "Late" in str(x) else x
        )
        train_data[target_column] = train_data[target_column].replace(mappings)

        test_data[target_column] = test_data[target_column].apply(
            lambda x: "Late" if "Late" in str(x) else x
        )
        test_data[target_column] = test_data[target_column].replace(mappings)

    train_data = train_data[train_data[target_column].isin(mappings.values())]
    test_data = test_data[test_data[target_column].isin(mappings.values())]

    for col in train_data.select_dtypes(include=["object"]).columns:
        if col != target_column:
            train_data[col], unique_values = pd.factorize(train_data[col])
            test_data[col] = test_data[col].apply(lambda x: unique_values.get_loc(x) if x in unique_values else -1)

    return train_data, test_data


def main():
    """Main function to run all preprocessing steps."""
    start_time = time.time()
    print("Starting preprocessing")

    data = load_and_sample_data()

    print(
        f"Sampled data from loan.csv: {time.time() - start_time}\nRemoving columns with missing values."
    )
    data = remove_columns_with_many_missing_values(data)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['loan_status'])

    print(f"Data split into train and test sets: {time.time() - start_time}\nNormalizing numerical features")
    train_data, test_data = normalize_numerical_features(train_data, test_data)

    print(f"Numerical features normalized: {time.time() - start_time}\nEncoding categorical values")
    loan_status_mappings = {
        "Current": 0,
        "Fully Paid": 1,
        "Charged Off": 3,
        "Late": 4,
        "In Grace Period": 5,
    }
    train_data, test_data = encode_categorical_variables(train_data, test_data, "loan_status", loan_status_mappings)

    train_file_path = "data/processed_train_data.csv"
    test_file_path = "data/processed_test_data.csv"
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    print(f"Preprocessing completed. Processed data saved to: {train_file_path} and {test_file_path}")

if __name__ == "__main__":
    main()
