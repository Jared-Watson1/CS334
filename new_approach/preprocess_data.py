import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler


def load_and_sample_data(
    file_path="../loan.csv", output_file_path="data/raw_loan_data.csv"
):
    """Load the dataset and take a random 50% sample because the dataset is too large"""
    data = pd.read_csv(file_path)
    sampled_data = data.sample(frac=0.5, random_state=1)
    sampled_data.to_csv(output_file_path, index=False)
    return output_file_path


def normalize_numerical_features(data):
    """Normalize numerical features in the dataset."""
    # Select columns that are int or float types for normalization
    numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data


def remove_columns_with_many_missing_values(data, threshold=0.5):
    """Remove columns with more than a specified threshold of missing values."""
    limit = len(data) * threshold
    data = data.dropna(thresh=limit, axis=1)
    return data


def encode_categorical_variables(data, target_column="loan_status", mappings=None):
    """Encode categorical variables including a specific mapping for the target column."""
    # Normalize 'Late' and other variations
    if target_column in data.columns:
        data[target_column] = data[target_column].apply(
            lambda x: "Late" if "Late" in str(x) else x
        )
        data[target_column] = data[target_column].replace(mappings)
    # Remove rows with loan statuses not in the mappings
    data = data[data[target_column].isin(mappings.values())]
    for col in data.select_dtypes(include=["object"]).columns:
        if col != target_column:
            data[col], _ = pd.factorize(data[col])
    return data


def main():
    """Main function to run all preprocessing steps."""
    start_time = time.time()
    print("Starting preprocessing")

    # Load and sample half data from loan.csv
    sampled_file_path = load_and_sample_data()
    data = pd.read_csv(sampled_file_path)

    print(
        f"Sampled data from loan.csv: {time.time() - start_time}\nNormalizing numerical features"
    )
    # Normalize numerical features first
    data = normalize_numerical_features(data)

    print(
        f"Numerical features normalized: {time.time() - start_time}\nRemoving columns with missing values."
    )
    # Remove columns with many missing values
    data = remove_columns_with_many_missing_values(data)

    print(
        f"Removed columns with many missing values: {time.time() - start_time}\nEncoding categorical values"
    )
    # Encode categorical variables
    loan_status_mappings = {
        "Current": 0,
        "Fully Paid": 1,
        "Charged Off": 3,
        "Late": 4,
        "In Grace Period": 5,
    }
    data = encode_categorical_variables(data, "loan_status", loan_status_mappings)

    # Save the final processed data
    final_file_path = "data/processed_loan_data.csv"
    data.to_csv(final_file_path, index=False)
    print(f"Encoded categorical variables: {time.time() - start_time}")

    print(f"Preprocessing completed. Processed data saved to: {final_file_path}")


if __name__ == "__main__":
    main()
