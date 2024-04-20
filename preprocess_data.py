import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time


def preprocess_chunk(chunk):
    """
    This function takes a chunk of the DataFrame and performs preprocessing steps.
    """
    chunk = remove_irrelevant_features(chunk)
    chunk = handle_missing_values(chunk)
    chunk = feature_engineering(chunk)
    chunk = encode_and_normalize(chunk)
    return chunk


def remove_irrelevant_features(data):
    """
    Remove irrelevant features such as IDs and specific timestamps.
    """
    columns_to_drop = ["id", "member_id", "url", "desc", "zip_code", "emp_title"]
    data.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    return data


def handle_missing_values(data):
    """
    Remove or impute missing values.
    """
    missing_threshold = 0.6
    data = data.loc[:, data.isnull().mean() < missing_threshold]

    for column in data.columns:
        if data[column].dtype == "object" or data[column].dtype.name == "category":
            data[column] = data[column].astype("category")  # Convert to category first
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            imputer = SimpleImputer(strategy="median")

        # Perform imputation and flatten the result to a 1D array before assignment
        data[column] = imputer.fit_transform(data[[column]]).ravel()

    return data


def feature_engineering(data):
    """
    Derive new features that might be helpful for the models.
    """
    # Add any feature engineering steps here
    return data


def encode_and_normalize(data):
    """
    Convert categorical variables using one-hot encoding and normalize numerical features.
    """
    # Isolate numerical columns for scaling
    numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Convert categorical variables to type 'category' to save memory
    categorical_columns = data.select_dtypes(include=["category"]).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data


if __name__ == "__main__":
    start_time = time.time()
    file_path = "loan.csv"
    chunksize = 10000  # Adjust this based on your machine's memory
    first_chunk = True

    # Use iterator to process chunks
    reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)

    for chunk in reader:
        processed_chunk = preprocess_chunk(chunk)

        # Define the mode in which to open the file ('w' for write first time, 'a' for append thereafter)
        mode = "w" if first_chunk else "a"
        # Write headers only on the first chunk
        header = first_chunk

        processed_chunk.to_csv(
            "preprocessed_loan_data.csv", mode=mode, header=header, index=False
        )

        if first_chunk:
            first_chunk = False  # Only write the header once, on the first chunk

    print(f"Finished preprocessing: {time.time() - start_time}s")
