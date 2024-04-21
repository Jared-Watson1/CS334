import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import time


def preprocess_chunk(chunk, scaler, encoder, initial_features=None, fit=False):
    """
    This function preprocesses each chunk of data.
    """
    chunk = remove_irrelevant_features(chunk)
    chunk = handle_missing_values(chunk)
    chunk = feature_engineering(chunk)
    chunk = encode_and_normalize(chunk, scaler, encoder, initial_features, fit=fit)
    return chunk


def remove_irrelevant_features(data):
    """
    Remove irrelevant features such as IDs and specific non-informative fields.
    """
    columns_to_drop = [
        "id",
        "member_id",
        "url",
        "desc",
        "zip_code",
        "emp_title",
        "addr_state",
        "settlement_date",
        "settlement_status",
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    return data


def handle_missing_values(data):
    """
    Impute missing values based on column data type.
    """
    data = data.loc[:, data.isnull().mean() < 0.6].copy()
    for column in data.columns:
        imputer_strategy = (
            "most_frequent" if data[column].dtype == "object" else "median"
        )
        imputer = SimpleImputer(strategy=imputer_strategy)
        data[column] = imputer.fit_transform(data[[column]]).ravel()
    return data


def feature_engineering(data):
    """
    Apply any feature engineering steps.
    """
    if "term" in data.columns:
        data["term"] = data["term"].str.extract("(\d+)").astype(float)
    return data


def encode_and_normalize(data, scaler, encoder, initial_features, fit=False):
    """
    One-hot encode categorical variables and normalize numerical features.
    """
    # Separate categorical and numerical data
    categorical_columns = data.select_dtypes(include=["object"]).columns
    numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns

    # Process categorical data
    if fit:
        data_encoded = encoder.fit_transform(data[categorical_columns])
    else:
        data_encoded = encoder.transform(data[categorical_columns])

    # Convert to DataFrame and handle feature names
    data_encoded = pd.DataFrame(
        data_encoded.toarray(),
        columns=encoder.get_feature_names_out(categorical_columns),
    )
    data = pd.concat([data.drop(categorical_columns, axis=1), data_encoded], axis=1)

    # Normalize numerical features
    if fit:
        scaler.fit(data[numerical_columns])
    data[numerical_columns] = scaler.transform(data[numerical_columns])
    return data


if __name__ == "__main__":
    start_time = time.time()
    file_path = "loan.csv"
    chunksize = 10000

    scaler = MinMaxScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")
    first_chunk = True

    reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        processed_chunk = preprocess_chunk(chunk, scaler, encoder, fit=first_chunk)
        mode = "w" if first_chunk else "a"
        header = first_chunk
        processed_chunk.to_csv(
            "preprocessed_loan_data.csv", mode=mode, header=header, index=False
        )
        first_chunk = False  # Update first_chunk flag after first use

    print(f"Finished preprocessing: {time.time() - start_time}s")
