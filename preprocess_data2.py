import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, low_memory=False)

    # Select relevant columns
    columns_of_interest = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "annual_inc",
        "dti",
        "purpose",
        "addr_state",
        "loan_status",
    ]
    data = data[columns_of_interest]

    # Define preprocessing for numerical attributes
    numerical_cols = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti"]
    num_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )

    # Define preprocessing for categorical attributes
    categorical_cols = ["term", "grade", "purpose", "addr_state", "loan_status"]
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    # Apply transformations
    data_preprocessed = preprocessor.fit_transform(data)

    # Generate column names for the transformed DataFrame
    categorical_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_cols)
    )
    columns_transformed = numerical_cols + list(categorical_features)

    # Check the shape of the data_preprocessed
    print("Shape of data_preprocessed:", data_preprocessed.shape)

    # Convert the output back to a DataFrame
    if data_preprocessed.ndim > 1 and data_preprocessed.shape[1] == len(
        columns_transformed
    ):
        data_preprocessed_df = pd.DataFrame(
            data_preprocessed, columns=columns_transformed
        )
    else:
        raise ValueError(
            "The shape of the transformed data does not match the number of expected columns."
        )

    # Save the processed data to a new CSV file
    data_preprocessed_df.to_csv("preprocessed_data2.csv", index=False)
    print("Data preprocessing complete and saved to 'preprocessed_data2.csv'.")


# Example usage
file_path = "loan.csv"
preprocess_data(file_path)
