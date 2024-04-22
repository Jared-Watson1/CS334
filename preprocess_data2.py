import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(file_path):
    # Load the dataset with low_memory=False to handle DtypeWarnings
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
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Define preprocessing for categorical attributes
    categorical_cols = ["term", "grade", "purpose", "addr_state", "loan_status"]
    cat_transformer = Pipeline(
        steps=[
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

    # Extract feature names for the categorical features
    # Must use the fitted ColumnTransformer's named step for 'cat' and the 'encoder'
    cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(input_features=categorical_cols)
    )
    all_features = numerical_cols + list(cat_features)

    # Convert the processed data back to a DataFrame
    data_preprocessed_df = pd.DataFrame(
        data_preprocessed.toarray(), columns=all_features
    )

    # Save the processed data to a new CSV file
    data_preprocessed_df.to_csv("preprocessed_data2.csv", index=False)
    print("Data preprocessing complete and saved to 'preprocessed_data.csv'.")


file_path = "loan.csv"
preprocess_data(file_path)
