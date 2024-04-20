import pandas as pd
from sklearn.model_selection import train_test_split
import time


def load_data(file_path):
    """
    Load the preprocessed dataset
    """
    return pd.read_csv(file_path)


def split_data(data, train_size=0.6, test_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    """
    # first split to separate out the training set
    train_data, remaining_data = train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    # split the remaining data equally for validation and test sets
    validation_data, test_data = train_test_split(
        remaining_data, test_size=0.5, random_state=random_state
    )

    return train_data, validation_data, test_data


if __name__ == "__main__":
    start_time = time.time()
    # Load the preprocessed data
    file_path = "preprocessed_loan_data.csv"
    data = load_data(file_path)

    # Split the data
    train_data, validation_data, test_data = split_data(data)

    train_data.to_csv("train.csv", index=False)
    validation_data.to_csv("validation.csv", index=False)
    test_data.to_csv("test.csv", index=False)

    print(f"Data has been split and saved successfully: {time.time()-start_time}s")
