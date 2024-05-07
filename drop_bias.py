import pandas as pd

bias_columns = [
    "zip_code",
    "addr_state",
    "emp_title",
    "home_ownership",
    "emp_length",
    "annual_inc",
    "verification_status",
    "mths_since_recent_inq",
    "inq_last_6mths",
    "inq_last_12m",
    "loan_amnt"
]

def drop_bias_columns(input_file, output_file):
    """Drop specified columns from the dataset and save the cleaned data"""
    data = pd.read_csv(input_file)
    data_cleaned = data.drop(columns=bias_columns, errors='ignore')
    data_cleaned.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")

def main():
    train_input_file = "data/training_cleaned.csv"
    test_input_file = "data/testing_cleaned.csv"
    train_output_file = "data/train_bias_cleaned.csv"
    test_output_file = "data/test_bias_cleaned.csv"

    print("Processing training data to remove bias columns...")
    drop_bias_columns(train_input_file, train_output_file)

    print("Processing test data to remove bias columns...")
    drop_bias_columns(test_input_file, test_output_file)

if __name__ == "__main__":
    main()
