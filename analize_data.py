import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """
    Load the dataset from a specified file path.
    """
    return pd.read_csv(file_path)


def summarize_missing_values(data):
    """
    Generate a summary of missing values in the dataset.
    """
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_summary = pd.DataFrame(
        {"Number of Missing Values": missing_values, "Percentage": missing_percentage}
    )
    print("Summary of Missing Values:\n", missing_summary)


def describe_data(data):
    """
    Provide basic descriptive statistics of the dataset.
    """
    print("Basic Descriptive Statistics:\n", data.describe())


def visualize_data(data):
    """
    Visualize important characteristics and distributions of variables.
    Focus on demographic and loan performance variables.
    """
    # Visualization of Loan Amount Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data["loan_amnt"], kde=True)
    plt.title("Distribution of Loan Amounts")
    plt.xlabel("Loan Amount")
    plt.ylabel("Frequency")

    # Visualization of Annual Income Distribution (limited to certain range for better visualization)
    plt.figure(figsize=(10, 5))
    sns.histplot(data[data["annual_inc"] < 200000]["annual_inc"], kde=True)
    plt.title("Distribution of Annual Income")
    plt.xlabel("Annual Income")
    plt.ylabel("Frequency")

    # Visualization of Loan Status (Count Plot)
    plt.figure(figsize=(10, 5))
    sns.countplot(y="loan_status", data=data)
    plt.title("Loan Status Counts")
    plt.xlabel("Count")
    plt.ylabel("Loan Status")

    plt.show()


if __name__ == "__main__":
    # Change this to your CSV file path
    file_path = "loan.csv"
    data = load_data(file_path)
    summarize_missing_values(data)
    describe_data(data)
    visualize_data(data)
