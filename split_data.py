import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
data = pd.read_csv("modified_preprocessed_data2.csv")

# Split the dataset into training (75%) and testing (25%) sets
train_set, test_set = train_test_split(data, test_size=0.25, random_state=42)

# Save the training and testing sets to new CSV files
train_set.to_csv("data/training_set.csv", index=False)
test_set.to_csv("data/testing_set.csv", index=False)

print("Data has been split and saved successfully.")
