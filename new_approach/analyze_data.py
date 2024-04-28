import pandas as pd

df = pd.read_csv("data/processed_loan_data.csv")

sampled_df = df.sample(n=1000)
out_file = "data/processed_loan_data_sample.csv"
# Save the sampled data to a new CSV file
sampled_df.to_csv(out_file, index=False)

print(f"Sampled data has been saved to {out_file}.")
