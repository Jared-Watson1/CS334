import pandas as pd

df = pd.read_csv("data/testing.csv")

sampled_df = df.sample(n=1000)
out_file = "data/testing_sample.csv"
# Save the sampled data to a new CSV file
sampled_df.to_csv(out_file, index=False)

print(f"Sampled data has been saved to {out_file}.")
