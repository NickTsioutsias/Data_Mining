import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/processed/dataset_1_balanced_sampled.csv')

# Print the first 5 rows (default)
print(df.head())