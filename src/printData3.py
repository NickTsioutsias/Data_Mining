import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/processed/dataset_3_dbscan.csv')

# Print the first 5 rows (default)
print(df.head())