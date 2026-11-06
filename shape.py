print("Dataset shape:", nvda.shape)
print("Columns:", nvda.columns)
print("Missing values:")
print(nvda.isnull().sum())

# Save the dataset as a CSV file
nvda.to_csv("NVDA_raw_data.csv")
