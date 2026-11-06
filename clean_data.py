# Check for duplicate dates
print("Duplicate dates:", nvda.index.duplicated().sum())

# Check for missing values
print("\nMissing values:")
print(nvda.isnull().sum())

# Check the date range
print("\nFirst date:", nvda.index.min())
print("Last date:", nvda.index.max())

# Check dataset dimensions
print("\nDataset shape:", nvda.shape)
