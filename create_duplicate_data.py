# Create a copy of the original dataset
df = nvda.copy()

# Sort data chronologically
df = df.sort_index()

# Remove duplicate dates
df = df[~df.index.duplicated(keep="first")]

# Remove rows with missing raw data
df = df.dropna()

print("Cleaned dataset shape:", df.shape)
print(df.head())
