# Select final machine learning columns
processed_df = df[features + ["Target"]].copy()

# Save processed dataset
processed_df.to_csv("NVDA_processed_data.csv")

print("Processed dataset saved successfully!")
print("Dataset shape:", processed_df.shape)
print(processed_df.head())
