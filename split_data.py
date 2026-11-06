# Separate input features and target
X = processed_df[features]
y = processed_df["Target"]

# Calculate 80% training split
split_index = int(len(processed_df) * 0.8)

# Split chronologically with a one-day boundary gap
X_train = X.iloc[:split_index - 1]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index - 1]
y_test = y.iloc[split_index:]

# Display dataset sizes
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nTesting period:")
print(X_test.index.min(), "to", X_test.index.max())
