import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset          
df = pd.read_csv("data/cleaned_dataset.csv")

# Select numeric columns only
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

X = df[num_cols]

# ---------------------------------
# 1. Min-Max Scaling
# ---------------------------------
minmax_scaler = MinMaxScaler()
minmax_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(X),
    columns=num_cols
)
minmax_scaled.to_csv("data/minmax_scaled.csv", index=False)

# ---------------------------------
# 2. Max Absolute Scaling
# ---------------------------------
maxabs_scaler = MaxAbsScaler()
maxabs_scaled = pd.DataFrame(
    maxabs_scaler.fit_transform(X),
    columns=num_cols
)
maxabs_scaled.to_csv("data/maxabs_scaled.csv", index=False)

# ---------------------------------
# 3. Z-score Standardization
# ---------------------------------
standard_scaler = StandardScaler()
standard_scaled = pd.DataFrame(
    standard_scaler.fit_transform(X),
    columns=num_cols
)
standard_scaled.to_csv("data/standard_scaled.csv", index=False)

# ---------------------------------
# 4. Vector Normalization
# ---------------------------------
normalizer = Normalizer()
normalized = pd.DataFrame(
    normalizer.fit_transform(X),
    columns=num_cols
)
normalized.to_csv("data/normalized.csv", index=False)

# ---------------------------------
# Optional: Train-Test Split
# ---------------------------------
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

print("All scaling techniques applied successfully!")