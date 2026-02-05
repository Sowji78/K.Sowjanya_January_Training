import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

# Load dataset
df = pd.read_csv("data/datasets.csv")

# Identify categorical columns
cat_cols = df.select_dtypes(include="object").columns

# -------------------------------
# 1. One-Hot Encoding
# -------------------------------
one_hot_df = pd.get_dummies(df, columns=cat_cols)

# -------------------------------
# 2. Label Encoding
# -------------------------------
label_df = df.copy()
le = LabelEncoder()

for col in cat_cols:
    label_df[col] = le.fit_transform(label_df[col].astype(str))

# -------------------------------
# 3. Ordinal Encoding
# -------------------------------
ordinal_df = df.copy()
oe = OrdinalEncoder()
ordinal_df[cat_cols] = oe.fit_transform(ordinal_df[cat_cols].astype(str))

# -------------------------------         
# 4. Frequency Encoding
# -------------------------------
freq_df = df.copy()

for col in cat_cols:
    freq_map = freq_df[col].value_counts(normalize=True)
    freq_df[col] = freq_df[col].map(freq_map)

# -------------------------------
# 5. Target Encoding
# -------------------------------
# Replace 'target' with your actual target column name
target_column = df.columns[-1]

te = TargetEncoder()
target_df = df.copy()
target_df[cat_cols] = te.fit_transform(target_df[cat_cols], df[target_column])

# -------------------------------
# Save encoded datasets
# -------------------------------
one_hot_df.to_csv("data/one_hot_encoded.csv", index=False)
label_df.to_csv("data/label_encoded.csv", index=False)
ordinal_df.to_csv("data/ordinal_encoded.csv", index=False)
freq_df.to_csv("data/frequency_encoded.csv", index=False)
target_df.to_csv("data/target_encoded.csv", index=False)

print("All categorical encoding techniques applied successfully!")