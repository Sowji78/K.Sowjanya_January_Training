import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression # Changed from Linear
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 2. Load Dataset
df = pd.read_csv("data/heart_disease_cleaned.csv")

# 3. Data Cleaning & Preprocessing
# Note: In your CSV, the target column is 'num'
target_col = 'num'

# Convert 'num' to binary (0 = no disease, 1 = disease) if it's multi-class
# In this dataset, 0 is healthy, 1-4 are heart disease stages.
df[target_col] = (df[target_col] > 0).astype(int)

# 3.1 Handle Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# 3.3 Remove Duplicate Records
df.drop_duplicates(inplace=True)

# 3.4 Detect & Treat Outliers using IQR
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# 3.5 Encode Categorical Variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = label_encoder.fit_transform(df[col])

# 3.7 Feature Scaling
X = df.drop(['id', target_col], axis=1) # Drop 'id' as it's just a sequence
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.8 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

results_list = []

# 5 & 6. Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results_list.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1 Score": f1_score(y_test, preds, zero_division=0)
    })

# 7. Model Comparison Table
results_df = pd.DataFrame(results_list)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# 8. Conclusion
best_model = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
print("\nBest Performing Model:")
print(best_model)