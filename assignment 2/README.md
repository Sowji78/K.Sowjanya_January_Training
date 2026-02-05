Missing Value Handling
Median imputation performed best for numerical features as it was robust to outliers, whereas mode imputation was more suitable for categorical variables because it preserved the most frequent category without introducing bias.

Categorical Encoding Techniques
One-Hot Encoding worked best for nominal features with fewer unique values, as it avoided introducing artificial order. Label and Ordinal Encoding were effective for ordinal features where ranking was meaningful. Frequency Encoding handled high-cardinality features efficiently by preserving occurrence information. Target Encoding was the most effective for high-cardinality categorical variables when a target relationship was present, as it captured predictive information.

Feature Scaling
Min-Max Scaling was most effective for distance-based models because it preserved the original distribution within a fixed range. Z-score standardization performed best for algorithms assuming normally distributed data. Max Absolute Scaling was useful for sparse datasets, while Vector Normalization was ideal for magnitude-sensitive models such as cosine similarity-based algorithms.

Outlier Treatment and Skewness Transformation
Outlier detection revealed extreme values that negatively affected scaling. Log transformation significantly reduced skewness in highly skewed numerical features, resulting in improved feature distribution and model stability.

Final Preprocessing Choice
The final preprocessing pipeline used median imputation, appropriate categorical encoding based on feature type, Z-score standardization for numerical features, and log transformation for skewed variables. These choices ensured data consistency, reduced bias, and improved overall model readiness.      