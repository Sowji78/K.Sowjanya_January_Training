## Heart Disease Prediction Using Machine Learning

##  Project Overview

This project focuses on predicting the presence of heart disease using multiple machine learning algorithms. The dataset is preprocessed, cleaned, and evaluated using different classification models to identify the best-performing model based on standard evaluation metrics.

The goal is to compare models such as Logistic Regression, Decision Tree, Random Forest, KNN, and SVM, and select the most suitable one for heart disease prediction.

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve treatment outcomes. Traditional diagnosis can be time-consuming and dependent on expert availability. This project aims to automate heart disease prediction using machine learning techniques for faster and more accurate decision-making.

---

## 🗂 Dataset Description

* *Dataset Name:* heart_disease_cleaned.csv
* *Target Column:* num

  * 0 → No heart disease
  * 1–4 → Presence of heart disease (converted to binary 1)
* *ID Column:* id (removed during training)

---

##  Technologies Used

* *Programming Language:* Python
* *Libraries:*

  * pandas, numpy
  * scikit-learn

---

##  Data Preprocessing Steps

1. Converted target variable into binary classification
2. Handled missing values using mean imputation
3. Removed duplicate records
4. Outlier detection and removal using IQR method
5. Encoded categorical variables using Label Encoding
6. Feature scaling using StandardScaler
7. Train-test split (80% training, 20% testing)

---

##  Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

---

## Model Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

## Results Summary

Based on the evaluation results, the *Random Forest Classifier* performed best among all models.

*Best Model Performance:*

* Accuracy: ~82%
* Precision: ~77%
* Recall: ~81%
* F1 Score: ~79%

Model Comparison:
              Model  Accuracy  Precision   Recall  F1 Score
Logistic Regression  0.814286   0.775862 0.775862  0.775862
      Decision Tree  0.764286   0.719298 0.706897  0.713043
      Random Forest  0.821429   0.770492 0.810345  0.789916
                KNN  0.800000   0.741935 0.793103  0.766667
                SVM  0.785714   0.725806 0.775862  0.750000

Best Performing Model:
Model        Random Forest
Accuracy          0.821429
Precision         0.770492
  for col in df.select_dtypes(include='object'):


Model Comparison:
              Model  Accuracy  Precision   Recall  F1 Score
Logistic Regression  0.814286   0.775862 0.775862  0.775862
      Decision Tree  0.764286   0.719298 0.706897  0.713043
      Random Forest  0.821429   0.770492 0.810345  0.789916
                KNN  0.800000   0.741935 0.793103  0.766667
                SVM  0.785714   0.725806 0.775862  0.750000

Best Performing Model:
Model        Random Forest
Accuracy          0.821429
Precision         0.770492
Recall            0.810345
F1 Score          0.789916
Name: 2, dtype: object


## 🏆 Conclusion

The Random Forest model achieved the highest accuracy and balanced performance across all metrics, making it the most suitable model for this heart disease prediction task. Ensemble methods performed better than individual models due to their ability to reduce overfitting and capture complex patterns.