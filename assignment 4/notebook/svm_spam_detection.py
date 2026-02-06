import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_excel("data/Spam Email Detection.xlsx")

print(df.head())
print(df.info())

df = df[['v1','v2']]   # keep only label and message
df.columns = ['label','message']

# convert label to numeric
df['label'] = df['label'].map({'ham':0,'spam':1})

df=df.dropna()
df['message']=df['message'].astype(str)

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

sample = ["Congratulations! You won a free lottery. Click now"]

sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("Spam" if prediction[0]==1 else "Not Spam")