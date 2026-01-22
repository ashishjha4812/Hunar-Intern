import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv(r"E:\Users\oma\Downloads\breast cancer.csv")

print(df.head())
print(df.info())



if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)




print("Missing values before:")
print(df.isna().sum())

df.fillna(df.median(numeric_only=True), inplace=True)

print("Missing values after:")
print(df.isna().sum())



df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})



X = df.drop(columns=['diagnosis'])
y = df['diagnosis']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)




y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))





sample = X_test[0].reshape(1, -1)
prediction = knn.predict(sample)

if prediction[0] == 1:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")




for k in range(3, 11, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"k={k}, Accuracy={accuracy_score(y_test, y_pred)}")


