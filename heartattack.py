# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("heart.csv")  

print("Columns:", data.columns)


if data['target'].nunique() > 2:
    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

data = data.dropna()

X = data.drop("target", axis=1)
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\n Model Evaluation:") 
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred), 2))
print("Recall:", round(recall_score(y_test, y_pred), 2))
print("F1 Score:", round(f1_score(y_test, y_pred), 2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


new_patient = [[55, 1, 0, 140, 250, 0, 1, 150, 0, 1.0, 2, 0, 2]]

prediction = model.predict(new_patient)


print("\n Heart Attack Risk Prediction for New Patient:")
if prediction[0] == 1:
    print(" Heart Disease Risk Detected!")
else:
    print(" No Heart Disease Risk Detected.")
