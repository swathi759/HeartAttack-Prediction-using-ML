import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox


data = pd.read_csv("heart.csv")

data.rename(columns={'num': 'target'}, inplace=True)

data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)


X = data.drop("target", axis=1)
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

features = list(X.columns)


def predict():
    try:
        
        input_data = []
        for entry in entries:
            val = float(entry.get())
            input_data.append(val)

        
        result = model.predict([input_data])[0]

        if result == 1:
            messagebox.showwarning("Prediction", " Heart Disease Risk Detected!")
        else:
            messagebox.showinfo("Prediction", " No Heart Disease Risk Detected!")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Heart Attack Risk Predictor")
root.geometry("400x700")

tk.Label(root, text="Enter Patient Data", font=("Arial", 16)).pack(pady=10)


entries = []
for feature in features:
    tk.Label(root, text=feature).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

tk.Button(root, text="Predict", command=predict, bg="green", fg="white", font=("Arial", 14)).pack(pady=20)

root.mainloop()
