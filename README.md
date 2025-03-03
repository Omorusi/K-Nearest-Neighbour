# K-Nearest Neighbors (KNN) Model - Development Stages

This project implements the **K-Nearest Neighbors (KNN)** algorithm for classification, following a structured approach to **data preprocessing, model training, evaluation, and optimization**.

---

## **1️⃣ Data Collection & Exploration**

### **Stage Overview:**

- The dataset used in this project is related to **diabetes prediction**.
- It was loaded using `pandas` and inspected for missing values, data types, and class distributions.
- **Visualization** techniques, including histograms and scatter plots, were used to analyze feature relationships.

### **Key Steps:**

```python
import pandas as pd

# Load dataset
data = pd.read_csv('diabetes.csv')

# Display first few rows
print(data.head())

# Rename columns for better readability (if needed)
data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Check dataset information
print(data.info())
print(data.describe())
```

---

## **2️⃣ Data Preprocessing & Missing Value Handling**

### **Stage Overview:**

- The dataset was **checked for missing values** and inconsistencies.
- **Feature scaling** was applied to standardize numerical values for better KNN performance.
- The dataset was **split into training and testing sets** to evaluate the model effectively.

### **Key Steps:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for missing values
print(data.isnull().sum())

# Split dataset into features (X) and target variable (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---

## **3️⃣ Model Training & Initial Evaluation**

### **Stage Overview:**

- The **KNN classifier** was implemented using `sklearn.neighbors.KNeighborsClassifier`.
- A **baseline model** was trained with `k=5`.
- Initial performance was evaluated using accuracy, confusion matrix, and classification report.

### **Key Steps:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## **4️⃣ Model Optimization - Choosing the Best k**

### **Stage Overview:**

- **Hyperparameter tuning** was performed to find the optimal `k` value.
- A loop tested multiple values of `k`, and **cross-validation** was used to determine the best value.

### **Key Steps:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Find the optimal k value
k_values = range(1, 21)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean() for k in k_values]

# Plot accuracy vs. k value
plt.plot(k_values, scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding the Optimal k')
plt.show()
```

✅ **The best ****`k`**** was selected based on the highest accuracy.**

---

## **5️⃣ Final Model Evaluation & Performance Metrics**

### **Stage Overview:**

- The model was retrained with the **best ****`k`**** value**.
- Performance was measured using **precision, recall, F1-score**, and **ROC-AUC curves**.

### **Key Steps:**

```python
from sklearn.metrics import roc_curve, auc

# Train with the best k value
best_k = 7  # Example selected from previous tuning
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Make final predictions
y_pred_best = knn_best.predict(X_test)

# Generate ROC curve
fpr, tpr, _ = roc_curve(y_test, knn_best.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## **6️⃣ Future Improvements & Deployment**

### **Next Steps:**

- Implement **k-fold cross-validation** for stability.
- Deploy the model using **Streamlit** for real-time predictions.
- Compare KNN performance with other models like **Decision Trees or SVM**.

---

## **📌 How to Run the Project**

### **Clone the Repository:**

```bash
git clone https://github.com/yourusername/KNN-Project.git
cd KNN-Project
```

### **Install Dependencies:**

```bash
pip install -r requirements.txt
```

### **Run the Jupyter Notebook or Python Script:**

```bash
jupyter notebook KNN.ipynb
```

---

## **🚀 Optional Deployment with Streamlit**

### **To run the model as a web app:**

1. **Install Streamlit:**

   ```bash
   pip install streamlit
   ```

2. **Create a Streamlit App (****`app.py`****):**

   ```python
   import streamlit as st
   import joblib

   # Load the trained model
   model = joblib.load("knn_model.pkl")

   st.title("Diabetes Prediction Using KNN")

   pregnancies = st.slider("Pregnancies", min_value=0, max_value=15)
   glucose = st.number_input("Glucose Level")
   bmi = st.number_input("BMI")

   if st.button("Predict"):
       prediction = model.predict([[pregnancies, glucose, bmi]])
       st.write(f"Predicted Outcome: {prediction[0]}")
   ```

3. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

---

## **📝 License**

This project is licensed under the MIT License.

Edward omorusi 
