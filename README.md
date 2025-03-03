Diabetes Prediction with KNN Classifier
This repository demonstrates the process of predicting diabetes using a K-Nearest Neighbors (KNN) classifier. The project follows a standard workflow including data collection, preprocessing, exploratory data analysis (EDA), model training, evaluation, and model saving.

Project Overview
The main objective is to predict whether a person has diabetes based on various health-related features using a K-Nearest Neighbors (KNN) classification model. The dataset used is the Pima Indians Diabetes Database, which contains data about several medical attributes like glucose levels, blood pressure, BMI, and age.

Stages and Code
1. Data Collection
Purpose: Collect the dataset to be used for analysis and model building.

python
Copy
Edit
import pandas as pd

# Load the diabetes dataset
data = pd.read_csv('/content/diabetes.csv', header=None)
data.head()  # Show the first few rows of the dataset
2. Data Preprocessing
Purpose: Clean and preprocess the data by handling missing values, scaling features, and encoding categorical variables if necessary.

python
Copy
Edit
# Add column names to the dataset
-- python
data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Outcome']
data.head()  # View the first few rows after adding column names
--
# Handle missing values: replace zero with the median in specific columns
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_replace:
    data[col] = data[col].replace(0, data[col].median())
    
# Checking dataset shape (rows and columns)
rows, columns = data.shape
print(f"The dataset has {rows} rows and {columns} columns.")
3. Exploratory Data Analysis (EDA)
Purpose: Understand the data's patterns, distributions, and relationships by visualizing the data.

python
Copy
Edit
# Mapping Outcome (0 -> Non-Diabetes, 1 -> Diabetes)
data_target = data['Outcome'].map({0: "Non-Diabetes", 1: "Diabetes"})
data_target.head()  # Display mapped target labels

# Visualization: Countplot of Outcome (Diabetes vs Non-Diabetes)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Outcome"], palette=["blue", "red"])
plt.xticks(ticks=[0, 1], labels=["Non-Diabetic", "Diabetic"])
plt.xlabel("Diabetes Status")
plt.ylabel("Count")
plt.title("Distribution of Diabetes in the Dataset")
plt.show()

# Split dataset into diabetic and non-diabetic groups for plotting
data0 = data[data["Outcome"] == 0]  # Non-Diabetes
data1 = data[data["Outcome"] == 1]  # Diabetes

# Scatter plot: Glucose vs. DiabetesPedigreeFunction
plt.figure(figsize=(8, 6))
ax = data0.plot.scatter("Glucose", "DiabetesPedigreeFunction", marker='+', color="green", label="Non-Diabetic")
ax = data1.plot.scatter("Glucose", "DiabetesPedigreeFunction", marker='.', color="red", ax=ax, label="Diabetic")
plt.xlabel("Glucose")
plt.ylabel("DiabetesPedigreeFunction")
plt.title("Diabetes Classification: Glucose vs. DiabetesPedigreeFunction")
plt.legend()
plt.show()
4. Model Selection and Training
Purpose: Choose a machine learning model (e.g., K-Nearest Neighbors) and train it using the training data.

python
Copy
Edit
# Feature and target selection
X = data.drop(["Outcome"], axis=1)  # Features (all columns except 'Outcome')
Y = data["Outcome"]  # Target column ('Outcome')

# Split dataset into training and test sets (80-20 split)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# K-Nearest Neighbors (KNN) Classifier with n_neighbors=3
from sklearn.neighbors import KNeighborsClassifier
mymodel = KNeighborsClassifier(n_neighbors=3)
mymodel.fit(X_train, Y_train)
5. Model Evaluation
Purpose: Evaluate the model's performance using appropriate metrics (accuracy, confusion matrix, classification report).

python
Copy
Edit
# Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on the test set
Y_pred = mymodel.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report (precision, recall, f1-score)
class_report = classification_report(Y_test, Y_pred)
print("Classification Report:")
print(class_report)
6. Model Deployment and Monitoring
Purpose: Deploy the model and monitor its performance in a real-world setting (not covered in detail here but this would typically involve saving the model, creating an API, or embedding it into a larger system).

python
Copy
Edit
import joblib

# Save the trained model
joblib.dump(mymodel, 'knn_diabetes_model.pkl')

# To load the model later
loaded_model = joblib.load('knn_diabetes_model.pkl')
Requirements
To run this code, you'll need the following Python libraries:

pandas
sklearn
matplotlib
seaborn
joblib
You can install the necessary dependencies using pip:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn joblib
