-Nearest Neighbors (KNN) Model - Diabetes Prediction
This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using the diabetes dataset. The dataset contains medical attributes used to predict whether a patient has diabetes.

📌 Project Stages
1️⃣ Data Collection & Exploration
Loaded the diabetes dataset using pandas.
Added meaningful column names to improve readability.
Checked for missing values and the total shape of the dataset.
Displayed a count plot to visualize the distribution of diabetic vs. non-diabetic patients.
Created scatter plots to analyze feature relationships.
Code Snippet
python
Copy
Edit
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Load dataset & Assign column names 
![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20121313.png)


# Check for any missing values 

![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20121427.png?raw=true
)
# Visualizing Outcome Distribution  
![image alt]()

2️⃣ Data Preprocessing & Cleaning
Changing Diabetes & Non-Diabetes to numerical value 
Handled missing values by replacing zeros with the median in key medical attributes (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
Separated features (X) and target variable (y) for model training.
Performed feature scaling using StandardScaler() since KNN is a distance-based algorithm.
Code Snippet
python
Copy
Edit
from sklearn.preprocessing import StandardScaler  
# Change Diabetes = 1 & Non-Diabetes = 0  to numerical value 
![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20121959.png?raw=true)
Representing the data of non-diabetes vs diabetes
![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20122042.png?raw=true)
# Replace zero values with median in selected columns  
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']  
for col in columns_to_replace:  
    data[col] = data[col].replace(0, data[col].median())  

# Split dataset into features and target  
X = data.drop(columns=['Outcome'])  
y = data['Outcome']  

# Scale features  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
3️⃣ Train-Test Split & Model Training
Split the dataset into 80% training and 20% testing using train_test_split().
Initialized KNN classifier with n_neighbors=5.
Trained the model using KNN fit function.
Code Snippet
python
Copy
Edit
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  

# Split dataset into training and testing sets (80-20)  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  

# Initialize KNN Classifier with K=5  
knn = KNeighborsClassifier(n_neighbors=5)  

# Train the model  
knn.fit(X_train, y_train)  
4️⃣ Model Evaluation & Performance Metrics
Accuracy Score: Evaluated the overall performance of the KNN model.
Confusion Matrix: Analyzed how well the model classified diabetic and non-diabetic patients.
Classification Report: Checked precision, recall, and F1-score for each class.
Code Snippet
python
Copy
Edit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  

# Make predictions  
y_pred = knn.predict(X_test)  

# Evaluate model performance  
accuracy = accuracy_score(y_test, y_pred)  
conf_matrix = confusion_matrix(y_test, y_pred)  
class_report = classification_report(y_test, y_pred)  

# Print results  
print(f"Accuracy: {accuracy * 100:.2f}%")  
print("Confusion Matrix:\n", conf_matrix)  
print("Classification Report:\n", class_report)  
5️⃣ Visualizing Model Performance
Confusion Matrix Heatmap: A graphical representation of how well the model classified instances.
Code Snippet
python
Copy
Edit
import seaborn as sns  
import matplotlib.pyplot as plt  

# Plot confusion matrix  
plt.figure(figsize=(5,4))  
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Diabetic", "Diabetic"],  
            yticklabels=["Non-Diabetic", "Diabetic"])  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.title("Confusion Matrix")  
plt.show()


👨‍💻 Author
🚀 Developed by [Your Name]

