Changelog.md
# Initial Implementation

Assigned  column names as the dataset originally lacked them, making it clearer to understand the data.
Converted Outcome Labels to Numerical Format:
 ![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20121322.png?raw=true)
Changed "non-diabetes" → 0 and "diabetes" → 1 for the outcome column.
Reason: Machine learning models require numerical values for computation, as they cannot process categorical text labels directly.
Data Visualization:

Created a bar chart to visualize the distribution of diabetes and non-diabetes cases in the dataset.
 ![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20122042.png?raw=true)
KNN Model Implementation:

Built a K-Nearest Neighbors (KNN) classifier to predict diabetes based on features in the dataset.
Split the data into training and testing sets.
Model Evaluation:

Checked the accuracy of the model to assess its performance.
Considered using different values of K to optimize the model’s accuracy.
 ![image alt](https://github.com/Omorusi/K-Nearest-Neighbour/blob/main/Screenshot%202025-03-03%20232234.png?raw=true)
Next Steps
Tune K value to find the best fit.
Implement confusion matrix, precision, recall, and F1-score for better evaluation.
Normalize/scale data if needed for better KNN performance.
