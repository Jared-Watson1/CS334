import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the datasets
train_data = pd.read_csv("train.csv")
validation_data = pd.read_csv("validation.csv")
test_data = pd.read_csv("test.csv")

# Prepare the datasets
X_train = train_data.drop("loan_status", axis=1)
y_train = train_data["loan_status"]
X_validation = validation_data.drop("loan_status", axis=1)
y_validation = validation_data["loan_status"]
X_test = test_data.drop("loan_status", axis=1)
y_test = test_data["loan_status"]

# Create a Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=0)

# Train the model
decision_tree.fit(X_train, y_train)

# Predict on validation set
y_pred_validation = decision_tree.predict(X_validation)

# Evaluate the model
accuracy = accuracy_score(y_validation, y_pred_validation)
conf_matrix = confusion_matrix(y_validation, y_pred_validation)

# Output the evaluation results
print("Accuracy on validation set: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:")
print(conf_matrix)

# You can also add code here to evaluate the fairness metric (True Positive Rate across different groups)
