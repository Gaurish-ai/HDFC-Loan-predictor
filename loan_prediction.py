import pickle

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 1: Load train and test data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


# Step 2: Remove Loan_ID because it is only an ID, not useful for prediction
train_data = train_data.drop("Loan_ID", axis=1)
test_data = test_data.drop("Loan_ID", axis=1)


# Step 3: Fill missing values
for column in train_data.columns:
    if is_numeric_dtype(train_data[column]):
        train_data[column] = train_data[column].fillna(train_data[column].median())
    else:
        train_data[column] = train_data[column].fillna(train_data[column].mode()[0])

for column in test_data.columns:
    if is_numeric_dtype(test_data[column]):
        test_data[column] = test_data[column].fillna(test_data[column].median())
    else:
        test_data[column] = test_data[column].fillna(test_data[column].mode()[0])


# Step 4: Separate input columns and output column
X_train = train_data.drop("Loan_Status", axis=1)
y_train = train_data["Loan_Status"]

X_test = test_data.drop("Loan_Status", axis=1)
y_test = test_data["Loan_Status"]


# Step 5: Convert text values into numbers
y_train = y_train.map({"Rejected": 0, "Approved": 1})
y_test = y_test.map({"Rejected": 0, "Approved": 1})

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# Step 6: Make sure train and test have the same columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# Step 7: Create and train the logistic regression model
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)


# Step 8: Test the model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 9: Save the model and column names
model_file = open("model/loan_model.pkl", "wb")
pickle.dump({"model": model, "columns": X_train.columns.tolist()}, model_file)
model_file.close()

print("\nModel saved as model/loan_model.pkl")


# Step 10: Create a simple correlation heatmap image
numeric_data = train_data.select_dtypes(include=["int64", "float64"])
correlation = numeric_data.corr()

plt.figure(figsize=(9, 6))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()


# Step 11: Create a simple model workflow image
plt.figure(figsize=(10, 3))
steps = ["Load Data", "Clean Data", "Encode Data", "Train Model", "Predict"]

for i, step in enumerate(steps):
    plt.text(
        i,
        0.5,
        step,
        ha="center",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "edgecolor": "black"},
    )
    if i < len(steps) - 1:
        plt.arrow(i + 0.25, 0.5, 0.45, 0, head_width=0.05, head_length=0.05)

plt.axis("off")
plt.title("Loan Prediction Model Workflow")
plt.tight_layout()
plt.savefig("images/model_workflow.png")
plt.close()

print("Images saved in images folder")


# Step 12: Example prediction
sample = X_test.head(1)
sample_prediction = model.predict(sample)

if sample_prediction[0] == 1:
    print("\nSample prediction: Loan Approved")
else:
    print("\nSample prediction: Loan Rejected")
