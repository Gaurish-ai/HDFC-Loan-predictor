# HDFC Loan Prediction System

This is a simple beginner-friendly machine learning project.

The project predicts whether a loan will be:

- Approved
- Rejected

The algorithm used is Logistic Regression.

## Folder Structure

```text
HDFC-Loan-Prediction-System/
|
|-- data/
|   |-- train.csv
|   |-- test.csv
|
|-- notebook/
|   |-- loan_prediction.ipynb
|
|-- model/
|   |-- loan_model.pkl
|
|-- images/
|   |-- correlation_heatmap.png
|   |-- model_workflow.png
|
|-- loan_prediction.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## How To Run

First install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the project:

```bash
python loan_prediction.py
```

## What The Code Does

1. Loads the train and test CSV files.
2. Removes the `Loan_ID` column.
3. Fills missing values.
4. Converts text columns into numbers.
5. Trains a Logistic Regression model.
6. Checks model accuracy.
7. Saves the model as `model/loan_model.pkl`.
8. Saves two simple images in the `images` folder.

## Dataset Columns

- Gender
- Married
- Dependents
- Education
- Employment_Status
- Applicant_Income
- Coapplicant_Income
- Loan_Amount
- Loan_Term
- Credit_History
- Property_Area
- Age
- Loan_Status

## Note

This project is for learning purposes only.
