# Loan Approval Prediction

This project demonstrates how to build a loan approval prediction model using a Support Vector Machine (SVM) classifier.

## Project Description

The goal of this project is to predict whether a loan will be approved or not based on various applicant features. The model is trained on a dataset containing information about past loan applicants.

## Dataset

The dataset used in this project is assumed to be located at `/content/train_u6lujuX_CVtuZ9i (1).csv`. It contains features such as Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, and Loan_Status.

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn
- seaborn
- Google Colaboratory or Jupyter Notebook

## Setup and Usage

1. **Open the notebook:** Open the provided code in Google Colaboratory or a Jupyter Notebook.
2. **Load the data:** Ensure the dataset file (`train_u6lujuX_CVtuZ9i (1).csv`) is accessible at the specified path.
3. **Run the code:** Execute the cells sequentially.

The notebook performs the following steps:

- **Data Loading and Exploration:** Reads the dataset and performs initial exploration.
- **Data Preprocessing:** Handles missing values, performs label encoding for categorical features, and addresses outliers.
- **Data Visualization:** Visualizes the relationship between key features and the loan status.
- **Data Splitting:** Splits the dataset into training and testing sets.
- **Model Training:** Trains an SVM classifier on the training data.
- **Model Evaluation:** Evaluates the model's accuracy on both the training and testing data.
- **Prediction:** Demonstrates how to make predictions on new, unseen data.

## Code Explanation

- **Importing Libraries:** Necessary libraries like pandas, numpy, scikit-learn, and seaborn are imported.
- **Loading Data:** `pd.read_csv()` is used to load the dataset.
- **Data Exploration:** `.head()`, `.shape()`, `.describe()`, and `.isnull().sum()` are used to understand the data's structure, summary statistics, and identify missing values.
- **Data Preprocessing:**
    - Missing values are dropped using `.dropna()`.
    - Label encoding is applied to the `Loan_Status` column to convert 'N' and 'Y' to 0 and 1.
    - The '3+' value in the 'Dependents' column is replaced with 4.
    - Categorical features are converted to numerical using `.replace()`.
- **Data Visualization:** `seaborn.countplot()` is used to visualize the distribution of loan status based on Education and Married status.
- **Splitting Data:** `train_test_split` from `sklearn.model_selection` is used to split the data into training and testing sets. `stratify=Y` ensures that the proportion of loan statuses is the same in both sets.
- **Training SVM Model:** `svm.SVC(kernel='linear')` creates an SVM classifier with a linear kernel, and `.fit()` trains the model.
- **Model Evaluation:** `accuracy_score` from `sklearn.metrics` is used to calculate the accuracy of the model's predictions on the training and testing data.
- **Prediction:** A sample input is created, converted to a numpy array, reshaped, and then used with `classifier.predict()` to get the loan approval prediction.

## Results

The output of the notebook will show the accuracy of the model on both the training and testing datasets, as well as the prediction for a sample input.

## Future Improvements

- Experiment with different SVM kernels and hyperparameters to potentially improve accuracy.
- Explore other classification algorithms (e.g., Logistic Regression, Random Forest) and compare their performance.
- Implement more sophisticated methods for handling missing values.
- Perform feature engineering to create new features that might improve model performance.
- Implement cross-validation for a more robust evaluation of the model.
