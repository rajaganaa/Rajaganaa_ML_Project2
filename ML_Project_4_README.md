# Loan Default Prediction Using Random Forest Classifier

This project focuses on predicting loan default using a Random Forest Classifier. It involves data preprocessing, model training, evaluation, and visualization of key metrics.

## Project Overview

The project includes the following steps:

1. **Data Loading**: The dataset (`loan_default_prediction_project_complete.csv`) containing loan-related information is loaded using pandas.

2. **Handling Missing Values**: Missing values are filled with median values for numeric features and mode values for categorical features.

3. **Feature Encoding**: Categorical variables ('Gender', 'Employment_Status', 'Location') are encoded to numerical format using `LabelEncoder`.

4. **Target Variable Encoding**: The target variable 'Loan_Status' is encoded using `LabelEncoder`.

5. **Data Splitting**: The dataset is split into training and test sets using `train_test_split` from scikit-learn.

6. **Normalization**: Data is standardized using `StandardScaler` to ensure all features contribute equally to the model.

7. **Model Building**: A Random Forest Classifier is trained on the training data to predict loan default.

8. **Predictions**: Loan default predictions are made on the test set, and probabilities are predicted using `predict_proba`.

9. **Model Evaluation**: Various metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score are computed to evaluate the model's performance.

10. **Visualization**: 
    - Confusion matrix is plotted to visualize the true positive, false positive, true negative, and false negative predictions.
    - ROC Curve is plotted to visualize the trade-off between true positive rate and false positive rate.
    - Feature Importance is visualized using a bar plot to understand which features contribute most to the model.

## Files

- `loan_default_prediction_project_complete.csv`: Dataset containing loan-related information and loan status (default or not).
- `loan_default_prediction.py`: Python script containing the code for data preprocessing, model training, evaluation, and visualization.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Clone the repository:

2. Install dependencies:

3. Run the script:

## Results

- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1 Score**: XX%
- **ROC-AUC**: XX%

## Author

- [RAJAGANAPATHY](https://github.com/your-rajaganaa)

Feel free to contribute, report issues, or provide feedback!

