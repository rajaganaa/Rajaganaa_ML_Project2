# Anomaly Detection Using Isolation Forest

This project focuses on detecting anomalies in sensor data using the Isolation Forest algorithm. It involves preprocessing data, training an anomaly detection model, evaluating its performance, and visualizing key metrics.

## Project Overview

The project includes the following steps:

1. **Data Loading**: The dataset (`synthetic_sensor_data.csv`) containing sensor readings is loaded using pandas.

2. **Handling Missing Values**: Missing values are filled with median values for numeric features and mode values for categorical features.

3. **Feature Engineering**: Additional features are created if needed. In this example, we'll use existing features.

4. **Data Splitting**: The dataset is split into training and test sets using `train_test_split` from scikit-learn.

5. **Normalization**: Data is standardized using `StandardScaler` to ensure all features contribute equally to the model.

6. **Model Training**: An Isolation Forest model is trained on the training data to detect anomalies.

7. **Anomaly Prediction**: Anomalies are predicted on both training and test sets. Predictions are converted from -1 (anomaly) and 1 (normal) to binary format (0 for normal, 1 for anomaly).

8. **Model Evaluation**: Various metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score are computed to evaluate the model's performance.

9. **Visualization**: 
   - Confusion matrix is plotted to visualize the true positive, false positive, true negative, and false negative predictions.
   - ROC Curve is plotted to visualize the trade-off between true positive rate and false positive rate.

## Files

- `synthetic_sensor_data.csv`: Dataset containing sensor readings and anomaly labels.
- `anomaly_detection.py`: Python script containing the code for data preprocessing, model training, evaluation, and visualization.

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

