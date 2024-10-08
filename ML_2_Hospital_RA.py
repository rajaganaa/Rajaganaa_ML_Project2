<a href="https://colab.research.google.com/github/rajaganaa/Rajaganaa_ML_Project2/blob/main/ML_2_hospital_readmission_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#Predicting Hospital Readmissions

#The primary goal of this project is to build a predictive model that can identify patients
# who are at high risk of hospital readmission within 30 days after their initial discharge.



import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

raw_df = pd.read_csv(r"/content/drive/Othercomputers/My Laptop (1)/Desktop/raj007_projects/RAJA_ML_PROJEC_2/synthetic_hospital_readmissions_data.csv")

raw_df.head()

raw_df.shape

raw_df.info()



raw_df.isnull().sum()

raw_df.nunique()

new = raw_df.drop(columns="Patient_ID", axis=1)
for column in new.columns:
    unique_values = raw_df[column].unique()
    print(f"'{column}':\n {unique_values}\n")

###Handling Missing Values

raw_df['A1C_Result'].fillna('Unknown', inplace=True)

raw_df['A1C_Result']

raw_df.isnull().sum()

## Handling Categorical Features

# 'Gender': ['Other' 'Female' 'Male']

raw_df["Gender"]= LabelEncoder().fit_transform(raw_df["Gender"])

raw_df["Gender"].unique()

# 'Admission_Type': ['Emergency' 'Urgent' 'Elective']

raw_df["Admission_Type"]= LabelEncoder().fit_transform(raw_df["Admission_Type"])

raw_df["Admission_Type"].unique()


# 'Diagnosis': ['Heart Disease' 'Diabetes' 'Injury' 'Infection']

raw_df["Diagnosis"]= LabelEncoder().fit_transform(raw_df["Diagnosis"])

raw_df["Diagnosis"].unique()

# 'A1C_Result': ['Unknown' 'Normal' 'Abnormal']

raw_df["A1C_Result"]= LabelEncoder().fit_transform(raw_df["A1C_Result"])

raw_df["A1C_Result"].unique()

# 'Readmitted': ['Yes' 'No']

raw_df["Readmitted"]= LabelEncoder().fit_transform(raw_df["Readmitted"])

raw_df["Readmitted"].unique()

raw_df.head()

## Handling Data Types

raw_df.info()

## Handling Duplicate Values

raw_df.duplicated().sum()

## Save the Dataframe

# Save the Dataframe
raw_df.to_csv("hospital_readmissions_only_int.csv", index= False)

## Read the Dataframe

df_1 = pd.read_csv("hospital_readmissions_only_int.csv")

df_1.head()

# Finding the Unknown(2) values of A1C_Result

# Drop rows where 'A1C_Result' has the value 'Unknown(2)'
A1C_Not_Null = df_1[df_1['A1C_Result'] != 2]

A1C_Not_Null.head()

A1C_Not_Null['A1C_Result'].unique()

A1C_Not_Null.shape

## Handling Outliers

# Find outlier using Boxplot

# Function for box plot
def plot_box_plots(df, cols):

    plt.figure(figsize=(10, 12))

    for i, col in enumerate(cols):
        plt.subplot(4, 4, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

columns = A1C_Not_Null.columns
plot_box_plots(A1C_Not_Null, columns)

# Find outlier using IQR

# Calculate quartiles and IQR
Q1 = A1C_Not_Null.quantile(0.25)
Q3 = A1C_Not_Null.quantile(0.75)
IQR = Q3 - Q1

# Calculate upper and lower bounds for outliers
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Identify outliers
outliers = A1C_Not_Null[(A1C_Not_Null < lower_bound) | (A1C_Not_Null > upper_bound)]

# Count outliers
num_outliers = outliers.count()

print("Number of outliers:")
print(num_outliers)

# find outlier using Z-score
df_age = A1C_Not_Null["Age"]

import numpy as np
outliers = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers# Driver code

sample_outliers = detect_outliers_zscore(df_age)
print("Outliers from Z-scores method: ", sample_outliers)

# converts all the values above the upper threshold to the upper threshold value
# converts all the values below the lower threshold to the lower threshold value

def outlier(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1

    upper_threshold = Q3 + (1.5*IQR)
    lower_threshold = Q1 - (1.5*IQR)

    df["Age_New"] = df[col].clip(lower_threshold, upper_threshold)

outlier(A1C_Not_Null,"Age")

# Box plot after handling outlier

plt.figure(figsize=(3, 5))
sns.boxplot(data=A1C_Not_Null, y=A1C_Not_Null["Age_New"])

A1C_Not_Null.describe().T

# droping the 'age' column

A1C_Not_Null_1 = A1C_Not_Null.drop(columns=["Age"], axis=1)

## Handling Skwness

# Function for histogram
def plot_histograms(df, cols):

    plt.figure(figsize=(8, 15))

    for i, col in enumerate(cols):
        plt.subplot(7,2, i+1)
        sns.histplot(df[col],kde= True, bins=30, color="salmon")
        plt.title(col)
    plt.tight_layout()
    plt.show()

columns = A1C_Not_Null_1.columns
plot_histograms(A1C_Not_Null_1, columns)

A1C_Not_Null_1.skew()

# Skewness is a measure of lack of symmetry
# Skewness value range from -1 to 1:
# If the skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# If the skewness is less than -0.5, the distribution is negatively skewed (left-skewed).
# If the skewness is greater than 0.5, the distribution is positively skewed (right-skewed).

# Feature Selection

# head map
plt.figure(figsize=(12,6))
sns.heatmap(A1C_Not_Null_1.corr(), annot=True, cmap="Reds")
plt.show()

# Import necessary libraries
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif




calc_vif(A1C_Not_Null_1)

# Variance Inflation Factor
# -  VIF values below 5 indicate that multicollinearity is not a significant concern, and the predictor variables are likely not highly correlated with each other.
# - VIF values between 5 and 10 suggest moderate multicollinearity.
# - VIF values above 10 indicate potentially severe multicollinearity.

# Droping the multicollinearity & unwanted columns

A1C_Not_Null_2 = A1C_Not_Null_1.drop(columns=["Age_New","Patient_ID"], axis=1)

calc_vif(A1C_Not_Null_2)

A1C_Not_Null_2.isnull().sum()

A1C_Not_Null_2.head()

A1C_Not_Null_2.shape

# saving with actual A1C values

A1C_Not_Null_2.to_csv("hospital_with_actual_A1C.csv", index= False)

#   Model to Predict A1C Values

!pip install xgboost


# import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, classification_report

from imblearn.combine import SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pickle

A1C_Not_Null_2.columns

A1C_Not_Null_2["A1C_Result"].value_counts()

# Data Splitting

x = A1C_Not_Null_2.drop(columns=["A1C_Result"],axis=1) #independent variables.
y = A1C_Not_Null_2["A1C_Result"] #dependent variable

x

y

# ### Handling Imbalanced feature &rarr; "SMOTE-Tomek"

# This method combines
# * SMOTE ability to generate synthetic data for minority class
# * Tomek has ability to remove the data that are identified as Tomek links from the majority class
# * Tomek links are pairs of instances from different classes that are very close to each other, but they are of different classes.

# balancing using smotetomek
x_new, y_new = SMOTETomek().fit_resample(x,y)

print(len(x_new))
print(len(y_new))

x_new

y_new



# Logistic Regression

# splitting train & test
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "Logistic Regression",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)

# SVM Classification

# splitting train & test
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
model = svm.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "SVM",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)

# Other classification algorithms

def accuracy_checking(x_data, y_data, algorithm):

    # splitting train & test
    x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size= 0.2, random_state=50)

    model = algorithm().fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    #checking the accuracy_score
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    metrics = {"Algorithm": algorithm.__name__,
               "Accuracy_Train": accuracy_train,
               "Accuracy_Test": accuracy_test}
    return metrics

print(accuracy_checking(x_new,y_new,DecisionTreeClassifier))
print(accuracy_checking(x_new,y_new,RandomForestClassifier))
print(accuracy_checking(x_new,y_new,ExtraTreesClassifier))
print(accuracy_checking(x_new,y_new,AdaBoostClassifier))
print(accuracy_checking(x_new,y_new,GradientBoostingClassifier))
print(accuracy_checking(x_new,y_new,XGBClassifier))

## Cross Validation

# StratifiedKFold Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Instantiate the classification model
A1C_Model = GradientBoostingClassifier()

# Instantiate Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Stratified K-Fold Cross-Validation and calculate accuracy for each fold
accuracy_scores = cross_val_score(model, x_new, y_new, scoring='accuracy', cv=skf)
mean_accuracy = np.mean(accuracy_scores)

# Print
print("Accuracy scores for each fold:", accuracy_scores)
print("Mean Accuracy:", mean_accuracy)


# Selected Model
x_train, x_test, y_train, y_test= train_test_split(x_new, y_new, test_size= 0.2, random_state= 42)

A1C_Model = GradientBoostingClassifier().fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

### Performance Metrics

# accuracy_score for train and test

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy score for Train and Test")
print("----------------------------------")
print("Accuracy_Train: ",accuracy_train)
print("Accuracy_Test: ",accuracy_test)

# confution matrics

print("Confution_matrix for Test")
print("--------------------------")
print(confusion_matrix(y_true = y_test, y_pred = y_pred_test))

# classification report typically includes metrics such as precision, recall, F1-score, and support

print("Classification_report for Test")
print("-------------------------------")
print(classification_report(y_true= y_test, y_pred= y_pred_test))

# Receiver Operating Characteristic (ROC) Curve

FP, TP, Threshold = roc_curve(y_true=y_test, y_score=y_pred_test)

print(FP)
print(TP)
print(Threshold)


# Area Under the Curve (AUC)

auc_curve = auc(x=FP, y=TP)
print("auc_curve: ", auc_curve)

# create a plot for ROC and AUC curve

roc_point= {"ROC Curve (area)":round(auc_curve, 2)}
plt.plot(FP,TP,label= roc_point)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.plot([0,1],[0,1],"k--")
plt.legend(loc= "lower right")
plt.show()

# Saving the Model unsing pickle
with open("A1C_Model.pkl","wb") as m:
    pickle.dump(A1C_Model, m)

# --------------------------------------------------------------------------------------------------------------------------------------------------#

## Read the Dataframe

df_1 = pd.read_csv("hospital_readmissions_only_int.csv")

df_1.head()

# Replacing the Unknown(2) values of A1C_Result

# Drop rows where 'A1C_Result' has the value 'Unknown(2)'
A1C_Null = df_1[df_1['A1C_Result'] == 2]

A1C_Null.head(25)

A1C_Null['A1C_Result'].unique()

A1C_Null.shape

## Handling Outliers

# Function for box plot
def plot_box_plots(df, cols):

    plt.figure(figsize=(10, 12))

    for i, col in enumerate(cols):
        plt.subplot(4, 4, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

columns = A1C_Null.columns
plot_box_plots(A1C_Null, columns)

# Find outlier using IQR

# Calculate quartiles and IQR
Q1 = A1C_Null.quantile(0.25)
Q3 = A1C_Null.quantile(0.75)
IQR = Q3 - Q1

# Calculate upper and lower bounds for outliers
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Identify outliers
outliers = A1C_Null[(A1C_Null < lower_bound) | (A1C_Null > upper_bound)]

# Count outliers
num_outliers = outliers.count()

print("Number of outliers:")
print(num_outliers)

# find outlier using Z-score

df_age = A1C_Null["Age"]

import numpy as np
outliers = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers# Driver code

sample_outliers = detect_outliers_zscore(df_age)
print("Outliers from Z-scores method: ", sample_outliers)

# converts all the values above the upper threshold to the upper threshold value
# converts all the values below the lower threshold to the lower threshold value

def outlier(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1

    upper_threshold = Q3 + (1.5*IQR)
    lower_threshold = Q1 - (1.5*IQR)

    df["Age_New"] = df[col].clip(lower_threshold, upper_threshold)

outlier(A1C_Null,"Age")

# Box plot after handling outlier

plt.figure(figsize=(3, 5))
sns.boxplot(data=A1C_Null, y=A1C_Null["Age_New"])

A1C_Null.describe().T

# droping the 'age' column

A1C_Null_1 = A1C_Null.drop(columns=["Age"], axis=1)

## Handling Skwness

# Function for histogram
def plot_histograms(df, cols):

    plt.figure(figsize=(8, 15))

    for i, col in enumerate(cols):
        plt.subplot(7,2, i+1)
        sns.histplot(df[col],kde= True, bins=30, color="salmon")
        plt.title(col)
    plt.tight_layout()
    plt.show()

columns = A1C_Null_1.columns
plot_histograms(A1C_Null_1, columns)

A1C_Null_1.skew()

# Skewness is a measure of lack of symmetry
# Skewness value range from -1 to 1:
# If the skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# If the skewness is less than -0.5, the distribution is negatively skewed (left-skewed).
# If the skewness is greater than 0.5, the distribution is positively skewed (right-skewed).

# Feature Selection

# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(A1C_Null_1)

# Variance Inflation Factor
# -  VIF values below 5 indicate that multicollinearity is not a significant concern, and the predictor variables are likely not highly correlated with each other.
# - VIF values between 5 and 10 suggest moderate multicollinearity.
# - VIF values above 10 indicate potentially severe multicollinearity.

# Droping the multicollinearity & unwanted columns

A1C_Null_2 = A1C_Null_1.drop(columns=["Age_New","Patient_ID","A1C_Result"], axis=1)

calc_vif(A1C_Null_2)

A1C_Null_2.isnull().sum()

A1C_Null_2.head()

#   Predict Hospital Re-admissions Model

import pickle

# Loading the MODEL
with open("A1C_Model.pkl","rb") as m1:
    A1C_Model = pickle.load(m1)

new_A1C = A1C_Model.predict(A1C_Null_2)

A1C_Null_2['A1C_Result'] = new_A1C

# Using pop replace 'Readmitted' placing to the last position
A1C_Result_column  = A1C_Null_2.pop('Readmitted')

A1C_Null_2['Readmitted'] = A1C_Result_column

A1C_Null_2.tail()

A1C_Null_2.shape

A1C_Null_2["A1C_Result"].unique()

# saving with actual A1C values

A1C_Null_2.to_csv("hospital_with_predicted_A1C.csv", index= False)

## Read the Dataframe

df1 = pd.read_csv("hospital_with_actual_A1C.csv")
df2 = pd.read_csv("hospital_with_predicted_A1C.csv")

# Concatenate the two DataFrames along the rows (axis=0)
final_df = pd.concat([df1, df2], axis=0)

# Reset the index of the concatenated DataFrame
final_df.reset_index(drop=True, inplace=True)

# Display the concatenated DataFrame
final_df.head()

final_df['A1C_Result'].value_counts()

#shape
final_df.shape

# info
final_df.info()

#null
final_df.isnull().sum()

#duplicates
final_df.duplicated().sum()

#unique
final_df.nunique()

for column in final_df.columns:
    unique_values = final_df[column].unique()
    print(f"'{column}':\n {unique_values}\n")

# Save the Dataframe
final_df.to_csv("hospital_readmissions_final.csv", index= False)

## Handling Outliers

# Calculate quartiles and IQR
Q1 = final_df.quantile(0.25)
Q3 = final_df.quantile(0.75)
IQR = Q3 - Q1

# Calculate upper and lower bounds for outliers
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Identify outliers
outliers = final_df[(final_df < lower_bound) | (final_df > upper_bound)]

# Count outliers
num_outliers = outliers.count()

print("Number of outliers:")
print(num_outliers)

## Handling Skwness

# Function for histogram
def plot_histograms(df, cols):

    plt.figure(figsize=(8, 15))

    for i, col in enumerate(cols):
        plt.subplot(7,2, i+1)
        sns.histplot(df[col],kde= True, bins=30, color="salmon")
        plt.title(col)
    plt.tight_layout()
    plt.show()

columns = final_df.columns
plot_histograms(final_df, columns)

final_df.skew()

#     Skewness is a measure of lack of symmetry
#     Skewness value range from -1 to 1:

# - If the skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# - If the skewness is less than -0.5, the distribution is negatively skewed (left-skewed).
# - If the skewness is greater than 0.5, the distribution is positively skewed (right-skewed)

# Checking for multicollinearity

# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(final_df)

# Variance Inflation Factor

# - VIF values below 5 indicate that multicollinearity is not a significant concern, and the predictor variables are likely not highly correlated with each other.
# - VIF values between 5 and 10 suggest moderate multicollinearity.
# - VIF values above 10 indicate potentially severe multicollinearity.



#  Model to Readmission

# import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, classification_report

from imblearn.combine import SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pickle

final_df.columns

final_df["Readmitted"].value_counts()

# Data Splitting

x_new = final_df.drop(columns=["Readmitted"],axis=1) #independent variables.
y_new = final_df["Readmitted"] #dependent variable

# Logistic Regression

# splitting train & test
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "Logistic Regression",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)

# SVM Classification

# splitting train & test
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
model = svm.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "SVM",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)

# Other classification algorithms

def accuracy_checking(x_data, y_data, algorithm):

    # splitting train & test
    x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size= 0.2, random_state=40)

    model = algorithm().fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    #checking the accuracy_score
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    metrics = {"Algorithm": algorithm.__name__,
               "Accuracy_Train": accuracy_train,
               "Accuracy_Test": accuracy_test}
    return metrics

print(accuracy_checking(x_new,y_new,DecisionTreeClassifier))
print(accuracy_checking(x_new,y_new,RandomForestClassifier))
print(accuracy_checking(x_new,y_new,ExtraTreesClassifier))
print(accuracy_checking(x_new,y_new,AdaBoostClassifier))
print(accuracy_checking(x_new,y_new,GradientBoostingClassifier))
print(accuracy_checking(x_new,y_new,XGBClassifier))

## Cross Validation

# StratifiedKFold Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Instantiate the classification model
A1C_Model = GradientBoostingClassifier()

# Instantiate Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Stratified K-Fold Cross-Validation and calculate accuracy for each fold
accuracy_scores = cross_val_score(model, x_new, y_new, scoring='accuracy', cv=skf)
mean_accuracy = np.mean(accuracy_scores)

# Print
print("Accuracy scores for each fold:", accuracy_scores)
print("Mean Accuracy:", mean_accuracy)


# Selected Model
x_train, x_test, y_train, y_test= train_test_split(x_new, y_new, test_size= 0.2, random_state= 50)

Readmission_Model = GradientBoostingClassifier().fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

### Performance Metrics

# accuracy_score for train and test

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy score for Train and Test")
print("----------------------------------")
print("Accuracy_Train: ",accuracy_train)
print("Accuracy_Test: ",accuracy_test)

# confution matrics

print("Confution_matrix for Test")
print("--------------------------")
print(confusion_matrix(y_true = y_test, y_pred = y_pred_test))

# classification report typically includes metrics such as precision, recall, F1-score, and support

print("Classification_report for Test")
print("-------------------------------")
print(classification_report(y_true= y_test, y_pred= y_pred_test))

# Receiver Operating Characteristic (ROC) Curve

FP, TP, Threshold = roc_curve(y_true=y_test, y_score=y_pred_test)

print(FP)
print(TP)
print(Threshold)


# Area Under the Curve (AUC)

auc_curve = auc(x=FP, y=TP)
print("auc_curve: ", auc_curve)

# create a plot for ROC and AUC curve

roc_point= {"ROC Curve (area)":round(auc_curve, 2)}
plt.plot(FP,TP,label= roc_point)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.plot([0,1],[0,1],"k--")
plt.legend(loc= "lower right")
plt.show()

# Saving the Model unsing pickle
with open("Readmission_Model.pkl","wb") as m:
    pickle.dump(Readmission_Model, m)

final_df.head()

# END

# testing
user_data = np.array([[0,2,1,75,29,4,0,3,5,1]])
prediction = Readmission_Model.predict(user_data)
prediction[0]

x_new.columns

# Min & Max of each Column
min_values = x_new.min()
max_values = x_new.max()

# Concatenate min_values and max_values along the columns axis
min_max_df = pd.concat([min_values, max_values], axis=1)
min_max_df.columns = ['Minimum', 'Maximum']

print("Minimum and Maximum values of all columns:")
print(" ")
print(min_max_df)



# ---------------------------------------------------------------------------------------------------------------------------------------------------#

!pip install streamlit-option-menu

!pip install streamlit
# streamlit part

# import
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

import pickle
from PIL import Image

import warnings
warnings.filterwarnings("ignore")



def predict_readmission(Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
       Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
       Num_Emergency_Visits, Num_Diagnoses, A1C_Result):

    with open("Readmission_Model.pkl","rb") as m:
        model = pickle.load(m)

    data = np.array([[Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
       Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
       Num_Emergency_Visits, Num_Diagnoses, A1C_Result]])
    prediction = model.predict(data)
    out = prediction[0]
    return out



st.markdown("<h1 style='text-align: center; color: #fa6607;'>Predicting Hospital Readmissions</h1>", unsafe_allow_html=True)
st.write("")

select = option_menu(None,["Home", "Readmission"],
                    icons =["hospital-fill","ticket-detailed"], orientation="horizontal",
                    styles={"container": {"padding": "0!important", "background-color": "#fafafa"},
                            "icon": {"color": "#fdfcfb", "font-size": "20px"},
                            "nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#fa6607"}})

if select == "Home":
    st.title("Welcome to the Hospital Readmissions Prediction Project!")

    st.write('''
**Objective:**
The primary goal of this project is to develop a predictive model that accurately determines whether a patient will require readmission within 30 days after their initial discharge. By leveraging advanced machine learning techniques, this project aims to enhance patient care and optimize healthcare resources.

**Key Features:**
- **Predictive Modeling:** This machine learning model analyzes patient data to predict the likelihood of hospital readmission within 30 days. This predictive capability enables proactive intervention and personalized care planning for at-risk patients.

- **Data-Driven Insights:** By processing comprehensive healthcare records, this model extracts valuable insights and identifies key factors contributing to readmission risk. These insights empower healthcare providers to make informed decisions and implement targeted interventions.

**How It Works:**
1. **Data Input:** This model accepts input data comprising patient demographics, medical history, previous hospitalizations, diagnoses, and medications.

2. **Predictive Analysis:** Leveraging state-of-the-art machine learning algorithms, the model processes the input data to generate predictions on whether readmission is required or not.

3. **Actionable Recommendations:** Based on the model's predictions, healthcare providers can proactively engage with high-risk patients, implement preventive measures, and optimize post-discharge care plans to minimize readmission rates.

**Technological Stack:**
This predictive model is developed using Python, a versatile programming language for data science and machine learning. It leverages industry-standard libraries such as scikit-learn for modeling and pandas for data manipulation to ensure robust performance and reliability.

**Stay Informed:**
Follow along as this project explores the intricacies of hospital readmissions prediction and strives to make a positive impact on patient outcomes and healthcare delivery.

''')

elif select == "Readmission":

    st.write("")
    st.header("Fill all the details below to know the prediction")
    st.write("")

    col1,col2,col3 = st.columns([5,1,5])
    with col1:
        selected_gender = st.selectbox('Select a Gender:', ["Female", "Male", "Other"])
        if selected_gender == "Female":
            Gender = 0
        elif selected_gender == "Male":
            Gender = 1
        else:
            Gender = 2


        Selected_Admission_Type  = st.selectbox('Select a Admission Type:', ['Emergency','Urgent', 'Elective'])
        if Selected_Admission_Type  == "Emergency":
            Admission_Type = 1
        elif Selected_Admission_Type == "Urgent":
            Admission_Type = 2
        else:
            Admission_Type = 0

        Selected_Diagnosis  = st.selectbox('Select a Diagnosis:', ['Heart Disease', 'Diabetes', 'Injury', 'Infection'])
        if Selected_Admission_Type  == "Heart Disease":
            Diagnosis = 1
        elif Selected_Admission_Type == "Diabetes":
            Diagnosis = 0
        elif Selected_Admission_Type == "Injury":
            Diagnosis = 3
        else:
            Diagnosis = 2

        Num_Lab_Procedures  = st.selectbox('Select a Number of Lab Procedures:', range(1,100))

        Num_Medications  = st.selectbox('Select a Number of Medications:', range(1,36))

    with col3:
        Num_Outpatient_Visits  = st.selectbox('Select a Number of Outpatient Visits:', range(0,5))

        Num_Inpatient_Visits  = st.selectbox('Select a Number of Inpatient Visits:', range(0,5))

        Num_Emergency_Visits  = st.selectbox('Select a Number of Emergency Visits:', range(0,5))

        Num_Diagnoses  = st.selectbox('Select a Number of Diagnoses:', range(1,10))

        A1C = st.selectbox('Select a Number of A1C Result:', ['Normal','Abnormal'])
        if A1C  == "Normal":
            A1C_Result = 1
        else:
            A1C_Result = 0

    st.write("")
    st.write("")

    col1,col2,col3 = st.columns([3,4,3])
    with col2:
        button = st.button(":red[PREDICT THE READMISSION]",use_container_width= True)

        if button:
            admission = predict_readmission(Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
       Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
       Num_Emergency_Visits, Num_Diagnoses, A1C_Result)
            if admission == 1:
                st.write("## :red[Readmission is Required]")
            else:
                st.write("## :green[Readmission is Not Required]")

