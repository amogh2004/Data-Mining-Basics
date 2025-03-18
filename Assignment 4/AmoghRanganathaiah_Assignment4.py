#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
  
# fetch dataset 
data = pd.read_csv("processed_cleveland.csv")


# In[2]:


# Renaming columns for clarity
columns = ['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data.columns = columns


# In[3]:


# Remove observations with missing values
data.replace('?', np.nan, inplace=True)
data.dropna(how='any', inplace=True)

# # Clean and convert problematic columns
# data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
# data['thal'] = pd.to_numeric(data['thal'], errors='coerce')

# Create a new response variable, y
data['num'] = data['num'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)


# In[4]:


# Specify the columns to be one-hot encoded
columns_to_encode = ['cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Perform one-hot encoding for all categorical variables
data_encoded = pd.get_dummies(data, columns=columns_to_encode)

# Convert all data to float type to ensure all are numeric
for col in data_encoded.columns:
    data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')


# In[5]:


from sklearn.model_selection import train_test_split

# Split the data
X = data_encoded.drop(['id', 'num'], axis=1)
y = data_encoded['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[6]:


X_train


# In[7]:


# Convert DataFrame to numpy array to ensure compatibility
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


# In[8]:


import statsmodels.api as sm

# Add a constant to the training and testing features for statsmodels
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)


# In[9]:


import warnings
warnings.filterwarnings('ignore')

# Fit logistic regression model
logit_model = sm.Logit(y_train.astype(float), X_train_const.astype(float))
logit_result = logit_model.fit()

# Print summary of the model
logit_result.summary()


# In[10]:


# interpretations of the coefficients 
print("Significant coefficients:")
significant_coeffs = logit_result.params[logit_result.pvalues < 0.05]
print(significant_coeffs)


# The logistic regression model revealed three significant coefficients. The coefficient for "sex" is approximately 1.43, indicating that being male is associated with an increase in the log-odds of the outcome variable compared to being female. For "trestbps," the coefficient is approximately 0.033, suggesting that for every one-unit increase in resting blood pressure, there is a corresponding increase in the log-odds of the outcome variable. Lastly, the coefficient for "ca" is around 1.21, indicating that the number of major vessels colored by fluoroscopy is positively associated with the log-odds of the outcome variable. These coefficients provide valuable insights into the relationship between these variables and the predicted outcome.

# In[11]:


# Split the data
X = data_encoded.drop(['id', 'num'], axis=1)
y = data_encoded['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Define a logistic regression model
logistic_model = LogisticRegression()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')

# Initialize variables to store the best thresholds and corresponding scores
best_threshold_f_score = None
best_threshold_tpr = None
best_threshold_youden = None
best_f_score = 0
max_tpr = 0
max_youden = 0

# Define the threshold range
threshold_range = np.linspace(0, 1, 100) 

# Iterate over different thresholds
for alpha in threshold_range:
    # Fit the model and obtain predictions
    logistic_model.fit(X_train, y_train)
    y_pred = (logistic_model.predict_proba(X_test)[:, 1] >= alpha).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate precision, recall, sensitivity, specificity, and Youden index
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = recall
    specificity = tn / (tn + fp)
    youden_index = sensitivity + specificity - 1

    # Calculate F-score
    f_score = 2 * (precision * recall) / (precision + recall)

    # Update best thresholds and corresponding scores
    if f_score > best_f_score:
        best_threshold_f_score = alpha
        best_f_score = f_score

    if sensitivity > max_tpr:
        best_threshold_tpr = alpha
        max_tpr = sensitivity

    if youden_index > max_youden:
        best_threshold_youden = alpha
        max_youden = youden_index

# Output the best thresholds and corresponding scores
print("Best Threshold for F-score:", best_threshold_f_score)
print("Best F-score:", best_f_score)
print()

print("Best Threshold for TPR:", best_threshold_tpr)
print("Max TPR:", max_tpr)
print()

print("Best Threshold for Youden Index:", best_threshold_youden)
print("Max Youden Index:", max_youden)


# In[12]:


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Initialize lists to store metrics for each threshold
threshold_metrics = []

# Iterate over each threshold value
for threshold in [best_threshold_f_score, best_threshold_tpr, best_threshold_youden]:
    # Perform prediction on the test dataset using the current threshold
    y_pred = (logistic_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate metrics
    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn)  # False Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate (Specificity)
    fnr = fn / (fn + tp)  # False Negative Rate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Append metrics to the list
    threshold_metrics.append({
        'Threshold': threshold,
        'True Positive Rate': tpr,
        'False Positive Rate': fpr,
        'True Negative Rate': tnr,
        'False Negative Rate': fnr,
        'Accuracy': accuracy,
        'Precision': precision
    })

# Print metrics for each threshold
for metrics in threshold_metrics:
    print("Threshold:", metrics['Threshold'])
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, (logistic_model.predict_proba(X_test)[:, 1] >= metrics['Threshold']).astype(int)))
    print("True Positive Rate (TPR):", metrics['True Positive Rate'])
    print("False Positive Rate (FPR):", metrics['False Positive Rate'])
    print("True Negative Rate (TNR):", metrics['True Negative Rate'])
    print("False Negative Rate (FNR):", metrics['False Negative Rate'])
    print("Accuracy:", metrics['Accuracy'])
    print("Precision:", metrics['Precision'])
    print()


# In[13]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Define an array of candidate values for the regularization parameter lambda
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Initialize lists to store average scores across folds for each lambda value
avg_scores = []

# Perform 5-fold cross-validation
for lambda_val in lambda_values:
    # Define logistic regression model with ridge penalty
    logistic_model = LogisticRegressionCV(Cs=[lambda_val], penalty='l2', cv=5, scoring='accuracy')
    
    # Fit model and obtain cross-validation scores
    cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=5)
    
    # Calculate average score across folds
    avg_score = np.mean(cv_scores)
    
    # Append average score to list
    avg_scores.append(avg_score)

# Choose the lambda value that maximizes the chosen metric
best_lambda = lambda_values[np.argmax(avg_scores)]
print("Best Lambda:", best_lambda)

# Make predictions on the test set using the chosen threshold value
logistic_model = LogisticRegression(C=best_lambda, penalty='l2')
logistic_model.fit(X_train, y_train)
y_pred = (logistic_model.predict_proba(X_test)[:, 1] >= best_threshold_f_score).astype(int)

# Calculate and display confusion matrix and other metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("True Positive Rate (TPR):", tpr)
print("False Positive Rate (FPR):", fpr)
print("True Negative Rate (TNR):", tnr)
print("False Negative Rate (FNR):", fnr)
print("Accuracy:", accuracy)
print("Precision:", precision)


# In[14]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Define an array of candidate values for the regularization parameter lambda
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Initialize lists to store average scores across folds for each lambda value
avg_scores = []

# Perform 5-fold cross-validation
for lambda_val in lambda_values:
    # Define logistic regression model with Lasso penalty
    logistic_model = LogisticRegressionCV(Cs=[lambda_val], penalty='l1', solver='liblinear', cv=5, scoring='accuracy')
    
    # Fit model and obtain cross-validation scores
    cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=5)
    
    # Calculate average score across folds
    avg_score = np.mean(cv_scores)
    
    # Append average score to list
    avg_scores.append(avg_score)

# Choose the lambda value that maximizes the chosen metric
best_lambda = lambda_values[np.argmax(avg_scores)]
print("Best Lambda:", best_lambda)

# Make predictions on the test set using the chosen threshold value
logistic_model = LogisticRegression(C=best_lambda, penalty='l1', solver='liblinear')
logistic_model.fit(X_train, y_train)
y_pred = (logistic_model.predict_proba(X_test)[:, 1] >= best_threshold_f_score).astype(int)

# Calculate and display confusion matrix and other metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("True Positive Rate (TPR):", tpr)
print("False Positive Rate (FPR):", fpr)
print("True Negative Rate (TNR):", tnr)
print("False Negative Rate (FNR):", fnr)
print("Accuracy:", accuracy)
print("Precision:", precision)

