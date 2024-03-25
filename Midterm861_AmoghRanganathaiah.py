#!/usr/bin/env python
# coding: utf-8

# # The Boston Housing Dataset
# ### The Boston Housing Dataset is a derived from information collected by the U.S.
# 
# Census Service concerning housing in the area of Boston MA. The following describes the dataset columns: <br>
# 01. CRIM - per capita crime rate by town <br>
# 02. ZN - proportion of residential land zoned for lots over 25,000 sq.ft. <br>
# 03. INDUS - proportion of non-retail business acres per town. <br>
# 04. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise) <br>
# 05. NOX - nitric oxides concentration (parts per 10 million) <br>
# 06. RM - average number of rooms per dwelling <br>
# 07. AGE - proportion of owner-occupied units built prior to 1940 <br>
# 08. DIS - weighted distances to five Boston employment centres <br>
# 09. RAD - index of accessibility to radial highways <br>
# 10. TAX - full-value property-tax rate per USD 10,000 <br>
# 11. PTRATIO - pupil-teacher ratio by town <br>
# 12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town <br>
# 13. LSTAT - % lower status of the population <br>
# 14. MEDV - Median value of owner-occupied homes in USD 1000's <br>

# ### Question 1
# For problem 1,2, and 3, split the data into 2 parts: X and X_test with ratio 3:1 ( 25% data for testing). You will use X for model training and X_test for testing purpose. For any data splitting that you perform, use the last 4 digits of your SFSU ID as random_state for all splits.

# In[1]:


import pandas as pd

# Reading the data and adding column names
df = pd.read_csv('housing.csv', header=None, delimiter='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df


# In[2]:


from sklearn.model_selection import train_test_split

# Splitting the data into features (X) and target variable (y)
XX = df.drop('MEDV', axis=1)  # Features
yy = df['MEDV']  # Target variable

# Splitting the data into training and testing sets
# My SFSU ID is 923290859. I can't use 0 as a prefix, hence I have just put up 859 for random state
X, X_test, y, y_test = train_test_split(XX, yy, test_size=0.25, random_state=859)


# ### Question 2
# A linear regression model to predict median value (a regression model) (10pts)
# 1. Fit a full model using all the observations. <br>
# 2. Perform model assessment. <br>
# 3. Provide interpretation to the significant coefficients. <br>
# 4. Report MSE value on the test set. <br>

# In[3]:


import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Fit the linear regression model
X_con = sm.add_constant(X)
model = sm.OLS(y, X_con).fit()

model.summary()


# #### Interpretation to Significant Coefficients
# 
# R-squared: 0.733 => Approximately 73.3% of the variance in the dependent variable (median value) is explained by the independent variables included in the model. <br>
# 
# Adj. R-squared: 0.723 => Adjusts for the number of predictors in the model, providing a more accurate measure of model fit for multiple regression. <br>
# 
# F-statistic: 77.08 and Prob (F-statistic): 4.17e-96 => The F-statistic is 77.08, and the associated p-value (Prob (F-statistic)) is very small (4.17e-96), indicating that the model as a whole is statistically significant. <br>
# 
# 1. RM: Coefficient: 4.1987, p-value: 0.000 <br> For each additional room in a dwelling, the median value of homes in the area tends to increase by approximately USD 4,198.7.
# 
# 2. DIS: Coefficient: -1.3559, p-value: 0.000 <br> For each unit increase in the weighted distance to employment centers, the median value of homes in the area tends to decrease by approximately USD 1,355.9.
# 
# 3. PTRATIO: Coefficient: -0.8875, p-value: 0.000 <br> For each one-unit increase in the pupil-teacher ratio, the median value of homes in the area tends to decrease by approximately USD 887.5.
# 
# 4. LSTAT: Coefficient: -0.4971, p-value: 0.000 <br> For each one-unit increase in the percentage of lower status population, the median value of homes in the area tends to decrease by approximately USD 497.1.
# 
# 5. NOX: Coefficient: -17.9243, p-value: 0.000 <br> For each unit increase in the concentration of nitric oxides, the median value of homes in the area tends to decrease by approximately USD 17,924.3.

# In[4]:


# Make predictions on the test set
X_test_con = sm.add_constant(X_test)  # Add constant term to test features
y_pred = model.predict(X_test_con)  # Predict y values for the test set

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) on the test set: {:.4f}".format(mse))


# In[5]:


# Model Assessment

# Calculate residuals
residuals = model.resid

import matplotlib.pyplot as plt
import scipy.stats as stats

# Plot residuals vs fitted values
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, residuals, alpha=0.5)
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()


# In[6]:


# QQ plot of residuals
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()


# ### Question 3
# Two regularized models to predict median value (a Lasso and a Ridge regressionmodel) (20 pts)
# 1. Fit regularized models and perform prediction on a test set. <br>
# 2. You may use any lambda value range of your choice with at least 20 values for lambda, you must select from more than 1 lambda option. <br>
# 3. Report the MSE value on the test set. <br>
# 4. Compare the coefficients here with the model from part 2, and report any differences/observations. <br>

# In[7]:


from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error

# Define the range of lambda values
lambda_range = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 125, 150, 175, 200, 225, 275, 300, 350, 400, 500, 750, 1000]

# Fit Lasso regression model
lasso_model = LassoCV(alphas=lambda_range, cv=5)
lasso_model.fit(X, y)

# Fit Ridge regression model
ridge_model = RidgeCV(alphas=lambda_range, cv=5)
ridge_model.fit(X, y)

# Make predictions on the test set
lasso_y_pred = lasso_model.predict(X_test)
ridge_y_pred = ridge_model.predict(X_test)


# In[8]:


print("MSE value on the test set (Lasso regression): {:.4f}".format(mean_squared_error(y_test, lasso_y_pred)))
print("MSE value on the test set (Ridge regression): {:.4f}".format(mean_squared_error(y_test, ridge_y_pred)))


# In[9]:


import pandas as pd

# Create DataFrame for coefficients of the OLS model
coefficients_df = pd.DataFrame({'Feature': X_con.columns, 'OLS Coefficient': model.params})

# Create DataFrame for coefficients of the Lasso regression model
lasso_coefficients_df = pd.DataFrame({'Feature': X.columns, 'Lasso Coefficient': lasso_model.coef_})

# Create DataFrame for coefficients of the Ridge regression model
ridge_coefficients_df = pd.DataFrame({'Feature': X.columns, 'Ridge Coefficient': ridge_model.coef_})

# Merge the DataFrames on the 'Feature' column
merged_df = coefficients_df.merge(lasso_coefficients_df, on='Feature').merge(ridge_coefficients_df, on='Feature')

# Display the merged DataFrame
print(merged_df)


# The magnitudes of the coefficients tend to be slightly smaller in Lasso and Ridge regression compared to OLS regression. This is because Lasso and Ridge regression introduce a penalty term to the coefficient values, which can lead to shrinkage of the coefficients towards zero.
# 
# We observe differences in the Lasso coefficients compared to OLS and Ridge coefficients. For example, the coefficient for the feature INDUS (Proportion of Non-Retail Business Acres per Town) becomes zero in Lasso regression, indicating that Lasso has effectively eliminated this feature from the model.
# 
# Lasso regression, by shrinking some coefficients to exactly zero, provides a more interpretable model with a subset of important features. In contrast, OLS and Ridge regression may include all features in the model, potentially leading to overfitting or difficulties in interpretation, especially when dealing with a large number of features.

# ### Question 4
# A KNN model to predict low, medium or high median house value (a classification model) (10 pts)
# 1. Create a categorical response variable based on TARGET_houseValue. <br>
# 2. This variable should contain 3 levels (low, medium, high) that divides the data into 3 (roughly) equal chunks, i.e. each level should have (roughly) the same amount of observations. <br>
# 3. Split the data into 2 parts: X and X_test with ratio 3:1 ( 25% data for testing). You will use X for model training and X_test for testing purpose. <br>
# 4. Report the accuracy on the test set. <br>

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('housing.csv', header=None, delimiter='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Create categorical response variable based on TARGET_houseValue
df['TARGET_houseValue_category'] = pd.qcut(df['MEDV'], q=3, labels=['low', 'medium', 'high'])

df


# In[11]:


# Split the data into features (X) and target variable (y)
X = df.drop(['MEDV', 'TARGET_houseValue_category'], axis=1)
y = df['TARGET_houseValue_category']

# Split the data into training and testing sets
# My SFSU ID is 923290859. I can't use 0 as a prefix, hence I have just put up 859 for random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=859)

# Fit a KNN classifier model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Perform prediction on the test set
y_pred_test = knn_model.predict(X_test)

# Report the accuracy on the test set
accuracy = round(accuracy_score(y_test, y_pred_test),5)
print("Accuracy on the test set:", accuracy)

