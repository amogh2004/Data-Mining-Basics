#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from time import perf_counter

# Read data from CSV file
data = pd.read_csv('super_train-1.csv')
data = data.head(5000)


# In[2]:


# Extract predictors (X) and target variable (y)
X = data.drop(columns=['critical_temp'])
y = data['critical_temp']

# Split a universal holdout set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=1)

# Scale your data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ## Hyperparameters

# In[3]:


import warnings
warnings.filterwarnings('ignore')

# Define candidate values for hyperparameters
alphas = np.logspace(-3, 2, 10)
max_iters = np.linspace(1000, 20000, 10, dtype=int)


# ## Model 1

# In[4]:


best_score1 = -np.inf
best_params1 = {}
start_time = perf_counter()

for alpha in alphas:
    for max_iter in max_iters:
        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_val_scaled, y_val)
        if score > best_score1:
            best_score1 = score
            best_params1['alpha'] = alpha
            best_params1['max_iter'] = max_iter
            
end_time = perf_counter()


# In[5]:


execution_time1 = end_time - start_time
print("Hyperparameter selection execution time for Model 1: {:.2f} seconds".format(execution_time1))


# In[6]:


# Train the final model using best hyperparameters
final_model1 = Lasso(alpha=best_params1['alpha'], max_iter=best_params1['max_iter'])
final_model1.fit(X_train_scaled, y_train)

# Evaluate the final model
test_score1 = final_model1.score(X_test_scaled, y_test)
print("Test set score for Model 1: {:.2f}".format(test_score1))

# Calculate mean squared error for Model 1
predictions1 = final_model1.predict(X_test_scaled)
mse1 = mean_squared_error(predictions1, y_test)
print("Mean Squared Error for Model 1: {:.2f}".format(mse1))

# Extract coefficients for Model 1
coef1 = final_model1.coef_


# ## Model 2

# In[7]:


best_score2 = -np.inf
best_params2 = {}
start_time = perf_counter()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for alpha in alphas:
    for max_iter in max_iters:
        model = Lasso(alpha=alpha, max_iter=max_iter)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=kf)
        score = np.mean(scores)
        if score > best_score2:
            best_score2 = score
            best_params2['alpha'] = alpha
            best_params2['max_iter'] = max_iter
            
end_time = perf_counter()


# In[8]:


execution_time2 = end_time - start_time
print("Hyperparameter selection execution time for Model 2: {:.2f} seconds".format(execution_time2))


# In[9]:


# Train the final model using best hyperparameters
final_model2 = Lasso(alpha=best_params2['alpha'], max_iter=best_params2['max_iter'])
final_model2.fit(X_train_scaled, y_train)

# Evaluate the final model
test_score2 = final_model2.score(X_test_scaled, y_test)
print("Test set score for Model 2: {:.2f}".format(test_score2))

# Calculate mean squared error for Model 2
predictions2 = final_model2.predict(X_test_scaled)
mse2 = mean_squared_error(predictions2, y_test)
print("Mean Squared Error for Model 2: {:.2f}".format(mse2))

# Extract coefficients for Model 2
coef2 = final_model2.coef_


# ## Model 3

# In[10]:


best_score3 = -np.inf
best_params3 = {}
start_time = perf_counter()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for alpha in alphas:
    for max_iter in max_iters:
        model = Lasso(alpha=alpha, max_iter=max_iter)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=kf)
        score = np.mean(scores)
        if score > best_score3:
            best_score3 = score
            best_params3['alpha'] = alpha
            best_params3['max_iter'] = max_iter
            
end_time = perf_counter()


# In[11]:


execution_time3 = end_time - start_time
print("Hyperparameter selection execution time for Model 3: {:.2f} seconds".format(execution_time3))


# In[12]:


# Train the final model using best hyperparameters
final_model3 = Lasso(alpha=best_params3['alpha'], max_iter=best_params3['max_iter'])
final_model3.fit(X_train_scaled, y_train)

# Evaluate the final model
test_score3 = final_model3.score(X_test_scaled, y_test)
print("Test set score for Model 3: {:.2f}".format(test_score3))

# Calculate mean squared error for Model 3
predictions3 = final_model3.predict(X_test_scaled)
mse3 = mean_squared_error(predictions3, y_test)
print("Mean Squared Error for Model 3: {:.2f}".format(mse3))

# Extract coefficients for Model 3
coef3 = final_model3.coef_


# ## Analysis

# In[13]:


# Compare the performance of all three models
results = {
    "Model 1": {"Test Score": test_score1, "MSE": mse1, "Execution Time": execution_time1},
    "Model 2": {"Test Score": test_score2, "MSE": mse2, "Execution Time": execution_time2},
    "Model 3": {"Test Score": test_score3, "MSE": mse3, "Execution Time": execution_time3}
}

# Print the results
print("Performance Comparison:")
for model, metrics in results.items():
    print(f"\n{model}:")
    print(f"Test Score: {metrics['Test Score']:.2f}")
    print(f"Mean Squared Error: {metrics['MSE']:.2f}")
    print(f"Execution Time: {metrics['Execution Time']:.2f} seconds")


# In[14]:


# Print coefficients for all three models
print("\nCoefficients:")
print("Model 1:")
print(coef1)

print("\nModel 2:")
print(coef2)

print("\nModel 3:")
print(coef3)


# ## Inference

# Model 1 achieved the highest test score among the three models, indicating that it has the best predictive performance on the test data. Additionally, it has the lowest mean squared error, suggesting that it produces more accurate predictions compared to the other models. Moreover, it has the shortest execution time, indicating that it is the most computationally efficient model. <br>
# 
# Model 2 has a slightly lower test score and higher mean squared error compared to Model 1. However, it also has a significantly longer execution time, which may not be desirable if computational efficiency is important. <br>
# 
# Model 3 has similar performance to Model 2 in terms of test score and mean squared error. However, it has the longest execution time among the three models, making it the least desirable choice if computational efficiency is a concern. <br>
# 
# __In summary, Model 1 appears to be the best choice among the three models, as it achieves the highest test score, lowest mean squared error, and shortest execution time.__

# ## Implementing OLS Method to choose the Best 20 Coefficients

# In[19]:


import statsmodels.api as sm

# Fit the model using StatsModels
model = sm.OLS(y_train, sm.add_constant(X_train))
results = model.fit()

# Get the p-values and coefficients
p_values = results.pvalues[1:]  # Exclude intercept
coefficients = results.params[1:]

# Sort the coefficients by p-values
sorted_indices = np.argsort(p_values)
sorted_coefficients = coefficients[sorted_indices]

# Select the top 20 coefficients with the lowest p-values
top_20_indices = sorted_indices[:20]
top_20_coefficients = sorted_coefficients[:20]

# Print the top 20 coefficients with their names
print("Top 20 coefficients based on p-values:")
for index, coefficient in zip(top_20_indices, top_20_coefficients):
    print(f"{feature_names[index]}: {coefficient} (p-value: {p_values[index]})")

