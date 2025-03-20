#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import arff
import pandas as pd

data=arff.loadarff('1year.arff')

data=pd.DataFrame(data[0])

data['class'] = data['class'].str.decode('utf-8').astype(int)
data.dropna(inplace=True)


# In[2]:


data


# In[3]:


# Normalizing the data is not strictly required for implementing the DecisionTreeClassifier, 
# as decision trees are not sensitive to the scale of the features. 
# Decision trees make splits based on individual feature values, 
# so the scale of the features does not impact the algorithm's performance.

from sklearn.model_selection import train_test_split

X = data.drop(columns=['class'])  # Features
y = data['class']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the parameter grid
param_grid = {
    'max_depth': np.linspace(1, 20, 10, dtype=int),  # 10 candidate values for max_depth
    'max_leaf_nodes': np.linspace(2, 50, 10, dtype=int),  # 10 candidate values for max_leaf_nodes
    'ccp_alpha': np.linspace(0, 0.1, 10)  # 10 candidate values for alpha
}

# Create the decision tree classifier
dt_classifier = DecisionTreeClassifier()

# Create the grid search object
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)

# Perform grid search
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Get the best decision tree model from grid search
best_dtree = grid_search.best_estimator_

# Use the best model to make predictions
dtree_predictions = best_dtree.predict(X_test)

# Calculate accuracy
dtree_accuracy = accuracy_score(y_test, dtree_predictions)

# Calculate confusion matrix
dtree_conf_matrix = confusion_matrix(y_test, dtree_predictions)

# Print decision tree results
print("\nDecision Tree Results:")
print("Accuracy:", dtree_accuracy)
print("Confusion Matrix:")
print(dtree_conf_matrix)


# In[6]:


# Feature Importance
feature_importance = best_dtree.feature_importances_
print("\nFeature Importance:")
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1}: {importance}")


# Feature 9 has the highest importance with a value of approximately 0.632, indicating that it plays a significant role in the decision-making process of the classifier. <br>
# Features 34, 46, and 57 also have notable importance values, though comparatively smaller than feature 9. <br>
# The remaining features (1 to 8 and 10 to 64) have zero importance, suggesting that they do not significantly contribute to the decision-making process of the classifier. <br>

# In[7]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid_bagging = {
    'n_estimators': np.linspace(10, 100, 10, dtype=int),  # 10 candidate values for n_estimators
    'base_estimator__max_depth': np.linspace(1, 20, 10, dtype=int)  # 10 candidate values for max_depth
}

# Create the base decision tree estimator
base_estimator = DecisionTreeClassifier()

# Create the BaggingClassifier
bagging_classifier = BaggingClassifier(base_estimator=base_estimator)

# Create the grid search object
grid_search_bagging = GridSearchCV(bagging_classifier, param_grid_bagging, cv=5)

# Perform grid search
grid_search_bagging.fit(X, y)

# Get the best BaggingClassifier model from grid search
best_bagging = grid_search_bagging.best_estimator_

# Use the best model to make predictions
bagging_predictions = best_bagging.predict(X_test)

# Calculate accuracy
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

# Calculate confusion matrix
bagging_conf_matrix = confusion_matrix(y_test, bagging_predictions)

# Print bagging tree results
print("\nBagging Tree Results:")
print("Accuracy:", bagging_accuracy)
print("Confusion Matrix:")
print(bagging_conf_matrix)

# Get feature importances
feature_importances = np.mean([tree.feature_importances_ for tree in best_bagging.estimators_], axis=0)


# In[13]:


print("\nFeature Importance:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance}")


# Feature 36 has the highest importance with a value of approximately 0.4, indicating it plays a significant role in the classifier's decision-making process. <br>
# Feature 46 follows closely with an importance value of approximately 0.35. <br>
# Features 9, 21, 27, and 30 also have non-zero importance values, though comparatively smaller than features 36 and 46. <br>

# In[9]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid_gradient_boosting = {
    'learning_rate': np.logspace(-3, 0, 10),  # 10 candidate values for learning_rate
    'n_estimators': np.linspace(10, 100, 10, dtype=int)  # 10 candidate values for n_estimators
}

# Create the GradientBoostingClassifier
gradient_boosting_classifier = GradientBoostingClassifier()

# Create the grid search object
grid_search_gradient_boosting = GridSearchCV(gradient_boosting_classifier, param_grid_gradient_boosting, cv=5)

# Perform grid search
grid_search_gradient_boosting.fit(X, y)

# Get the best GradientBoostingClassifier model from grid search
best_gradient_boosting = grid_search_gradient_boosting.best_estimator_

# Use the best model to make predictions
gradient_boosting_predictions = best_gradient_boosting.predict(X_test)

# Calculate accuracy
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_predictions)

# Calculate confusion matrix
gradient_boosting_conf_matrix = confusion_matrix(y_test, gradient_boosting_predictions)

# Print gradient boosting tree results
print("\nGradient Boosting Tree Results:")
print("Accuracy:", gradient_boosting_accuracy)
print("Confusion Matrix:")
print(gradient_boosting_conf_matrix)

# Get feature importances
feature_importances_gb = best_gradient_boosting.feature_importances_


# In[14]:


print("\nFeature Importance:")
for i, importance in enumerate(feature_importances_gb):
    print(f"Feature {i+1}: {importance}")


# Features 36, 46, and 34 have the highest importance values, with values of approximately 0.352, 0.168, and 0.134, respectively. These features are likely crucial in the classifier's decision-making process. <br>
# Features 9, 5, and 20 also have notable importance values, though comparatively smaller than features 36, 46, and 34. <br>
# Additionally, features 37, 60, and 61 have been assigned non-zero importance values, though smaller compared to the aforementioned features. <br>

# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid_random_forest = {
    'n_estimators': np.linspace(10, 100, 10, dtype=int),  # 10 candidate values for n_estimators
    'max_features': ['auto', 'sqrt', 'log2', None]  # Candidate values for max_features
}

# Create the RandomForestClassifier
random_forest_classifier = RandomForestClassifier()

# Create the grid search object
grid_search_random_forest = GridSearchCV(random_forest_classifier, param_grid_random_forest, cv=5)

# Perform grid search
grid_search_random_forest.fit(X, y)

# Get the best RandomForestClassifier model from grid search
best_random_forest = grid_search_random_forest.best_estimator_

# Use the best model to make predictions
random_forest_predictions = best_random_forest.predict(X_test)

# Calculate accuracy
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)

# Calculate confusion matrix
random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predictions)

# Print random forest results
print("\nRandom Forest Results:")
print("Accuracy:", random_forest_accuracy)
print("Confusion Matrix:")
print(random_forest_conf_matrix)


# In[15]:


# Get feature importances
feature_importances_rf = best_random_forest.feature_importances_

print("\nFeature Importance:")
for i, importance in enumerate(feature_importances_rf):
    print(f"Feature {i+1}: {importance}")


# Features such as 9, 46, 3, 4, 26, and 36 have relatively higher importance values compared to others, suggesting they play crucial roles in the classifier's decision-making process. <br>
# Other features also have notable importance values, though smaller compared to the aforementioned features.  <br>Features like 18, 38, 39, 55, 56, 57, and 54 have relatively higher importance values compared to others. <br>
# Some features, such as 33 and 6, have very low importance values, suggesting they contribute less to the classification process. <br>
