#!/usr/bin/env python
# coding: utf-8

# # Part 1 : Exploratory Analysis

# In[1]:


import pandas as pd
df = pd.read_csv('flights.csv')
df


# In[2]:


df.describe()


# ### 1
# How many observations are there? How many features are there?

# In[3]:


df.shape[0]


# In[4]:


df.shape[1]


# ### 2
# How many flights arrived at SFO? How many airlines fly to SFO?

# In[5]:


# Filter the DataFrame for flights arriving at SFO
arrivals_at_sfo = df[df['DESTINATION_AIRPORT'] == 'SFO']

# Count and print the number of rows in the filtered DataFrame
print("Number of flights arrived at SFO:", len(arrivals_at_sfo))

# Get the unique airlines flying to SFO
airlines_to_sfo = arrivals_at_sfo['AIRLINE'].unique()

# Count and print the number of airlines in the filtered DataFrame
print("Number of airlines flying to SFO:", len(airlines_to_sfo))


# ### 3
# How many missing values are there in the departure delays? How about arrival delays? Do they match? Why or why not? Remove these observations afterwards

# In[6]:


# Count and print the number of rows where DEPARTURE_DELAY value is missing
print("Number of missing values in departure delays:", df['DEPARTURE_DELAY'].isnull().sum())

# Count and print the number of rows where ARRIVAL_DELAY value is missing
print("Number of missing values in arrival delays:", df['ARRIVAL_DELAY'].isnull().sum())


# The number of missing values in departure and arrival delays **do not match** because there will be legitimate differences due to natural variability in flight operations. Also, the data collection and data entry for departure and arrivals could be different. 

# In[7]:


# Dropping all the rows where values are missing in DEPARTURE_DELAY and ARRIVAL_DELAY columns
df = df.dropna(subset=['DEPARTURE_DELAY', 'ARRIVAL_DELAY'])
df


# ### 4
# What is the average and median departure and arrival delay? What do you observe?

# In[8]:


print("The average of departure delay(in minutes) is: ",round(df['DEPARTURE_DELAY'].mean(),2))
print("The median of departure delay(in minutes) is: ",df['DEPARTURE_DELAY'].median())
print("The average of arrival delay(in minutes) is: ",round(df['ARRIVAL_DELAY'].mean(),2))
print("The median of arrival delay(in minutes) is: ",df['ARRIVAL_DELAY'].median())


# In departures/arrivals, negative represents early and positive represents late.
# So, the average arrival and departure is later than expected; but the median of arrival and departure is earlier than expected.
# Also, since the mean and median are not close, we can say the data set has non-symmetrical distribution.

# ### 5
# Display graphically the departure delays and arrival delays for each airline. What do you notice? Explain.

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a combined DataFrame for departure and arrival delays
df_delays = pd.concat([df['DEPARTURE_DELAY'], df['ARRIVAL_DELAY']], axis=1)

# Melt the DataFrame to have delays in a single column with an additional column for the type of delay
df_delays = df_delays.melt(var_name='Delay Type', value_name='Delay')

# Plot the delays for each airline
plt.figure(figsize=(12, 6))
sns.boxplot(x='Delay Type', y='Delay', data=df_delays)
plt.title('Departure and Arrival Delays for Each Airline')
plt.xlabel('Delay Type')
plt.ylabel('Delay (minutes)')
plt.xticks([0, 1], ['Departure Delay', 'Arrival Delay'])
plt.grid(True)
plt.show()


# We see that the minimum and maximum lines of delays are very near to 0, and most of the other points are beyond those lines. By this we understand that there is no significant delay in the almost all flights, except for some exceptional cases.

# ### 6
# Now calculate the 5 number summary (min, Q1, median, Q3, max) of departure delay for each airline. Arrange it by median delay (descending order). Do the same for arrival delay.

# In[10]:


# Calculate the five-number summary for departure delay for each airline
summary_departure_delay = df.groupby('AIRLINE')['DEPARTURE_DELAY'].describe(percentiles=[.25, .5, .75])

# Sort the summary by median delay in descending order
summary_departure_delay_sorted = summary_departure_delay.sort_values(by='50%', ascending=False)

# Rename the columns for better clarity
summary_departure_delay_sorted = summary_departure_delay_sorted.rename(columns={
    'min': 'Minimum',
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'max': 'Maximum'
})

print("Five-Number Summary of Departure Delay for Each Airline (Arranged by Median Delay - Descending Order):")
summary_departure_delay_sorted


# In[11]:


# Calculate the five-number summary for arrival delay for each airline
summary_arrival_delay = df.groupby('AIRLINE')['ARRIVAL_DELAY'].describe(percentiles=[.25, .5, .75])

# Sort the summary by median delay in descending order
summary_arrival_delay_sorted = summary_arrival_delay.sort_values(by='50%', ascending=False)

# Rename the columns for better clarity
summary_arrival_delay_sorted = summary_arrival_delay_sorted.rename(columns={
    'min': 'Minimum',
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'max': 'Maximum'
})

print("Five-Number Summary of Arrival Delay for Each Airline (Arranged by Median Delay - Descending Order):")
summary_arrival_delay_sorted


# ### 7
# Which airline has the most averaged departure delay? Give me the top 10 airlines. 

# In[12]:


# Calculate the average departure delay for each airline
average_departure_delay = df.groupby('AIRLINE')['DEPARTURE_DELAY'].mean()

# Sort the average departure delay in descending order
top_10_airlines = average_departure_delay.sort_values(ascending=False).head(10)

print("Top 10 Airlines with the Highest Average Departure Delay:")
top_10_airlines


# ### 8
# Do you expect the departure delay has anything to do with distance of trip? What about arrival delay and distance? Prove your claims.

# In[13]:


# Drop rows with missing departure and arrival delays
df_cleaned = df.dropna(subset=['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE'])

# Plot departure delay vs. distance
plt.figure(figsize=(12, 6))
plt.scatter(df_cleaned['DISTANCE'], df_cleaned['DEPARTURE_DELAY'], alpha=0.5)
plt.title('Departure Delay vs. Distance')
plt.xlabel('Distance (miles)')
plt.ylabel('Departure Delay (minutes)')
plt.grid(True)
plt.show()


# In[14]:


# Plot arrival delay vs. distance
plt.figure(figsize=(12, 6))
plt.scatter(df_cleaned['DISTANCE'], df_cleaned['ARRIVAL_DELAY'], alpha=0.5)
plt.title('Arrival Delay vs. Distance')
plt.xlabel('Distance (miles)')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True)
plt.show()


# From the above graphs, we can conclude that delay is inversely proportional to distance covered. This may be because of the fuel situation and longhaul flight that they offer.

# ### 9
# What about day of week vs departure delay?

# In[15]:


# Calculate the average departure delay for each day of the week
average_departure_delay_by_day = df.groupby('DAY_OF_WEEK')['DEPARTURE_DELAY'].mean()

# Define the days of the week
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plot departure delay vs. day of the week
plt.figure(figsize=(10, 6))
plt.bar(days_of_week, average_departure_delay_by_day, color='skyblue')
plt.title('Average Departure Delay by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Departure Delay (minutes)')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()


# From the above graph, we can conclude that the delay in time is not going to depend on the day of week.

# ### 10
# If there is a departure delay (i.e. positive values for departure delay), does distance have anything to do with arrival delay? Explain.

# In[16]:


# Filter the DataFrame for flights with positive departure delays
df_filtered = df[(df['DEPARTURE_DELAY'] > 0) & df['ARRIVAL_DELAY'].notnull() & df['DISTANCE'].notnull()]

# Plot arrival delay vs. distance for flights with positive departure delays
plt.figure(figsize=(12, 6))
plt.scatter(df_filtered['DISTANCE'], df_filtered['ARRIVAL_DELAY'], alpha=0.5)
plt.title('Arrival Delay vs. Distance for Flights with Positive Departure Delay')
plt.xlabel('Distance (miles)')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True)
plt.show()


# We see that arrival delay is inversely proportional to distance(miles), when there is a departure delay.

# ### 11
# Are there any seasonal (monthly) patterns in departure delays for all flights?

# In[17]:


import warnings
warnings.filterwarnings('ignore')

# Convert 'MONTH' column to datetime format
df['MONTH'] = pd.to_datetime(df['MONTH'], format='%m')

# Group the DataFrame by month and calculate the average departure delay for each month
monthly_departure_delay = df.groupby(df['MONTH'].dt.month)['DEPARTURE_DELAY'].mean()

# Plot monthly departure delay patterns
plt.figure(figsize=(10, 6))
monthly_departure_delay.plot(marker='o')
plt.title('Average Departure Delay by Month')
plt.xlabel('Month')
plt.ylabel('Average Departure Delay (minutes)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()


# From this graph, we are able to understand that delays are very less during Sept, Oct and Nov months.

# # Part 2 : Regression Analysis

# ## Subpart 1

# ### 1
# Your response is ARRIVAL DELAY. First, remove all the missing data in the WEATHER DELAY column. Once you do this, there shouldn't be any more missing values in the data set (except for the cancellation reason feature).

# In[18]:


df = df.dropna(subset=['WEATHER_DELAY'])
df.dtypes


# ### 2
# Build a regression model using all the observations, and the following predictors:
# [LATE AIRCRAFT DELAY, AIR SYSTEM DELAY, DEPARTURE DELAY , WEATHER DELAY, SECURITY DELAY, DAY OF WEEK,  DISTANCE, AIRLINE] a total of 8 predictors.

# In[19]:


from sklearn.preprocessing import LabelEncoder

# Convert 'AIRLINE' to categorical variable
df['AIRLINE_CAT'] = df['AIRLINE'].astype('category')

# Use label encoding to convert 'AIRLINE_CAT' to numerical values
label_encoder = LabelEncoder()
df['AIRLINE_CAT'] = label_encoder.fit_transform(df['AIRLINE_CAT'])

df


# In[20]:


import statsmodels.api as sm

# Define the predictors and response variable
X = df[['LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'DEPARTURE_DELAY', 'WEATHER_DELAY', 'SECURITY_DELAY', 'DAY_OF_WEEK', 'DISTANCE', 'AIRLINE_CAT']]
y = df['ARRIVAL_DELAY']

# Add a constant to the predictors matrix (required for OLS regression)
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()


# ### 3
# Perform model diagnostics. What do you observe? Explain.

# In[21]:


# Print the model summary
print(model.summary())


# ### 4
# Provide interpretations for a few of the coefficients, and comment on whether they make sense.

# 1) R-squared = 0.935: 
# This indicates that approximately 93.5% of the variability in the response variable (ARRIVAL_DELAY) is explained by the predictors included in the model. This is a high R-squared value, suggesting that the model fits the data well. 
# 
# 2) Adjusted R-squared = 0.935: 
# The value provides a more accurate measure of model fit.
# 
# 3) F-statistic and p-value: 
# The statistically significant F-statistic and low p-value indicate that the overall regression model is significant, meaning that at least one predictor variable has a significant effect on arrival delays.
# 
# 4) Standard Error: 
# The standard errors (provided in the notes) measure the variability of the coefficient estimates.
# 
# 5) Condition Number: 
# A large condition number (3.48e+03) indicates potential multicollinearity or numerical problems in the model. Multicollinearity occurs when predictor variables are highly correlated with each other, which can make it difficult to determine the individual effect of each predictor on the response variable.
# 
# Since there is multicolinearity issues, we need to ensure the reliability of the coefficient estimates. Also, Multicollinearity inflates the standard errors of the coefficient estimates. This makes it harder to detect statistically significant effects, as the confidence intervals around the coefficients become wider. As a result, predictors that are actually important may appear to be non-significant in the presence of multicollinearity. **Therefore, this isn't a reliable model.**

# ## Subpart 2
# If you have done the above steps correctly, you will notice a lot of things "doesn't seem right". We will try to fix a couple of these things here.

# ### 1
# Removing outliers: first is to remove outliers. Using the boxplot method, remove the outliers in the ARRIVAL DELAY variable.

# In[22]:


import seaborn as sns

# Plot a boxplot of the 'ARRIVAL_DELAY' variable
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['ARRIVAL_DELAY'])
plt.title('Boxplot of Arrival Delay')
plt.xlabel('Arrival Delay (minutes)')
plt.show()

# Determine the threshold for identifying outliers based on IQR method
Q1 = df['ARRIVAL_DELAY'].quantile(0.25)
Q3 = df['ARRIVAL_DELAY'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers based on the threshold
df_cleaned = df[(df['ARRIVAL_DELAY'] >= lower_bound) & (df['ARRIVAL_DELAY'] <= upper_bound)]


# ### 2
# Refit the linear regression model, but now with log(ARRIVAL DELAY) as your response. Also, remove the non-significant predictors from the previous model (with p-values larger than 0.05). (Remember that when removing non-significant predictors one can only eliminate one variable per step, but for now we will ignore this rule and remove everything in one step).
# Also take the log transform of a DELAY variable and the square of another DELAY variable of your choice.  

# In[23]:


import numpy as np

# Dropping SECURITY_DELAY and DISTANCE columns as they are non-significant(p-values larger than 0.05)
df_cleaned = df_cleaned.dropna(subset=['SECURITY_DELAY', 'DISTANCE'])

# Create a new column 'WEATHER_DELAY_LOG' with logarithm values of 'WEATHER_DELAY'
df_cleaned['WEATHER_DELAY_LOG'] = np.log(df_cleaned['WEATHER_DELAY'] + 1)  # Adding 1 to avoid logarithm of zero

# Create a new column 'DEPARTURE_DELAY_SQ' with squared values of 'DEPARTURE_DELAY'
df_cleaned['DEPARTURE_DELAY_SQ'] = df_cleaned['DEPARTURE_DELAY'] ** 2


# In[24]:


# Define the predictors and response variable
X = df_cleaned[['LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'DEPARTURE_DELAY_SQ', 'WEATHER_DELAY_LOG', 'DAY_OF_WEEK', 'AIRLINE_CAT']]
y = df_cleaned['ARRIVAL_DELAY']

# Add a constant to the predictors matrix (required for OLS regression)
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()


# ### 3
# Perform model diagnostics. Did anything improve? 

# In[25]:


# Print the model summary
print(model.summary())


# ### 4
# Provide interpretations to a few of the coefficients. Do you think they make sense?

# The regression model with the additional predictors ('DEPARTURE_DELAY_SQ' and 'WEATHER_DELAY_LOG') appears to be statistically significant, as indicated by the low p-value of the F-statistic.
# 
# The R-squared value suggests that the model explains a reasonable amount of variability in arrival delays, but the improvement over the previous model is modest.
# 
# The presence of a large condition number indicates potential multicollinearity issues, particularly between 'DEPARTURE_DELAY' and its squared term. Further investigation is warranted to assess the impact of multicollinearity on the reliability of coefficient estimates and model interpretation.

# ### 5
# Obviously there's still a lot that needs to be done. Provide a few suggestions on how we can further improve the model fit (you don't need to implement them).

# 1) We can drop the AIRLINE_CAT column because the p-value is very high.
# 
# 2) We should address multicollinearity by removing highly correlated variables, using dimensionality reduction techniques, or redefining variables to avoid multicollinearity.
# 
# 3) We can do transformations of existing features that may better capture the relationship with the response variable. This could include interactions between variables, polynomial features, or other transformations.
# 
# 4) We should validate the assumptions of the regression model, such as linearity, homoscedasticity, and normality of residuals.
# 
# 5) We should consider alternative regression techniques. This could include regularization techniques like Ridge or Lasso regression, nonlinear regression models, or machine learning algorithms.
