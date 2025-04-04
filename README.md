# Data-Mining-Basics
Data Mining Startoff Projects

## Assignment 1

1. Write a function col_stats(matrix). The function input is a matrix. The function should return the following statistics: sum, mean, and standard deviation of each column of the matrix. 
Write a test case by generating a random matrix from a standard normal distribution with size 10 x 10, and test run your function.

2. A prime number is a number that is only divisble by 1 and itself. </br>
  a. Write a function is_prime(number) that takes a number as an input and return True if the number is prime, False otherwise. </br>
  b. Write a function prime_list(number) that takes a number as input and return a list (or array) of all primes between 1 and that number. </br>
  c. Test your function prime_list() with number=10,000 or 100,000. What do you observe? </br>

3. For this exercise, please use the following file: data sero3-1.csvDownload data sero3-1.csv. The data set "data sero3.csv" is a data set collected from a children's hospital. Read the data set as a pandas data frame. Print out the number of rows and columns of the data set, and the names of the columns. The column "aom" is a column that has a value of 0 or 1. If the value is 1, that means the subject is having an ear infection. Print out the number of observations with such conditions. Create a new column with the name "aom.col" which has a value equal to the product of the aom value and col value.

4. There are 2 columns in the data set that are supposed to date time object (column "Birth date" and "Visit date"). Use the package datetime and the function to_datetime to convert these 2 columns to datetime object in Python. Calculate a new column called "Age" which is the difference between these 2 columns. Print out the number of observations that are older than 450 days. Create a new column with name "pre.vaccine" which has value "1" if the child was born before "01-01-2010", and value "0" otherwise. 

5. Write a piece of python code to simulate the central limit theorem. The distribution you are drawing from is an uneven die of 6 faces with uneven probabilities for each side. You can choose how you want to distribute the probabilities. Comment on the sampling distribution.

6. Import the file IMDB Dataset.csv Download IMDB Dataset.csvas a data frame in pandas. </br>
a. Create a list called review from the column review of the data set. How many reviews does the dataset contain. </br>
b. Create 2 lists: positive_words and negative_words. Each list contains 10 words that can be considered positive (or negative) about a movie. For example, "incredible", "wonderful" can be on the positive list. "terrible", "boring" can be on the negative list. Check if a review contains a positive word (or negative). </br>
c. Determine the number of reviews in the dataset  that can be classified as positive (negative). Which fraction of the reviews are positive (negative). The sentiment score can be calculated as the difference of the positive fraction and negative fraction. Compute the sentiment score. </br>

## Assignment 2

### Part 1

1. How many observations are there? How many features are there?
2. How many flights arrived at SFO? How many airlines fly to SFO?
3. How many missing values are there in the departure delays? How about arrival delays? Do they match? Why or why not? Remove these observations afterwards.
4. What is the average and median departure and arrival delay? What do you observe?
5. Display graphically the departure delays and arrival delays for each airline. What do you notice? Explain.
6. Now calculate the 5 number summary (min, Q1, median, Q3, max) of departure delay for each airline. Arrange it by median delay (descending order). Do the same for arrival delay.
7. Which airline has the most averaged departure delay? Give me the top 10 airlines.
8. Do you expect the departure delay has anything to do with distance of trip? What about arrival delay and distance? Prove your claims.
9. What about day of week vs departure delay?
10. If there is a departure delay (i.e. positive values for departure delay), does distance have anything to do with arrival delay? Explain. (My experience has been that longer distance flights can make up more time.)
11. Are there any seasonal (monthly) patterns in departure delays for all flights?

### Part 2

Now we want to build a model to analyze the arrival delay. We will use linear regression here.

#### Subpart I
1. Your response is ARRIVAL DELAY. First, remove all the missing data in the WEATHER DELAY column. Once you do this, there shouldn't be any more missing values in the data set (except for the cancellation reason feature). Check that. </br>
2. Build a regression model using all the observations, and the following predictors: [LATE AIRCRAFT DELAY, AIR SYSTEM DELAY, DEPARTURE DELAY , WEATHER DELAY, SECURITY DELAY, DAY OF WEEK,  DISTANCE, AIRLINE] a total of 8 predictors. </br>
3. Perform model diagnostics. What do you observe? Explain. </br>
4. Provide interpretations for a few of the coefficients, and comment on whether they make sense. </br>

#### Subpart II
1. If you have done the above steps correctly, you will notice a lot of things "doesn't seem right". We will try to fix a couple of these things here. Removing outliers: first is to remove outliers. Using the boxplot method, remove the outliers in the ARRIVAL DELAY variable. </br>
2. Refit the linear regression model, but now with log(ARRIVAL DELAY) as your response. Also, remove the non-significant predictors from the previous model (with p-values larger than 0.05). (Remember that when removing non-significant predictors one can only eliminate one variable per step, but for now we will ignore this rule and remove everything in one step.) </br>
3. Also take the log transform of a DELAY variable and the square of another DELAY variable of your choice. </br>
4. Perform model diagnostics. Did anything improve? Provide interpretations to a few of the coefficients. Do you think they make sense? Obviously there's still a lot that needs to be done. Provide a few suggestions on how we can further improve the model fit. </br>

## Assignment 3
For this homework, we will be analyzing superconductivity data for 21263 superconductors and their relevant features. The original data set can be found on the UCI machine learning data repository (https://archive.ics.uci.edu/dataset/464/superconductivty+data). This data set consists of 81 features extracted from superconductors. The goal is to use these attributes to predict the critical temperature (last column of the data set). The original data set consists of 21263 observations and 81 attributes. For this assignment, we are only using the first 5000 observations of the data set. The main task of this assignment is to create a Lasso regression model that uses the 81 predictors to predict the critical temperature.

## Assignment 4
The Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0). The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

## Assignment 5
Developing an accurate financial distress prediction model using Decision Trees, Bagging, and Boosting techniques to classify companies based on financial stability and risk.
