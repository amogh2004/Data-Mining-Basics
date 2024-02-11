#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:28:47 2024

@author: amogh
"""

# QUESTION 1

import numpy as np

def col_stats(matrix):
    # Calculate sum, mean, and standard deviation for each column
    column_sums = np.round(np.sum(matrix, axis=0),4)
    column_means = np.round(np.mean(matrix, axis=0),4)
    column_stds = np.round(np.std(matrix, axis=0),4)

    # Construct dictionary to hold column-wise statistics
    stats_dict = {
        'sum': column_sums,
        'mean': column_means,
        'std_dev': column_stds
    }

    return stats_dict

# Generate a random matrix of size 10 x 10 from standard normal distribution so that we also get decimal values
random_matrix = np.random.standard_normal((10, 10))

# Test run the function with the random matrix
column_statistics = col_stats(random_matrix)

# Print column-wise statistics
print("Column-wise Statistics:")
for col_index in range(random_matrix.shape[1]):
    sum_val = column_statistics['sum'][col_index]
    mean_val = column_statistics['mean'][col_index]
    std_dev_val = column_statistics['std_dev'][col_index]
    print(f"Column {col_index + 1}: Sum={sum_val}, Mean={mean_val}, Std Dev={std_dev_val}")
    
    
    
# QUESTION 2

def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True

def prime_list(number):
    primes = []
    for num in range(2, number + 1):
        if is_prime(num):
            primes.append(num)
    return primes


n = int(input('Enter a number until which you need prime numbers: '))
primes_up_to = prime_list(n)
print(f'Prime Numbers until {n}: ', primes_up_to)

# Test the prime_list function with number=10,000
primes_up_to_10000 = prime_list(10000)
# print('Prime Numbers until 10,000: ", primes_up_to_10000)
print("Number of prime numbers up to 10,000:", len(primes_up_to_10000))

# Test the prime_list function with number=100,000
primes_up_to_100000 = prime_list(100000)
# print('Prime Numbers until 100,000: ", primes_up_to_100000)
print("Number of prime numbers up to 100,000:", len(primes_up_to_100000))


""" 
As the upper limit becomes larger, the computation time required to generate the list of prime numbers also increases significantly. 
This is because checking primality becomes more computationally expensive for larger numbers.
For number=10,000, there are 1229 prime numbers, whereas for number=100,000, there are 9592 prime numbers. 
This shows the increasing density of prime numbers as we move towards larger numbers.
"""



# QUESTION 3

import pandas as pd

# Read the dataset into a pandas dataframe
df = pd.read_csv("datasero3-1.csv")

# Print number of rows and columns
num_rows, num_columns = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# Print names of the columns
print("Column names:", df.columns.tolist())

# Count the number of observations where 'aom' column has value 1
num_aom_infection = df['aom'].sum()
print("Number of observations with ear infection (aom=1):", num_aom_infection)

# Create a new column 'aom.col' with the product of 'aom' and 'col' values
df['aom.col'] = df['aom'] * df['col']

# Display the first few rows of the dataframe to verify the changes
print("\nFirst few rows of the dataframe with the new column 'aom.col':\n")
print(df.head())



# QUESTION 4

import pandas as pd
from datetime import datetime

# Read the dataset into a pandas dataframe
df = pd.read_csv("datasero3-1.csv")

# Convert 'Birth Date' and 'Visit Date' columns to datetime objects
df['Birth Date'] = pd.to_datetime(df['Birth Date'])
df['Visit Date'] = pd.to_datetime(df['Visit Date'])

# Calculate the age of each subject at the time of visit
df['Age'] = (df['Visit Date'] - df['Birth Date']).dt.days

# Print out the number of observations that are older than 450 days
num_older_than_450 = (df['Age'] > 450).sum()
print("Number of observations older than 450 days:", num_older_than_450)

# Create a new column 'pre.vaccine' based on birth date
df['pre.vaccine'] = (df['Birth Date'] < datetime(2010, 1, 1)).astype(int)

# Display the first few rows of the dataframe to verify the changes
print("\nFirst few rows of the dataframe with the new column 'pre.vaccine':")
print(df.head())



# QUESTION 5

import numpy as np
import matplotlib.pyplot as plt

# Define probabilities for each side of the die
probabilities = [0.2, 0.15, 0.25, 0.15, 0.1, 0.15]

# Number of samples to draw for each experiment
num_samples = 1000

# Number of experiments to run
num_experiments = 1000

# Perform experiments and store the sample means
sample_means = []
for _ in range(num_experiments):
    # Draw samples from the uneven die
    samples = np.random.choice(np.arange(1, 7), size=num_samples, p=probabilities)
    # Calculate the mean of the samples
    sample_mean = np.mean(samples)
    # Store the sample mean
    sample_means.append(sample_mean)

# Plot the sampling distribution of the sample means
plt.hist(sample_means, bins=30, density=True, edgecolor='black')
plt.title('Sampling Distribution of Sample Means')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.grid(True)
plt.show()

"""
According to the central limit theorem, the sampling distribution of the sample means should approach a normal distribution as the sample size increases, regardless of the shape of the population distribution.
In this simulation, even though the underlying distribution of the die is uneven, the sampling distribution of the sample means tends to become approximately normal as the number of experiments increases.
The shape of the sampling distribution becomes more bell-shaped with fewer outliers as the number of experiments increases, which demonstrates the central limit theorem in action.
"""



# QUESTION 6
import pandas as pd

# Load the dataset into a pandas dataframe
df = pd.read_csv("IMDBDataset.csv")

# a. Create a list called review from the column 'review' of the dataset
review = df['review'].tolist()

# Calculate the number of reviews in the dataset
num_reviews = len(review)

# b. Create lists of positive and negative words
positive_words = ["incredible", "wonderful", "amazing", "fantastic", "awesome", "great", "excellent", "good", "superb", "brilliant"]
negative_words = ["terrible", "boring", "awful", "horrible", "bad", "disappointing", "poor", "disgusting", "worse", "dreadful"]

# c. Determine the number of positive and negative reviews
num_positive_reviews = sum(df['review'].str.contains('|'.join(positive_words), case=False))
num_negative_reviews = sum(df['review'].str.contains('|'.join(negative_words), case=False))

# Calculate the fraction of positive and negative reviews
fraction_positive = num_positive_reviews / num_reviews
fraction_negative = num_negative_reviews / num_reviews

# Compute the sentiment score
sentiment_score = fraction_positive - fraction_negative

# Display the results
print("\nNumber of reviews in the dataset:", num_reviews)
print("Number of positive reviews:", num_positive_reviews)
print("Number of negative reviews:", num_negative_reviews)
print("Fraction of positive reviews:", fraction_positive)
print("Fraction of negative reviews:", fraction_negative)
print("Sentiment score:", sentiment_score)