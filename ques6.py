#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:25:39 2024

@author: amogh
"""

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
