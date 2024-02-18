#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:10:51 2024

@author: amogh
"""

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