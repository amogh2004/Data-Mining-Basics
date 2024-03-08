#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:03:12 2024

@author: amogh
"""

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
