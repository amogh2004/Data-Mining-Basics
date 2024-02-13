#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:38:23 2024

@author: amogh
"""

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