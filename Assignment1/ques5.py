#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:15:48 2024

@author: amogh
"""

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