#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:48:54 2024

@author: amogh
"""

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