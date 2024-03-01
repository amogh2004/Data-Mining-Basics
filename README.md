# Data-Mining-Basics
Data Mining Startoff Projects

1. Write a function col_stats(matrix). The function input is a matrix. The function should return the following statistics: sum, mean, and standard deviation of each column of the matrix. 
Write a test case by generating a random matrix from a standard normal distribution with size 10 x 10, and test run your function.

2. A prime number is a number that is only divisble by 1 and itself. </br>
  a. Write a function is_prime(number) that takes a number as an input and return True if the number is prime, False otherwise. </br>
  b. Write a function prime_list(number) that takes a number as input and return a list (or array) of all primes between 1 and that number. </br>
  c. Test your function prime_list() with number=10,000 or 100,000. What do you observe? </br>

3. For this exercise, please use the following file: data sero3-1.csvDownload data sero3-1.csv. The data set "data sero3.csv" is a data set collected from a children's hospital. Read the data set as a pandas data frame. Print out the number of rows and columns of the data set, and the names of the columns. The column "aom" is a column that has a value of 0 or 1. If the value is 1, that means the subject is having an ear infection. Print out the number of observations with such conditions. Create a new column with the name "aom.col" which has a value equal to the product of the aom value and col value.
