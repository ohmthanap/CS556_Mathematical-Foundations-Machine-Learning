# Thanapoom Phatthanaphan
# CWID: 20011296
# CS 556-A
# Part 2: Least Square Approximation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Import csv file")
df = pd.read_csv("salary_data.csv")
print(df)

print("\nPrint the first 5 items")
print(df.head())

print("\nCreate a scatter plot")
df.plot(kind="scatter",
        x="YearsExperience",
        y="Salary",
        xlabel="YearsExperience",
        ylabel="Salary",
        figsize=(4,3),
        grid=True)
plt.show()

print("\nCompute Least Squares Approximation")
# Compute the Least Squares Approximation: finds value c and m as the equation y = c + mx
# According to the equation Ax = b,

# Finds matrix A
A = df.copy()
A.pop('Salary')
A.insert(0, '1\'s', 1)

# Finds the transpose of A
transpose_A = np.transpose(A)

# Compute to find x in the equation to get the value of c and m
x = np.linalg.inv(transpose_A.dot(A)).dot(transpose_A).dot(df['Salary'])

# From the equation y = c + mx : c = x[0] and m = x[1]
print(x)

print("\nCreate scatter plot with a line plot of Least Square Approximation")
df.plot(kind="scatter",
        x="YearsExperience",
        y="Salary",
        xlabel="YearsExperience",
        ylabel="Salary",
        figsize=(4,3),
        grid=True)

# From y = c + mx:
y = x[0] + x[1] * df['YearsExperience']
plt.plot(df['YearsExperience'], y, 'r-')
plt.show()
