'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
from pulp import LpMaximize, LpProblem, LpVariable, value

# Define the problem
model = LpProblem("Maximize_Profit", LpMaximize)

# Decision variables
A = LpVariable("Product_A", lowBound=0, cat='Integer')
B = LpVariable("Product_B", lowBound=0, cat='Integer')

# Objective function
model += 30 * A + 50 * B, "Total_Profit"

# Constraints
model += 3 * A + 2 * B <= 120, "Machine_Hours"
model += 2 * A + 4 * B <= 100, "Labor_Hours"

# Solve the problem
model.solve()

# Output the results
print(f"Produce {A.varValue} units of Product A")
print(f"Produce {B.varValue} units of Product B")
print(f"Maximum Profit: ${value(model.objective)}")

import numpy as np
import matplotlib.pyplot as plt

A_vals = np.linspace(0, 40, 400)
B1 = (120 - 3*A_vals) / 2     # Machine constraint
B2 = (100 - 2*A_vals) / 4     # Labor constraint

plt.plot(A_vals, B1, label='Machine Constraint')
plt.plot(A_vals, B2, label='Labor Constraint')
plt.fill_between(A_vals, np.minimum(B1, B2), alpha=0.3)
plt.xlabel('Product A')
plt.ylabel('Product B')
plt.legend()
plt.grid(True)
plt.title('Feasible Region for Production')
plt.show()