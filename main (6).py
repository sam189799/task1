'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
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