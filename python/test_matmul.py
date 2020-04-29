import numpy as np

A = np.arange(2*3*5*5).reshape(2, -1) + 1
B = np.ones(3*5*5*5).reshape(5, -1) * 0.5

res = A.dot(B.T)
print(res)
