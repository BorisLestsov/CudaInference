import numpy as np

A = np.arange(5*3).reshape(3, 5) + 1
B = np.ones(5*4).reshape(4, 5) * 0.5

res = A.dot(B.T)
print(res)
