import numpy as np

a = np.arange(2*3*3).reshape(2, 3, 3)
print(a)
res = a.transpose(2, 1, 0)
print(res)
