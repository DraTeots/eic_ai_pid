import numpy as np

a = np.array([[1,2], [3,4], [5,6]])
b = np.array([[10,20], [30,40], [50,60]])

c = np.concatenate((a,b), axis=1)

print(c)