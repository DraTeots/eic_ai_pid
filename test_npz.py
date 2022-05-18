import numpy as np

data = [np.arange(8).reshape(2, 4), np.arange(10).reshape(2, 5)]
np.savez('mat.npz', *data)

container = np.load('mat.npz')
data = [container[key] for key in container]
print(container)
print(data)