import numpy as np
import matplotlib.pyplot as plt

A = np.zeros((100, 100))


A[40:60, 40:60] = 1

filter = np.array([[-1,-1],[1,1]])
output = np.zeros((99,99))
for i in range(99):
    for j in range(99):
        output[i][j] = np.sum(A[i:i+2,j:j+2]*filter)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(A)
axes[1].imshow(output)
plt.show()