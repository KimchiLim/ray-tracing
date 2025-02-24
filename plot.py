import matplotlib.pyplot as plt
import numpy as np
import struct

N = 1000

A = np.zeros((N, N), dtype=np.float32)

with open('sphere.bin', 'rb') as f:
    for i in range(N):
        for j in range(N):
            A[i][j] = struct.unpack('f', f.read(4))[0]

plt.axis('off')
plt.imshow(A, cmap='gray')
plt.savefig(f'./sphere.png')
    