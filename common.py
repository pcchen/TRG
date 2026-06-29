import numpy as np

temp = 1.0
shape = (2, 2, 2, 2)
a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
c = np.cosh(1.0 / temp)
s = np.sinh(1.0 / temp)
for idx in np.ndindex(shape):
    if sum(idx) == 0:
        a[idx] = 2 * c * c
    elif sum(idx) == 2:
        a[idx] = 2 * c * s
    elif sum(idx) == 4:
        a[idx] = 2 * s * s

print(a)