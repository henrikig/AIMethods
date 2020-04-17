import numpy as np
import matplotlib.pyplot as plt
import os
import inspect


working_dir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())))

npzfile = np.load(os.path.join(working_dir, "t1dataset1.npz"))

x = npzfile['x']
y = npzfile['y']

plt.close('all')
plt.figure()

plt.plot(x, y, 'g-')
plt.legend(['polynomial'])
plt.xlabel('x')
plt.ylabel('y')

N = x.shape[0]

x2 = np.linspace(-5, 5, N)

w0, w1, w2 = -2, 0.4, 0.22
y2 = w2*x2**2 + w1*x2 + w0

plt.plot(x2, y2)
plt.show()
