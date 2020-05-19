import numpy as np
import os
import inspect
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

working_dir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    ))
npzfile = np.load(os.path.join(working_dir, "t8dataset1.npz"))
x = npzfile["x"]
y = npzfile["y"]


clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(12, 12, 12, 8), activation="relu", max_iter=500)

clf.fit(x, y)

yhat = clf.predict(x)

mse = np.mean(np.square(y-yhat))

print("MSE:", mse)

plt.close("all")
plt.figure()

plt.plot(y, "k.")
plt.plot(yhat, "g-")

plt.show()