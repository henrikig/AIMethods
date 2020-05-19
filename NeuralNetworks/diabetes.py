from sklearn import datasets
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

dataset = datasets.load_diabetes()

x = dataset.data
y = dataset.target


clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(16, 32, 16), activation="relu", max_iter=500)

num_points, num_attributes = x.shape

split = int(np.floor(num_points * 0.8))

xtrain = x[:split, :]
ytrain = y[:split]

xtest = x[split:, :]
ytest = y[split:]

clf.fit(xtrain, ytrain)

yhat = clf.predict(xtest)

mse = np.mean(np.square(ytest-yhat))

print(mse)

plt.close("all")
plt.figure()

plt.plot(ytest, "k.")
plt.plot(yhat, "g-")

plt.show()

