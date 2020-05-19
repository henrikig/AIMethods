import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier


def data_plot(x, y=None):
    num_points, num_attributes = x.shape

    # Expecting 8x8 image
    im_height = 8
    im_width = 8

    # plt.close("all")
    figure_handle = plt.figure()
    plot_handles = []

    # Plot the first 16 images
    n = 0
    for r in range(4):
        for c in range(4):
            if n >= num_points:
                continue

            # Reshape n'th image of 64 dimensions to 8z8 matrix
            im = x[n, :].reshape(im_height, im_width)
            n += 1
            # Add a subfigure
            ph = figure_handle.add_subplot(4, 4, n)
            # Show the 8x8 matrix as image
            plot_handles.append(ph)
            ph.imshow(im)

            ph.xaxis.set_visible(False)
            ph.yaxis.set_visible(False)

    for n in range(len(plot_handles)):
        if np.sum(y[n, :]) != 1:
            class_label = "?"
        else:
            class_label = np.argmax(y[n, :])

        plot_handles[n].set_title(class_label)

    plt.show()


def run_epoch(classification, digits, labels):
    num_points, num_attributes = digits.shape

    i = np.random.permutation(num_points)
    digits = digits[i, :]
    labels = labels[i, :]

    split = int(np.floor(num_points * 0.8))

    digits_train = digits[0:split, :]
    labels_train = labels[0:split, :]

    digits_test = digits[split:, :]
    labels_test = labels[split:, :]

    classification.fit(digits_train, labels_train)

    yhat_test = classification.predict(digits_test)

    num = digits_test.shape[0]
    errors = 0
    for n in range(num):
        if np.sum(np.abs(yhat_test[n,:] - labels_test[n, :])) > 0:
            errors += 1

    errors = float(errors)/float(num)

    accuracy = 1 - errors

    return accuracy, digits_test, yhat_test



digits = datasets.load_digits()

x = digits.data
y = digits.target

y = np.expand_dims(y, axis=1)
enc = preprocessing.OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(16, 16), activation="relu", max_iter=500)

accuracy, xtest, yhat_test = run_epoch(clf, x, y)
data_plot(xtest, yhat_test)
print(accuracy)


