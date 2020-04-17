import numpy as np
import matplotlib.pyplot as plt
import time

plot_handles = []


def show_classification(x, y, w, b):
    global plot_handles

    # get indices of points labelled as 0
    I0 = np.where(y == 0)[0]
    # get indices of points labelled 1
    I1 = np.where(y == 1)[0]

    # draw separator line
    x1s = np.array([-0.5, 1.5])
    x2s = (b - w[0] * x1s) / w[1]

    # add shade to plot on one side of separator line
    x_neg = (500 + b - w[0] * x1s) / w[1]

    if not plot_handles:
        plt.close('all')
        plt.figure()
        plt.ion()
        plt.show()

        plt.plot(x[I0, 0], x[I0, 1], 'r.')
        plt.plot(x[I1, 0], x[I1, 1], 'b.')

    for ph in plot_handles:
        try:
            ph.remove()
        except ValueError:
            continue
    plot_handles = []

    ph, = plt.plot(x1s, x2s, 'k-')
    plot_handles.append(ph)

    plt.fill_between(x1s, x2s, x_neg, facecolor='blue', alpha=0.01)
    plot_handles.append(ph)

    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])

    plt.pause(0.1)
    time.sleep(0.1)


# Takes data, labels, weights and bias as inputs and returns new weights and bias
def perceptron_epoch(x, y, w, b, a=0.05):
    def h(xn, w, b):
        if xn @ np.transpose(w) > b:
            return 1
        return 0

    errors = 0
    for i in range(np.shape(x)[0]):
        e = y[i] - h(x[i], w, b)
        if e != 0:
            errors += 1
            for j in range(np.shape(x)[1]):
                dw = e * x[i][j]
                w[j] = w[j] + dw
            db = -e
            b = b + a*db

    return w, b, errors


x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

w = np.random.normal(0, 1, 2)
b = np.random.normal(0, 1, 1)
N = 50

for i in range(N):
    w, b, errors = perceptron_epoch(x, y, w, b)
    show_classification(x, y, w, b)
    if errors == 0:
        print("# iterations:", i)
        break
    if i == N-1:
        print("Classification ended without finding a proper classification")

plt.ioff()
plt.show()


