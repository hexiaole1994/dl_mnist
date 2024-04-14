import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x -= np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        y = y.T
        return(y)
    x -= np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    return y
def cross_entropy_error(x, t):
    delta = 1e-7
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)
    batch_size = x.shape[0]
    y = - np.sum(np.log(x+delta) * t) / batch_size
    return y

def numerical_gradient(f, x):
    h = 1e-4
    grads = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        save = x[i]
        x[i] = save - h
        y1 = f(x)
        x[i] = save + h
        y2 = f(x)
        grads[i] = (y2-y1) / (2*h)
        x[i] = save
        it.iternext()
    return grads