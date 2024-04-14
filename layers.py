import numpy as np
from functions import *
class affine_layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        y = np.dot(self.x, self.W) + self.b
        return y
    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        return dx
class sigmoid_layer:
    def __init__(self):
        self.y = None
    def forward(self, x):
        self.y = sigmoid(x)
        return self.y
    def backward(self, dout):
        dx = dout * self.y * (1.0-self.y)
        return dx
class softmax_with_loss_layer:
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        out = cross_entropy_error(self.y, self.t)
        return out
    def backward(self, dout):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y-self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size
        return dx
