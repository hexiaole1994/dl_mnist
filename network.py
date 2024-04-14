import numpy as np
import pickle
import matplotlib.pyplot as plt
import os.path
from layers import *
from functions import *
from collections import OrderedDict

weight_file = "weights.pkl"
acc_limit = 0.995

class network:
    def __init__(self, input_size=784, hidden_size1=100, hidden_size2=50, output_size=10, weight_init_std=0.01):
        if os.path.exists(weight_file):
            print("load params from " + weight_file)
            with open(weight_file, 'rb') as f:
                self.params = pickle.load(f)
        else:
            self.params = {}
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
            self.params['b1'] = np.zeros(hidden_size1)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
            self.params['b2'] = np.zeros(hidden_size2)
            self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
            self.params['b3'] = np.zeros(output_size)
        self.layers = OrderedDict()
        self.layers['Affine1'] = affine_layer(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = sigmoid_layer()
        self.layers['Affine2'] = affine_layer(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = sigmoid_layer()
        self.layers['Affine3'] = affine_layer(self.params['W3'], self.params['b3'])
        self.lastlayer = softmax_with_loss_layer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return(x)

    def loss(self, x, t):
        x = self.predict(x)
        y = self.lastlayer.forward(x, t)
        return(y)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        out = np.sum(y == t) / float(x.shape[0])
        return(out)

    def showacc(self, train_acc, test_acc):
        x = np.arange(len(train_acc))
        plt.plot(x, train_acc, label='train acc')
        plt.plot(x, test_acc, label='test acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

    def numerical_gradient(self, x, t):
        f = lambda w: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(f, self.params['W1'])
        grads['b1'] = numerical_gradient(f, self.params['b1'])
        grads['W2'] = numerical_gradient(f, self.params['W2'])
        grads['b2'] = numerical_gradient(f, self.params['b2'])
        grads['W3'] = numerical_gradient(f, self.params['W3'])
        grads['b3'] = numerical_gradient(f, self.params['b3'])
        return(grads)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        return grads

    def gradient_check(self, img_batch, label_batch):
        grads_nf = self.numerical_gradient(img_batch, label_batch)
        grads_bp = self.gradient(img_batch, label_batch)
        for key in grads_nf.keys():
            diff = np.average(np.abs(grads_nf[key] - grads_bp[key]))
            print(key + ": " + str(diff))

    def sgd(self, train_img, train_label, test_img, test_label, it_num=10000, batch_size=100, lr=0.1, showacc=False):
        if os.path.exists(weight_file):
            print("train had finished.")
            return
        train_size = train_img.shape[0]
        it_per_epoch = max(train_size/batch_size, 1)
        train_acc = []
        test_acc = []
        acc1 = acc2 = 0
        for i in range(it_num):
            batch_mask = np.random.choice(train_size, batch_size)
            train_batch = train_img[batch_mask]
            label_batch = train_label[batch_mask]
            grads = self.gradient(train_batch, label_batch)
            for key in self.params.keys():
                self.params[key] -= lr * grads[key]
            if i % it_per_epoch == 0:
                acc1 = self.accuracy(train_img, train_label)
                acc2 = self.accuracy(test_img, test_label)
                train_acc.append(acc1)
                test_acc.append(acc2)
                print("train_acc=%{}, test_acc=%{}“.".format(acc1*100, acc2*100))
                if acc1 > acc_limit and acc2 > acc_limit-0.2:
                    break
        if showacc:
            self.showacc(train_acc, test_acc)
        if acc1 > acc_limit and acc2 > acc_limit-0.2:
            with open(weight_file, 'wb') as f:
                pickle.dump(self.params, f, -1)