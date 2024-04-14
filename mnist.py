import os.path
import urllib.request
import numpy as np
import gzip
import pickle

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

save_file = "mnist.pkl"
img_size = 784

def _download(fname):
    if os.path.exists(fname):
        return
    print("downloading " + fname)
    urllib.request.urlretrieve(url_base + fname, fname)
    print("done")

def download_mnist():
    for fname in key_file.values():
        _download(fname)

def _load_img(fname):
    print("converting " + fname + " to numpy array")
    with gzip.open(fname, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("done")
    return data

def _load_label(fname):
    print("converting " + fname + " to numpy array")
    with gzip.open(fname, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset

def _change_one_hot_lable(x):
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1
    return t

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("creating pickle file")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("done")

def load_mnist(normalize=True, flatten=True, one_hot_label=True):
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = _change_one_hot_lable(dataset[key])
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
