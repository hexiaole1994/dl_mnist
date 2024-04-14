import tkinter as tk
from tkinter import filedialog
from mnist import load_mnist
from PIL import Image, ImageDraw, ImageFont
from network import *

def open_file():
    r = tk.Tk()
    r.withdraw()
    fpath = filedialog.askopenfilename()
    return fpath

def img_preprocess(fpath):
    img = Image.open(fpath)
    x = np.array(img)
    x = x.reshape(1, 784)
    x = x.astype('float32')
    x /= 255.0
    return x

(train_img, train_label), (test_img, test_label) = load_mnist(normalize=True)
nw = network(input_size=784, hidden_size1=100, hidden_size2=50, output_size=10)
nw.sgd(train_img, train_label, test_img, test_label, it_num=100000, showacc=False)

while 1:
    fpath = open_file()
    if not fpath:
        break
    x = img_preprocess(fpath)
    y = nw.predict(x)
    print(fpath + " is " + str(np.argmax(y)))




