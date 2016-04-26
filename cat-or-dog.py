#!/usr/bin/env python

import random, sys, subprocess, os.path

import matplotlib
matplotlib.use('Agg')
import pylab

import numpy as np
from PIL import Image

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

batchsize = 10
imgsize = 64

model_filename = 'cat-or-dog.model'
state_filename = 'cat-or-dog.state'


def read_command(cmd):
    stdout, stderr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return stdout

catfns = read_command('ls PetImages/Cat/*').split()
dogfns = read_command('ls PetImages/Dog/*').split()

class CatDog(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self):
        super(CatDog, self).__init__(
            l1=L.Convolution2D(3,64,3,stride=2),
            l2=L.Convolution2D(64,128,3,stride=2),
            l3=L.Convolution2D(128,256,3,stride=2),
            l4=L.Linear(256*7*7,2)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return F.reshape(self.l4(h3), (x.data.shape[0],2))



def load_image(fns = None):
    if fns is not None:
        my_batchsize = len(fns)
    else:
        my_batchsize = batchsize

    ret = np.zeros((my_batchsize, 3, imgsize, imgsize), dtype=np.float32)
    ret_ans = np.zeros((my_batchsize), dtype=np.int32)


    for j in range(my_batchsize):
        ans = np.random.randint(2)
        img = None
        if fns is not None:
            with open(fns[j],'r') as fp:
                img=Image.open(fp).convert('RGB')
            ans=-1
        else:
            while True:
                try:
                    if ans==0:
                        fn = random.choice(catfns)
                    else:
                        fn = random.choice(dogfns)

                    with open(fn,'r') as fp:
                        img=Image.open(fp).convert('RGB')
                    break
                except:
                    continue

        img=img.rotate(np.random.random()*20.0-10.0, Image.BICUBIC)
        w,h=img.size
        scale = 80.0/min(w,h)*(1.0+0.2*np.random.random())
        img=img.resize((int(w*scale),int(h*scale)),Image.BICUBIC)

        img = np.asarray(img).astype(np.float32).transpose(2, 0, 1)

        # offset the image about the center of the image.
        oy = (img.shape[1]-imgsize)/2
        ox = (img.shape[2]-imgsize)/2
        oy=oy/2+np.random.randint(oy)
        ox=ox/2+np.random.randint(ox)

        # optionally, mirror the image.
        if np.random.randint(2)==0:
            img[:,:,:] = img[:,:,::-1]

        ret[j,:,:,:] = (img[:,oy:oy+imgsize,ox:ox+imgsize]-128.0)/128.0
        ret_ans[j] = ans
    return (chainer.Variable(ret), chainer.Variable(ret_ans))


def write_image(fn, img, ans):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    pylab.imshow((img.transpose(1,2,0) + 1)/2)
    if ans == 0:
        pylab.title("cat")
    else:
        pylab.title("dog")
    pylab.savefig(fn)






model = L.Classifier(CatDog())
optimizer = optimizers.Adam()
optimizer.setup(model)


if os.path.exists(model_filename) and os.path.exists(state_filename):
    print('Load model from', model_filename, state_filename)
    serializers.load_npz(model_filename, model)
    serializers.load_npz(state_filename, optimizer)

def save():
    print('save the model')
    serializers.save_npz(model_filename, model)
    print('save the optimizer')
    serializers.save_npz(state_filename, optimizer)

def learn():
    e = 0
    while True:
        e+=1
        print e
        imgs, anss = load_image()
        optimizer.update(model, imgs, anss)

        write_image('test.png', imgs.data[0], anss.data[0])

        if e & (e-1) == 0 or e%1000 == 0:
            save()
    save()

def test():
    fns = sys.argv[1:]
    imgs, anss = load_image(fns)
    model(imgs,anss)
    print model.y.data

if len(sys.argv) > 1:
    test()
else:
    learn()
