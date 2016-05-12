#!/usr/bin/env python

import random, sys

import matplotlib
matplotlib.use('Agg')
import pylab

import numpy as np
from PIL import Image

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

batchsize = 10
imgsize = 64


class CatDog(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self):
        super(CatDog, self).__init__(
            l1=L.Convolution2D(3,64,3,stride=2)
            l2=L.Convolution2D(64,128,3,stride=2)
            l3=L.Convolution2D(128,256,3,stride=2)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)



def load_image(fns):
    ret = np.zeros((batchsize, 3, imgsize, imgsize), dtype=np.float32)

    for j in range(batchsize):
        fn = random.choice(fns)
        with open(fn,'r') as fp:
            img=Image.open(fp).convert('RGB')
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
    return ret


def write_image(fn, img):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    pylab.imshow((img.transpose(1,2,0) + 1)/2)
    pylab.savefig(fn)



imgs = load_image(sys.argv[1:])

model = L.Classifier(CatDog())

print model.forward(imgs).shape

write_image('test.png', imgs[0])
