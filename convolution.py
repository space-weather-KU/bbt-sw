#!/usr/bin/env python
import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

img = np.zeros((1,1,10,10), dtype=np.float32)
for y in range(10):
    for x in range(10):
        img[0,0,y,x] = x+10*y

w = np.zeros((1,1,3,3), dtype=np.float32)
w[0,0,0,1] = 1
w[0,0,1,0] = 100


v = chainer.Variable(img)
f = L.Convolution2D(1,1,3,stride=1, initialW = w)



print v.data
print f(v).data
