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
for y in range(2,8):
    for x in range(2,8):
        img[0,0,y,x] = 99

w = np.zeros((3,1,2,2), dtype=np.float32)
w[0,0,0,0] =  1
w[0,0,1,0] = -1

w[1,0,0,0] =  1
w[1,0,0,1] = -1

w[2,0,0,0] =  1
w[2,0,1,0] = -1
w[2,0,0,1] = -1
w[2,0,1,1] =  1


v = chainer.Variable(img)
f = L.Convolution2D(1,3,2,stride=1, initialW = w)


print "Input:"
print v.data

print "Output:"
print f(v).data
