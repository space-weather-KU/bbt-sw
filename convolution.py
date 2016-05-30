#!/usr/bin/env python
# -*- coding: utf-8 -*-

# chainerのconvolution関数を使ってみます
import numpy as np
import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

# (x,y)の位置にx+10*yが格納された配列imgを作ります
img = np.zeros((1,1,10,10), dtype=np.float32)
for y in range(10):
    for x in range(10):
        img[0,0,y,x] = x+10*y

# convolutionの係数はは4次元配列で定義されます
# ４つの配列添字は、それぞれ
# 「出力画像の色(channel)」「入力画像の色(channel)」「y座標」「x座標」に対応しています
w = np.zeros((1,1,3,3), dtype=np.float32)
w[0,0,0,1] = 1
w[0,0,1,0] = 100

# 上記のwを使ってconvolutionを行います
v = chainer.Variable(img)
f = L.Convolution2D(1,1,3,stride=1, initialW = w)

# convolutionを行う前と、後の配列の内容を出力してみます
print "Original data"
print v.data
print "The data after convolution"
print f(v).data
