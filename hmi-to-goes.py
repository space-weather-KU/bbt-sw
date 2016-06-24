#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 太陽磁場画像を取得するサンプルプログラムです

# 必要なライブラリのimport文です
import datetime, StringIO, urllib, random, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from astropy.io import fits
from observational_data import *

batchsize = 10

# 256^2サイズの画像から１つの値を予測するニューラルネットワークです
class Predictor(chainer.Chain):
    def __init__(self):
        super(Predictor, self).__init__(
            l1=L.Convolution2D(  1, 16,3,stride=2),
            l2=L.Convolution2D( 16, 32,3,stride=2),
            l3=L.Convolution2D( 32, 64,3,stride=2),
            l4=L.Convolution2D( 64,128,3,stride=2),
            l5=L.Convolution2D(128,256,3,stride=2),
            l6=L.Convolution2D(256,512,3,stride=2),
            l7=L.Convolution2D(512,1024,3,stride=2),
            l9=L.Linear(1024,1)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return F.reshape(self.l4(h3), (x.data.shape[0],2))



class InOutPair:
    pass


batch = []
while len(batch) < batchsize:
    # 2011年初頭から5年間のあいだでランダムな時刻tを生成します
    step = random.randrange(5*365*24)
    t = datetime.datetime(2011,1,1,0,0) + datetime.timedelta(hours=step)

    # 時刻tのHMI画像の取得を試みます
    img = get_hmi_image(t)
    if img is None:
        continue # だめだったら別のtを試す

    # 時刻、画像、GOESライトカーブなどの情報を持ったInOutPairを作ります。
    p = InOutPair()
    p.time = t
    p.hmi_img = img
    p.goes_max = max(1e-8, get_goes_max(t, datetime.timedelta(days=1)))

    p.goes_lightcurve_t = []
    p.goes_lightcurve_y = []
    t2 = t - datetime.timedelta(days=1)
    while t2 < t + datetime.timedelta(days=2):
        x2 = get_goes_flux(t2)
        if x2 is not None:
            p.goes_lightcurve_t.append(t2)
            p.goes_lightcurve_y.append(x2)
        t2 += datetime.timedelta(minutes=1)

    batch.append(p)

for p in batch:
    print p
