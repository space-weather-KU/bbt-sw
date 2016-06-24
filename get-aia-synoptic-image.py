#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリのimport文です
import datetime, StringIO, urllib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from astropy.io import fits
from observational_data import *

# 2011年1月1日、00:00に相当するdatetime型の値を作ります。
#t = datetime.datetime(2011,1,1,0,0)
t = datetime.datetime(2016,5,23,23,0)
print t

# 画像データを取得します
img = get_aia_image(304, t)

print np.max(img)
print np.min(img)

img = np.minimum(1, np.maximum(0, img / 50.0))



# とれた画像の中身、型、寸法を表示します
print img
print type(img)
print img.shape

img3 = np.zeros((1024,1024,3), dtype=np.float32)
img3[:,:,0] = img
img3[:,:,1] = img
img3[:,:,2] = img

# 画像データを'test-aia.png'というファイル名で出力してみます
pylab.rcParams['figure.figsize'] = (6.4,6.4)
pylab.clf()
pylab.imshow(img3)
pylab.savefig('test-aia.png')
