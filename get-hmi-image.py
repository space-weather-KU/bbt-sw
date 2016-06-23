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

# 2016年6月23日、00:00に相当するdatetime型の値を作ります。
t = datetime.datetime(2015,06,23,0,0)
print t

# 画像データを取得します
img = get_hmi_image(t)

print np.max(img)
print np.min(img)

img = np.arctan(img / 300.0)

print np.max(img)
print np.min(img)



# とれた画像の中身、型、寸法を表示します
print img
print type(img)
print img.shape

img3 = np.zeros((1024,1024,3), dtype=np.float32)
img3[:,:,0] = np.minimum(1,np.maximum(-1,img))/2+0.5
img3[:,:,1] = np.minimum(1,np.maximum(-1,img))/2+0.5
img3[:,:,2] = np.minimum(1,np.maximum(-1,img))/2+0.5

# 画像データを'test-sun.png'というファイル名で出力してみます
pylab.rcParams['figure.figsize'] = (6.4,6.4)
pylab.clf()
pylab.imshow(img3)
pylab.savefig('test-hmi.png')
