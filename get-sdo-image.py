#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリのimport文です
import datetime, StringIO, urllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab

# 時刻tにおける、波長wavelengthの太陽画像を取得します
# SDO衛星が撮影した元データは http://sdo.gsfc.nasa.gov/data/ にあります。
#
# 村主へのメモ
# http://jsoc2.stanford.edu/data/aia/synoptic/を使った方がいいかも？
def get_image(wavelength,t):
    url = 'http://sdo.s3-website-us-west-2.amazonaws.com/aia{}/720s-x1024/{:04}/{:02}/{:02}/{:02}{:02}.npz'.format(wavelength, t.year, t.month, t.day, t.hour, t.minute)
    resp = urllib.urlopen(url)
    strio = StringIO.StringIO(resp.read())
    return np.load(strio)

# 2011年1月1日、00:00に相当するdatetime型の値を作ります。
t = datetime.datetime(2011,1,1,0,0)
print t

# 画像データを取得します
img = get_image(171, t)['img']

# とれた画像の中身、型、寸法を表示します
print img
print type(img)
print img.shape

# 画像データを'test-sun.png'というファイル名で出力してみます
pylab.rcParams['figure.figsize'] = (6.4,6.4)
pylab.clf()
pylab.imshow(img)
pylab.savefig('test-sun.png')
