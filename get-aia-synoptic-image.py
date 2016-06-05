#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリのimport文です
import datetime, StringIO, urllib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from astropy.io import fits

# 時刻tにおける、波長wavelengthの太陽画像を取得します
# SDO衛星が撮影した元データは http://sdo.gsfc.nasa.gov/data/ にあります。
#
# 村主へのメモ
# http://jsoc2.stanford.edu/data/aia/synoptic/を使った方がいいかも？
def get_aia_image(wavelength,t):
    try:
        url = 'http://jsoc2.stanford.edu/data/aia/synoptic/{:04}/{:02}/{:02}/H{:02}00/AIA{:04}{:02}{:02}_{:02}{:02}_{:04}.fits'.format(t.year, t.month, t.day,t.hour, t.year, t.month, t.day, t.hour, t.minute, wavelength)

        resp = urllib.urlopen(url)
        strio = StringIO.StringIO(resp.read())

        hdulist=fits.open(strio)
        hdulist.verify('fix')
        img=hdulist[1].data
        exptime=hdulist[1].header['EXPTIME']
        if (exptime<=0):
            sys.stderr.write("non-positive EXPTIME\n")
            return None
        img = np.where( np.isnan(img), 0.0, img)

        return img / exptime
    except Exception as e:
        sys.stderr.write(e.message)
        return None

# 2011年1月1日、00:00に相当するdatetime型の値を作ります。
t = datetime.datetime(2011,1,2,0,0)
print t

# 画像データを取得します
img = get_aia_image(304, t)

print np.max(img)
print np.min(img)

img = np.minimum(1, (np.maximum(1, img)) / (500.0))



# とれた画像の中身、型、寸法を表示します
print img
print type(img)
print img.shape

img3 = np.zeros((1024,1024,3), dtype=np.float32)
img3[:,:,0] = img
img3[:,:,1] = img
img3[:,:,2] = img

# 画像データを'test-sun.png'というファイル名で出力してみます
pylab.rcParams['figure.figsize'] = (6.4,6.4)
pylab.clf()
pylab.imshow(img3)
pylab.savefig('test-sun.png')
