#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 太陽磁場画像を取得するサンプルプログラムです

# 必要なライブラリのimport文です
import datetime, StringIO, urllib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from astropy.io import fits
from observational_data import *


for fn in sys.argv[1:]:
    fnbody,ext = os.path.splitext(fn)

    img = np.load(fn)['img']

    # とれた画像の中身、型、寸法、値域を表示します
    print img
    print type(img)
    print img.shape
    print "span before scaling: ", np.min(img), np.max(img)

    # 値の範囲をarctan関数を使って調整します
    img = np.arctan(img / 300.0)

    print "span after scaling; ", np.min(img), np.max(img)


    # 三原色(RGB)形式の画像データを作ります。
    # RGB形式では、各色の範囲を0以上1以下の数で指定します
    img3 = np.zeros((1024,1024,3), dtype=np.float32)
    img3[:,:,0] = np.minimum(1,np.maximum(-1,img))/2+0.5
    img3[:,:,1] = np.minimum(1,np.maximum(-1,img))/2+0.5
    img3[:,:,2] = np.minimum(1,np.maximum(-1,img))/2+0.5

    # 画像データを'*-gray.png'というファイル名で出力してみます
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    pylab.imshow(img3)
    pylab.savefig(fnbody + '-gray.png')
