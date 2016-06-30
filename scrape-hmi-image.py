#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 太陽磁場画像を取得するサンプルプログラムです

# 必要なライブラリのimport文です
import datetime, StringIO, urllib, sys, os, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from astropy.io import fits
from observational_data import *


for dhour in range(366*24*4):
    t = datetime.datetime(2011,01,01,0,0) + datetime.timedelta(hours=dhour)
    print t

    # 画像データを取得します
    img = get_hmi_image(t)
    if img is not None:
        dir = t.strftime('data/hmi/%Y/%m/%d')
        subprocess.call(['mkdir','-p',dir])
        fn = dir + '/' + t.strftime('%H%M.npz')
        np.savez_compressed(fn,img=img)
