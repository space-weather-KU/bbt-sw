#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 太陽磁場画像を取得するサンプルプログラムです

# 必要なライブラリのimport文です

import datetime, StringIO, urllib, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
from astropy.io import fits
from astropy import units as u

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import sunpy.map

from observational_data import *

# for h in range(24):
#     for m in range(0,60,12):
#         url = 'http://jsoc2.stanford.edu/data/aia/synoptic/2013/11/03/H{:02}00/AIA20131103_{:02}{:02}_0193.fits'.format(h,h,m)
#         subprocess.call(["wget",url,"-O","data/fits/aia0193-20131103-{:02}{:02}.fits".format(h,m)] )
#
# exit()

# aiamap = sunpy.map.Map('http://jsoc2.stanford.edu/data/aia/synoptic/2013/11/03/H0400/AIA20131103_0424_0193.fits')
# aiamap.peek()
#
# exit()


for fn in sys.argv[1:]:
    fnbody,ext = os.path.splitext(fn)

    img = sunpy.map.Map(fn)

    print img.data.shape
    print "dt = ", img.exposure_time
    print type(img.data)

    thre = 0
    img.data = np.maximum(0,img.data -thre) / (img.exposure_time / u.second)


    print np.min(img.data)
    print np.max(img.data)


    #pylab.rcParams['figure.figsize'] = (6.4,6.4)
    #pylab.clf()
    img.plot()
    plt.colorbar()
    plt.clim(0,10000)
    plt.savefig(fnbody + '.png')
