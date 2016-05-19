#!/usr/bin/env python

import datetime, StringIO, urllib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab


def get_image(wavelength,t):
    url = 'http://sdo.s3-website-us-west-2.amazonaws.com/aia{}/720s-x1024/{:04}/{:02}/{:02}/{:02}{:02}.npz'.format(wavelength, t.year, t.month, t.day, t.hour, t.minute)
    resp = urllib.urlopen(url)
    strio = StringIO.StringIO(resp.read())
    return np.load(strio)

t = datetime.datetime(2011,1,1,0,0)
print t
img = get_image(171, t)['img']

print type(img)
print img.shape

pylab.rcParams['figure.figsize'] = (6.4,6.4)
pylab.clf()
pylab.imshow(img)
pylab.savefig('test-sun.png')
