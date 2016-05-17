#!/usr/bin/env python

import numpy as np
import cv2, datetime, hashlib, os, urllib, StringIO
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import pylab

def get_sdo_image(wavelength,t):
    url = 'http://sdo.s3-website-us-west-2.amazonaws.com/aia{}/720s-x1024/{:04}/{:02}/{:02}/{:02}{:02}.npz'.format(wavelength, t.year, t.month, t.day, t.hour, t.minute)
    cache_fn = "/tmp/sdo-" + hashlib.sha224(url).hexdigest()
    if os.path.exists(cache_fn):
        cached = True
        with open(cache_fn, 'r') as fp:
            content = fp.read()
    else:
        cached = False
        resp = urllib.urlopen(url)
        content = resp.read()

    strio = StringIO.StringIO(content)
    ret = np.load(strio)['img']

    lo = 5.0
    hi = 125.0
    x2 = np.maximum(lo,ret)
    ret = (np.log(x2)-np.log(lo)) / (np.log(hi) - np.log(lo))


    if not cached:
        with open(cache_fn, 'w') as fp:
            fp.write(content)


    return ret

# A scaling for human perception of SDO-AIA 193 image.
# c.f. page 11 of
# http://helio.cfa.harvard.edu/trace/SSXG/ynsu/Ji/sdo_primer_V1.1.pdf
#
# AIA orthodox color table found at
# https://darts.jaxa.jp/pub/ssw/sdo/aia/idl/pubrel/aia_lct.pro
def plot_sdo_image(fn,img):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    c0 = np.minimum(1.0, img)
    c1 = np.sqrt(c0)
    c2 = c0 ** 2
    c3 = (c1 + c2/2.0)/1.5

    h,w = img.shape
    rgb = np.zeros((h,w,3), dtype=np.float32)
    rgb[:,:,0] = np.minimum(1.0, np.maximum(0.0, 3.0 * c0))
    rgb[:,:,1] = np.minimum(1.0, np.maximum(0.0, 3.0 * c0 - 1.0))
    rgb[:,:,2] = np.minimum(1.0, np.maximum(0.0, 3.0 * c0 - 2.0))
    pylab.imshow(rgb)
    pylab.savefig(fn)
    pylab.close('all')


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    mapfun = np.zeros(flow.shape, dtype= np.float32)
    mapfun[:,:,:] = flow[:,:,:]
    mapfun[:,:,0] += np.arange(w)
    mapfun[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, mapfun, None, cv2.INTER_LINEAR)
    return res


def load_image(fn):
    with open(fn, 'r') as fp:
        img=Image.open(fp).convert('RGB')

    img = np.asarray(img).astype(np.float32).transpose(2, 0, 1)
    return img[0,:,:] + img[1,:,:] + img[2,:,:]


for i in range(600):
    try:
        img = get_sdo_image(304, datetime.datetime(2011,1,1,0,36)+datetime.timedelta(hours=i))
        plot_sdo_image("sun{:03}.png".format(i), img)
        print i
    except:
        pass
exit(0)


img0 = get_sdo_image(304, datetime.datetime(2011,1,1,0,0))
img1 = get_sdo_image(304, datetime.datetime(2011,1,1,1,0))

print img0.shape
print img1.shape

flow = cv2.calcOpticalFlowFarneback(img1 * 256, img0 * 256, 0.5, 3, 15, 3, 5, 1.2, 0)
print flow.shape



counter = 0
for img in [img0, img1]:
    plot_sdo_image("sun{:02}.png".format(counter), img)
    counter += 1

img = img1
for i in range(24):
    img = warp_flow(img, flow)
    print np.max(flow)
    print np.min(flow)
    plot_sdo_image("sun{:02}.png".format(counter), img)
    newflow = np.zeros(flow.shape, dtype = np.float32)
    newflow[:,:,0] = warp_flow(flow[:,:,0], flow)
    newflow[:,:,1] = warp_flow(flow[:,:,1], flow)
    flow = newflow
    counter += 1
