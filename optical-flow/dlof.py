#!/usr/bin/env python

import numpy as np
import cv2, datetime, hashlib, os, urllib, StringIO, random, copy, scipy.ndimage
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import pylab

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.math import sum

model_filename = 'model.save'
state_filename = 'state.save'
batchsize = 25

# A physics
class Physics(chainer.Chain):
    def __init__(self):
        super(Physics, self).__init__(
            c1=L.Convolution2D(3,6,3,stride=2),
            c2=L.Convolution2D(6,12,3,stride=2),
            c3=L.Convolution2D(12,24,3,stride=2),
            d3=L.Deconvolution2D(24,12,3,stride=2),
            d2=L.Deconvolution2D(12,6,3,stride=2),
            d1=L.Deconvolution2D(6,3,4,stride=2)
        )

    def __call__(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = F.relu(self.d3(h))
        h = F.relu(self.d2(h))
        h = self.d1(h)
        return x+0.01*h

    def get_loss_func(self):
        def lf(x,y):
            self.loss = sum.sum((x-y)**2)
            return self.loss
        return lf

# Prepare the classifier model and the optimizer
model = Physics()
optimizer = optimizers.Adam()
optimizer.setup(model)

# If the save files are found, continue from the previous learning state.
if os.path.exists(model_filename) and os.path.exists(state_filename):
    print 'Load model from', model_filename, state_filename
    serializers.load_npz(model_filename, model)
    serializers.load_npz(state_filename, optimizer)

# Save the learning state to files
def save():
    print('save the model')
    serializers.save_npz(model_filename, model)
    print('save the optimizer')
    serializers.save_npz(state_filename, optimizer)


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


    return np.maximum(0,scipy.ndimage.interpolation.zoom(ret,0.25))

# A scaling for human perception of SDO-AIA 193 image.
# c.f. page 11 of
# http://helio.cfa.harvard.edu/trace/SSXG/ynsu/Ji/sdo_primer_V1.1.pdf
#
# AIA orthodox color table found at
# https://darts.jaxa.jp/pub/ssw/sdo/aia/idl/pubrel/aia_lct.pro
def plot_sdo_image(fn,img):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    c0 = np.maximum(0.0,np.minimum(1.0, img))
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




for epoch in range(99999999):
    print "EPOCH: " , epoch

    t_begin = random.randrange(600)
    train_imgs = 26*[0]
    try:
        for i in range(26):
            train_imgs[i] = get_sdo_image(304, datetime.datetime(2011,1,1,0,36)+datetime.timedelta(hours=t_begin+i))
    except Exception as e:
        print e.message
        print "dameyatta"
        continue
    h,w = train_imgs[0].shape

    train_flows = 25*[0]
    for i in range(25):
        img1 = train_imgs[i+1]
        img0 = train_imgs[i]
        train_flows[i] = cv2.calcOpticalFlowFarneback(img1 * 256, img0 * 256, 0.5, 3, 15, 3, 5, 1.2, 0)


    predict_imgs = 25 * [0]
    predict_flows = 25 * [0]
    predict_imgs[0] = copy.deepcopy(train_imgs[0])
    predict_flows[0] = copy.deepcopy(train_flows[0])

    for i in range(24):
        flow = predict_flows[i]
        predict_imgs[i+1] = warp_flow(predict_imgs[i], flow)
        newflow = np.zeros(train_flows[0].shape, dtype = np.float32)
        newflow[:,:,0] = warp_flow(flow[:,:,0], flow)
        newflow[:,:,1] = warp_flow(flow[:,:,1], flow)
        predict_flows[i+1]=newflow

        # use the model to fix the prediction
        var = np.zeros((1,3,h,w), dtype = np.float32)
        var[0, 0,:,:]   = predict_imgs[i+1]
        var[0, 1:3,:,:] = np.transpose(predict_flows[i+1], (2,0,1))
        var2 = model(chainer.Variable(var)).data
        predict_imgs[i+1] = var2[0,0,:,:]
        predict_flows[i+1] = np.transpose(var2[0,1:3,:,:], (1,2,0))

    if epoch % 100 == 0:
        for i in range(25):
            plot_sdo_image('{:06}-lag-{:02}.png'.format(epoch,i), predict_imgs[i])

    # learn
    model.zerograds()

    trains = np.zeros((batchsize,3,h,w), dtype = np.float32)
    predicts = np.zeros((batchsize,3,h,w), dtype = np.float32)
    for i in range(batchsize):
        trains[i, 0,:,:] = train_imgs[i]
        trains[i, 1:3,:,:] = np.transpose(train_flows[i], (2,0,1))
        predicts[i, 0,:,:] = predict_imgs[i]
        predicts[i, 1:3,:,:] = np.transpose(predict_flows[i], (2,0,1))

    trains = chainer.Variable(trains)
    predicts = chainer.Variable(predicts)

    mp = model(predicts)
    optimizer.update( model.get_loss_func(), mp, trains)
