#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 太陽磁場画像を取得するサンプルプログラムです

# 必要なライブラリのimport文です
import datetime, math,os, random,scipy.ndimage, StringIO, sys, urllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import pylab
from astropy.io import fits
from observational_data import *

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

# パラメータ群
batchsize = 10
input_size = 72

workdir = 'goes-to-goes'
try:
    os.mkdir(workdir)
except Exception as e:
    pass

model_filename = workdir + '/model.save'
state_filename = workdir + '/state.save'




# input_size 個の値から１つの値を予測するニューラルネットワークです
class Predictor(chainer.Chain):
    def __init__(self):
        super(Predictor, self).__init__(
            l1=L.Linear(input_size,1024),
            l2=L.Linear(1024,1024),
            l3=L.Linear(1024,1024),
            l9=L.Linear(1024,1)
        )

    def __call__(self, x):
        x = F.log(x) + 13.0
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        return F.exp(self.l9(h)-13.0)

################################################################
# メインプログラム開始
################################################################

# ニューラルネットワークによるモデルと、モデルの最適化機を作ります
model = Predictor()
optimizer = optimizers.Adam(alpha = 0.0001)
optimizer.setup(model)

# セーブファイルが存在する場合は、セーブファイルから状態を読み込みます
if os.path.exists(model_filename) and os.path.exists(state_filename):
    print 'Load model from', model_filename, state_filename
    serializers.load_npz(model_filename, model)
    serializers.load_npz(state_filename, optimizer)

# 現在までの学習状態をファイルに書き出す関数です。
def save():
    print('save the model')
    serializers.save_npz(model_filename, model)
    print('save the optimizer')
    serializers.save_npz(state_filename, optimizer)
    print('saved.')




class InOutPair:
    def visualize(self, fn_base):
        goes_fn = fn_base + '-goes.png'

        pylab.rcParams['figure.figsize'] = (6.4,4.8)
        pylab.gca().set_yscale('log')
        days    = mdates.DayLocator()  # every day
        daysFmt = mdates.DateFormatter('%Y-%m-%d')
        hours   = mdates.HourLocator()
        pylab.gca().xaxis.set_major_locator(days)
        pylab.gca().xaxis.set_major_formatter(daysFmt)
        pylab.gca().xaxis.set_minor_locator(hours)
        pylab.gca().grid()
        pylab.gcf().autofmt_xdate()
        pylab.plot(self.goes_lightcurve_t, self.goes_lightcurve_y, 'b', zorder=300)

        input_t = []
        input_y = []
        for i in range(input_size):
            input_t.append(self.past_lightcurve_t[i])
            input_y.append(self.past_lightcurve_y[i])
            input_t.append(self.past_lightcurve_t[i]+datetime.timedelta(hours=1))
            input_y.append(self.past_lightcurve_y[i])

        pylab.plot(input_t, input_y, 'g', zorder=400)

        predict_t = [self.time, self.time+datetime.timedelta(days=1)]
        predict_y = [self.goes_max_predict, self.goes_max_predict]
        observe_t = [self.time, self.time+datetime.timedelta(days=1)]
        observe_y = [self.goes_max, self.goes_max]


        pylab.plot(predict_t, predict_y, color=(1,0,0), lw=1, zorder = 200,marker='o',linestyle='--')
        pylab.plot(observe_t, observe_y, color=(1,0.66,0.66), lw=2, zorder = 100)

        pylab.gca().set_xlabel('International Atomic Time')
        pylab.gca().set_ylabel(u'GOES Long[1-8A] Xray Flux')

        pylab.savefig(goes_fn)
        pylab.close('all')

def visualize_log():
    logs = []
    predicts = []
    observes = []
    with open(workdir + '/log.txt','r') as fp:
        for l in iter(fp.readline, ''):
            ws = l.split()
            t = datetime.datetime.strptime(ws[0],"%Y-%m-%dT%H:%M")
            p = float(ws[1])
            o = float(ws[2])
            predicts.append(p)
            observes.append(o)

    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.gca().set_xscale('log')
    pylab.gca().set_yscale('log')
    pylab.plot(predicts,observes, color=(1,0,0), zorder = 100,marker='.',linestyle='.',markersize=1)
    pylab.plot(predicts[-100:],observes[-100:], color=(1,0,0), zorder = 200,marker='o',markersize=5,linestyle='.')
    pylab.plot(predicts[-10:],observes[-10:], color=(1,0,0), zorder = 300,marker='o',markersize=10,linestyle='.')

    pylab.gca().set_xlabel('prediction')
    pylab.gca().set_ylabel('observation')
    pylab.gca().grid()
    pylab.savefig(workdir+'/log.png')
    pylab.close('all')


def learn():
    batch = []
    while len(batch) < batchsize:
        # 2011年初頭から5年間のあいだでランダムな時刻tを生成します
        step = random.randrange(5*365*24)
        t = datetime.datetime(2011,1,1,0,0) + datetime.timedelta(hours=step)

        # 時刻、画像、GOESライトカーブなどの情報を持ったInOutPairを作ります。
        p = InOutPair()
        p.time = t
        p.goes_max = max(1e-8, get_goes_max(t, datetime.timedelta(days=1)))

        # 予測に使う、GOESの過去３日ぶん、１時間おきのライトカーブを作ります
        p.past_lightcurve_t = []
        p.past_lightcurve_y = []
        t2 = t - datetime.timedelta(days=3)
        while t2 <= t - datetime.timedelta(hours=1):
            x2 = max(1e-8,get_goes_max(t2, datetime.timedelta(hours=1)))
            if x2 is not None:
                p.past_lightcurve_t.append(t2)
                p.past_lightcurve_y.append(x2)
            t2 += datetime.timedelta(hours=1)


        # GOES の前後１日のライトカーブを記録させます
        p.goes_lightcurve_t = []
        p.goes_lightcurve_y = []
        t2 = t - datetime.timedelta(days=1)
        while t2 < t + datetime.timedelta(days=2):
            x2 = get_goes_flux(t2)
            if x2 is not None:
                p.goes_lightcurve_t.append(t2)
                p.goes_lightcurve_y.append(x2)
            t2 += datetime.timedelta(minutes=1)

        batch.append(p)


    input = np.ndarray((batchsize,input_size), dtype=np.float32)
    for i in range(batchsize):
        input[i,:] = batch[i].past_lightcurve_y
    input_v = chainer.Variable(input)
    predict_v = model(input_v)
    predict = predict_v.data
    for i in range(batchsize):
        batch[i].goes_max_predict = predict[i,0]

    for i in range(batchsize):
        batch[i].visualize('{}/{:02}'.format(workdir,i))

    observe = np.ndarray((batchsize,1), dtype=np.float32)
    for i in range(batchsize):
        observe[i] = batch[i].goes_max
    observe_v = chainer.Variable(observe)

    def square_norm(x,y):
        return F.sum((F.log(x)-F.log(y))**2)/batchsize

    optimizer.update(square_norm, predict_v, observe_v)

    with open(workdir + '/log.txt','a') as fp:
        for p in batch:
            fp.write(' '.join([p.time.strftime("%Y-%m-%dT%H:%M"),str(p.goes_max_predict),str(p.goes_max),"\n"]))
    save()
    visualize_log()

while True:
    try:
        learn()
    except Exception as e:
        print str(e.message)
