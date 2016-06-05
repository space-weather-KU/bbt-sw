#!/usr/bin/env python
# -*- coding: utf-8 -*-

from observational_data import *
import datetime, random
import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers


#

n_input = 2048
n_output = 120



class CatDog(chainer.Chain):
    def __init__(self):
        super(CatDog, self).__init__(
            l1=L.Linear(n_input,n_hidden)
            l2=L.Linear(n_hidden,n_hidden)
            l3=L.Linear(n_hidden,n_hidden)
            l4=L.Linear(n_hidden, n_output)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return F.reshape(self.l4(h3), (x.data.shape[0],2))



while True:
    t0 = datetime.datetime(2011,1,1,0,0)+datetime.timedelta(minutes = random.randrange(365*24*60))
    input_curve = np.zeros((1,1,n_input), dtype=np.float32)
    output_curve = np.zeros((1,1,n_output), dtype=np.float32)
    for i in range(n_input):
        input_curve[0,0,i] = get_goes_flux(t0 + datetime.timedelta(seconds = 720 * i))
    for i in range(n_output):
        output_curve[0,0,i] = get_goes_flux(t0 + datetime.timedelta(seconds = 720 * (i + n_output)))
    print input_curve
    print output_curve

    exit(0)
