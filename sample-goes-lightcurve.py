#!/usr/bin/env python

import datetime
from observational_data import *

dt = datetime.timedelta(minutes=12)

set_data_path('data')

with open("goes-data-12min.txt","w") as fp:
    for step in range(6*366*24*5):
        t = datetime.datetime(2011,1,1,0,0) + dt * step
        x = max(1e-8, get_goes_max(t, dt))
        tstr = t.strftime("%Y-%m-%dT%H:%M")
        if step % 120 == 0:
            print tstr,x
        fp.write("{} {}\n".format(tstr,x))
