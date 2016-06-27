#!/usr/bin/env python

import datetime
from observational_data import *

with open("goes-data-12min.txt","w") as fp:
    for step in range(5*366*24*5):
        t = datetime.datetime(2011,1,1,0,0) + datetime.timedelta(minutes=12*step)
        x = max(1e-8, get_goes_max(t, datetime.timedelta(hours=1)))
        tstr = t.strftime("%Y-%m-%dT%H:%M")
        if step % 120 == 0:
            print tstr,x
        fp.write("{} {}\n".format(tstr,x))
