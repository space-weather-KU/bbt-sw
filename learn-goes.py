#!/usr/bin/env python
# -*- coding: utf-8 -*-

from observational_data import *
import datetime

print get_goes_flux(datetime.datetime(2013,5,5,0,0))
print get_goes_flux(datetime.datetime(2013,5,5,0,1))
print get_goes_flux(datetime.datetime(2013,5,5,0,2))
print get_goes_max(datetime.datetime(2013,5,6), datetime.timedelta(hours=24))
