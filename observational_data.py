#!/usr/bin/env python
# module for loading observational data.

import astropy.time as time
import calendar, datetime, os, subprocess
from astropy.io import fits

def cmd(str):
    print str
    subprocess.call(str, shell = True)

def aia193(t):
    ymd = t.strftime('%Y/%m/%d')
    data_path = "data/aia193/" + ymd
    fn=data_path + t.strftime('/%H%M.fits')

    if not(os.path.exists(data_path)):
        cmd('aws s3 sync s3://sdo/aia193/720s/{}/ {}/ --region=us-west-2'.format(ymd,data_path))
    if not(os.path.exists(fn)):
        return None

    h = fits.open(fn)
    h[1].verify('fix')
    exptime = h[1].header['EXPTIME']
    if exptime <=0:
        print "Warning: non-positive EXPTIME: ", h[1].header['EXPTIME']
        return None

    # adjust the pixel luminosity with the exposure time.
    return h[1].data / exptime

global goes_raw_data, goes_loaded_files
goes_raw_data = {}
goes_loaded_files = set()
def goes(t0):
    global goes_raw_data, goes_loaded_files
    if t0 in goes_raw_data:
        return goes_raw_data[t0]

    day31 = calendar.monthrange(t0.year,t0.month)[1]
    fn = 'g15_xrs_1m_{y:4}{m:02}{d:02}_{y:4}{m:02}{d31:02}.csv'.format(y=t0.year, m=t0.month, d=01, d31=day31)
    localpath = 'data/goes/' + fn
    if localpath in goes_loaded_files:
        return None

    if not(os.path.exists(localpath)):
        url = 'http://satdat.ngdc.noaa.gov/sem/goes/data/new_avg/{y}/{m:02}/goes15/csv/'.format(y=t0.year, m=t0.month) + fn
        cmd('wget ' + url + ' -O ' + localpath)
    if not(os.path.exists(localpath)):
        return None

    goes_loaded_files.add(localpath)
    with (open(localpath, "r")) as fp:
        while True:
            con = fp.readline()
            if con[0:5]=='data:':
                break
        fp.readline()

        while True:
            con = fp.readline()
            if con=='':
                break
            ws = con.split(',')
            t = time.Time(ws[0]).datetime
            goes_raw_data[t] = float(ws[6])

    return goes(t)

goes_max_epoch = time.Time("2000-01-01 00:00:00").datetime

def goes_max(t, timedelta):
    i = int((t - goes_max_epoch).total_seconds())
    j = int((t - goes_max_epoch + timedelta).total_seconds())
    start=min(i,j)
    end  =max(i,j)
    delta=1;
    while delta < end:
        delta*=2
    return goes_max_for_secondrange(delta,start, end)

global goes_max_for_secondrange_memo
goes_max_for_secondrange_memo={}
def goes_max_for_secondrange(delta, start ,end):
    global goes_max_for_secondrange_memo

    box0 = delta*(int(start + delta-1)/int(delta))
    box1 = delta*(int(end)/int(delta))
    key = (start,end)


    if start >= end:
        return None
    if end-start < 60:
        t = goes_max_epoch + datetime.timedelta(seconds = (int(start)/60)*60)
        return goes(t)
    if key in goes_max_for_secondrange_memo:
        return goes_max_for_secondrange_memo[key]

    if box0 > box1:
        return goes_max_for_secondrange(delta/2, start, end)
    if box0 == box1:
        return max(goes_max_for_secondrange(delta/2, start, box0),
                   goes_max_for_secondrange(delta/2, box0, end))
    bm_mid = max(goes_max_for_secondrange(delta/2, box0, box0+delta/2)
                 ,goes_max_for_secondrange(delta/2, box0+delta/2, box1))
    goes_max_for_secondrange_memo[(box0, box1)]=bm_mid

    return  max(goes_max_for_secondrange(delta/2, start, box0),
                bm_mid,
                goes_max_for_secondrange(delta/2, box1, end))
