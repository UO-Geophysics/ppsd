#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:17:35 2023

@author: loispapin

Last time checked on Mon May 29

"""

import fcts
import warnings
import numpy as np

import datetime
from datetime import date as date_n

from matplotlib import cm
from matplotlib import mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
client = Client("IRIS")
from obspy.signal.util import prev_pow_2
from obspy.imaging.cm import obspy_sequential 

def centered_data(stream,ncomp,nsamp):
    """
    Centers seismic data around zero by removing the mean
    Args:
    - stream: seismic data stream
    - ncomp: number of components (traces) in the stream
    - nsamp: number of samples in each traces
    Returns:
    - centered_data
    """
    centered_data = np.zeros((ncomp,nsamp))
    for i,tr in enumerate(stream):
        tr.detrend()
        centered_data[i] = tr.data - np.nanmean(tr.data)
    return centered_data

######################################################################
#                               DATA                                 #
######################################################################

# Start of the data and how long
date = date_n(2014,11,29)
day  = date.timetuple().tm_yday 
day1 = day
num  = 1 #8 = 1 semaine

# Nom du fichier
sta = 'DOSE'
net = 'UW'
cha = 'EHZ'
yr  = str(date.timetuple().tm_year)

segm = 3600 #1h cut
skip_on_gaps=False

for iday in np.arange(day,day+num,dtype=int):
    
    if len(str(iday)) == 1:
        day = ('00' + str(iday))
    elif len(str(iday)) == 2:
        day = ('0' + str(iday))
    elif len(str(iday)) == 3:
        day = (str(iday))

    datebis=datetime.datetime(int(yr),1,1)+datetime.timedelta(days=int(iday-1))
    mth = str(datebis.timetuple().tm_mon)
    tod = str(datebis.timetuple().tm_mday)
    if len(str(mth)) == 1:
        mth = ('0' + str(mth))
    if len(str(tod)) == 1:
        tod = ('0' + str(tod))
        
    # Read the file
    path = "/Users/loispapin/Documents/Work/PNSN/"
    if net=='PB' or net=='UW':
        filename = (path + yr + '/Data/' + sta + '/' + sta 
                    + '.' + net + '.' + yr + '.' + day)
    elif net=='CN' or net=='NTKA':
        filename = (path + yr + '/Data/' + sta + '/' + yr + mth + 
                    tod + '.' + net + '.' + sta + '..' + cha + '.mseed')
    
    stream = read(filename)
    stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
    trace  = stream[2] #The number is the channel
    stats         = trace.stats
    network       = trace.stats.network
    station       = trace.stats.station
    channel       = trace.stats.channel
    sampling_rate = trace.stats.sampling_rate

    starttime     = UTCDateTime(datetime.datetime(int(yr),int(mth),int(tod)))
    endtime       = starttime+((24*3600)-(1/sampling_rate))
    
    starttime = starttime+(3600*20)#+(300*6)#+(27.05/60)))
    endtime   = starttime+(3600*4)
    trace=trace.trim(starttime=starttime,endtime=endtime)
    print(trace)
    fig=trace.plot()
    # ax.set_ylim(1000,-1000)
