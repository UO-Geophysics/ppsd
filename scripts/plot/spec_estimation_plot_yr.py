# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:47:41 2023
Update  on Wed Jan 24

@author: papin

This script does the same computation as the spec_estimation_yr.py but shows
the results in a plot (curve) form and not in a probabilistic way. 
Possibility to add the 5th & 95th percentiles.

Last time checked on Tue Mar  7

Optimization : need to find a way to have proper hours in the title of the figures

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
from datetime import date as date_n
from matplotlib import mlab

from obspy import read
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.util import prev_pow_2
from obspy.clients.fdsn import Client
client = Client("IRIS")

# Functions called in this script #Mac & Windows
runfile('/Users/loispapin/Documents/Work/PNSN/fcts.py',
        wdir='/Users/loispapin/Documents/Work/PNSN')

"""
    Read the data with the function read of the Obspy module. Identify the 
    necessary infos from it and also get the metadata of the station response.
    
"""

# Start of the data and how long
date = date_n(2015,1,1)
day  = date.timetuple().tm_yday 
day1 = day
num  = 365 #8 = 1 semaine
timeday = np.arange(day,day+num,dtype=int)

# Period of time for computations per segm
h2 = 20; h1 = 24;
timehr=np.arange(h1,h1,1,dtype=int)

# Nom du fichier
sta = 'B926'
net = 'PB'
cha = 'EHZ'
yr  = str(date.timetuple().tm_year)

# Parameters 
segm = 3600 #1h cut
ppsd_length                    = segm 
overlap                        = 0
period_smoothing_width_octaves = 1.0
period_step_octaves            = 0.0125
db_bins                        = (-170, -110, 0.5)

# Calculation on 1-10Hz
f1 = 1; f2 = 10; 
period_limits = (1/f2,1/f1)

# Initialisation of the parameters
grid=True
period_lim=(f1,f2) 
beg=None #1st date
daynull=None
cptday=0
skip_on_gaps=False
time_error=[]
time_unv=[] #Lack of data

# Create figure
fig, ax = plt.subplots() 
fig.grid = grid

# Start of the figure
plt.ioff()

for iday in timeday:
    
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
    elif net=='CN':
        filename = (path + yr + '/Data/' + sta + '/' + yr + mth + 
                    tod + '.' + net + '.' + sta + '..' + cha + '.mseed')
    
    try: #if stream is empty or the wanted hours are missing
        # 1 day 
        stream = read(filename)
        stream.merge(merge_method(skip_on_gaps),fill_value=0)
        # Choix de la composante (channel)
        cpttr=0
        while stream[cpttr].stats.channel!=cha:
            cpttr+=1
        trace=stream[cpttr]
    except:
        print('Probleme de premiere lecture du file')
        print('A verifier si pb de channel ou autre')
        name=net+'.'+sta+'..'+cha+'.'+day
        time_unv.append(name)
        print('Break sur '+name)
        break
    
    stats         = trace.stats
    network       = trace.stats.network
    station       = trace.stats.station
    sampling_rate = trace.stats.sampling_rate
    starttime     = trace.stats.starttime
    endtime       = trace.stats.endtime
    
    for ihour in timehr:

        # Cut of the data on choosen times
        starttimenew = UTCDateTime(datetime.datetime(int(yr),int(mth),int(tod),int(ihour),30))+(starttime.datetime.microsecond/1000000)
        endtimenew   = starttimenew+segm
        
        try: #if stream is empty or the wanted hours are missing
            # 1 day 
            stream = read(filename)
            stream.merge(merge_method(skip_on_gaps),fill_value=0)
            # Choix de la composante (channel)
            cpttr=0
            while stream[cpttr].stats.channel!=cha:
                cpttr+=1
            trace = stream[cpttr]
            trace=trace.trim(starttime=starttimenew,endtime=endtimenew)
        except:
            name=net+'.'+sta+'..'+cha+'.'+day
            time_unv.append(name)
            break
        
        if len(trace)==0 or len(trace)<3600*sampling_rate:
            break
        
        # First calculated time
        if beg==None:
            beg=starttimenew
        
        print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
    
        iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        try:
            metadata = client.get_stations(network=network,station=station,
                                           starttime=starttimenew,endtime=endtimenew,level='response')
        except:
            time_error.append(trace)
    
        # FFT calculations
        nfft=prev_pow_2((ppsd_length*sampling_rate)/4.0)
        nlap=int(0.75*nfft)            
        leng=int(sampling_rate*ppsd_length)
        _,freq=mlab.psd(np.ones(leng),nfft,sampling_rate,noverlap=nlap) 
        freq=freq[1:]
        psd_periods=1.0/freq[::-1]
    
        period_binning = setup_period_binning(psd_periods,
                                              period_smoothing_width_octaves,
                                              period_step_octaves,period_limits)
    
        period_xedges = np.concatenate([period_binning[1,0:1],
                                        period_binning[3,:]])
    
        period_bin_left_edges  = period_binning[0,:]
        period_bin_centers     = period_binning[2,:]
        period_bin_right_edges = period_binning[4,:]
    
        #set up the binning for the db scale
        num_bins = int((db_bins[1]-db_bins[0])/db_bins[2])
        db_bin_edges   = np.linspace(db_bins[0],db_bins[1],num_bins+1,endpoint=True)
        db_bin_centers = (db_bin_edges[:-1]+db_bin_edges[1:])/2.0
    
        # Init
        times_processed = []
        times_data      = []
        times_gaps      = []
        binned_psds     = []
        current_times_used            = [] 
        current_times_all_details     = []
    
        # save information on available data and gaps
        times_data = insert_data_times(times_data,stream)
        times_gaps = insert_gap_times (times_gaps,stream)
        
        # Read the all stream by the defined segments
        t1 = trace.stats.starttime
        t2 = trace.stats.endtime
        while t1 + ppsd_length - trace.stats.delta <= t2:
            slice = trace.slice(t1, t1 + ppsd_length -
                                trace.stats.delta)
            success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                              period_bin_left_edges,period_bin_right_edges,
                              times_processed,binned_psds,
                              metadata,iid,trace=slice)
            t1 += (1 - overlap) * ppsd_length  # advance
    
        selected = stack_selection(current_times_all_details, times_processed,
                                   starttime=starttimenew, endtime=endtimenew)
        used_indices = selected.nonzero()[0]
        used_count   = len(used_indices)
        used_times   = np.array(times_processed)[used_indices]
        num_period_bins = len(period_bin_centers)
        num_db_bins = len(db_bin_centers)
        
        inds = db_bin_edges.searchsorted(np.hstack([binned_psds[i] for i in used_indices]), side="left") - 1
        inds[inds == -1] = 0
        inds[inds == num_db_bins] -= 1
        inds = inds.reshape((used_count, num_period_bins)).T
        
        hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
        for i, inds_ in enumerate(inds):
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)
    
        current_hist_stack = hist_stack
        current_times_used = used_times
        
        # Last calculated time
        end=endtimenew
        
        # Creation of a plot representing the PPSD
        sz=len(current_hist_stack) #Size=number of frequencies
        b=np.flipud(np.transpose(current_hist_stack))
        curve=np.zeros(sz)
        # Creation of the curve
        for ib in np.linspace(0,sz-1,sz):
            indx=np.nonzero(b[:,int(ib)])
            indx=int(indx[0])
            val=db_bin_edges
            val=val[::-1]
            curve[int(ib)]=val[indx]
        curves=np.flip(curve)
        # print(curves)
        xedges=1.0/period_xedges
        x=np.linspace(min(xedges),max(xedges),sz)
        plot=plt.plot(x,curves,c='lightgrey')
        
        # Curves stock
        if iday==day1:
            cpthr=0
        else:
            cpthr=cptday*len(timehr)
        for itime in timehr:
            if itime==ihour:
                break
            else:
                cpthr+=1
        if ihour==timehr[0] and iday==day1: #1st time
            newcurves=np.zeros((sz,len(timehr)*num))
            newcurves[:,0]=curves
        else:
            newcurves[:,cpthr]=curves
    cptday+=1

# Changing column of 0 in nan for percentiles
df=pd.DataFrame(newcurves)
df.replace(0,np.nan,inplace=True)
newcurves=df.to_numpy()

# 5th & 95th percentiles
curve5 =np.zeros(sz)
curve95=np.zeros(sz)
for ip in np.linspace(0,sz-1,sz):
    curve5[int(ip)] =np.nanpercentile(newcurves[int(ip)], 5)
    curve95[int(ip)]=np.nanpercentile(newcurves[int(ip)],95)

plt.plot(x,curve5,'b',x,curve95,'b')

# Grid
color = {"color": "0.7"}
ax.grid(True, which="major", **color)
ax.grid(True, which="minor", **color)

# Axis and title
ax.set_xlabel('Frequency [Hz]')
ax.invert_xaxis()
ax.set_xlim(period_lim)

ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
ax.set_ylim(db_bin_edges[0],db_bin_edges[-1])

if np.abs(end.datetime.hour-12)>=12:
    t='pm'
else:
    t='am'
title = "%s   %s--%s   (from %s to %s %s) "
########################
# th1=beg.datetime.hour
# th2=end.datetime.hour
# if th1<
########################
title = title % (iid,beg.date,(end-1).date,
                  np.abs(beg.datetime.hour-12),
                  np.abs(end.datetime.hour-12),
                  t)
ax.set_title(title)

# Show the figure
plt.ion()
plt.savefig('fig.jpg', dpi=300, bbox_inches='tight')
plt.show()
