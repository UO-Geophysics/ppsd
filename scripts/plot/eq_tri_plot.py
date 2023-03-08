#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:50:14 2023
Update  on Fri Feb 10

@author: loispapin

This script is a combination of spec_estimation_plot_yr.py by Lois Papin and 
the CNN scripts used by Amanda Thomas>

It reads and creates probabilties of picking P and S values on choosen signals
by using  CNN. By choosing a threshold, it allows to clean signals that are 
most likely an earthquake. Signals after that, are mostly (maybe only) composed
of noise. Then PPSD computations are used on that new clean signal to study the 
amplitude of a range of frequencies.

Last time checked on Fri Feb 17

"""

import os
import fcts
import pickle
import datetime
import unet_tools
import numpy as np
import pandas as pd
import run_cnn_alldata
from obspy import read
import tensorflow as tf
from matplotlib import mlab
import matplotlib.pyplot as plt
from datetime import date as date_n
from scipy.signal import find_peaks
from obspy.clients.fdsn import Client
from obspy.signal.util import prev_pow_2
from obspy import Stream, Trace, UTCDateTime

client = Client("IRIS")

"""
    Initialization of the parameters from the spec_estimation_plot_yr script
    that will help with the building of the model and the use of an eq CNN.
    
"""

# Start of the data and how long
date = date_n(2015,12,18)
day  = date.timetuple().tm_yday 
day1 = day
num  = 1 #8 = 1 semaine
timeday = np.arange(day,day+num,dtype=int)
tmp=timeday

# Period of time for computations per segm
h1 = 20; h2 = 24
timehr=np.arange(h1,h2,1,dtype=int)

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
cptday=0
newcurves=None
skip_on_gaps=False
time_error=[] #Erreur de connexion (except)
hr_out=[]
trace_out=[] #Trace with pvals or svals >= threshold
time_unv=[] #Lack of data

"""
    Initialization of the parameters from the apply_cnn_toeqs.py script.
    
"""

drop=True # you can edit this
large=0.5 # you can edit this
std=0.05 # you can edit this
shift=15 # set this
winlen=15 # leave this
epos=50 # epochs, leave this
epsilon=1e-6 # leave this
sr=100 # sample rate, leave this
nwin=int(sr*winlen) # leave this
nshift=int(sr*shift) # leave this
plots=1

# make stack
wid_sec = 15
sr = 100
# epsilon value shouldn't change
epsilon = 1e-6
thrhold=0.5

# SET MODEL FILE NAME    
model_save_file="unet_3comp_logfeat_b_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"                  
if large:
    fac=large
    model_save_file="large_"+'{:3.1f}'.format(fac)+"_"+model_save_file
if drop:
    model_save_file="drop_"+model_save_file

# BUILD THE MODEL
if drop:
    model=unet_tools.make_large_unet_drop_b(fac,sr,ncomps=3)     
else:
    model=unet_tools.make_large_unet_b(fac,sr,ncomps=3)  

# LOAD THE MODEL
model.load_weights("/Users/loispapin/Documents/Work/AI/"+model_save_file)  

"""
    Start of the figure : first calculation of pvals and svals values ; then
    taking out the hour-segments with potentials earthquakes ; then plotting
    the PPSD curves for all the data frame.
    
"""

# Create figure
fig, ax = plt.subplots() 
fig.grid = grid

# Start of the figure
plt.ioff()

# LOAD DATA TO RUN MODEL ON
for iday in timeday:
    
    if len(str(iday)) == 1:
        day = ('00' + str(iday))
    elif len(str(iday)) == 2:
        day = ('0' + str(iday))
    elif len(str(iday)) == 3:
        day = (str(iday))
        
    D_Z, D_E, D_N=run_cnn_alldata.rover_data_process('/Users/loispapin/Documents/Work/PNSN/2015/Data/'
                                                     +sta+'/'+sta+'.'+net+'.2015.'+day, 'p_and_s')
    times=D_Z.times()
    t_start = D_Z.stats.starttime
    D_Z=D_Z.data
    D_E=D_E.data
    D_N=D_N.data
    sav_data = []
    sav_data_Z = []
    sav_data_E = []
    sav_data_N = []

    # what decision value are you using
    wid_pts = wid_sec * sr
    i_st = 0
    i_ed = i_st + wid_pts
    
    while i_ed<=8640000+1:
        data_Z = D_Z[i_st:i_ed]
        data_E = D_E[i_st:i_ed]
        data_N = D_N[i_st:i_ed]
        norm_val=max(max(np.abs(data_Z)),max(np.abs(data_E)),max(np.abs(data_N)))
        #normalize data by the 3-Comps max value, this is what the model was trained
        TT = times[i_st:i_ed]
        tr_starttime = t_start + TT[0]
        tr_endtime = t_start + TT[-1]
        i_st = i_ed  #the first point of the 2nd round is same as the last point of the 1st round
        i_ed = i_st + wid_pts
        data_inp = run_cnn_alldata.ZEN2inp(data_Z,data_E,data_N,epsilon)  #normalize the data
        sav_data.append(data_inp)  # run model prediction in batch
        sav_data_Z.append(data_Z)  #save raw data (without normalized and feature engineering)
        sav_data_E.append(data_E)
        sav_data_N.append(data_N)
    #==== batch prediction, this is way faster====
    sav_data = np.array(sav_data)
    tmp_y = model.predict(sav_data) #prediction in 2d array
    pvals=tmp_y[:,:,0].ravel()
    svals=tmp_y[:,:,1].ravel()

    #### CUT POUR h1 till h2 ####

    indx=np.where((times[:-1]>=h1*3600) & (times[:-1]<=h2*3600))
    times=times[indx]
    pvals=pvals[indx]
    svals=svals[indx]
    
    #### DETECTE LES VALEURS >= threshold ####
    
    indx=np.where((pvals>=thrhold)|(svals>=thrhold))
    
    h=[]
    timehr=np.arange(h1,h2,1,dtype=int)
    # Corresponding the times to the hour-segment
    for iindx in indx[0]:
        if iindx<=1*360000: #1st hour
            h.append(0)
            continue
        elif iindx>=1*360000+1 and iindx<=2*360000: #2nd hour
            h.append(1)
            continue
        elif iindx>=2*360000+1 and iindx<=3*360000: #3rd hour
            h.append(2)
            continue
        elif iindx>=3*360000+1 and iindx<=4*360000: #4th hour
            h.append(3)
            continue
    
    # Cutting out the segments with earthquakes
    hr_out=np.unique(timehr[h])
    timehr=np.delete(timehr,h)
    print(timehr) #Hours processed
    
    # Data file
    path = "/Users/loispapin/Documents/Work/PNSN/"
    filename = (path + yr + '/Data/' + sta + '/' + sta 
                + '.' + net + '.' + yr + '.' + day)
    
    try: #if stream is empty or the wanted hours are missing
        # 1 day 
        stream = read(filename)
        stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
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
    channel       = trace.stats.channel
    starttime     = trace.stats.starttime
    endtime       = trace.stats.endtime
    sampling_rate = trace.stats.sampling_rate
    
    # Keeping the trace with supposed earthquakes in trace_out
    for ihourout in hr_out:
        starttimeout=starttime+(3600*ihourout)
        endtimeout=starttimeout+segm
        copy=trace.copy()
        copy.trim(starttime=starttimeout,endtime=endtimeout)
        trace_out.append(copy)
        # print(trace_out)
    
    for ihour in timehr:

        # Cut of the data on choosen times
        starttimenew = starttime+(3600*ihour)
        endtimenew   = starttimenew+segm
        
        try: #if stream is empty or the wanted hours are missing
            # 1 hour
            stream = read(filename,starttime=starttimenew,endtime=endtimenew)
            stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
            # Choix de la composante (channel)
            cpttr=0
            while stream[cpttr].stats.channel!=cha:
                cpttr+=1
            trace = stream[cpttr]
        except:
            name=net+'.'+sta+'..'+cha+'.'+day
            time_unv.append(name)
            break
        
        # First calculated time (need for title)
        if beg==None:
            beg=starttimenew
        
        print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
    
        iid = "%(network)s.%(station)s..%(channel)s" % stats 
        try: #Except for a XLMSyntaxError
            metadata = client.get_stations(network=network,station=station,
                                           starttime=starttimenew,endtime=endtimenew,level='response')
        except: #Keep the trace with error
            time_error.append(trace)
    
        # FFT calculations
        nfft=prev_pow_2((ppsd_length*sampling_rate)/4.0)
        nlap=int(0.75*nfft)            
        leng=int(sampling_rate*ppsd_length)
        _,freq=mlab.psd(np.ones(leng),nfft,sampling_rate,noverlap=nlap) 
        freq=freq[1:]
        psd_periods=1.0/freq[::-1]
    
        period_binning = fcts.setup_period_binning(psd_periods,
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
        
        # merge depending on skip_on_gaps
        skip_on_gaps= False
        stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
    
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
    
        # Calculation of the histogram used for the plots
        selected = fcts.stack_selection(current_times_all_details, times_processed,
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
        
        # Last calculated time (need for title)
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
        xedges=1.0/period_xedges
        x=np.linspace(min(xedges),max(xedges),sz)
        plot=plt.plot(x,curves,c='lightgrey')
        
        # Curves stock
        if iday==day1:
            cpthr=0
        else:
            cpthr=cptday*4
        for itime in timehr:
            if itime==ihour:
                break
            else:
                cpthr+=1
        if ihour==timehr[0] and iday==day1: #1st time
            newcurves=np.zeros((sz,4*num))
            newcurves[:,0]=curves
        else:
            newcurves[:,cpthr]=curves
    cptday+=1

"""
    This section plots the processed data and say something if no data was
    calculated. This part stays as print but goes for logging in talapas.
    
"""

if newcurves is None:
    print('No data to plot (see var newcurves)')
    print('Name of the segments which the data were unavailable : '+str(time_unv))
else:
    print('Number of segments with earthquakes taking out : '+str(len(trace_out)))
    print('Number of segments plotted : '+str((num*(h2-h1)-len(trace_out)))+' out of '+str(num*(h2-h1)))
    if time_unv!=[]:
        print('Name of the segments which the data were unavailable : '+str(time_unv))
    
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
    
    if (h1-12)<=12:
        t='pm'
    else:
        t='am'
    title = "%s   %s--%s   (from %s to %s %s) "
    title = title % (iid,beg.date,(end-1).date,
                      np.abs(h1-12),np.abs(h2-12),t)
    ax.set_title(title)
    
    # Show the figure
    plt.ion()
    plt.savefig(f'{net}.{sta}.{cha}_fig.jpg', dpi=300, bbox_inches='tight')
    plt.savefig('fig.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
