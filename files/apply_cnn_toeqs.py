#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 16:37:26 2022

An earthquake CNN applied to earthquakes

@author: amt
"""

import gc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import unet_tools
import datetime
from obspy.core import read, Stream
from obspy import UTCDateTime
from scipy.signal import find_peaks
import run_cnn_alldata
import pandas as pd

# OPTIONS
drop=True # you can edit this
large=0.5 # you can edit this
std=0.05 # you can edit this
# shift=15 # set this
# winlen=15 # leave this
epos=50 # epochs, leave this
epsilon=1e-6 # leave this
sr=100 # sample rate, leave this
# nwin=int(sr*winlen) # leave this
# nshift=int(sr*shift) # leave this
# plots=1

# SET MODEL FILE NAME    
model_save_file="unet_3comp_logfeat_b_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"                  
if large:
    fac=large
    model_save_file="large_"+'{:3.1f}'.format(fac)+"_"+model_save_file
if drop:
    model_save_file="drop_"+model_save_file

# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=unet_tools.make_large_unet_drop_b(fac,sr,ncomps=3)     
else:
    model=unet_tools.make_large_unet_b(fac,sr,ncomps=3)  

# LOAD THE MODEL
print('Loading training results from '+model_save_file)
model.load_weights("/Users/loispapin/Documents/Work/AI/"+model_save_file)  

##### FROM HERE #####

# LOAD DATA TO RUN MODEL ON
day=2 # to modify
t1=datetime.datetime(2014,11,day,0,0,0)
# t2=datetime.datetime(2015,12,day+30,0,0,0)
station='DOSE'
network='UW'
day=int(str(t1.timetuple().tm_yday))
num  = 31 
timeday = np.arange(day,day+num,dtype=int)
tt=t1-datetime.timedelta(days=1)
h1=20; h2=24
for iday in timeday:
    if len(str(iday)) == 1:
        day = ('00' + str(iday))
    elif len(str(iday)) == 2:
        day = ('0' + str(iday))
    elif len(str(iday)) == 3:
        day = (str(iday))
        
    datebis=datetime.datetime(2015,1,1)+datetime.timedelta(days=int(iday-1))
    mth = str(datebis.timetuple().tm_mon)
    tod = str(datebis.timetuple().tm_mday)
    if len(str(mth)) == 1:
        mth = ('0' + str(mth))
    if len(str(tod)) == 1:
        tod = ('0' + str(tod))        
        
    if network=='PB' or network=='UW':
        D_Z, D_E, D_N=run_cnn_alldata.rover_data_process('/Users/loispapin/Documents/Work/PNSN/2014/Data/'
                                                         +station+'/'+station+'.'+network+'.2014.'+day, 'p_and_s')
    elif network=='CN':
        D_Z, D_E, D_N=run_cnn_alldata.rover_data_process('/Users/loispapin/Documents/Work/PNSN/2014/Data/'
                                                         +station+ '/' + '2014' + mth + tod + '.' + network + '.' + station + '..EHZ.mseed', 'p_and_s')
    
    times=D_Z.times()
    t_start = D_Z.stats.starttime
    D_Z=D_Z.data
    D_E=D_E.data
    D_N=D_N.data
    sav_data = []
    sav_data_Z = []
    sav_data_E = []
    sav_data_N = []
            
    # make stack
    wid_sec = 15
    sr = 100
    # epsilon value shouldn't change
    epsilon = 1e-6
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
    
    fig, ax=plt.subplots(1,1)
    ax.plot(times[:-1],pvals,color='tab:red')
    ax.plot(times[:-1],svals,color='tab:blue')
    ax.plot(times,D_Z/np.max(np.abs(D_Z))+3, color=(0.5,0.5,0.5), alpha=0.5)
    ax.plot(times,D_E /np.max(np.abs(D_E))+2, color=(0.5,0.5,0.5), alpha=0.5)
    ax.plot(times,D_N/np.max(np.abs(D_N))+1, color=(0.5,0.5,0.5), alpha=0.5)
    # y1=np.linspace(0.5,0.5,len(times))
    # ax.plot(times,y1,'--k')
    y2=np.linspace(0.6,0.6,len(times))
    ax.plot(times,y2,'--k')
    ax.set_xlim(h1*3600,h2*3600)
    ax.set_xticks((h1*3600,(h1+1)*3600,(h1+2)*3600,(h1+3)*3600,h2*3600))
    # ax.set_xlim((h2-1)*3600,(h2-0.5)*3600)
    # ax.set_xticks(((h2-1)*3600,(h2-0.5)*3600))
    ax.set_xticklabels(('8pm', '9pm', '10pm', '11pm', '12pm'))
    # ax.set_xticklabels(('9.5am', '10.5am', '11.5am', '12.5am', '1.5pm'))
    ax.set_ylim(-0.05,4)
    title = "%s.%s -- %s"
    tt=tt+datetime.timedelta(days=1)
    title = title % (network, station, tt.strftime("%m/%d/%Y"))
    ax.set_title(title)

    plt.savefig(f'{tt.timetuple().tm_yday}fig.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    fig.clf()
    plt.close(fig)
    gc.collect()
    del sav_data, sav_data_Z, sav_data_E, sav_data_N, D_Z, D_E, D_N
