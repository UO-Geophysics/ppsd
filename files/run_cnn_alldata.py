#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:47:30 2021

Comb through data and run generator

@author: amthomas
Modified by loispapin
"""

import fcts
import glob
import os
import pandas as pd
import obspy
from obspy import UTCDateTime
import numpy as np
import unet_tools
import warnings
import datetime
import statistics
from scipy.signal import find_peaks

def check_sr(D, drop=True):
    tmp=np.zeros(len(D))
    for ii in range(len(D)):
        tmp[ii]=D[ii].stats.sampling_rate
    if np.min(tmp)!=np.max(tmp):
        try:
            mode_sr=statistics.mode(tmp)
        except:
            mode_sr=int(np.round(tmp[0]))
        if drop:
            for trace in D:
                if trace.stats.sampling_rate != mode_sr:
                    D.remove(trace)
        else:
            for trace in D:
                if trace.stats.sampling_rate != mode_sr:
                    trace.resample(mode_sr)
    return D

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def fill_zeros(D_Z, D_E, D_N):
    tmp0=zero_runs(D_Z.data)
    tmp1=zero_runs(D_E.data)
    tmp2=zero_runs(D_N.data)
    # EAST COMP
    for ii in range(len(tmp1)):
        if tmp1[ii][1]-tmp1[ii][0] > 1:
            if np.sum(np.abs(D_Z.data[tmp1[ii][0]:tmp1[ii][1]]))>0:
                D_E.data[tmp1[ii][0]:tmp1[ii][1]]=D_Z.data[tmp1[ii][0]:tmp1[ii][1]]    
            else:
                D_E.data[tmp1[ii][0]:tmp1[ii][1]]=D_N.data[tmp1[ii][0]:tmp1[ii][1]]     
    # NORTH COMP
    for ii in range(len(tmp2)):
        if tmp2[ii][1]-tmp2[ii][0] > 1:
            if np.sum(np.abs(D_Z.data[tmp2[ii][0]:tmp2[ii][1]]))>0:
                D_N.data[tmp2[ii][0]:tmp2[ii][1]]=D_Z.data[tmp2[ii][0]:tmp2[ii][1]]    
            else:
                D_N.data[tmp2[ii][0]:tmp2[ii][1]]=D_E.data[tmp2[ii][0]:tmp2[ii][1]]        
    # VERTICAL COMP
    for ii in range(len(tmp0)):
        if tmp0[ii][1]-tmp0[ii][0] > 1:
            if np.sum(np.abs(D_E.data[tmp0[ii][0]:tmp0[ii][1]]))>0:
                D_Z.data[tmp0[ii][0]:tmp0[ii][1]]=D_E.data[tmp0[ii][0]:tmp0[ii][1]]
            else:
                D_Z.data[tmp0[ii][0]:tmp0[ii][1]]=D_N.data[tmp0[ii][0]:tmp0[ii][1]]
    return D_Z, D_E, D_N
    
def rover_data_process(filePath,decttype,sampl=100):
    '''
        load and process daily .mseed data downloaded with rover
        filePath: absolute path of .mseed file.
        sampl: sampling rate
        other processing such as detrend, taper, filter are hard coded in the script, modify them accordingly
    '''
    files=glob.glob(filePath+'*')
    # print(files)
    skip_on_gaps=False
    D=obspy.Stream()
    for file in files:
        D += obspy.read(file)
        try:
            # D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
            D.merge(fcts.merge_method(skip_on_gaps),fill_value=0) 
        except:
            D=check_sr(D)
            # D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')   
            D.merge(fcts.merge_method(skip_on_gaps),fill_value=0) 
    #t1 = D[0].stats.starttime
    #t2 = D[0].stats.endtime
    # t1, t2 are not necessarily a whole day. Get the t1,t2 from file name instead
    t1 = UTCDateTime(D[0].stats.starttime.year,D[0].stats.starttime.month,D[0].stats.starttime.day)
    t2 = t1 + 86400
    if decttype=='lfe':
        D.detrend('linear')
        D.taper(0.02)
        D.filter('highpass',freq=1.0)
        D.trim(starttime=t1-1, endtime=t2+1, nearest_sample=True, pad=True, fill_value=0)
        D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        D.trim(starttime=t1, endtime=t2)   
    elif decttype=='plfe':
        D.detrend(type='simple')
        D.filter('highpass',freq=1.0,zerophase=True)
        D.trim(starttime=t1-1, endtime=t2+1, nearest_sample=True, pad=True, fill_value=0)
        D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        D.trim(starttime=t1, endtime=t2)        
    else:
        D.trim(starttime=t1-1, endtime=t2+1, nearest_sample=True, pad=True, fill_value=0)
        D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        D.detrend()   
        D.trim(starttime=t1, endtime=t2)
    # D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    print(D)
    # print(D[0].stats.network)
    if len(D)==2 and D[0].stats.channel[-1] == D[1].stats.channel[-1]:
        if len(np.where(D[1].data==0)[0]) > len(np.where(D[0].data==0)[0]):
            D.pop(1)
        else:
            D.pop(0)
    if len(D)==1:
        D_Z=D[0].copy()
        D_E=D[0].copy()
        D_N=D[0].copy()
    if D[0].stats.network=="BP":
        for ii in range(len(D)):
            if D[ii].stats.channel[-1]=='1':
                D_Z=D[ii].copy()
            if D[ii].stats.channel[-1]=='2':
                D_E=D[ii].copy()
            if D[ii].stats.channel[-1]=='3':
                D_N=D[ii].copy()     
    else:
        for ii in range(len(D)):
            if D[ii].stats.channel[-1]=='Z':
                D_Z=D[ii].copy()
            if D[ii].stats.channel[-1]=='E' or D[ii].stats.channel[-1]=='1':
                D_E=D[ii].copy()
            if D[ii].stats.channel[-1]=='N' or D[ii].stats.channel[-1]=='2':
                D_N=D[ii].copy()    
    # check one last time that you have all variables assigned
    try:
        D_N
    except:
        D_N=D_E
    try:
        D_E
    except:
        D_E=D_N
    try:
        D_Z
    except:
        D_Z=D_E
    return D_Z, D_E, D_N

def ZEN2inp(Z,E,N,epsilon):
    # convert raw ZEN data to input feature
    data_Z_sign = np.sign(Z)
    data_E_sign = np.sign(E)
    data_N_sign = np.sign(N)
    data_Z_val = np.log(np.abs(Z)+epsilon)
    data_E_val = np.log(np.abs(E)+epsilon)
    data_N_val = np.log(np.abs(N)+epsilon)
    data_inp = np.hstack([data_Z_val.reshape(-1,1),data_Z_sign.reshape(-1,1),
                          data_E_val.reshape(-1,1),data_E_sign.reshape(-1,1),
                          data_N_val.reshape(-1,1),data_N_sign.reshape(-1,1),])
    return data_inp

def QC(data,Type='data'):
    '''
        quality control of data
        return true if data pass all the checking filters, otherwise false
        '''
    #nan value in data check
    if np.isnan(data).any():
        return False
    #if they are all zeros
    if np.max(np.abs(data))==0:
        return False
    # #normalize the data to maximum 1
    # data = data/np.max(np.abs(data))
    # #set QC parameters for noise or data
    # if Type == 'data':
    #     N1,N2,min_std,CC = 30,30,0.01,0.98
    # else:
    #     N1,N2,min_std,CC = 30,30,0.05,0.98
    # # std window check, std too small probably just zeros
    # wind = len(data)//N1
    # for i in range(N1):
    #     #print('data=',data[int(i*wind):int((i+1)*wind)])
    #     #print('std=',np.std(data[int(i*wind):int((i+1)*wind)]))
    #     if np.std(data[int(i*wind):int((i+1)*wind)])<min_std :
    #         return False
    return True

def get_the_job_done(file, model, stainfo, outdir, decttype, thresh):
    D_Z, D_E, D_N=rover_data_process(file,decttype)
    D_Z, D_E, D_N=fill_zeros(D_Z, D_E, D_N)
#    print(D_Z.data[:5])
#    print(D_E.data[:5])
#    print(D_N.data[:5])
    # csv file to save detection information
    net=D_Z.stats.network
    sta=D_Z.stats.station
    chn=D_Z.stats.channel[:2]
    if decttype=='obs':
        file_csv = outdir+'obs_cut_daily_%s.%s.%s.%s.%s.csv'%(net,sta,str(D_Z.stats.starttime.year),str(D_Z.stats.starttime.month),str(D_Z.stats.starttime.day))   
    if decttype=='p_and_s':
        file_csv = outdir+'msh_cut_daily_%s.%s.%s.%s.%s.csv'%(net,sta,str(D_Z.stats.starttime.year),str(D_Z.stats.starttime.month),str(D_Z.stats.starttime.day))
        pvsv_file= outdir+'pvsv_msh_cut_daily_%s.%s.%s.%s.%s.csv'%(net,sta,str(D_Z.stats.starttime.year),str(D_Z.stats.starttime.month),str(D_Z.stats.starttime.day))
    if decttype=='lfe':
        file_csv = outdir+'lfe_cut_daily_%s.%s.%s.%s.%s.csv'%(net,sta,str(D_Z.stats.starttime.year),str(D_Z.stats.starttime.month),str(D_Z.stats.starttime.day))
    if decttype=='plfe':
        file_csv = outdir+'plfe_cut_daily_%s.%s.%s.%s.%s.csv'%(net,sta,str(D_Z.stats.starttime.year),str(D_Z.stats.starttime.month),str(D_Z.stats.starttime.day))
              
    # get station loc
    try:
        stlon = stainfo[(stainfo['STATION']==sta) & (stainfo['NETWORK']==net)]['LON'].iloc[0]
        stlat = stainfo[(stainfo['STATION']==sta) & (stainfo['NETWORK']==net)]['LAT'].iloc[0]
        stdep = stainfo[(stainfo['STATION']==sta) & (stainfo['NETWORK']==net)]['ELEVATION'].iloc[0]
    except:
        stlon,stlat,stdep = -1,-1,-1 # no station location information
    
        # create output file
    OUT1 = open(file_csv,'w')
    OUT1.write('network,sta,chn,stlon,stlat,stdep,starttime,endtime,y,idx_max_y,id,phase\n')
    OUT1.close()
    OUT1 = open(file_csv,'a')
    times = D_Z.times()
    t_start = D_Z.stats.starttime
    #t_end = D_Z.stats.endtime
    assert len(D_Z.data)==len(D_E.data)==len(D_N.data)==8640001, "Check data_process!"
    #only need numpy array
    D_Z = D_Z.data
    D_E = D_E.data
    D_N = D_N.data
    
    # cut daily data into 15 s data
    #t_st = D_Z[0].stats.starttime
    wid_sec = 15
    sr = 100
    # epsilon value shouldn't change
    epsilon = 1e-6
    # what decision value are you using
    wid_pts = wid_sec * sr
    i_st = 0
    i_ed = i_st + wid_pts
    sav_data = []
    sav_data_Z = []
    sav_data_E = []
    sav_data_N = []
    #time_bef = datetime.datetime.now()
    while i_ed<=8640001:
        data_Z = D_Z[i_st:i_ed]
        data_E = D_E[i_st:i_ed]
        data_N = D_N[i_st:i_ed]
        norm_val=max(max(np.abs(data_Z)),max(np.abs(data_E)),max(np.abs(data_N)))
        #normalize data by the 3-Comps max value, this is what the model was trained
        TT = times[i_st:i_ed]
        tr_starttime = t_start + TT[0]
        tr_endtime = t_start + TT[-1]
        #tr_id = '.'.join([net,sta,chn])+'_'+tr_starttime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]
        #print('T=',TT[0],TT[-1],tr_starttime,tr_endtime)
        i_st = i_ed  #the first point of the 2nd round is same as the last point of the 1st round
        i_ed = i_st + wid_pts
        # dealing with data
        #DD = np.concatenate([data_Z,data_E,data_N])
        #data_inp = ZEN2inp(data_Z,data_E,data_N,epsilon)
        if decttype=='lfe':
            data_inp = ZEN2inp(data_Z/norm_val,data_E/norm_val,data_N/norm_val,epsilon)
        else:
            data_inp = ZEN2inp(data_Z,data_E,data_N,epsilon)  #normalize the data
        sav_data.append(data_inp)  # run model prediction in batch
        sav_data_Z.append(data_Z)  #save raw data (without normalized and feature engineering)
        sav_data_E.append(data_E)
        sav_data_N.append(data_N)

    #==== batch prediction, this is way faster====
    sav_data = np.array(sav_data)
#    print(sav_data)
    tmp_y = model.predict(sav_data) #prediction in 2d array
    if decttype=='obs':
        pvals=tmp_y.ravel()
        svals=np.zeros_like(pvals)
    if decttype=='lfe' or decttype=='plfe':
        svals=tmp_y.ravel()
        pvals=np.zeros_like(svals)
    if decttype=='p_and_s':
        pvals=tmp_y[:,:,0].ravel()
        svals=tmp_y[:,:,1].ravel()
    OUT2 = open(pvsv_file,'w')
    OUT2.write('pvals,svals\n')
    for pv, sv in zip(pvals,svals):
        OUT2.write('%.6f,%.6f\n'%(pv,sv))
    OUT2.close()
    spk=find_peaks(svals, height=thresh, distance=100)
    ppk=find_peaks(pvals, height=thresh, distance=100)
    sdects=np.hstack((spk[0].reshape(-1,1)/sr,spk[1]['peak_heights'].reshape(-1,1),np.zeros((len(spk[0]),1))))
    pdects=np.hstack((ppk[0].reshape(-1,1)/sr,ppk[1]['peak_heights'].reshape(-1,1),np.ones((len(ppk[0]),1))))
    dects=np.concatenate((pdects,sdects))
    dects=dects[np.argsort(dects[:,0]),:] # contains dectection time as second of day, amplitude, and phase
    # get each lfe info
    for ii in range(dects.shape[0]):
        tmp_b=dects[ii,1] # detection amplitude
        i_win=dects[ii,0]//(wid_sec) # window id
        i_win=int(i_win)
        idx_maxy=int(np.round(sr*dects[ii,0]))
        # QC check first
        D_merge = np.hstack([sav_data_Z[i_win],sav_data_E[i_win],sav_data_N[i_win]])
        D_merge = D_merge/np.max(np.abs(D_merge)) #normalized
        if not QC(D_merge):
            continue # reject 
        # find the maximum 
        if dects[ii,2]==0:
            phase='S'
        else:
            phase='P'
        #get time from index and max y for each trace
        tr_starttime = t_start + i_win*wid_sec 
        tr_endtime = t_start + (i_win+1)*wid_sec - (1.0/sr)
        lfe_time = t_start + dects[ii,0]
        tr_id = '.'.join([net,sta,chn])+'_'+lfe_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4] #trace id
        OUT1.write('%s,%s,%s,%.4f,%.4f,%f,%s,%s,%.2f,%d,%s,%s\n'%(net,sta,chn,stlon,stlat,stdep,tr_starttime,tr_endtime,tmp_b,idx_maxy,tr_id,phase))

    #save result every sta,daily
    OUT1.close()
    return None
    
# ---------------------- SCRIPT START ------------------------------
    
warnings.filterwarnings("ignore",message=r"Passing",category=FutureWarning)

# ---------------------- SETUP PARALLELIZATION ---------------------- 

decttype='p_and_s'
thresh=0.1
sr=100
drop=False
## get all files associated with particular network
files=glob.glob('/Users/amt/Documents/cascadia_data_mining/MSH/data/*')
stainfo=pd.read_csv('stations_masterfile.dat',delim_whitespace=True)

# --- model parameters

if decttype=='p_and_s':
    fac=0.5
    model_path='/Users/loispapin/Documents/Work/AI/drop_large_0.5_unet_3comp_logfeat_b_eps_50_sr_100_std_0.05.tf'
    # model_path='/Users/amt/Documents/cascadia_data_mining/MSH/result_files/large_0.5_msh_3comp_logfeat_b_eps_50_sr_100_std_0.05.tf'
    # model_path='/home/lpapin/Work/drop_large_0.5_unet_3comp_logfeat_b_eps_50_sr_100_std_0.05.tf'
# --- build model artechitecture

if decttype=='p_and_s':
    if drop:
        model=unet_tools.make_large_unet_drop_b(fac,sr,ncomps=3)  
        outdir='/Users/amt/Documents/cascadia_data_mining/MSH/detections/drop_dects/'
    else:
        model=unet_tools.make_large_unet_b(fac,sr,ncomps=3)    
        outdir='/Users/amt/Documents/cascadia_data_mining/MSH/detections/no_drop_dects/'

# --- load weights

model.load_weights(model_path)
for file in files:
    get_the_job_done(file,model,stainfo,outdir,decttype,thresh)
