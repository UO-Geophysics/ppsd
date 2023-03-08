#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:31:00 2023
Update  on Thu Feb 16

@author: loispapin

Last time checked on Tue Mar  8

Updates to do : need to verify that logging and argparse is working and use of
the code on a terminal to see how it goes + modification of the paths when 
using the calculator (also in the functions)

"""

import os
import fcts
import pickle
import logging
import argparse
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

def get_args():
    """
    Extract arguments from command line
    Returns
    -------
    parse.parse_args(): dict of the arguments
    """
    
    parse = argparse.ArgumentParser(description="Noise without earthquakes")
    
    parse.add_argument('yr'      , type=int, help="Year of processing")
    parse.add_argument('mth'     , type=int, help="Month of processing")
    parse.add_argument('day'     , type=int, help="Day of processing")
    parse.add_argument('num'     , type=int, help="Number of days of processing")
    parse.add_argument('net'     , type=str, help="Station of processing")
    parse.add_argument('sta'     , type=str, help="Network of processing")
    parse.add_argument('cha'     , type=str, help="Channel of processing")
    parse.add_argument('h1'      , type=int, help="Start time hour of processing (hour)")
    parse.add_argument('h2'      , type=int, help="End time of processing (hour)")
    parse.add_argument('f1'      , type=int, help="First frequency for the range")
    parse.add_argument('f2'      , type=int, help="Last frequency for the range")
    parse.add_argument('thrhold' , type=float, help="Threshold for P and S waves picks")
    parse.add_argument('yr2'     , type=int, help="Year of comparison")
    parse.add_argument('mth2'    , type=int, help="Month of comparison")
    parse.add_argument('day2'    , type=int, help="Day of comparison")
    parse.add_argument('num2'    , type=int, help="Number of days of comparison")
    return parse.parse_args()

"""
    Initialization of the parameters from the spec_estimation_plot_yr script
    that will help with the building of the model and the use of an eq CNN.
    
"""

def main():
    
    # To replace the print ### to adjust #####################
    logging.basicConfig(
            format='[%(levelname)s] %(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO,
            handlers=[
                logging.FileHandler("out.log"),
                logging.StreamHandler()
            ]
    )
    
    args = get_args()
    ### A MODIFIER
    logging.info('----- DATA INFOS -----')
    logging.info(str(args.mth)+'-'+str(args.day)+'-'+str(args.yr)+' + '+str(args.num)+' days')
    logging.info('from '+str(args.h1)+' to '+str(args.h2)+' for '+args.net+'.'+args.sta)
    logging.info('from '+str(args.f1)+' to '+str(args.f2)+' Hz')
    
    # Days parameters
    date = date_n(args.yr,args.mth,args.day)
    day  = date.timetuple().tm_yday #Julian day
    
    # Period of time for computations per day
    timeday = np.arange(day,day+args.num,dtype=int)
    
    # Period of time for computations per segm
    timehr=np.arange(args.h1,args.h2,1,dtype=int)
    
    # Parameters 
    segm = 3600 #1h cut
    ppsd_length                    = segm 
    overlap                        = 0
    period_smoothing_width_octaves = 1.0
    period_step_octaves            = 0.0125
    db_bins                        = (-170, -110, 0.5)
    
    # Calculation on f1-f2Hz
    period_limits = (1/args.f2,1/args.f1)
    period_lim=(args.f1,args.f2)
    
    # Initialisation of the parameters
    grid=True
    beg=None #1st date
    cptday=0
    day1 = day
    newcurves=None
    skip_on_gaps=False
    time_error=[] #Erreur de connexion (except)
    hrout=[]
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
        datebis=datetime.datetime(int(args.yr),1,1)+datetime.timedelta(days=int(iday-1))
        mth = str(datebis.timetuple().tm_mon)
        tod = str(datebis.timetuple().tm_mday)
        if len(str(mth)) == 1:
            mth = ('0' + str(mth))
        if len(str(tod)) == 1:
            tod = ('0' + str(tod))
            
        D_Z, D_E, D_N=run_cnn_alldata.rover_data_process('/Users/loispapin/Documents/Work/PNSN/2015/Data/'
                                                         +args.sta+'/'+args.sta+'.'+args.net+'.2015.'+day, 'p_and_s')
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
    
        indx=np.where((times[:-1]>=args.h1*3600) & (times[:-1]<=args.h2*3600))
        times=times[indx]
        pvals=pvals[indx]
        svals=svals[indx]
        
        #### DETECTE LES VALEURS >= threshold ####
        
        indx=np.where((pvals>=args.thrhold)|(svals>=args.thrhold))
        
        h=[]
        timehr=np.arange(args.h1,args.h2,1,dtype=int)
        # Corresponding the times to the hour-segment (for 4 hours)
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
        hrout=np.unique(timehr[h])
        timehr=np.delete(timehr,h)
        logging.info(timehr) #Hours processed
        
        # Read the file #####################
        path = "/Users/loispapin/Documents/Work/PNSN/"
        if args.net=='PB' or args.net=='UW':
            filename = (path + args.yr + '/Data/' + args.sta + '/' + args.sta 
                        + '.' + args.net + '.' + args.yr + '.' + day)
        elif args.net=='CN':
            filename = (path + args.yr + '/Data/' + args.sta + '/' + args.yr + mth + 
                        tod + '.' + args.net + '.' + args.sta + '..' + args.cha + '.mseed')
        
        try: #if stream is empty or the wanted hours are missing
            # 1 day 
            stream = read(filename)
            stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
            # Choix de la composante (channel)
            cpttr=0
            while stream[cpttr].stats.channel!=args.cha:
                cpttr+=1
            trace=stream[cpttr]
        except:
            logging.error('Probleme de premiere lecture du file')
            logging.error('A verifier si pb de channel ou autre')
            name=args.net+'.'+args.sta+'..'+args.cha+'.'+day
            time_unv.append(name)
            logging.error('Break sur '+name)
            break
        
        stats         = trace.stats
        network       = trace.stats.network
        station       = trace.stats.station
        starttime     = trace.stats.starttime
        endtime       = trace.stats.endtime
        sampling_rate = trace.stats.sampling_rate
        
        # Keeping the trace with supposed earthquakes in trace_out
        for ihourout in hrout:
            starttimeout=starttime+(3600*ihourout)
            endtimeout=starttimeout+segm
            copy=trace.copy()
            copy.trim(starttime=starttimeout,endtime=endtimeout)
            trace_out.append(copy)
            # logging.info(trace_out)
        
        for ihour in timehr:
    
            # Cut of the data on choosen times
            starttimenew = UTCDateTime(datetime.datetime(int(args.yr),int(mth),int(tod),int(ihour),30))+(starttime.datetime.microsecond/1000000)
            endtimenew   = starttimenew+segm
            
            try: #if stream is empty or the wanted hours are missing
                # 1 hour
                stream = read(filename,starttime=starttimenew,endtime=endtimenew)
                stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
                # Choix de la composante (channel)
                cpttr=0
                while stream[cpttr].stats.channel!=args.cha:
                    cpttr+=1
                trace = stream[cpttr]
            except:
                name=args.net+'.'+args.sta+'..'+args.cha+'.'+args.day
                time_unv.append(name)
                break
            
            if len(trace)==0 or len(trace)<3600*sampling_rate:
                break
            
            # First calculated time (need for title)
            if beg==None:
                beg=starttimenew
            
            logging.info(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
        
            iid = "%(network)s.%(station)s..%(channel)s" % stats 
            try: 
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
            binned_psds     = []
            current_times_used            = [] 
            current_times_all_details     = []
            
            # Read the all stream by the defined segments
            t1 = trace.stats.starttime
            t2 = trace.stats.endtime
            while t1 + ppsd_length - trace.stats.delta <= t2:
                slice = trace.slice(t1, t1 + ppsd_length -
                                    trace.stats.delta)
                success = fcts.process(leng,nfft,sampling_rate,nlap,psd_periods,
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
            plot1=plt.plot(x,curves,c='lightgrey')
            
            # Curves stock for percentiles
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
                newcurves=np.zeros((sz,4*args.num))
                newcurves[:,0]=curves
            else:
                newcurves[:,cpthr]=curves
        cptday+=1
    
    """
        This section used the processed data to create the 5th and 95th percentiles
        to plot with a few prints to see here we at.
        
    """
    
    # For title
    endlast=end
    
    if plot1 is None:
        logging.info('----- PERIOD OF DATA -----')
        logging.info('No data to plot')
        logging.info('Name of some of the segments which the data were unavailable : '+str(time_unv))
    else:
        logging.info('----- PERIOD OF DATA -----')
        logging.info('Number of segments with earthquakes taking out : '+str(len(trace_out)))
        logging.info('Number of segments plotted : '+str((args.num*(args.h2-args.h1)-len(trace_out)))+' out of '+str(args.num*(args.h2-args.h1)))
        if time_unv!=[]:
            logging.info('Name of the segments which the data were unavailable : '+str(time_unv))
        
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
    
    th1=beg.datetime.hour
    th2=end.datetime.hour
    tth1='am';tth2='am'
    if th2==0:
        th2=12
        tth2='pm'
    if th1>12:
        th1=th1-12
        tth1='pm'
    if th2>12:
        th2=th2-12
        tth2='pm'
    title = "%s   %s--%s   (from %s to %s %s-%s) "
    title = title % (iid,beg.date,(end-1).date,
                      th1,th2,tth1,tth2)
    ax.set_title(title)
    
    # Show the figure
    plt.ion()
    plt.savefig(f'{args.net}.{args.sta}..{args.cha}_fig.jpg', dpi=300, bbox_inches='tight')
    plt.savefig('fig.jpg', dpi=300, bbox_inches='tight')
    
    pickle.dump(fig, open('myplot.pickle', 'wb'))
    
    """
        Here it's about what day.s we want to compare to the set of data previously
        plotted (greys plots : data ; blue plots : percentiles).
        
    """
    
    # Days parameters
    date = date_n(args.yr2,args.mth2,args.day2)
    day  = date.timetuple().tm_yday #Julian day
    
    # Period of time for computations per day
    timeday = np.arange(day,day+args.num2,dtype=int)
    
    # Period of time for computations per segm
    timehr=np.arange(args.h1,args.h2,1,dtype=int)
    
    # Initialisation of parameters
    cptday=0
    cpttrout=0
    cpttrout2=0
    
    for iday in timeday:
        
        # Load the previous figure 
        fig2 = pickle.load(open('myplot.pickle','rb'))
        ax2  = fig2.axes[0]
        
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
            stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
            # Choix de la composante (channel)
            cpttr=0
            while stream[cpttr].stats.channel!=args.cha:
                cpttr+=1
            trace=stream[cpttr]
        except:
            logging.error('Probleme de premiere lecture du file')
            logging.error('A verifier si pb de channel ou autre')
            name=args.net+'.'+args.sta+'..'+args.cha+'.'+day
            time_unv.append(name)
            logging.error('Break sur '+name)
            break
        
        stats         = trace.stats
        network       = trace.stats.network
        station       = trace.stats.station
        starttime     = trace.stats.starttime
        endtime       = trace.stats.endtime
        sampling_rate = trace.stats.sampling_rate
        
        for ihour in timehr:
    
            # Cut of the data on choosen times
            starttimenew = UTCDateTime(datetime.datetime(int(args.yr),int(mth),int(tod),int(ihour),30))+(starttime.datetime.microsecond/1000000)
            endtimenew   = starttimenew+segm
            
            try: #if stream is empty or the wanted hours are missing
                # 1 hour
                stream = read(filename,starttime=starttimenew,endtime=endtimenew)
                stream.merge(fcts.merge_method(skip_on_gaps),fill_value=0)
                # Choix de la composante (channel)
                cpttr=0
                while stream[cpttr].stats.channel!=args.cha:
                    cpttr+=1
                trace = stream[cpttr]
            except:
                name=args.net+'.'+args.sta+'..'+args.cha+'.'+args.day
                time_unv.append(name)
                break
            
            if len(trace)==0 or len(trace)<3600*sampling_rate:
                break
            
            logging.info(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
        
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
            binned_psds     = []
            current_times_used            = [] 
            current_times_all_details     = []
            
            # Read the all stream by the defined segments
            t1 = trace.stats.starttime
            t2 = trace.stats.endtime
            while t1 + ppsd_length - trace.stats.delta <= t2:
                slice = trace.slice(t1, t1 + ppsd_length -
                                    trace.stats.delta)
                success = fcts.process(leng,nfft,sampling_rate,nlap,psd_periods,
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
            
            inds = np.hstack([binned_psds[i] for i in used_indices])
            inds = db_bin_edges.searchsorted(inds, side="left") - 1
            inds[inds == -1] = 0
            inds[inds == num_db_bins] -= 1
            inds = inds.reshape((used_count, num_period_bins)).T
            
            hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
            for i, inds_ in enumerate(inds):
                hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)
                
            current_hist_stack = hist_stack
            current_times_used = used_times
            
            # Last calculated time (need for title)
            end=starttimenew
            
            # Has the trace an eq ?
            for itrace in np.arange(len(trace_out)):
                if trace.stats.station==trace_out[itrace].stats.station and trace.stats.starttime==trace_out[itrace].stats.starttime and trace.stats.endtime==trace_out[itrace].stats.endtime:
                    cpttrout+=1
                    cpttrout2=2
    
            if cpttrout2==2: #Trace not plotted
                cpttrout2=0
                continue
            else:
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
                plot2=ax2.plot(x,curves,'--r')
        
        # Date for title and name of the fig
        dayt = str(end.day)
        mtht = str(end.month)
        yrt  = str(end.year)
        if len(str(end.day))<2:
            dayt='0'+str(end.day)
        elif len(str(end.month))<2:
            mtht='0'+str(end.month)
        title = "%s   %s--%s   (from %s to %s %s-%s) \n day to compare : %s-%s-%s"
        title = title % (iid,beg.date,(endlast-1).date,
                          th1,th2,tth1,tth2,yrt,mtht,dayt)
        ax2.set_title(title)
        fig2.savefig(f'{args.net}.{args.sta}..{args.cha}_fig_.{yrt}{mtht}{dayt}.jpg', dpi=300, bbox_inches='tight')
        cptday+=1
    
    if plot2 is None:
        logging.info('----- DAY(S) OF COMPARISON -----')
        logging.info('No data to plot')
        logging.info('Name of some of the segments which the data were unavailable : '+str(time_unv))
    else:
        logging.info('----- DAY(S) OF COMPARISON -----')
        logging.info('Number of segments with earthquakes taking out : '+str(cpttrout))
        logging.info('Number of segments plotted : '+str((args.num*(args.h2-args.h1)-len(trace_out)))+' out of '+str(args.num*(args.h2-args.h1)))
        if time_unv!=[]:
            logging.info('Name of the segments which the data were unavailable : '+str(time_unv))
    
    return 0

if __name__ == '__main__':
    main()
