# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:47:41 2023

@author: papin
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from datetime import date as date_n
from matplotlib import mlab
from matplotlib.ticker import FormatStrFormatter

from obspy import read
from obspy import UTCDateTime
from obspy.imaging.cm import obspy_sequential 
from obspy.signal.util import prev_pow_2
from obspy.clients.fdsn import Client
client = Client("IRIS")

from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# Functions called in this script #Mac & Windows
runfile('/Users/loispapin/Documents/Work/PNSN/fcts.py',
        wdir='/Users/loispapin/Documents/Work/PNSN')
# runfile('C:/Users/papin/Documents/Spec/fcts.py', 
#         wdir='C:/Users/papin/Documents/Spec')

######################################################################
#                               DATA                                 #
######################################################################

# Start of the data and how long
date = date_n(2015,1,1)
day  = date.timetuple().tm_yday 
day1 = day
num  = 365 #8 = 1 semaine

# Nom du fichier
sta = 'B926'
net = 'PB'
yr  = str(date.timetuple().tm_year)

# Parameters 
segm = 3600 #1h cut
ppsd_length                    = segm 
overlap                        = 0
period_smoothing_width_octaves = 1.0
period_step_octaves            = 0.0125
db_bins                        = (-160, -90, 0.5)

# Calculation on 0.01-16Hz
f1 = 1; f2 = 10; 
period_limits = (1/f2,1/f1)

# Initialisation of the parameters
grid=True
period_lim=(f1,f2) 

# Create figure
fig, ax = plt.subplots() 
xlim = ax.get_xlim()
fig.grid = grid

# Start of the figure
plt.ioff()
x=np.linspace(0.99355938,10.04341567,267)
beg = None

for iday in np.arange(day,day+num,dtype=int):
    
    if len(str(iday)) == 1:
        day = ('00' + str(iday))
    elif len(str(iday)) == 2:
        day = ('0' + str(iday))
    elif len(str(iday)) == 3:
        day = (str(iday))

    # Mac
    path = "/Users/loispapin/Documents/Work/PNSN/"
    filename = (path + yr + '/Data/' + sta + '/' + sta 
                + '.' + net + '.' + yr + '.' + day)

    # # Windows
    # path = r"C:\Users\papin\Documents\Spec\Data"
    # filename = (path + "\\" + sta + "\\" + sta + '.' + net + '.' + yr + '.' + day)
    
    # 1 day 
    stream = read(filename)
    trace  = stream[2] #Composante Z
    
    if len(stream)==3:
    
        stats         = trace.stats
        network       = trace.stats.network
        station       = trace.stats.station
        channel       = trace.stats.channel
        starttime     = trace.stats.starttime
        endtime       = trace.stats.endtime
        sampling_rate = trace.stats.sampling_rate
    
        # Cut of the data on choosen times
        starttime = starttime+(3600*20)
        endtime   = starttime+segm
        stream = read(filename,starttime=starttime,endtime=endtime)
        trace  = stream[2] #Composante Z
    
        print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
    
        iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
    
        metadata = client.get_stations(network=network,station=station,
                                       starttime=starttime,endtime=endtime,level='response')
    
        # FFT calculations
        nfft=ppsd_length*sampling_rate 
        nfft=nfft/4.0                  
        nfft=prev_pow_2(nfft)          
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
        current_hist_stack            = None
        current_hist_stack_cumulative = None
        current_times_used            = [] 
        current_times_all_details     = []
    
        # Initialisation of the parameters
        verbose     = False #Show the time data computed
        skip_on_gaps= False
    
        times_data = insert_data_times(times_data,stream)
        times_gaps = insert_gap_times (times_gaps,stream)
        stream.merge(merge_method(skip_on_gaps),fill_value=0)
    
        # Read the all stream by the defined segments
        for trace in stream:
            if not sanity_check(trace,iid,sampling_rate):
                msg = "merde"
                warnings.warn(msg)
                continue
            t1 = trace.stats.starttime
            t2 = trace.stats.endtime
            if t1 + ppsd_length - trace.stats.delta > t2:
                msg = "merde"
                warnings.warn(msg)
                continue
            while t1 + ppsd_length - trace.stats.delta <= t2:
                if check_time_present(times_processed,ppsd_length,overlap,t1):
                    msg = "merde"
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    slice = trace.slice(t1, t1 + ppsd_length -
                                        trace.stats.delta)
                    success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                                      period_bin_left_edges,period_bin_right_edges,
                                      times_processed,binned_psds,
                                      metadata,iid,trace=slice)
                    if success:
                        if verbose:
                            print(t1)
                        changed = True
                t1 += (1 - overlap) * ppsd_length  # advance
    
        # Init
        if changed:
            current_hist_stack            = None
            current_hist_stack_cumulative = None
            current_times_used            = [] 
            current_times_all_details     = []
    
        selected = stack_selection(current_times_all_details, times_processed,
                                   starttime=starttime, endtime=endtime)
        used_indices = selected.nonzero()[0]
        used_count   = len(used_indices)
        used_times   = np.array(times_processed)[used_indices]
        num_period_bins = len(period_bin_centers)
        num_db_bins = len(db_bin_centers)
    
        hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
        if not used_count:
            current_hist_stack = hist_stack
            current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
            current_times_used = used_times
        inds = np.hstack([binned_psds[i] for i in used_indices])
        inds = db_bin_edges.searchsorted(inds, side="left") - 1
        inds[inds == -1] = 0
        inds[inds == num_db_bins] -= 1
        inds = inds.reshape((used_count, num_period_bins)).T
        for i, inds_ in enumerate(inds):
            # count how often each bin has been hit for this period bin,
            # set the current 2D histogram column accordingly
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)
    
        current_hist_stack = hist_stack
        current_times_used = used_times
    
        # Creation of a plot representing the PPSD
        ### only working for 1 segment 
        b=np.transpose(current_hist_stack)
        b=np.flipud(b)
        i=0
        ib=np.linspace(0,266,267)
        curve=np.zeros(len(ib))
        for i in ib:
            indx=np.nonzero(b[:,int(i)])
            indx=int(indx[0])
            val=db_bin_edges
            val=val[::-1]
            curve[int(i)]=val[indx]
        curve1=np.flip(curve) 
        
        if beg==None:
            beg=starttime
        
        # 1 day 
        stream = read(filename)
        trace  = stream[2] #Composante Z
        
        stats         = trace.stats
        network       = trace.stats.network
        station       = trace.stats.station
        channel       = trace.stats.channel
        starttime     = trace.stats.starttime
        endtime       = trace.stats.endtime
        sampling_rate = trace.stats.sampling_rate
        
        # Cut of the data on choosen times
        starttime = starttime+(3600*21)
        endtime   = starttime+segm
        stream = read(filename,starttime=starttime,endtime=endtime)
        trace  = stream[2] #Composante Z
        
        print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
        
        iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        
        metadata = client.get_stations(network=network,station=station,
                                       starttime=starttime,endtime=endtime,level='response')
        
        # FFT calculations
        nfft=ppsd_length*sampling_rate 
        nfft=nfft/4.0                  
        nfft=prev_pow_2(nfft)          
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
        current_hist_stack            = None
        current_hist_stack_cumulative = None
        current_times_used            = [] 
        current_times_all_details     = []
        
        # Initialisation of the parameters
        verbose     = False #Show the time data computed
        skip_on_gaps= False
        
        times_data = insert_data_times(times_data,stream)
        times_gaps = insert_gap_times (times_gaps,stream)
        stream.merge(merge_method(skip_on_gaps),fill_value=0)
        
        # Read the all stream by the defined segments
        for trace in stream:
            if not sanity_check(trace,iid,sampling_rate):
                msg = "merde"
                warnings.warn(msg)
                continue
            t1 = trace.stats.starttime
            t2 = trace.stats.endtime
            if t1 + ppsd_length - trace.stats.delta > t2:
                msg = "merde"
                warnings.warn(msg)
                continue
            while t1 + ppsd_length - trace.stats.delta <= t2:
                if check_time_present(times_processed,ppsd_length,overlap,t1):
                    msg = "merde"
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    slice = trace.slice(t1, t1 + ppsd_length -
                                        trace.stats.delta)
                    success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                                      period_bin_left_edges,period_bin_right_edges,
                                      times_processed,binned_psds,
                                      metadata,iid,trace=slice)
                    if success:
                        if verbose:
                            print(t1)
                        changed = True
                t1 += (1 - overlap) * ppsd_length  # advance
        
        # Init
        if changed:
            current_hist_stack            = None
            current_hist_stack_cumulative = None
            current_times_used            = [] 
            current_times_all_details     = []
        
        selected = stack_selection(current_times_all_details, times_processed,
                                   starttime=starttime, endtime=endtime)
        used_indices = selected.nonzero()[0]
        used_count   = len(used_indices)
        used_times   = np.array(times_processed)[used_indices]
        num_period_bins = len(period_bin_centers)
        num_db_bins = len(db_bin_centers)
        
        hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
        if not used_count:
            current_hist_stack = hist_stack
            current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
            current_times_used = used_times
        inds = np.hstack([binned_psds[i] for i in used_indices])
        inds = db_bin_edges.searchsorted(inds, side="left") - 1
        inds[inds == -1] = 0
        inds[inds == num_db_bins] -= 1
        inds = inds.reshape((used_count, num_period_bins)).T
        for i, inds_ in enumerate(inds):
            # count how often each bin has been hit for this period bin,
            # set the current 2D histogram column accordingly
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)
        
        current_hist_stack = hist_stack
        current_times_used = used_times
        
        # Creation of a plot representing the PPSD
        ### only working for 1 segment 
        b=np.transpose(current_hist_stack)
        b=np.flipud(b)
        i=0
        ib=np.linspace(0,266,267)
        curve=np.zeros(len(ib))
        for i in ib:
            indx=np.nonzero(b[:,int(i)])
            indx=int(indx[0])
            val=db_bin_edges
            val=val[::-1]
            curve[int(i)]=val[indx]
        curve2=np.flip(curve) ############
        
        # 1 day 
        stream = read(filename)
        trace  = stream[2] #Composante Z
        
        stats         = trace.stats
        network       = trace.stats.network
        station       = trace.stats.station
        channel       = trace.stats.channel
        starttime     = trace.stats.starttime
        endtime       = trace.stats.endtime
        sampling_rate = trace.stats.sampling_rate
        
        # Cut of the data on choosen times
        starttime = starttime+(3600*22)
        endtime   = starttime+segm
        stream = read(filename,starttime=starttime,endtime=endtime)
        trace  = stream[2] #Composante Z
        
        print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
        
        iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        
        metadata = client.get_stations(network=network,station=station,
                                       starttime=starttime,endtime=endtime,level='response')
        
        # FFT calculations
        nfft=ppsd_length*sampling_rate 
        nfft=nfft/4.0                  
        nfft=prev_pow_2(nfft)          
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
        current_hist_stack            = None
        current_hist_stack_cumulative = None
        current_times_used            = [] 
        current_times_all_details     = []
        
        # Initialisation of the parameters
        verbose     = False #Show the time data computed
        skip_on_gaps= False
        
        times_data = insert_data_times(times_data,stream)
        times_gaps = insert_gap_times (times_gaps,stream)
        stream.merge(merge_method(skip_on_gaps),fill_value=0)
        
        # Read the all stream by the defined segments
        for trace in stream:
            if not sanity_check(trace,iid,sampling_rate):
                msg = "merde"
                warnings.warn(msg)
                continue
            t1 = trace.stats.starttime
            t2 = trace.stats.endtime
            if t1 + ppsd_length - trace.stats.delta > t2:
                msg = "merde"
                warnings.warn(msg)
                continue
            while t1 + ppsd_length - trace.stats.delta <= t2:
                if check_time_present(times_processed,ppsd_length,overlap,t1):
                    msg = "merde"
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    slice = trace.slice(t1, t1 + ppsd_length -
                                        trace.stats.delta)
                    success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                                      period_bin_left_edges,period_bin_right_edges,
                                      times_processed,binned_psds,
                                      metadata,iid,trace=slice)
                    if success:
                        if verbose:
                            print(t1)
                        changed = True
                t1 += (1 - overlap) * ppsd_length  # advance
        
        # Init
        if changed:
            current_hist_stack            = None
            current_hist_stack_cumulative = None
            current_times_used            = [] 
            current_times_all_details     = []
        
        selected = stack_selection(current_times_all_details, times_processed,
                                   starttime=starttime, endtime=endtime)
        used_indices = selected.nonzero()[0]
        used_count   = len(used_indices)
        used_times   = np.array(times_processed)[used_indices]
        num_period_bins = len(period_bin_centers)
        num_db_bins = len(db_bin_centers)
        
        hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
        if not used_count:
            current_hist_stack = hist_stack
            current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
            current_times_used = used_times
        inds = np.hstack([binned_psds[i] for i in used_indices])
        inds = db_bin_edges.searchsorted(inds, side="left") - 1
        inds[inds == -1] = 0
        inds[inds == num_db_bins] -= 1
        inds = inds.reshape((used_count, num_period_bins)).T
        for i, inds_ in enumerate(inds):
            # count how often each bin has been hit for this period bin,
            # set the current 2D histogram column accordingly
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)
        
        current_hist_stack = hist_stack
        current_times_used = used_times
        
        # Creation of a plot representing the PPSD
        ### only working for 1 segment 
        b=np.transpose(current_hist_stack)
        b=np.flipud(b)
        i=0
        ib=np.linspace(0,266,267)
        curve=np.zeros(len(ib))
        for i in ib:
            indx=np.nonzero(b[:,int(i)])
            indx=int(indx[0])
            val=db_bin_edges
            val=val[::-1]
            curve[int(i)]=val[indx]
        curve3=np.flip(curve) ############
        
        # 1 day 
        stream = read(filename)
        trace  = stream[2] #Composante Z
        
        stats         = trace.stats
        network       = trace.stats.network
        station       = trace.stats.station
        channel       = trace.stats.channel
        starttime     = trace.stats.starttime
        endtime       = trace.stats.endtime
        sampling_rate = trace.stats.sampling_rate
        
        # Cut of the data on choosen times
        starttime = starttime+(3600*23)
        endtime   = starttime+segm
        stream = read(filename,starttime=starttime,endtime=endtime)
        trace  = stream[2] #Composante Z
        
        print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))
        
        iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        
        metadata = client.get_stations(network=network,station=station,
                                       starttime=starttime,endtime=endtime,level='response')
        
        # FFT calculations
        nfft=ppsd_length*sampling_rate 
        nfft=nfft/4.0                  
        nfft=prev_pow_2(nfft)          
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
        current_hist_stack            = None
        current_hist_stack_cumulative = None
        current_times_used            = [] 
        current_times_all_details     = []
        
        # Initialisation of the parameters
        verbose     = False #Show the time data computed
        skip_on_gaps= False
        
        times_data = insert_data_times(times_data,stream)
        times_gaps = insert_gap_times (times_gaps,stream)
        stream.merge(merge_method(skip_on_gaps),fill_value=0)
        
        # Read the all stream by the defined segments
        for trace in stream:
            if not sanity_check(trace,iid,sampling_rate):
                msg = "merde"
                warnings.warn(msg)
                continue
            t1 = trace.stats.starttime
            t2 = trace.stats.endtime
            if t1 + ppsd_length - trace.stats.delta > t2:
                msg = "merde"
                warnings.warn(msg)
                continue
            while t1 + ppsd_length - trace.stats.delta <= t2:
                if check_time_present(times_processed,ppsd_length,overlap,t1):
                    msg = "merde"
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    slice = trace.slice(t1, t1 + ppsd_length -
                                        trace.stats.delta)
                    success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                                      period_bin_left_edges,period_bin_right_edges,
                                      times_processed,binned_psds,
                                      metadata,iid,trace=slice)
                    if success:
                        if verbose:
                            print(t1)
                        changed = True
                t1 += (1 - overlap) * ppsd_length  # advance
        
        # Init
        if changed:
            current_hist_stack            = None
            current_hist_stack_cumulative = None
            current_times_used            = [] 
            current_times_all_details     = []
        
        selected = stack_selection(current_times_all_details, times_processed,
                                   starttime=starttime, endtime=endtime)
        used_indices = selected.nonzero()[0]
        used_count   = len(used_indices)
        used_times   = np.array(times_processed)[used_indices]
        num_period_bins = len(period_bin_centers)
        num_db_bins = len(db_bin_centers)
        
        hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
        if not used_count:
            current_hist_stack = hist_stack
            current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
            current_times_used = used_times
        inds = np.hstack([binned_psds[i] for i in used_indices])
        inds = db_bin_edges.searchsorted(inds, side="left") - 1
        inds[inds == -1] = 0
        inds[inds == num_db_bins] -= 1
        inds = inds.reshape((used_count, num_period_bins)).T
        for i, inds_ in enumerate(inds):
            # count how often each bin has been hit for this period bin,
            # set the current 2D histogram column accordingly
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)
        
        current_hist_stack = hist_stack
        current_times_used = used_times
        
        # Creation of a plot representing the PPSD
        ### only working for 1 segment 
        b=np.transpose(current_hist_stack)
        b=np.flipud(b)
        i=0
        ib=np.linspace(0,266,267)
        curve=np.zeros(len(ib))
        for i in ib:
            indx=np.nonzero(b[:,int(i)])
            indx=int(indx[0])
            val=db_bin_edges
            val=val[::-1]
            curve[int(i)]=val[indx]
        curve4=np.flip(curve)
        
        stp=endtime
        
        j=0
        curves=np.zeros((len(current_hist_stack),4))
        for j in np.linspace(0,266,267):
            
            if (curves==0).all():
                curves[0]=np.array([curve1[0],curve2[0],curve3[0],curve4[0]])
            else:
                curves[int(j)]=np.array([curve1[int(j)],curve2[int(j)],
                                         curve3[int(j)],curve4[int(j)]])
        if iday==day1: #1st time
            newcurves=curves
        else:
            newcurves=np.concatenate((newcurves,curves),axis=1)
        
        plot=plt.plot(x,curve1,x,curve2,x,curve3,x,curve4,c='lightgrey')
        
    else:
        print('len(stream)>3 donc jour pas valide')
        print(trace.stats.starttime)

k=0
curve5 =np.zeros(267)
curve95=np.zeros(267)
for k in np.linspace(0,265,266):
    curve5[int(k)] =np.percentile(newcurves[int(k)], 5)
    curve95[int(k)]=np.percentile(newcurves[int(k)],95)

plt.plot(x,curve5,'k',x,curve95,'k')

######################################################################

# Day of data
date = date_n(2015,12,18)

# Nom du fichier
yr  = str(date.timetuple().tm_year)
day = str(date.timetuple().tm_yday)
if len(day) == 1:
    day = ('00' + day)
elif len(day) == 2:
    day = ('0' + day)

#Mac
path = "/Users/loispapin/Documents/Work/PNSN/"
filename = (path + yr + '/Data/' + sta + '/' + sta 
            + '.' + net + '.' + yr + '.' + day)

# # Windows
# path = r"C:\Users\papin\Documents\Spec\Data"
# filename = (path + "\\" + sta + "\\" + sta + '.' + net + '.' + yr + '.' + day)

# 1 day 
stream = read(filename)
trace  = stream[2] #Composante Z

# Cut of the data on choosen times
starttime = UTCDateTime(date)+(3600*20)
endtime   = starttime+segm
stream = read(filename,starttime=starttime,endtime=endtime)
trace  = stream[2] #Composante Z

stats         = trace.stats
network       = trace.stats.network
station       = trace.stats.station
channel       = trace.stats.channel
starttime     = trace.stats.starttime
endtime       = trace.stats.endtime
sampling_rate = trace.stats.sampling_rate

print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))

iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats

metadata = client.get_stations(network=network,station=station,
                               starttime=starttime,endtime=endtime,level='response')

# FFT calculations
nfft=ppsd_length*sampling_rate 
nfft=nfft/4.0                  
nfft=prev_pow_2(nfft)          
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
current_hist_stack            = None
current_hist_stack_cumulative = None
current_times_used            = [] 
current_times_all_details     = []

# Initialisation of the parameters
verbose     = False #Show the time data computed
skip_on_gaps= False

times_data = insert_data_times(times_data,stream)
times_gaps = insert_gap_times (times_gaps,stream)
stream.merge(merge_method(skip_on_gaps),fill_value=0)

# Read the all stream by the defined segments
for trace in stream:
    if not sanity_check(trace,iid,sampling_rate):
        msg = "Skipping incompatible trace."
        warnings.warn(msg)
        continue
    t1 = trace.stats.starttime
    t2 = trace.stats.endtime
    if t1 + ppsd_length - trace.stats.delta > t2:
        msg = (f"Trace is shorter than this PPSD's 'ppsd_length' "
               f"({str(ppsd_length)} seconds). Skipping trace: "
               f"{str(trace)}")
        warnings.warn(msg)
        continue
    while t1 + ppsd_length - trace.stats.delta <= t2:
        if check_time_present(times_processed,ppsd_length,overlap,t1):
            msg = "Already covered time spans detected (e.g. %s), " + \
                  "skipping these slices."
            msg = msg % t1
            warnings.warn(msg)
        else:
            slice = trace.slice(t1, t1 + ppsd_length -
                                trace.stats.delta)
            success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                              period_bin_left_edges,period_bin_right_edges,
                              times_processed,binned_psds,
                              metadata,iid,trace=slice)
            if success:
                if verbose:
                    print(t1)
                changed = True
        t1 += (1 - overlap) * ppsd_length  # advance

# Init
if changed:
    current_hist_stack            = None
    current_hist_stack_cumulative = None
    current_times_used            = [] 
    current_times_all_details     = []

selected = stack_selection(current_times_all_details, times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]
num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
    current_times_used = used_times
inds = np.hstack([binned_psds[i] for i in used_indices])
inds = db_bin_edges.searchsorted(inds, side="left") - 1
inds[inds == -1] = 0
inds[inds == num_db_bins] -= 1
inds = inds.reshape((used_count, num_period_bins)).T
for i, inds_ in enumerate(inds):
    # count how often each bin has been hit for this period bin,
    # set the current 2D histogram column accordingly
    hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

current_hist_stack = hist_stack
current_times_used = used_times

# Creation of a plot representing the PPSD
### only working for 1 segment 
b=np.transpose(current_hist_stack)
b=np.flipud(b)
i=0
ib=np.linspace(0,266,267)
curve=np.zeros(len(ib))
for i in ib:
    indx=np.nonzero(b[:,int(i)])
    indx=int(indx[0])
    val=db_bin_edges
    val=val[::-1]
    curve[int(i)]=val[indx]
curve1c=np.flip(curve) 

# Day of data
date = date_n(2015,12,18)

# Nom du fichier
yr  = str(date.timetuple().tm_year)
day = str(date.timetuple().tm_yday)
if len(day) == 1:
    day = ('00' + day)
elif len(day) == 2:
    day = ('0' + day)

# Mac
path = "/Users/loispapin/Documents/Work/PNSN/"
filename = (path + yr + '/Data/' + sta + '/' + sta 
            + '.' + net + '.' + yr + '.' + day)

# # Windows
# path = r"C:\Users\papin\Documents\Spec\Data"
# filename = (path + "\\" + sta + "\\" + sta + '.' + net + '.' + yr + '.' + day)

# 1 day 
stream = read(filename)
trace  = stream[2] #Composante Z

# Cut of the data on choosen times
starttime = UTCDateTime(date)+(3600*21)
endtime   = starttime+segm
stream = read(filename,starttime=starttime,endtime=endtime)
trace  = stream[2] #Composante Z

stats         = trace.stats
network       = trace.stats.network
station       = trace.stats.station
channel       = trace.stats.channel
starttime     = trace.stats.starttime
endtime       = trace.stats.endtime
sampling_rate = trace.stats.sampling_rate

print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))

iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats

metadata = client.get_stations(network=network,station=station,
                               starttime=starttime,endtime=endtime,level='response')

# FFT calculations
nfft=ppsd_length*sampling_rate 
nfft=nfft/4.0                  
nfft=prev_pow_2(nfft)          
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
current_hist_stack            = None
current_hist_stack_cumulative = None
current_times_used            = [] 
current_times_all_details     = []

# Initialisation of the parameters
verbose     = False #Show the time data computed
skip_on_gaps= False

times_data = insert_data_times(times_data,stream)
times_gaps = insert_gap_times (times_gaps,stream)
stream.merge(merge_method(skip_on_gaps),fill_value=0)

# Read the all stream by the defined segments
for trace in stream:
    if not sanity_check(trace,iid,sampling_rate):
        msg = "Skipping incompatible trace."
        warnings.warn(msg)
        continue
    t1 = trace.stats.starttime
    t2 = trace.stats.endtime
    if t1 + ppsd_length - trace.stats.delta > t2:
        msg = (f"Trace is shorter than this PPSD's 'ppsd_length' "
               f"({str(ppsd_length)} seconds). Skipping trace: "
               f"{str(trace)}")
        warnings.warn(msg)
        continue
    while t1 + ppsd_length - trace.stats.delta <= t2:
        if check_time_present(times_processed,ppsd_length,overlap,t1):
            msg = "Already covered time spans detected (e.g. %s), " + \
                  "skipping these slices."
            msg = msg % t1
            warnings.warn(msg)
        else:
            slice = trace.slice(t1, t1 + ppsd_length -
                                trace.stats.delta)
            success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                              period_bin_left_edges,period_bin_right_edges,
                              times_processed,binned_psds,
                              metadata,iid,trace=slice)
            if success:
                if verbose:
                    print(t1)
                changed = True
        t1 += (1 - overlap) * ppsd_length  # advance

# Init
if changed:
    current_hist_stack            = None
    current_hist_stack_cumulative = None
    current_times_used            = [] 
    current_times_all_details     = []

selected = stack_selection(current_times_all_details, times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]
num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
    current_times_used = used_times
inds = np.hstack([binned_psds[i] for i in used_indices])
inds = db_bin_edges.searchsorted(inds, side="left") - 1
inds[inds == -1] = 0
inds[inds == num_db_bins] -= 1
inds = inds.reshape((used_count, num_period_bins)).T
for i, inds_ in enumerate(inds):
    # count how often each bin has been hit for this period bin,
    # set the current 2D histogram column accordingly
    hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

current_hist_stack = hist_stack
current_times_used = used_times

# Creation of a plot representing the PPSD
### only working for 1 segment 
b=np.transpose(current_hist_stack)
b=np.flipud(b)
i=0
ib=np.linspace(0,266,267)
curve=np.zeros(len(ib))
for i in ib:
    indx=np.nonzero(b[:,int(i)])
    indx=int(indx[0])
    val=db_bin_edges
    val=val[::-1]
    curve[int(i)]=val[indx]
curve2c=np.flip(curve) 

# Day of data
date = date_n(2015,12,18)

# Nom du fichier
yr  = str(date.timetuple().tm_year)
day = str(date.timetuple().tm_yday)
if len(day) == 1:
    day = ('00' + day)
elif len(day) == 2:
    day = ('0' + day)

#Mac
path = "/Users/loispapin/Documents/Work/PNSN/"
filename = (path + yr + '/Data/' + sta + '/' + sta 
            + '.' + net + '.' + yr + '.' + day)

# # Windows
# path = r"C:\Users\papin\Documents\Spec\Data"
# filename = (path + "\\" + sta + "\\" + sta + '.' + net + '.' + yr + '.' + day)

# 1 day 
stream = read(filename)
trace  = stream[2] #Composante Z

# Cut of the data on choosen times
starttime = UTCDateTime(date)+(3600*22)
endtime   = starttime+segm
stream = read(filename,starttime=starttime,endtime=endtime)
trace  = stream[2] #Composante Z

stats         = trace.stats
network       = trace.stats.network
station       = trace.stats.station
channel       = trace.stats.channel
starttime     = trace.stats.starttime
endtime       = trace.stats.endtime
sampling_rate = trace.stats.sampling_rate

print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))

iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats

metadata = client.get_stations(network=network,station=station,
                               starttime=starttime,endtime=endtime,level='response')

# FFT calculations
nfft=ppsd_length*sampling_rate 
nfft=nfft/4.0                  
nfft=prev_pow_2(nfft)          
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
current_hist_stack            = None
current_hist_stack_cumulative = None
current_times_used            = [] 
current_times_all_details     = []

# Initialisation of the parameters
verbose     = False #Show the time data computed
skip_on_gaps= False

times_data = insert_data_times(times_data,stream)
times_gaps = insert_gap_times (times_gaps,stream)
stream.merge(merge_method(skip_on_gaps),fill_value=0)

# Read the all stream by the defined segments
for trace in stream:
    if not sanity_check(trace,iid,sampling_rate):
        msg = "Skipping incompatible trace."
        warnings.warn(msg)
        continue
    t1 = trace.stats.starttime
    t2 = trace.stats.endtime
    if t1 + ppsd_length - trace.stats.delta > t2:
        msg = (f"Trace is shorter than this PPSD's 'ppsd_length' "
               f"({str(ppsd_length)} seconds). Skipping trace: "
               f"{str(trace)}")
        warnings.warn(msg)
        continue
    while t1 + ppsd_length - trace.stats.delta <= t2:
        if check_time_present(times_processed,ppsd_length,overlap,t1):
            msg = "Already covered time spans detected (e.g. %s), " + \
                  "skipping these slices."
            msg = msg % t1
            warnings.warn(msg)
        else:
            slice = trace.slice(t1, t1 + ppsd_length -
                                trace.stats.delta)
            success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                              period_bin_left_edges,period_bin_right_edges,
                              times_processed,binned_psds,
                              metadata,iid,trace=slice)
            if success:
                if verbose:
                    print(t1)
                changed = True
        t1 += (1 - overlap) * ppsd_length  # advance

# Init
if changed:
    current_hist_stack            = None
    current_hist_stack_cumulative = None
    current_times_used            = [] 
    current_times_all_details     = []

selected = stack_selection(current_times_all_details, times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]
num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
    current_times_used = used_times
inds = np.hstack([binned_psds[i] for i in used_indices])
inds = db_bin_edges.searchsorted(inds, side="left") - 1
inds[inds == -1] = 0
inds[inds == num_db_bins] -= 1
inds = inds.reshape((used_count, num_period_bins)).T
for i, inds_ in enumerate(inds):
    # count how often each bin has been hit for this period bin,
    # set the current 2D histogram column accordingly
    hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

current_hist_stack = hist_stack
current_times_used = used_times

# Creation of a plot representing the PPSD
### only working for 1 segment 
b=np.transpose(current_hist_stack)
b=np.flipud(b)
i=0
ib=np.linspace(0,266,267)
curve=np.zeros(len(ib))
for i in ib:
    indx=np.nonzero(b[:,int(i)])
    indx=int(indx[0])
    val=db_bin_edges
    val=val[::-1]
    curve[int(i)]=val[indx]
curve3c=np.flip(curve) 

# Day of data
date = date_n(2015,12,18)

# Nom du fichier
yr  = str(date.timetuple().tm_year)
day = str(date.timetuple().tm_yday)
if len(day) == 1:
    day = ('00' + day)
elif len(day) == 2:
    day = ('0' + day)

#Mac
path = "/Users/loispapin/Documents/Work/PNSN/"
filename = (path + yr + '/Data/' + sta + '/' + sta 
            + '.' + net + '.' + yr + '.' + day)

# # Windows
# path = r"C:\Users\papin\Documents\Spec\Data"
# filename = (path + "\\" + sta + "\\" + sta + '.' + net + '.' + yr + '.' + day)

# 1 day 
stream = read(filename)
trace  = stream[2] #Composante Z

# Cut of the data on choosen times
starttime = UTCDateTime(date)+(3600*23)
endtime   = starttime+segm
stream = read(filename,starttime=starttime,endtime=endtime)
trace  = stream[2] #Composante Z

stats         = trace.stats
network       = trace.stats.network
station       = trace.stats.station
channel       = trace.stats.channel
starttime     = trace.stats.starttime
endtime       = trace.stats.endtime
sampling_rate = trace.stats.sampling_rate

print(trace.stats.channel+' | '+str(trace.stats.starttime)+' | '+str(trace.stats.endtime))

iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats

metadata = client.get_stations(network=network,station=station,
                               starttime=starttime,endtime=endtime,level='response')

# FFT calculations
nfft=ppsd_length*sampling_rate 
nfft=nfft/4.0                  
nfft=prev_pow_2(nfft)          
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
current_hist_stack            = None
current_hist_stack_cumulative = None
current_times_used            = [] 
current_times_all_details     = []

# Initialisation of the parameters
verbose     = False #Show the time data computed
skip_on_gaps= False

times_data = insert_data_times(times_data,stream)
times_gaps = insert_gap_times (times_gaps,stream)
stream.merge(merge_method(skip_on_gaps),fill_value=0)

# Read the all stream by the defined segments
for trace in stream:
    if not sanity_check(trace,iid,sampling_rate):
        msg = "Skipping incompatible trace."
        warnings.warn(msg)
        continue
    t1 = trace.stats.starttime
    t2 = trace.stats.endtime
    if t1 + ppsd_length - trace.stats.delta > t2:
        msg = (f"Trace is shorter than this PPSD's 'ppsd_length' "
               f"({str(ppsd_length)} seconds). Skipping trace: "
               f"{str(trace)}")
        warnings.warn(msg)
        continue
    while t1 + ppsd_length - trace.stats.delta <= t2:
        if check_time_present(times_processed,ppsd_length,overlap,t1):
            msg = "Already covered time spans detected (e.g. %s), " + \
                  "skipping these slices."
            msg = msg % t1
            warnings.warn(msg)
        else:
            slice = trace.slice(t1, t1 + ppsd_length -
                                trace.stats.delta)
            success = process(leng,nfft,sampling_rate,nlap,psd_periods,
                              period_bin_left_edges,period_bin_right_edges,
                              times_processed,binned_psds,
                              metadata,iid,trace=slice)
            if success:
                if verbose:
                    print(t1)
                changed = True
        t1 += (1 - overlap) * ppsd_length  # advance

# Init
if changed:
    current_hist_stack            = None
    current_hist_stack_cumulative = None
    current_times_used            = [] 
    current_times_all_details     = []

selected = stack_selection(current_times_all_details, times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]
num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
    current_times_used = used_times
inds = np.hstack([binned_psds[i] for i in used_indices])
inds = db_bin_edges.searchsorted(inds, side="left") - 1
inds[inds == -1] = 0
inds[inds == num_db_bins] -= 1
inds = inds.reshape((used_count, num_period_bins)).T
for i, inds_ in enumerate(inds):
    # count how often each bin has been hit for this period bin,
    # set the current 2D histogram column accordingly
    hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

current_hist_stack = hist_stack
current_times_used = used_times

# Creation of a plot representing the PPSD
### only working for 1 segment 
b=np.transpose(current_hist_stack)
b=np.flipud(b)
i=0
ib=np.linspace(0,266,267)
curve=np.zeros(len(ib))
for i in ib:
    indx=np.nonzero(b[:,int(i)])
    indx=int(indx[0])
    val=db_bin_edges
    val=val[::-1]
    curve[int(i)]=val[indx]
curve4c=np.flip(curve)

###################################

plt.plot(x,curve1c,'--r',x,curve2c,'--r',x,curve3c,'--r',x,curve4c,'--r',lw=1)    

# Grid
color = {"color": "0.7"}
ax.grid(True, which="major", **color)
ax.grid(True, which="minor", **color)

# Axis and title
ax.set_xlabel('Frequency [Hz]')
ax.invert_xaxis()
# ax.set_xscale('log')
ax.set_xlim(period_lim)
ax.xaxis.set_major_formatter(FormatStrFormatter("%g")) #Pas de 10^

ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
ax.set_ylim(db_bin_edges[0],db_bin_edges[-1])

title = "%s   %s--%s   (from %s to %s pm)"
title = title % (iid,beg.date,(stp-1).date,
                  np.abs(beg.datetime.hour-12),
                  np.abs(stp.datetime.hour-12))
ax.set_title(title)

# Show the figure
plt.ion()
plt.savefig('fig.jpg', dpi=300, bbox_inches='tight')
plt.show()
