#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:44:29 2022

@author: loispapin

This script was made to help compare 2 sets of data on a PPSD plot. It is 
using a combination of spec_estimation_mine.py and spec_estimation_mine_yr.py.
The output figure is 2 PPSD of different cmap and transparencies.

No class is defined here, only necessary functions are in defs.py file.

Sections :
    . Collect of the data 
    . PPSD fixed information
    . Process of the data with first PSD estimates
    . Calculation of the 2D-histogram
        Before : 2 times (2 sets of time/data)
    . Plot of the histograms
    
"""

"""
    Importation of the necessary librairies to execute the code and also,
    in case, to execute the functions available in defs.py.
    
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import date as date_n

from matplotlib import mlab
from matplotlib.ticker import FormatStrFormatter

from obspy import read
from obspy import Stream, Trace, UTCDateTime
from obspy.imaging.cm import obspy_sequential, pqlx 
from obspy.signal.util import prev_pow_2
from obspy.clients.fdsn import Client
client = Client("IRIS")

# Functions called in this script
runfile('/Users/loispapin/Documents/Work/PNSN/2011/fcts.py', 
        wdir='/Users/loispapin/Documents/Work/PNSN/2011')

"""
    Start of the script with time to choose : the period of time that we want 
    (var day & num) to compare a day with. 
"""

# Start of the data and how long
day = 1 #1er janvier
num = 365

# Temporary variables
temp_time=[]
temp_binned_psds=[None]*365
starts=[];ends=[] #Every start and end of times

for iday in np.arange(day,day+num,dtype=int):
    
    """
        Read the data with the function read of the Obspy module. Identify the 
        necessary infos from it and also get the metadata of the station response.
        
    """
    
    # Nom du fichier
    sta = 'B017'
    net = 'PB'
    yr  = '2011'
    
    if len(str(iday)) == 1:
        day = ('00' + str(iday))
    elif len(str(iday)) == 2:
        day = ('0' + str(iday))
    elif len(str(iday)) == 3:
        day = (str(iday))
    
    path = "/Users/loispapin/Documents/Work/PNSN/2011/Data/"
    filename = (path + sta + '/' + sta + '.' + net + '.' + yr + '.' + day)
        
    segm = 3600 #1h cut
    
    # 1 day 
    stream = read(filename)
    trace  = stream[2] #Composante Z

    # # Cut of the data on choosen times
    # starttime = UTCDateTime("2011-07-03T02:30:00.000")
    # endtime   = starttime+segm
    # stream = read(filename,starttime=starttime,endtime=endtime)
    # trace  = stream[2] #Composante Z
    
    stats         = trace.stats
    network       = trace.stats.network
    station       = trace.stats.station
    channel       = trace.stats.channel
    starttime     = trace.stats.starttime
    endtime       = trace.stats.endtime
    sampling_rate = trace.stats.sampling_rate
    
    iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
    
    metadata = client.get_stations(network=network,station=station,
                                   starttime=starttime,endtime=endtime,level='response')
    
    starts = np.append(starts,starttime)
    ends   = np.append(ends,endtime)
    
    """
        Define the PPSD informations such as the segments for the calculations,
        the frenquencies and periods, the bins for the histogram. Equivalent of
        the def __init__ in the PPSD class.
        
    """
    
    ppsd_length                    = segm 
    overlap                        = 0.5
    period_smoothing_width_octaves = 1.0
    period_step_octaves            = 0.125
    db_bins                        = (-200, -50, 1.)
    
    ##13 segments overlapping 75% and truncate to next lower power of 2
    #number of points
    nfft=ppsd_length*sampling_rate 
    #1 full segment length + 25% * 12 full segment lengths
    nfft=nfft/4.0                  
    #next smaller power of 2 for nfft
    nfft=prev_pow_2(nfft)          
    #use 75% overlap
    nlap=int(0.75*nfft)            
    #trace length for one psd segment
    leng=int(sampling_rate*ppsd_length)
    #make an initial dummy psd and to get the array of periods
    _,freq=mlab.psd(np.ones(leng),nfft,sampling_rate,noverlap=nlap) 
    #leave out first adn last entry (offset)
    freq=freq[1:]
    psd_periods=1.0/freq[::-1]
    
    # Calculation on 0.01-16Hz
    f1 = 0.01; f2 = 16; 
    period_limits = (1/f2,1/f1)
    
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
    
    """
        Process the data and create the first PSD data. Equivalent of the 
        def add in the PPSD class.
        
    """
    
    # Init
    verbose     = False #Show the time data computed
    skip_on_gaps= False
    
    # Verifications before the computations
    if metadata is None:
        msg = ("PPSD instance has no metadata attached, which are needed "
               "for processing the data. When using 'PPSD.load_npz()' use "
               "'metadata' kwarg to provide metadata.")
        raise Exception(msg)
    changed = False
    if isinstance(stream, Trace):
        stream = Stream([stream])
    if not stream:
        msg = 'Empty stream object provided to PPSD.add()'
        warnings.warn(msg)
    stream = stream.select(id=iid)
    if not stream:
        msg = 'No traces with matching SEED ID in provided stream object.'
        warnings.warn(msg)
    stream = stream.select(sampling_rate=sampling_rate)
    if not stream:
        msg = ('No traces with matching sampling rate in provided stream '
               'object.')
        warnings.warn(msg)
    
    # save information on available data and gaps
    times_data = insert_data_times(times_data,stream)
    times_gaps = insert_gap_times (times_gaps,stream)
    # merge depending on skip_on_gaps
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
            temp_binned_psds[iday-1] = binned_psds
            
    temp_time=np.append(temp_time,times_processed)
    
    # Suivi du temps
    if iday==50:
        print('50e jour')
    elif iday==100:
        print('100e jour')
    elif iday==150:
        print('150e jour')
    elif iday==200:
        print('200e jour')
    elif iday==250:
        print('250e jour')
    elif iday==300:
        print('300e jour')

    # Init
    if changed:
        current_hist_stack            = None
        current_hist_stack_cumulative = None
        current_times_used            = [] 
        current_times_all_details     = []

# All days are include in the variables for the histogram
times_processed=temp_time.tolist()
res = []
for val in temp_binned_psds:
    if val != None :
        res.append(val)
binned_psds = [item for sublist in res for item in sublist]
starttime=starts[0];endtime=ends[-1] #Start and end of the period

"""
    Calculation of the 2D-histogram based on the processed data (time).
    Equivalent of the def calculate_histogram of the PPSD class.
    
"""

if not times_processed:
    current_hist_stack = None
    current_hist_stack_cumulative = None
    current_times_used = []

# determine which psd pieces should be used in the stack,
# based on all selection criteria specified by user
selected = stack_selection(current_times_all_details, times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]

num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

# initial setup of 2D histogram
hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)

# empty selection, set all histogram stacks to zeros
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
    current_times_used = used_times

# concatenate all used spectra, evaluate index of amplitude bin each
# value belongs to
inds = np.hstack([binned_psds[i] for i in used_indices])

# we need minus one because searchsorted returns the insertion index in
# the array of bin edges which is the index of the corresponding bin
# plus one
inds = db_bin_edges.searchsorted(inds, side="left") - 1
inds[inds == -1] = 0
# same goes for values right of last bin edge
inds[inds == num_db_bins] -= 1
# reshape such that we can iterate over the array, extracting for
# each period bin an array of all amplitude bins we have hit
inds = inds.reshape((used_count, num_period_bins)).T
# inds=inds[:,0]
for i, inds_ in enumerate(inds):
    # count how often each bin has been hit for this period bin,
    # set the current 2D histogram column accordingly
    hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

# set everything that was calculated
current_hist_stack = hist_stack
# current_hist_stack_cumulative = hist_stack_cumul
current_times_used = used_times

if current_hist_stack is None:
    msg = 'No data accumulated'
    raise Exception(msg)

# Computations needed
current_histogram = current_hist_stack
current_histogram_count = len(current_times_used)
data_yr = (current_histogram * 100.0 / (current_histogram_count or 1))
xedges = period_xedges
xedges = 1.0 / xedges

"""
    Start of the script with time to choose : the day that we want to compare>
    Another read of the data with the function read of the Obspy module for a 
    day. Identify the necessary infos from it and also get the metadata of 
    the station response.
    
"""

# Day of data to compare
date = date_n(2011,8,31)

# Nom du fichier
sta = 'B017'
net = 'PB'
yr  = str(date.timetuple().tm_year)
day = str(date.timetuple().tm_yday)
if len(day) == 1:
    day = ('00' + day)
elif len(day) == 2:
    day = ('0' + day)
    
path = "/Users/loispapin/Documents/Work/PNSN/2011/Data/"
filename = (path + sta + '/' + sta + '.' + net + '.' + yr + '.' + day)

# 1 day 
stream = read(filename)
trace  = stream[2] #Composante Z

# Cut of the data on choosen times
starttime = UTCDateTime(date)+(3600*2.5) #2h30
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

iid = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats

metadata = client.get_stations(network=network,station=station,
                               starttime=starttime,endtime=endtime,level='response')

"""
    Define the PPSD informations such as the segments for the calculations,
    the frenquencies and periods, the bins for the histogram. Equivalent of
    the def __init__ in the PPSD class.
    
"""

##13 segments overlapping 75% and truncate to next lower power of 2
#number of points
nfft=ppsd_length*sampling_rate 
#1 full segment length + 25% * 12 full segment lengths
nfft=nfft/4.0                  
#next smaller power of 2 for nfft
nfft=prev_pow_2(nfft)          
#use 75% overlap
nlap=int(0.75*nfft)            
#trace length for one psd segment
leng=int(sampling_rate*ppsd_length)
#make an initial dummy psd and to get the array of periods
_,freq=mlab.psd(np.ones(leng),nfft,sampling_rate,noverlap=nlap) 
#leave out first adn last entry (offset)
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

"""
    Process the data and create the first PSD data. Equivalent of the 
    def add in the PPSD class.
    
"""

# Initialisation of the parameters
verbose     = False #Show the time data computed
skip_on_gaps= False

# Verifications before the computations
if metadata is None:
    msg = ("PPSD instance has no metadata attached, which are needed "
           "for processing the data. When using 'PPSD.load_npz()' use "
           "'metadata' kwarg to provide metadata.")
    raise Exception(msg)
changed = False
if isinstance(stream, Trace):
    stream = Stream([stream])
if not stream:
    msg = 'Empty stream object provided to PPSD.add()'
    warnings.warn(msg)
stream = stream.select(id=iid)
if not stream:
    msg = 'No traces with matching SEED ID in provided stream object.'
    warnings.warn(msg)
stream = stream.select(sampling_rate=sampling_rate)
if not stream:
    msg = ('No traces with matching sampling rate in provided stream '
           'object.')
    warnings.warn(msg)

# save information on available data and gaps
times_data = insert_data_times(times_data,stream)
times_gaps = insert_gap_times (times_gaps,stream)
# merge depending on skip_on_gaps
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

"""
    Calculation of the 2D-histogram based on the processed data (time).
    Equivalent of the def calculate_histogram of the PPSD class.
    
"""

if not times_processed:
    current_hist_stack = None
    current_hist_stack_cumulative = None
    current_times_used = []

# determine which psd pieces should be used in the stack,
# based on all selection criteria specified by user
selected = stack_selection(current_times_all_details, times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]

num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

# initial setup of 2D histogram
hist_stack = np.zeros((num_period_bins,num_db_bins),dtype=np.uint64)

# empty selection, set all histogram stacks to zeros
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack,dtype=np.float32)
    current_times_used = used_times

# concatenate all used spectra, evaluate index of amplitude bin each
# value belongs to
inds = np.hstack([binned_psds[i] for i in used_indices])

# we need minus one because searchsorted returns the insertion index in
# the array of bin edges which is the index of the corresponding bin
# plus one
inds = db_bin_edges.searchsorted(inds, side="left") - 1
inds[inds == -1] = 0
# same goes for values right of last bin edge
inds[inds == num_db_bins] -= 1
# reshape such that we can iterate over the array, extracting for
# each period bin an array of all amplitude bins we have hit
inds = inds.reshape((used_count, num_period_bins)).T
# inds=inds[:,0]
for i, inds_ in enumerate(inds):
    # count how often each bin has been hit for this period bin,
    # set the current 2D histogram column accordingly
    hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

# set everything that was calculated
current_hist_stack = hist_stack
# current_hist_stack_cumulative = hist_stack_cumul
current_times_used = used_times

if current_hist_stack is None:
    msg = 'No data accumulated'
    raise Exception(msg)

# Computations needed
current_histogram = current_hist_stack
current_histogram_count = len(current_times_used)
data = (current_histogram * 100.0 / (current_histogram_count or 1))
xedges = period_xedges
xedges = 1.0 / xedges

"""
    Plot of the 2D-histogram with all the parameters. Without modification,
    no limit of frequencies, no noise models, no coverage, colors of PSD
    goes to 30%, amplitude for no particular handling. Equivalent of the def 
    plot of the PPSD class.
    
    Possibility to use the same colomap as [McNamara2004], cf. cmap. 
    
    The plot will do first the PSSD of the period of time (data_yr), and 
    then the PSSD of the day that we compared. 2 differents cmap for the
    plot to better show the differences.
    
"""

# Initialisation of the parameters
grid=True
max_percentage=30
color_limits = (0, max_percentage)
label = "[%]"
period_lim=(f1,f2) 
xaxis_frequency=True #False

# Start of the figure
plt.ioff()

# Create figure
fig, ax = plt.subplots() 

fig.label = label
fig.max_percentage = max_percentage
fig.grid = grid
fig.xaxis_frequency = xaxis_frequency
fig.color_limits = color_limits

xlim = ax.get_xlim()
fig.meshgrid = np.meshgrid(xedges,db_bin_edges)
# PPSD
ppsd1=ax.pcolormesh(fig.meshgrid[0], fig.meshgrid[1], 
                      data_yr.T, cmap=pqlx, zorder=2, alpha=1)
ppsd2=ax.pcolormesh(fig.meshgrid[0], fig.meshgrid[1], 
                      data.T, cmap=obspy_sequential, zorder=2, alpha=0.75)

# Colorbar
cb = plt.colorbar(ppsd2,ax=ax)
cb.mappable.set_clim(*fig.color_limits)
cb.set_label(fig.label)
fig.colorbar = cb
ppsd1.set_clim(*fig.color_limits)

# Grid (doesn't work)
color = {"color": "0.7"}
ax.grid(True, which="major", **color)
ax.grid(True, which="minor", **color)

# Axis and title
ax.set_xlabel('Frequency [Hz]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xlim(period_lim)
ax.xaxis.set_major_formatter(FormatStrFormatter("%g")) #Pas de 10^

ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
ax.set_ylim(db_bin_edges[0],db_bin_edges[-1])

title = "%s   %s -- %s  (%i/%i segments)"
title = title % (iid,
                 UTCDateTime(ns=times_processed[0]).date,
                 UTCDateTime(ns=times_processed[-1]).date,
                 len(current_times_used),
                 len(times_processed))
ax.set_title(title)

# Show the figure
plt.ion()
plt.show()
