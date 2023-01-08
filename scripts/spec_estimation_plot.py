#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 2023
Update  on Sun Jan 08 

@author: loispapin

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
from obspy import UTCDateTime
from obspy.imaging.cm import obspy_sequential 
from obspy.signal.util import prev_pow_2
from obspy.clients.fdsn import Client
client = Client("IRIS")

from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# Functions called in this script
runfile('C:/Users/papin/Documents/Spec/fcts.py', 
        wdir='C:/Users/papin/Documents/Spec')

"""
    Read the data with the function read of the Obspy module. Identify the 
    necessary infos from it and also get the metadata of the station response.
    
"""

# Day of data
date = date_n(2015,12,18)

# Nom du fichier
sta = 'B926'
net = 'PB'
yr  = str(date.timetuple().tm_year)
day = str(date.timetuple().tm_yday)
if len(day) == 1:
    day = ('00' + day)
elif len(day) == 2:
    day = ('0' + day)

path = r"C:\Users\papin\Documents\Spec\Data"
filename = (path + "\\" + sta + "\\" + sta + '.' + net + '.' + yr + '.' + day)

# Parameters 
segm = 3600 #1h cut
ppsd_length                    = segm 
overlap                        = 0
period_smoothing_width_octaves = 1.0
period_step_octaves            = 0.0125
db_bins                        = (-170, -90, 0.5)

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

"""


"""

# FFT calculations
nfft=ppsd_length*sampling_rate 
nfft=nfft/4.0                  
nfft=prev_pow_2(nfft)          
nlap=int(0.75*nfft)            
leng=int(sampling_rate*ppsd_length)
_,freq=mlab.psd(np.ones(leng),nfft,sampling_rate,noverlap=nlap) 
freq=freq[1:]
psd_periods=1.0/freq[::-1]

# Calculation on 0.01-16Hz
f1 = 1; f2 = 10; 
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


"""

# Initialisation of the parameters
verbose     = False #Show the time data computed
skip_on_gaps= False

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
curve=np.flip(curve) 

"""

  
"""
if current_hist_stack is None:
    msg = 'No data accumulated'
    raise Exception(msg)

# Initialisation of the parameters
grid=True
period_lim=(f1,f2) 

# Start of the figure
plt.ioff()

# Create figure
fig, ax = plt.subplots() 
xlim = ax.get_xlim()
fig.grid = grid

# PPSD
x=np.linspace(0.99355938,10.04341567,267)
plt.scatter(x, curve,s=2)
# plt.savefig('fig.jpg', dpi=300, bbox_inches='tight')

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

title = "%s   %s    (from %s to %s)"
title = title % (iid,starttime.date,
                  starttime.datetime.hour+1,
                  endtime.datetime.hour+1)

ax.set_title(title)

# Show the figure
plt.ion()
plt.show()