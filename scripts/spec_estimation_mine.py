#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:03:38 2022
Working on Mon Nov 28 15:18:23 2022

@author: loispapin

This script is based on the PPSD class defined in the Obspy module. Without 
changing the parameters, the output figure of this script is the same as 
using the ppsd.plot of the module.

No class is defined here, only necessary functions are in defs.py file.

Code for 1 day of data or a section of it. Need to check 
spectral_estimation_mine_yr.py for a year of data.

Sections :
    . Collect of the data 
    . PPSD fixed information
    . Process of the data with first PSD estimates
    . Calculation of the 2D-histogram
    . Plot of the histogram
    
PS : other parameters/lines of code are available in trash.py (all related
to the plot part)
PSS : the original script of the PPSD class of the module can be found at
/opt/anaconda3/lib/python3.9/site-packages/obspy/signal/spectral_estimation.py

"""


"""
    Importation of the necessary librairies to execute the code and also,
    in case, to execute the functions available in defs.py.
    
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import mlab
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patheffects import withStroke

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
    Read the data with the function read of the Obspy module. Identify the 
    necessary infos from it and also get the metadata of the station response.
    
"""

# Nom du fichier
sta = 'B017'
net = '.PB'
yr  = '.2011'
day = '.184'

path = "/Users/loispapin/Documents/Work/PNSN/2011/Data/"
filename = (path + sta + '/' + sta + net + yr + day)

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

# To be modified to select the wanted frequencies
# period_limits = (psd_periods[0],
#                  psd_periods[-1])
# Calculation on 0.01-16Hz
f1 = 1; f2 = 12; 
period_limits = (1/f2,1/f1)

period_binning=setup_period_binning(psd_periods,
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
db_bin_centers = (db_bin_edges[:-1]+db_bin_edges[1:])/ 2.0

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
selected = stack_selection(current_times_all_details,times_processed,
                           starttime=starttime, endtime=endtime)
used_indices = selected.nonzero()[0]
used_count   = len(used_indices)
used_times   = np.array(times_processed)[used_indices]

num_period_bins = len(period_bin_centers)
num_db_bins = len(db_bin_centers)

# initial setup of 2D histogram
hist_stack = np.zeros((num_period_bins, num_db_bins), dtype=np.uint64)

# empty selection, set all histogram stacks to zeros
if not used_count:
    current_hist_stack = hist_stack
    current_hist_stack_cumulative = np.zeros_like(hist_stack, 
                                                  dtype=np.float32)
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

"""
    Plot of the 2D-histogram with all the parameters. Without modification,
    no limit of frequencies, no noise models, no coverage, colors of PSD
    goes to 30%, amplitude for no particular handling. Equivalent of the def 
    plot of the PPSD class.
    
    Possibility to use the same colomap as [McNamara2004], cf. cmap. 
    
    The #NOPE means we don't use the particular parameter. If some parameters 
    are missing, check the trash.py file with already written script.
    
"""

if current_hist_stack is None:
    msg = 'No data accumulated'
    raise Exception(msg)

# Initialisation of the parameters
filename=None

special_handling = None

grid=True
show=True
draw=False

max_percentage=None
label = "[%]"
period_lim=(f1,f2) 
xaxis_frequency=True #False

color=int(input('Choose of the colormap (1 is obspy, 2 is McNamara) : '))
if color==1:
    cmap = obspy_sequential
elif color==2:
    cmap = pqlx #McNamara color map (9)white background, rainbow color)
else: 
    msg = "Error on the choosen number for the colormap"
    warnings.warn(msg)
    cmap = obspy_sequential

cumulative=False
cumulative_number_of_colors=20

show_noise_models=False #True
show_earthquakes=None
show_histogram=True

fig = plt.figure()
ax = fig.add_subplot(111)

# Parameters

if show_noise_models: #NOPE
    for periods, noise_model in models:
        if xaxis_frequency:
            xdata = 1.0 / periods
        else:
            xdata = periods
        ax.plot(xdata, noise_model, '0.4', linewidth=2, zorder=10)

if show_earthquakes is not None: #NOPE
    if len(show_earthquakes) == 2:
        show_earthquakes = (show_earthquakes[0],
                            show_earthquakes[0] + 0.1,
                            show_earthquakes[1],
                            show_earthquakes[1] + 1)
    if len(show_earthquakes) == 3:
        show_earthquakes += (show_earthquakes[-1] + 1, )
    min_mag, max_mag, min_dist, max_dist = show_earthquakes
    for key, data in earthquake_models.items():
        magnitude, distance = key
        frequencies, accelerations = data
        accelerations = np.array(accelerations)
        frequencies = np.array(frequencies)
        periods = 1.0 / frequencies
        # Eq.1 from Clinton and Cauzzi (2013) converts
        # power to density
        ydata = accelerations / (periods ** (-.5))
        ydata = 20 * np.log10(ydata / 2)
        if not (min_mag <= magnitude <= max_mag and
                min_dist <= distance <= max_dist and
                min(ydata) < self.db_bin_edges[-1]):
            continue
        xdata = periods
        if xaxis_frequency:
            xdata = frequencies
        ax.plot(xdata, ydata, '0.4', linewidth=2)
        leftpoint = np.argsort(xdata)[0]
        if not ydata[leftpoint] < self.db_bin_edges[-1]:
            continue
        ax.text(xdata[leftpoint],
                ydata[leftpoint],
                'M%.1f\n%dkm' % (magnitude, distance),
                ha='right', va='top',
                color='w', weight='bold', fontsize='x-small',
                path_effects=[withStroke(linewidth=3,
                                         foreground='0.4')])

if cumulative: #NOPE
        label = "non-exceedance (cumulative) [%]"
        if max_percentage is not None:
            msg = ("Parameter 'max_percentage' is ignored when "
                   "'cumulative=True'.")
            warnings.warn(msg)
        max_percentage = 100
        if cumulative_number_of_colors is not None:
            cmap = LinearSegmentedColormap(
                name=cmap.name, segmentdata=cmap._segmentdata,
                N=cumulative_number_of_colors)
elif max_percentage is None: #OK
    # Set default only if cumulative is not True.
    max_percentage = 30

# Parameters of fig
fig.cumulative = cumulative
fig.cmap = cmap
fig.label = label
fig.max_percentage = max_percentage
fig.grid = grid
fig.xaxis_frequency = xaxis_frequency

if max_percentage is not None: #OK
    color_limits = (0, max_percentage)
    fig.color_limits = color_limits
    
    # PPSD figure
    plot_histogram(fig, current_hist_stack, current_times_used, 
                   period_xedges, db_bin_edges, draw, filename)

# Axis and title
if xaxis_frequency: #OK
    ax.set_xlabel('Frequency [Hz]')
    ax.invert_xaxis()
    ax.set_xlabel('Period [s]')
ax.set_xscale('log')
ax.set_xlim(period_lim)
ax.xaxis.set_major_formatter(FormatStrFormatter("%g")) #Pas de 10^

if special_handling is None: #OK
    ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
ax.set_ylim(db_bin_edges[0],db_bin_edges[-1])

title = "%s   %s -- %s  (%i/%i segments)"
title = title % (iid,
                 UTCDateTime(ns=times_processed[0]).date,
                 UTCDateTime(ns=times_processed[-1]).date,
                 len(current_times_used),
                 len(times_processed))
ax.set_title(title)

# Catch underflow warnings due to plotting on log-scale.
with np.errstate(all="ignore"):
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    elif show:
        plt.draw()
        plt.show()
    else:
        plt.draw()
