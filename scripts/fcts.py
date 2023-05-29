#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:56:05 2022

@author: loispapin

All the functions used in the scripts of spectral estimation (PPSD)
are here. This is made to avoid a bigger script.

Last time checked on Mon May 29

"""

import math
import bisect
import warnings
import numpy as np

from matplotlib import mlab
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
client = Client("IRIS")
from obspy.core.util import AttribDict
from obspy.core.inventory import Inventory
from obspy.signal.invsim import cosine_taper

def setup_period_binning(psd_periods,period_smoothing_width_octaves,
                          period_step_octaves, period_limits):
    """
    Set up period binning.
    """
    # we step through the period range at step width controlled by
    # period_step_octaves (default 1/8 octave)
    period_step_factor = 2 ** period_step_octaves
    # the width of frequencies we average over for every bin is controlled
    # by period_smoothing_width_octaves (default one full octave)
    period_smoothing_width_factor = \
        2 ** period_smoothing_width_octaves
    # calculate left/right edge and center of first period bin
    # set first smoothing bin's left edge such that the center frequency is
    # the lower limit specified by the user (or the lowest period in the
    # psd)
    per_left = (period_limits[0] /
                (period_smoothing_width_factor ** 0.5))
    per_right = per_left * period_smoothing_width_factor
    per_center = math.sqrt(per_left * per_right)
    # build up lists
    per_octaves_left = [per_left]
    per_octaves_right = [per_right]
    per_octaves_center = [per_center]
    # do this for the whole period range and append the values to our lists
    while per_center < period_limits[1]:
        # move left edge of smoothing bin further
        per_left *= period_step_factor
        # determine right edge of smoothing bin
        per_right = per_left * period_smoothing_width_factor
        # determine center period of smoothing/binning
        per_center = math.sqrt(per_left * per_right)
        # append to lists
        per_octaves_left.append(per_left)
        per_octaves_right.append(per_right)
        per_octaves_center.append(per_center)
    per_octaves_left = np.array(per_octaves_left)
    per_octaves_right = np.array(per_octaves_right)
    per_octaves_center = np.array(per_octaves_center)
    valid = per_octaves_right > psd_periods[0]
    valid &= per_octaves_left < psd_periods[-1]
    per_octaves_left = per_octaves_left[valid]
    per_octaves_right = per_octaves_right[valid]
    per_octaves_center = per_octaves_center[valid]
    period_binning = np.vstack([
        # left edge of smoothing (for calculating the bin value from psd
        per_octaves_left,
        # left xedge of bin (for plotting)
        per_octaves_center / (period_step_factor ** 0.5),
        # bin center (for plotting)
        per_octaves_center,
        # right xedge of bin (for plotting)
        per_octaves_center * (period_step_factor ** 0.5),
        # right edge of smoothing (for calculating the bin value from psd
        per_octaves_right])
    return period_binning

def insert_data_times(times_data,stream):
    """
    Gets gap information of stream and adds the encountered gaps to the gap
    list of the PPSD instance.

    :type stream: :class:`~obspy.core.stream.Stream`
    """
    times_data += \
        [[trace.stats.starttime._ns, trace.stats.endtime._ns]
            for trace in stream]
    return times_data

def insert_gap_times(times_gaps,stream):
    """
    Gets gap information of stream and adds the encountered gaps to the gap
    list of the PPSD instance.

    :type stream: :class:`~obspy.core.stream.Stream`
    """
    times_gaps += [[gap[4]._ns, gap[5]._ns]
                          for gap in stream.get_gaps()]
    return times_gaps

def merge_method(skip_on_gaps):
    if skip_on_gaps:
        return -1
    else:
        return 0

def sanity_check(trace,iid,sampling_rate):
    """
    Checks if trace is compatible for use in the current PPSD instance.
    Returns True if trace can be used or False if not.

    :type trace: :class:`~obspy.core.trace.Trace`
    """
    if trace.id != iid:
        return False
    if trace.stats.sampling_rate != sampling_rate:
        return False
    return True

def check_time_present(times_processed, ppsd_length, overlap, utcdatetime):
    """
    Checks if the given UTCDateTime is already part of the current PPSD
    instance. That is, checks if from utcdatetime to utcdatetime plus
    ppsd_length there is already data in the PPSD.
    Returns True if adding ppsd_length starting at the given time
    would result in an overlap of the ppsd data base, False if it is OK to
    insert this piece of data.
    """
    if not times_processed:
        return False
    # new data comes before existing data.
    if utcdatetime._ns < times_processed[0]:
        overlap_seconds = (
            (utcdatetime._ns + ppsd_length * 1e9) -
            times_processed[0]) / 1e9
        # the new data is welcome if any overlap that would be introduced
        # is less or equal than the overlap used by default on continuous
        # data.
        if overlap_seconds / ppsd_length > overlap:
            return True
        else:
            return False
    # new data exactly at start of first data segment
    elif utcdatetime._ns == times_processed[0]:
        return True
    # new data comes after existing data.
    elif utcdatetime._ns > times_processed[-1]:
        overlap_seconds = (
            (times_processed[-1] + ppsd_length * 1e9) -
            utcdatetime._ns) / 1e9
        # the new data is welcome if any overlap that would be introduced
        # is less or equal than the overlap used by default on continuous
        # data.
        if overlap_seconds / ppsd_length > overlap:
            return True
        else:
            return False
    # new data exactly at start of last data segment
    elif utcdatetime._ns == times_processed[-1]:
        return True
    # otherwise we are somewhere within the currently already present time
    # range..
    else:
        index1 = bisect.bisect_left(times_processed,
                                    utcdatetime._ns)
        index2 = bisect.bisect_right(times_processed,
                                     utcdatetime._ns)
        # if bisect left/right gives same result, we are not exactly at one
        # sampling point but in between to timestamps
        if index1 == index2:
            t1 = times_processed[index1 - 1]
            t2 = times_processed[index1]
            # check if we are overlapping on left side more than the normal
            # overlap specified during init
            overlap_seconds_left = (
                (t1 + ppsd_length * 1e9) - utcdatetime._ns) / 1e9
            # check if we are overlapping on right side more than the
            # normal overlap specified during init
            overlap_seconds_right = (
                (utcdatetime._ns + ppsd_length * 1e9) - t2) / 1e9
            max_overlap = max(overlap_seconds_left,
                              overlap_seconds_right) / ppsd_length
            if max_overlap > overlap:
                return True
            else:
                return False
        # if bisect left/right gives different results, we are at exactly
        # one timestamp that is already present
        else:
            return True
    raise NotImplementedError('This should not happen, please report on '
                              'github.')

def process(leng,nfft,sampling_rate,nlap,psd_periods,
            period_bin_left_edges,period_bin_right_edges,
            times_processed,binned_psds,metadata,iid,trace):
    """
    Processes a segment of data and save the psd information.
    Whether `Trace` is compatible (station, channel, ...) has to
    checked beforehand.

    :type tr: :class:`~obspy.core.trace.Trace`
    :param tr: Compatible Trace with data of one PPSD segment
    :returns: `True` if segment was successfully processed,
        `False` otherwise.
    """

    # XXX DIRTY HACK!!
    if len(trace) == leng + 1:
        trace.data = trace.data[:-1]
    # one last check..
    if len(trace) != leng:
        msg = ("Got a piece of data with wrong length. Skipping:\n" +
               str(trace))
        warnings.warn(msg)
        return False
    # being paranoid, only necessary if in-place operations would follow
    trace.data = trace.data.astype(np.float64)
    # if trace has a masked array we fill in zeros
    try:
        trace.data[trace.data.mask] = 0.0
    # if it is no masked array, we get an AttributeError
    # and have nothing to do
    except AttributeError:
        pass

    # restitution:
    # mcnamara apply the correction at the end in freq-domain,
    # does it make a difference?
    # probably should be done earlier on bigger chunk of data?!
    # Yes, you should avoid removing the response until after you
    # have estimated the spectra to avoid elevated lp noise

    spec, _freq = mlab.psd(trace.data, nfft, sampling_rate,
                           detrend=mlab.detrend_linear, window=fft_taper,
                           noverlap=nlap, sides='onesided',
                           scale_by_freq=True)

    # leave out first entry (offset)
    spec = spec[1:]

    # working with the periods not frequencies later so reverse spectrum
    spec = spec[::-1]
    
    try:
        resp = get_response(metadata,iid,nfft,trace)
    except Exception as e:
        msg = ("Error getting response from provided metadata:\n"
                  "%s: %s\n"
                  "Skipping time segment(s).")
        msg = msg % (e.__class__.__name__, str(e))
        warnings.warn(msg)
        return False

    resp = resp[1:]
    resp = resp[::-1]
    # Now get the amplitude response (squared)
    respamp = np.absolute(resp * np.conjugate(resp))
    # Make omega with the same conventions as spec
    
    # _freq=_freq[1966:5242]
    # resp=resp[1966:5242]
    # respamp=respamp[1966:5242]
    # spec=spec[1966:5242]
    
    w = 2.0 * math.pi * _freq[1:]
    w = w[::-1]
    # Here we do the response removal
    spec = (w ** 2) * spec / respamp

    # avoid calculating log of zero
    dtiny = np.finfo(0.0).tiny
    idx = spec < dtiny
    spec[idx] = dtiny

    # go to dB
    spec = np.log10(spec)
    spec *= 10
    
    smoothed_psd = []

    # do this for the whole period range and append the values to our lists
    for per_left, per_right in zip(period_bin_left_edges,
                                   period_bin_right_edges):
        specs = spec[(per_left <= psd_periods) &
                     (psd_periods <= per_right)]
        smoothed_psd.append(specs.mean())
    smoothed_psd = np.array(smoothed_psd, dtype=np.float32)
    insert_processed_data(times_processed,binned_psds,
                          trace.stats.starttime,smoothed_psd)
    return True

def fft_taper(data):
    """
    Cosine taper, 10 percent at each end (like done by [McNamara2004]_).

    .. warning::
        Inplace operation, so data should be float.
    """
    data *= cosine_taper(len(data), 0.2)
    return data

def get_response(metadata,iid,nfft,trace):
    # check type of metadata and use the correct subroutine
    # first, to save some time, tried to do this in __init__ like:
    #   self._get_response = self._get_response_from_inventory
    # but that makes the object non-picklable
    if isinstance(metadata, Inventory):
        return get_response_from_inventory(metadata,iid,nfft,trace)
    else:
        msg = "Unexpected type for `metadata`: %s" % type(metadata)
        raise TypeError(msg)

def get_response_from_inventory(metadata,iid,nfft,trace):
    inventory = metadata
    response = inventory.get_response(iid,trace.stats.starttime)
    resp, _ = response.get_evalresp_response(
        t_samp=trace.stats.delta, nfft=nfft, output="VEL")
    return resp

def insert_processed_data(times_processed,binned_psds,utcdatetime,spectrum):
    t = utcdatetime._ns
    ind = bisect.bisect(times_processed, t)
    times_processed.insert(ind, t)
    binned_psds.insert(ind, spectrum)

def stack_selection(current_times_all_details,times_processed, starttime=None,
                    endtime=None):
    """
    For details on restrictions see :meth:`calculate_histogram`.

    :rtype: :class:`numpy.ndarray` of bool
    :returns: Boolean array of which psd pieces should be included in the
        stack.
    """
    times_all = np.array(times_processed)
    selected = np.ones(len(times_all), dtype=bool)
    if starttime is not None:
        selected &= times_all >= starttime._ns
    if endtime is not None:
        selected &= times_all <= endtime._ns
    return selected

def get_times_all_details(current_times_all_details,times_processed):
    # check if we can reuse a previously cached array of all times as
    # day of week as int and time of day in float hours
    if len(current_times_all_details) == len(times_processed):
        return current_times_all_details
    # otherwise compute it and store it for subsequent stacks on the
    # same data (has to be recomputed when additional data gets added)
    else:
        dtype = np.dtype([('time_of_day', np.float32),
                          ('iso_weekday', np.int8),
                          ('iso_week', np.int8),
                          ('year', np.int16),
                          ('month', np.int8)])
        times_all_details = np.empty(shape=len(times_processed),
                                     dtype=dtype)
        utc_times_all = [UTCDateTime(ns=t) for t in times_processed]
        times_all_details['time_of_day'][:] = \
            [t._get_hours_after_midnight() for t in utc_times_all]
        times_all_details['iso_weekday'][:] = \
            [t.isoweekday() for t in utc_times_all]
        times_all_details['iso_week'][:] = \
            [t.isocalendar()[1] for t in utc_times_all]
        times_all_details['year'][:] = \
            [t.year for t in utc_times_all]
        times_all_details['month'][:] = [t.month for t in utc_times_all]
        current_times_all_details = times_all_details
        return times_all_details

def plot_histogram(fig, current_hist_stack, current_times_used,
                   period_xedges, db_bin_edges, draw, filename):
    """
    Reuse a previously created figure returned by `plot(show=False)`
    and plot the current histogram stack (pre-computed using
    :meth:`calculate_histogram()`) into the figure. If a filename is
    provided, the figure will be saved to a local file.
    Note that many aspects of the plot are statically set during the first
    :meth:`plot()` call, so this routine can only be used to update with
    data from a new stack.
    """
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    
    #Normalement utilisé avec la fonction check_histogram
    current_histogram=current_hist_stack
    current_histogram_count=len(current_times_used)
    
    if fig.cumulative is False: #OK
        # avoid divison with zero in case of empty stack
        data = (current_histogram * 100.0 / (current_histogram_count or 1))
    else:
        msg = "fig.cumulative est True"
        raise TypeError(msg)

    xedges = period_xedges
    if fig.xaxis_frequency:
        xedges = 1.0 / xedges
    
    ##fig.ppsd n'existe pas donc à voir
    fig.ppsd=AttribDict()
    if "meshgrid" not in fig.ppsd:
        fig.meshgrid = np.meshgrid(xedges,db_bin_edges)
        
    ppsd = ax.pcolormesh(fig.meshgrid[0], fig.meshgrid[1], data.T,
        cmap=fig.cmap, zorder=-1)
    fig.quadmesh = ppsd

    if "colorbar" not in fig.ppsd:
        cb = plt.colorbar(ppsd,ax=ax)
        cb.mappable.set_clim(*fig.color_limits)
        cb.set_label(fig.label)
        fig.colorbar = cb

    if fig.max_percentage is not None:
        ppsd.set_clim(*fig.color_limits)

    if fig.grid:
        if fig.cmap.name == "viridis": #OK
            color = {"color": "0.7"}
        else: #NOPE
            color = {}
        ax.grid(True, which="major", **color)
        ax.grid(True, which="minor", **color)

    ax.set_xlim(*xlim)

    if filename is not None:
        plt.savefig(filename)
    elif draw:
        with np.errstate(under="ignore"):
            plt.draw()
    return fig
