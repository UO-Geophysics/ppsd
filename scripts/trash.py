#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:36:51 2022
Update  on Mon Dec 28 

@author: loispapin
"""

#################

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","black"])

#################

it=0;
for traceZ in stream:
    print(traceZ)
    it+=1
    ch = traceZ.stats.channel
    if ch == 'EHZ':
        trace+=traceZ
        print(traceZ)

#################

# Explications bins : inds 1 et inds 2

# for "inds" now a number of ..
#   - 0 means below lowest bin (bin index 0)
#   - 1 means, hit lowest bin (bin index 0)
#   - ..
#   - len(self.db_bin_edges) means above top bin

# for "inds" now a number of ..
#   - -1 means below lowest bin (bin index 0)
#   - 0 means, hit lowest bin (bin index 0)
#   - ..
#   - (len(self.db_bin_edges)-1) means above top bin
# values that are left of first bin edge have to be moved back into the
# binning

#################

# Ligne 263

# calculate and set the cumulative version (i.e. going from 0 to 1 from
# low to high psd values for every period column) of the current
# histogram stack.
# sum up the columns to cumulative entries
hist_stack_cumul = hist_stack.cumsum(axis=1)
# normalize every column with its overall number of entries
# (can vary from the number of self.times_processed because of values
#  outside the histogram db ranges)
norm = hist_stack_cumul[:, -1].copy().astype(np.float64)
# avoid zero division
norm[norm == 0] = 1
hist_stack_cumul = (hist_stack_cumul.T / norm).T

#################

# LIGNE 296

percentiles=[0, 25, 50, 75, 100]
show_percentiles=False
show_mode=False
show_mean=False
show_coverage=False #True

# Parameters

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

if special_handling == "infrasound":
    # Use IDC global infrasound models
    models = (get_idc_infra_hi_noise(), get_idc_infra_low_noise())
else: #PB LECTURE FICHIER MAIS POSSIBLE
    # Use Peterson NHNM and NLNM
    models = (get_nhnm(), get_nlnm())

if show_coverage:
    ax = fig.add_axes([0.12, 0.3, 0.90, 0.6])
    ax2 = fig.add_axes([0.15, 0.17, 0.7, 0.04])
else: #OK
    ax = fig.add_subplot(111)

if show_percentiles: #NOPE
    # for every period look up the approximate place of the percentiles
    for percentile in percentiles:
        periods, percentile_values = \
            self.get_percentile(percentile=percentile)
        if xaxis_frequency:
            xdata = 1.0 / periods
        else:
            xdata = periods
        ax.plot(xdata, percentile_values, color="black", zorder=8)

if show_mode: #NOPE
    periods, mode_ = self.get_mode()
    if xaxis_frequency:
        xdata = 1.0 / periods
    else:
        xdata = periods
    if cmap.name == "viridis":
        color = "0.8"
    else:
        color = "black"
    ax.plot(xdata, mode_, color=color, zorder=9)

if show_mean: #NOPE
    periods, mean_ = self.get_mean()
    if xaxis_frequency:
        xdata = 1.0 / periods
    else:
        xdata = periods
    if cmap.name == "viridis":
        color = "0.8"
    else:
        color = "black"
    ax.plot(xdata, mean_, color=color, zorder=9)
    
if show_coverage: #NOPE
    self.__plot_coverage(ax2)
    # emulating fig.autofmt_xdate():
    for label in ax2.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(30)

