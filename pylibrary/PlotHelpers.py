#!/usr/bin/env python
# encoding: utf-8
"""
PlotHelpers.py

Routines to help use matplotlib and make cleaner plots
as well as get plots read for publication. 

Modified to allow us to use a list of axes, and operate on all of those, 
or to use just one axis if that's all that is passed.
Therefore, the first argument to these calls can either be an axes object,
or a list of axes objects.  2/10/2012 pbm.

Created by Paul Manis on 2010-03-09.
Copyright 2010-2014  Paul Manis
Distributed under MIT/X11 license. See license.txt for more infofmation.
"""

import sys
import os
import string

stdFont = 'Arial'
import seaborn  # a bit dangerous because it changes defaults, but it has wider capabiities also
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as mpl
seaborn.set_style('white')
seaborn.set_style('ticks')

def nice_plot(axl, spines = ['left', 'bottom'], position = 10, direction='inward', axesoff = False):
    """ Adjust a plot so that it looks nicer than the default matplotlib plot
        Also allow quickaccess to things we like to do for publication plots, including:
           using a calbar instead of an axes: calbar = [x0, y0, xs, ys]
           inserting a reference line (grey, 3pt dashed, 0.5pt, at refline = y position)
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        for loc, spine in ax.spines.iteritems():
            if loc in spines:
                spine.set_position(('outward', position))
                # outward by 10 points
            else:
                spine.set_color('none')
                # don't draw spine
        if axesoff is True:
            noaxes(ax)

        # turn off ticks where there is no spine, if there are axes
        if 'left' in spines and not axesoff:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines and not axesoff:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])
        if direction == 'inward':
            ax.tick_params(axis='y', direction='in')
            ax.tick_params(axis='x', direction='in')
        else:
            ax.tick_params(axis='y', direction='out')
            ax.tick_params(axis='x', direction='out')


def noaxes(axl, whichaxes = 'xy'):
    """ take away all the axis ticks and the lines"""
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if 'x' in whichaxes:
            ax.xaxis.set_ticks([])
        if 'y' in whichaxes:
            ax.yaxis.set_ticks([])
        if 'xy' == whichaxes:
            ax.set_axis_off()


def setY(ax1, ax2):
    if type(ax1) is list:
        print 'PlotHelpers: cannot use list as source to set Y axis'
        return
    if type(ax2) is not list:
        ax2 = [ax2]
    refy = ax1.get_ylim()
    for ax in ax2:
        ax.set_ylim(refy)


def setX(ax1, ax2):
    if type(ax1) is list:
        print 'PlotHelpers: cannot use list as source to set Y axis'
        return
    if type(ax2) is not list:
        ax2 = [ax2]
    refx = ax1.get_xlim()
    for ax in ax2:
        ax.set_xlim(refx)


def labelPanels(axl, axlist=None, font='Arial', fontsize=18, weight='normal', xy=(-1.05, 1.05)):
    if type(axl) is dict:
        axt = [axl[x] for x in axl]
        axlist = axl.keys()
        axl = axt
    if type(axl) is not list:
        axl = [axl]
    if axlist is None:
        axlist = string.uppercase[len(axl)]
        # assume we wish to go in sequence
        assert len(axlist) == len(axl)
    font = FontProperties()
    font.set_family('sans-serif')
    font.set_weight=weight
    font.set_size=fontsize
    font.set_style('normal')
    for i, ax in enumerate(axl):
        if ax is None:
            continue
        ax.annotate(axlist[i], xy=xy, xycoords='axes fraction',
                annotation_clip=False,
                color="k", verticalalignment='bottom',weight=weight, horizontalalignment='right',
                fontsize=fontsize, family='sans-serif',
                )
        # ax.annotate
        # at = TextArea(axlist[i], textprops=dict(color="k", verticalalignment='bottom',
        #     weight=weight, horizontalalignment='right', fontsize=fontsize, family='sans-serif'))
        # box = HPacker(children=[at], align="left", pad=0, sep=2)
        # ab = AnchoredOffsetbox(loc=3, child=box, pad=0., frameon=False, bbox_to_anchor=(-0.08, 1.1),
        #     bbox_transform=ax.transAxes, borderpad=0.)
        # ax.add_artist(ab)
        #text(-0.02, 1.05, axlist[i], verticalalignment='bottom', ha='right', fontproperties = font)


def listAxes(axd):
    """
    make a list of the axes from the dictionary
    """
    if type(axd) is not dict:
        if type(axd) is list:
            return axd
        else:
            print 'listAxes expects dictionary or list; type not known (fix the code)'
            raise
    axl = [axd[x] for x in axd]
    return axl


def cleanAxes(axl):
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        for loc, spine in ax.spines.iteritems():
            if loc in ['left', 'bottom']:
                pass
            elif loc in ['right', 'top']:
                spine.set_color('none')
                # do not draw the spine
            else:
                raise ValueError('Unknown spine location: %s' % loc)
            # turn off ticks when there is no spine
            ax.xaxis.set_ticks_position('bottom')
            #pdb.set_trace()
            ax.yaxis.set_ticks_position('left')  # stopped working in matplotlib 1.10
        update_font(ax)


def setTicks(axl, axis='x', ticks=np.arange(0, 1.1, 1.0)):
    if type(axl) is dict:
        axl = [axl[x] for x in axl.keys()]
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if axis == 'x':
            ax.set_xticks(ticks)
        if axis == 'y':
            ax.set_yticks(ticks)


def formatTicks(axl, axis='xy', fmt='%d', font='Arial'):
    """
    Convert tick labels to integers
    To do just one axis, set axis = 'x' or 'y'
    Control the format with the formatting string
    """
    if type(axl) is not list:
        axl = [axl]
    majorFormatter = FormatStrFormatter(fmt)
    for ax in axl:
        if ax is None:
            continue
        if 'x' in axis:
            ax.xaxis.set_major_formatter(majorFormatter)
        if 'y' in axis:
            ax.yaxis.set_major_formatter(majorFormatter)


def autoFormatTicks(axl, axis='xy', font='Arial'):
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if 'x' in axis:
        #    print ax.get_xlim()
            x0, x1= ax.get_xlim()
            setFormatter(ax,  x0, x1, axis = 'x')
        if 'y' in axis:
            y0, y1= ax.get_xlim
            setFormatter(ax, y0, y1, axis = 'y')


def setFormatter(ax, x0, x1, axis='x'):
    datarange = np.abs(x0-x1)
    mdata = np.ceil(np.log10(datarange))
    if mdata > 0 and mdata <= 4:
        majorFormatter = FormatStrFormatter('%d')
    elif mdata > 4:
        majorFormatter = FormatStrFormatter('%e')
    elif mdata <= 0 and mdata > -1:
        majorFormatter = FormatStrFormatter('%5.1f')
    elif mdata < -1 and mdata > -3:
        majorFormatatter = FormatStrFormatter('%6.3f')
    else:
        majorFormatter = FormatStrFormatter('%e')
    if axis == 'x':
        ax.xaxis.set_major_formatter(majorFormatter)
    else:
        ax.yaxis.set_major_formatter(majorFormatter)


def update_font(axl, size=9, font=stdFont):
    if type(axl) is not list:
        axl = [axl]
    fontProperties = {'family':'sans-serif', #'sans-serif': font,
            'weight' : 'normal', 'size' : size}
    for ax in axl:
        if ax is None:
            continue
        for tick in ax.xaxis.get_major_ticks():
              #tick.label1.set_family('sans-serif')
            #  tick.label1.set_fontname(stdFont)
              tick.label1.set_size(size)

        for tick in ax.yaxis.get_major_ticks():
             # tick.label1.set_family('sans-serif')
            #  tick.label1.set_fontname(stdFont)
              tick.label1.set_size(size)
        ax.set_xticklabels(ax.get_xticks(), fontProperties)
        ax.set_yticklabels(ax.get_yticks(), fontProperties)
        ax.xaxis.set_smart_bounds(True)
        ax.yaxis.set_smart_bounds(True) 
        ax.tick_params(axis = 'both', labelsize = size)


def lockPlot(axl, lims, ticks=None):
    """ 
        This routine forces the plot of invisible data to force the axes to take certain
        limits and to force the tick marks to appear. 
        call with the axis and lims (limits) = [x0, x1, y0, y1]
    """ 
    if type(axl) is not list:
        axl = [axl]
    plist = []
    for ax in axl:
        if ax is None:
            continue
        lpl = ax.plot([lims[0], lims[0], lims[1], lims[1]], [lims[2], lims[3], lims[2], lims[3]],
            color='none', marker='', linestyle='None')
        plist.extend(lpl)
        ax.axis(lims)
    return(plist)  # just in case you want to modify these plots later.


def adjust_spines(axl, spines = ['left', 'bottom'], direction = 'outward', distance=5, smart=True):
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])
        for loc, spine in ax.spines.iteritems():
            if loc in spines:
                spine.set_position((direction,distance)) # outward by 10 points
                if smart is True:
                    spine.set_smart_bounds(True)
                else:
                    spine.set_smart_bounds(False)
            else:
                spine.set_color('none')  # don't draw spine

        
def calbar(axl, calbar = None, axesoff = True, orient = 'left', unitNames=None):
    """
        draw a calibration bar and label it. The calibration bar is defined as:
        [x0, y0, xlen, ylen]
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if axesoff is True:
            noaxes(ax)
        Vfmt = '%.0f'
        if calbar[2] < 1.0:
            Vfmt = '%.1f'
        Hfmt = '%.0f'
        if calbar[3] < 1.0:
            Hfmt = '%.1f'
        if unitNames is not None:
            Vfmt = Vfmt + ' ' + unitNames['x']
            Hfmt = Hfmt + ' ' + unitNames['y']
        if calbar is not None:
            if orient == 'left':  # vertical part is on the left
                ax.plot([calbar[0], calbar[0], calbar[0]+calbar[2]], 
                    [calbar[1]+calbar[3], calbar[1], calbar[1]],
                    color = 'k', linestyle = '-', linewidth = 1.5)
                ax.text(calbar[0]+0.05*calbar[2], calbar[1]+0.5*calbar[3], Hfmt % calbar[3], 
                    horizontalalignment = 'left', verticalalignment = 'center',
                    fontsize = 11)
            elif orient == 'right':  # vertical part goes on the right
                ax.plot([calbar[0] + calbar[2], calbar[0]+calbar[2], calbar[0]], 
                    [calbar[1]+calbar[3], calbar[1], calbar[1]],
                    color = 'k', linestyle = '-', linewidth = 1.5)
                ax.text(calbar[0]+calbar[2]-0.05*calbar[2], calbar[1]+0.5*calbar[3], Hfmt % calbar[3], 
                    horizontalalignment = 'right', verticalalignment = 'center',
                    fontsize = 11)
            else:
                print "PlotHelpers.py: I did not understand orientation: %s" % (orient)
                print "plotting as if set to left... "
                ax.plot([calbar[0], calbar[0], calbar[0]+calbar[2]], 
                    [calbar[1]+calbar[3], calbar[1], calbar[1]],
                    color = 'k', linestyle = '-', linewidth = 1.5)
                ax.text(calbar[0]+0.05*calbar[2], calbar[1]+0.5*calbar[3], Hfmt % calbar[3], 
                    horizontalalignment = 'left', verticalalignment = 'center',
                    fontsize = 11)
            ax.text(calbar[0]+calbar[2]*0.5, calbar[1]-0.1*calbar[3], Vfmt % calbar[2], 
                horizontalalignment = 'center', verticalalignment = 'top',
                fontsize = 11)


def refline(axl, refline = None, color = '0.33', linestyle = '--' ,linewidth = 0.5):
    """
    draw a reference line at a particular level of the data on the y axis
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if refline is not None:
            xlims = ax.get_xlim()
            ax.plot([xlims[0], xlims[1]], [refline, refline], color = color, linestyle=linestyle, linewidth=linewidth)


def crossAxes(axl, xyzero=[0., 0.], limits=[None, None, None, None]):
    """
    Make plot(s) with crossed axes at the data points set by xyzero, and optionally
    set axes limits
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
#        ax.set_title('spines at data (1,2)')
#        ax.plot(x,y)
        ax.spines['left'].set_position(('data',xyzero[0]))
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position(('data',xyzero[1]))
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if limits[0] is not None:
            ax.set_xlim(left=limits[0], right=limits[2])
            ax.set_ylim(bottom=limits[1], top=limits[3])
            
def violin_plot(ax, data, pos, bp=False, median = False):
    '''
    create violin plots on an axis
    '''
    dist = max(pos)-min(pos)
    w = min(0.15*max(dist,1.0),0.5)
    for d,p in zip(data,pos):
        k = gaussian_kde(d)  #calculates the kernel density
        m = k.dataset.min()  #lower bound of violin
        M = k.dataset.max()  #upper bound of violin
        x = np.arange(m, M, (M-m)/100.)  # support for violin
        v = k.evaluate(x)  #violin profile (density curve)
        v = v / v.max() * w  #scaling the violin to the available space
        ax.fill_betweenx(x, p, v+p, facecolor='y', alpha=0.3)
        ax.fill_betweenx(x, p, -v+p, facecolor='y', alpha=0.3)
        if median:
            ax.plot([p-0.5, p+0.5], [np.median(d), np.median(d)], '-')
    if bp:
        bpf = ax.boxplot(data, notch=0, positions=pos, vert=1)
        mpl.setp(bpf['boxes'], color='black')
        mpl.setp(bpf['whiskers'], color='black', linestyle='-')


if __name__ == '__main__':
    from collections import OrderedDict
    hfig, ax = mpl.subplots(2, 3)
    axd = OrderedDict()
    for i, a in enumerate(ax.flatten()):
        label = string.uppercase[i]
        axd[label] = a
    for a in axd:
        axd[a].plot(np.random.random(10), np.random.random(10))
    nice_plot([axd[a] for a in axd])
    cleanAxes([axd['B'], axd['C']])
    calbar([axd['B'], axd['C']], calbar=[0.5, 0.5, 0.2, 0.2])
    labelPanels([axd[a] for a in axd], axd.keys())
    mpl.tight_layout(pad=2, w_pad=0.5, h_pad=2.0)
    mpl.show()
    
               