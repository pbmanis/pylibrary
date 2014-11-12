#!/usr/bin/env python
# encoding: utf-8
"""
pyqtgraphPlotHelpers.py

Routines to help use pyqtgraph and make cleaner plots
as well as get plots read for publication. 

copied from PlotHelpers.py for matplotlib.
Modified to allow us to use a list of axes, and operate on all of those,
or to use just one axis if that's all that is passed.
Therefore, the first argument to these calls can either be an axes object,
or a list of axes objects.  2/10/2012 pbm.

Created by Paul Manis on 2010-03-09.
Copyright (c) 2010 Paul B. Manis, Ph.D.. All rights reserved.
"""


import string

stdFont = 'Arial'

from scipy.stats import gaussian_kde
import numpy as np
try:
    import pyqtgraph as pg
    from PyQt4 import QtCore, QtGui
except:
    pass
import talbotetalTicks as ticks # logical tick formatting... 

def nice_plot(plotlist, spines = ['left', 'bottom'], position = 10, direction='inward', axesoff = False):
    """ Adjust a plot so that it looks nicer than the default matplotlib plot
        Also allow quickaccess to things we like to do for publication plots, including:
           using a calbar instead of an axes: calbar = [x0, y0, xs, ys]
           inserting a reference line (grey, 3pt dashed, 0.5pt, at refline = y position)
    """
    if type(plotlist) is not list:
        plotlist = [plotlist]
    for pl in plotlist:
        if axesoff is True:
            pl.hideAxis('bottom')
            pl.hideAxis('left')

def noaxes(plotlist, whichaxes = 'xy'):
    """ take away all the axis ticks and the lines"""
    if type(plotlist) is not list:
        plotlist = [plotlist]
    for pl in plotlist:
        if 'x' in whichaxes:
            pl.hideAxis('bottom')
        if 'y' in whichaxes:
            pl.hideAxis('left')

def setY(ax1, ax2):
    if type(ax1) is list:
        print 'PlotHelpers: cannot use list as source to set Y axis'
        return
    if type(ax2) is not list:
        ax2 = [ax2]
    y = ax1.getAxis('left')
    refy = y.range # return the current range
    for ax in ax2:
        ax.setRange(refy)
        
def setX(ax1, ax2):
    if type(ax1) is list:
        print 'PlotHelpers: cannot use list as source to set X axis'
        return
    if type(ax2) is not list:
        ax2 = [ax2]
    x = ax1.getAxis('bottom')
    refx = x.range
    for ax in ax2:
        ax.setrange(refx)

def labelPanels(axl, axlist=None, font='Arial', fontsize=18, weight = 'normal'):
    if type(axl) is dict:
        axt = [axl[x] for x in axl]
        axlist = axl.keys()
        axl = axt
    if type(axl) is not list:
        axl = [axl]
    if axlist is None:
        axlist = string.uppercase(1,len(axl)) # assume we wish to go in sequence

    for i, ax in enumerate(axl):
        labelText = pg.TextItem(axlist[i])
        y = ax.getAxis('left').range
        x = ax.getAxis('bottom').range
        ax.addItem(labelText)
        labelText.setPos(x[0], y[1])

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
    # does nothing at the moment, as axes are already "clean"
    # for ax in axl:
    #
    #    update_font(ax)

def formatTicks(axl, axis='xy', fmt='%d', font='Arial'):
    """
    Convert tick labels to intergers
    to do just one axis, set axis = 'x' or 'y'
    control the format with the formatting string
    """
    if type(axl) is not list:
        axl = [axl]

def autoFormatTicks(axl, axis='xy', font='Arial'):
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if 'x' in axis:
            b = ax.getAxis('bottom')
            x0 = b.range
 #           setFormatter(ax,  x0, x1, axis = 'x')
        if 'y' in axis:
            l = ax.getAxis('left')
            y0= l.range
 #           setFormatter(ax, y0, y1, axis = 'y')

def setFormatter(ax, x0, x1, axis='x'):
    datarange = np.abs(x0-x1)
    mdata = np.ceil(np.log10(datarange))
    # if mdata > 0 and mdata <= 4:
    #     majorFormatter = FormatStrFormatter('%d')
    # elif mdata > 4:
    #     majorFormatter = FormatStrFormatter('%e')
    # elif mdata <= 0 and mdata > -1:
    #     majorFormatter = FormatStrFormatter('%5.1f')
    # elif mdata < -1 and mdata > -3:
    #     majorFormatatter = FormatStrFormatter('%6.3f')
    # else:
    #     majorFormatter = FormatStrFormatter('%e')
    # if axis == 'x':
    #     ax.xaxis.set_major_formatter(majorFormatter)
    # else:
    #     ax.yaxis.set_major_formatter(majorFormatter)


def update_font(axl, size=6, font=stdFont):
    pass
    # if type(axl) is not list:
    #     axl = [axl]
    # fontProperties = {'family':'sans-serif','sans-serif':[font],
    #         'weight' : 'normal', 'size' : size}
    # for ax in axl:
    #     for tick in ax.xaxis.get_major_ticks():
    #           tick.label1.set_family('sans-serif')
    #           tick.label1.set_fontname(stdFont)
    #           tick.label1.set_size(size)
    #
    #     for tick in ax.yaxis.get_major_ticks():
    #           tick.label1.set_family('sans-serif')
    #           tick.label1.set_fontname(stdFont)
    #           tick.label1.set_size(size)
    #     ax.set_xticklabels(ax.get_xticks(), fontProperties)
    #     ax.set_yticklabels(ax.get_yticks(), fontProperties)
    #     ax.xaxis.set_smart_bounds(True)
    #     ax.yaxis.set_smart_bounds(True)
    #     ax.tick_params(axis = 'both', labelsize = 9)

def lockPlot(axl, lims, ticks=None):
    """ 
        This routine forces the plot of invisible data to force the axes to take certain
        limits and to force the tick marks to appear. 
        call with the axis and lims = [x0, x1, y0, y1]
    """ 
    if type(axl) is not list:
        axl = [axl]
    plist = []
    for ax in axl:
        y = ax.getAxis('left')
        x = ax.getAxis('bottom')
        x.setRange(lims[0], lims[1])
        y.setRange(lims[2], lims[3])

def adjust_spines(axl, spines = ('left', 'bottom'), direction = 'outward', distance=5, smart=True):
    pass
    # if type(axl) is not list:
    #     axl = [axl]
    # for ax in axl:
    #     # turn off ticks where there is no spine
    #     if 'left' in spines:
    #         ax.yaxis.set_ticks_position('left')
    #     else:
    #         # no yaxis ticks
    #         ax.yaxis.set_ticks([])
    #
    #     if 'bottom' in spines:
    #         ax.xaxis.set_ticks_position('bottom')
    #     else:
    #         # no xaxis ticks
    #         ax.xaxis.set_ticks([])
    #     for loc, spine in ax.spines.iteritems():
    #         if loc in spines:
    #             spine.set_position((direction,distance)) # outward by 10 points
    #             if smart is True:
    #                 spine.set_smart_bounds(True)
    #             else:
    #                 spine.set_smart_bounds(False)
    #         else:
    #             spine.set_color('none') # don't draw spine
    #     return
    #
def calbar(plotlist, calbar = None, axesoff = True, orient = 'left', unitNames=None):
    """ draw a calibration bar and label it up. The calibration bar is defined as:
        [x0, y0, xlen, ylen]
    """
    if type(plotlist) is not list:
        plotlist = [plotlist]
    for pl in plotlist:
        if axesoff is True:
            noaxes(pl)
        Vfmt = '%.0f'
        if calbar[2] < 1.0:
            Vfmt = '%.1f'
        Hfmt = '%.0f'
        if calbar[3] < 1.0:
            Hfmt = '%.1f'
        if unitNames is not None:
            Vfmt = Vfmt + ' ' + unitNames['x']
            Hfmt = Hfmt + ' ' + unitNames['y']
        Vtxt = pg.TextItem(Vfmt % calbar[2], anchor=(0.5, 0.5), color=pg.mkColor('k'))
        Htxt = pg.TextItem(Hfmt % calbar[3], anchor=(0.5, 0.5), color=pg.mkColor('k'))
       # print pl
        if calbar is not None:
            if orient == 'left': # vertical part is on the left
                pl.plot([calbar[0], calbar[0], calbar[0]+calbar[2]],
                    [calbar[1]+calbar[3], calbar[1], calbar[1]],
                    pen=pg.mkPen('k'), linestyle = '-', linewidth = 1.5)
                ht = Htxt.setPos(calbar[0]+0.05*calbar[2], calbar[1]+0.5*calbar[3])
            elif orient == 'right': # vertical part goes on the right
                pl.plot([calbar[0] + calbar[2], calbar[0]+calbar[2], calbar[0]],
                    [calbar[1]+calbar[3], calbar[1], calbar[1]],
                    pen=pg.mkPen('k'), linestyle = '-', linewidth = 1.5)
                ht = Htxt.setPos(calbar[0]+calbar[2]-0.05*calbar[2], calbar[1]+0.5*calbar[3])
            else:
                print "PlotHelpers.py: I did not understand orientation: %s" % (orient)
                print "plotting as if set to left... "
                pl.plot([calbar[0], calbar[0], calbar[0]+calbar[2]],
                    [calbar[1]+calbar[3], calbar[1], calbar[1]],
                    pen=pg.mkPen('k'), linestyle = '-', linewidth = 1.5)
                ht = Htxt.setPos(calbar[0]+0.05*calbar[2], calbar[1]+0.5*calbar[3])
                Htxt.setText(Hfmt % calbar[3])
            xc = float(calbar[0]+calbar[2]*0.5)  # always centered, below the line
            yc = float(calbar[1]-0.1*calbar[3])
            vt = Vtxt.setPos(xc, yc)
            Vtxt.setText(Vfmt % calbar[2])
            pl.addItem(Htxt)
            pl.addItem(Vtxt)

def refline(axl, refline = None, color = [64, 64, 64], linestyle = '--' ,linewidth = 0.5):
    """ draw a reference line at a particular level of the data on the y axis 
    """
    if type(axl) is not list:
        axl = [axl]
    if linestyle == '--':
        style = QtCore.Qt.DashLine
    elif linestyle == '.':
        style=QtCore.Qt.DotLine
    elif linestyle == '-':
        style=QtCore.Qt.SolidLine
    elif linestyle == '-.':
        style = QtCore.Qt.DsahDotLine
    elif linestyle == '-..':
        style = QtCore.Qt.DashDotDotLine
    else:
        style = QtCore.Qt.SolidLine # default is solid
    for ax in axl:
        if refline is not None:
            x = ax.getAxis('bottom')
            xlims = x.range
            ax.plot(xlims, [refline, refline], pen=pg.mkPen(color, width=linewidth, style=style))

def tickStrings(values, scale=1, spacing=None, tickPlacesAdd = 1):
    """Return the strings that should be placed next to ticks. This method is called 
    when redrawing the axis and is a good method to override in subclasses.
    The method is called with a list of tick values, a scaling factor (see below), and the 
    spacing between ticks (this is required since, in some instances, there may be only 
    one tick and thus no other way to determine the tick spacing)
    
    The scale argument is used when the axis label is displaying units which may have an SI scaling prefix.
    When determining the text to display, use value*scale to correctly account for this prefix.
    For example, if the axis label's units are set to 'V', then a tick value of 0.001 might
    be accompanied by a scale value of 1000. This indicates that the label is displaying 'mV', and 
    thus the tick should display 0.001 * 1000 = 1.
    from pyqtgraph; we need it here.
    """
    if spacing is None:
        spacing = np.mean(np.diff(values))
    places = max(0, np.ceil(-np.log10(spacing*scale))) + tickPlacesAdd
    strings = []
    for v in values:
        vs = v * scale
        if abs(vs) < .001 or abs(vs) >= 10000:
            vstr = "%g" % vs
        else:
            vstr = ("%%0.%df" % places) % vs
        strings.append(vstr)
    return strings


def crossAxes(axl, xyzero=[0., 0.], limits=[None, None, None, None], **kwds):
    """
    Make the plot(s) have crossed axes at the data points set by xyzero, and optionally
    set axes limits
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        make_crossedAxes(ax, xyzero, limits, **kwds)
 
def make_crossedAxes(ax, xyzero=[0., 0.], limits=[None, None, None, None], ndec=3,
            density=(1.0, 1.0), tickl = 0.0125, insideMargin=0.05, pointSize=12, tickPlacesAdd=(0,0)):
    # get axis limits
    aleft = ax.getAxis('left')
    abottom = ax.getAxis('bottom')
    aleft.setPos(pg.Point(3., 0.))
    yRange = aleft.range
    xRange =  abottom.range
    hl = pg.InfiniteLine(pos=xyzero[0], angle=90, pen=pg.mkPen('k'))
    ax.addItem(hl)
    vl = pg.InfiniteLine(pos=xyzero[1], angle=0, pen=pg.mkPen('k'))
    ax.addItem(vl)
    ax.hideAxis('bottom')
    ax.hideAxis('left')
    # now create substitue tick marks and labels, using Talbot et al algorithm
    xr = np.diff(xRange)[0]
    yr = np.diff(yRange)[0]
    xmin, xmax = (np.min(xRange) - xr * insideMargin, np.max(xRange) + xr * insideMargin)
    ymin, ymax = (np.min(yRange) - yr * insideMargin, np.max(yRange) + yr * insideMargin)
    xtick = ticks.Extended(density=density[0], figure=None, range=(xmin, xmax), axis='x')
    ytick = ticks.Extended(density=density[1], figure=None, range=(ymin, ymax), axis='y')
    xt = xtick()
    yt = ytick()
    ytk = yr*tickl
    xtk = xr*tickl
    y0 = xyzero[1]
    x0 = xyzero[0]
    tsx = tickStrings(xt, tickPlacesAdd=tickPlacesAdd[0])
    tsy = tickStrings(yt, tickPlacesAdd=tickPlacesAdd[1])
    for i, x in enumerate(xt):
        t = pg.PlotDataItem(x=x*np.ones(2), y=[y0-ytk, y0+ytk], pen=pg.mkPen('k'))
        ax.addItem(t)  # tick mark
        # put text in only if it does not overlap the opposite line
        if x == y0:
            continue
        txt = pg.TextItem(tsx[i], anchor=(0.5, 0), color=pg.mkColor('k')) #, size='10pt')
        txt.setFont(pg.QtGui.QFont('Arial', pointSize=pointSize))
        txt.setPos(pg.Point(x, y0-ytk))
        ax.addItem(txt) #, pos=pg.Point(x, y0-ytk))
    for i, y in enumerate(yt):
        t = pg.PlotDataItem(x=np.array([x0-xtk, x0+xtk]), y=np.ones(2)*y, pen=pg.mkPen('k'))
        ax.addItem(t)
        if y == x0:
            continue
        txt = pg.TextItem(tsy[i], anchor=(1, 0.5), color=pg.mkColor('k')) # , size='10pt')
        txt.setFont(pg.QtGui.QFont('Arial', pointSize=pointSize))
        txt.setPos(pg.Point(x0-xtk, y))
        ax.addItem(txt) #, pos=pg.Point(x, y0-ytk))

def polar(plot, r, theta, steps=4, rRange=None, vectors=False, sort=True, **kwds):
    """
    plot is the plot instance to plot into
    the plot will be converted to a polar graph
    r is a list or array of radii
    theta is the corresponding list of angles
    steps is the number of grid steps in r
    rRange is the max of r (max of data of not specified)
    vectors False means plots line of r, theta; true means plots vectors from origin to point
    sort allows ordered x (default)
    **kwds are passed to the data plot call.
    """

    plot.setAspectLocked()
    plot.hideAxis('bottom')
    plot.hideAxis('left')

    # sort r, theta by r
    i = np.argsort(theta)
    r = r[i]
    theta = theta[i]
    if rRange is None:
        rRange = np.max(r)
    # Add radial grid lines (theta)
    gridPen = pg.mkPen(width=0.5, color='k',  style=QtCore.Qt.DotLine)
    ringPen = pg.mkPen(width=0.75, color='k',  style=QtCore.Qt.SolidLine)  
    for th in np.linspace(0., np.pi*2, 8, endpoint=False):
        rx = np.cos(th)
        ry = np.sin(th)
        plot.plot(x=[0, rx], y=[0., ry], pen=gridPen)
        ang = th*360./(np.pi*2)
        # anchor is odd: 0,0 is upper left corner, 1,1 is loer right corner
        if ang < 90.:
            x=0.
            y=0.5
        elif ang == 90.:
            x=0.5
            y=1
        elif ang < 180:
            x=1.0
            y=0.5
        elif ang == 180.:
            x=1
            y=0.5
        elif ang < 270:
            x=1
            y=0
        elif ang== 270.:
            x=0.5
            y=0
        elif ang < 360:
            x=0
            y=0
        ti = pg.TextItem("%d" % (int(ang)), color=pg.mkColor('k'), anchor=(x,y))
        plot.addItem(ti)
        ti.setPos(rRange*rx, rRange*ry)
    # add polar grid lines (r)
    for gr in np.linspace(rRange/steps, rRange, steps):
        circle = pg.QtGui.QGraphicsEllipseItem(-gr, -gr, gr*2, gr*2)
        if gr < rRange:
            circle.setPen(gridPen)
        else:
            circle.setPen(ringPen)
        plot.addItem(circle)

    # Transform to cartesian and plot
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if vectors:  # plot r,theta as lines from origin
        for i in range(len(x)):
            plot.plot([0., x[i]], [0., y[i]], **kwds)
    else:
        plot.plot(x, y, **kwds)

def talbotTicks(axl, **kwds):
    """
    Adjust the tick marks using the talbot et al algorithm, on an existing plot.
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        do_talbotTicks(ax, **kwds)

def do_talbotTicks(ax, ndec=3,
                   density=(1.0, 1.0), insideMargin=0.05, pointSize=None, tickPlacesAdd=(0,0)):
    """
    Change the axis ticks to use the talbot algorithm for ONE axis
    Paramerters control the ticks, """
    # get axis limits
    aleft = ax.getAxis('left')
    abottom = ax.getAxis('bottom')
    yRange = aleft.range
    xRange =  abottom.range
    # now create substitue tick marks and labels, using Talbot et al algorithm
    xr = np.diff(xRange)[0]
    yr = np.diff(yRange)[0]
    xmin, xmax = (np.min(xRange) - xr * insideMargin, np.max(xRange) + xr * insideMargin)
    ymin, ymax = (np.min(yRange) - yr * insideMargin, np.max(yRange) + yr * insideMargin)
    xtick = ticks.Extended(density=density[0], figure=None, range=(xmin, xmax), axis='x')
    ytick = ticks.Extended(density=density[1], figure=None, range=(ymin, ymax), axis='y')
    xt = xtick()
    yt = ytick()
    xts = tickStrings(xt, scale=1, spacing=None, tickPlacesAdd = tickPlacesAdd[0])
    yts = tickStrings(yt, scale=1, spacing=None, tickPlacesAdd = tickPlacesAdd[1])
    xtickl = [[(x, xts[i]) for i, x in enumerate(xt)] , []]  # no minor ticks here
    ytickl = [[(y, yts[i]) for i, y in enumerate(yt)] , []]  # no minor ticks here

    #ticks format: [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
    aleft.setTicks(ytickl)
    abottom.setTicks(xtickl)
    
    
    

def violin_plot(ax, data, pos, bp=False):
    '''
    create violin plots on an axis
    '''
    dist = max(pos)-min(pos)
    w = min(0.15*max(dist,1.0),0.5)
    for d,p in zip(data,pos):
        k = gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = np.arange(m, M, (M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v / v.max() * w #scaling the violin to the available space
       # ax.fill_betweenx(x, p, v+p, facecolor='y', alpha=0.3)
       # ax.fill_betweenx(x, p, -v+p, facecolor='y', alpha=0.3)
    if bp:
       pass
       # bpf = ax.boxplot(data, notch=0, positions=pos, vert=1)
       # pylab.setp(bpf['boxes'], color='black')
       # pylab.setp(bpf['whiskers'], color='black', linestyle='-')

def labelAxes(plot, xtext, ytext, **kwargs):
    """
        helper to label up the plot
        Inputs: plot item
                text for x axis
                text for yaxis
                plot title (on top) OR
                plot panel label (for example, "A", "A1")
    """

    plot.setLabel('bottom', xtext, **kwargs)
    plot.setLabel('left', ytext, **kwargs)


def labelPanels(plot, label=None, **kwargs):
    """
        helper to label up the plot
        Inputs: plot item
                text for x axis
                text for yaxis
                plot title (on top) OR
                plot panel label (for example, "A", "A1")
    """

    if label is not None:
        setPlotLabel(plot, plotlabel="%s" % label, **kwargs)
    else:
        setPlotLabel(plot, plotlabel="")

def labelTitles(plot, title=None, **kwargs):
    """
    Set the title of a plotitem. Basic HTML formatting is allowed, along
    with "size", "bold", "italic", etc..
    If the title is not defined, then a blank label is used
    A title is a text label that appears centered above the plot, in 
    QGridLayout (position 0,2) of the plotitem.
    params
    -------
    :param plotitem: The plot item to label
    :param title: The text string to use for the label
    :kwargs: keywords to pass to the pg.LabelItem
    :return: None
    
    """

    if title is not None:
        plot.setTitle(title="<b><large>%s</large></b>" % title, visible=True, **kwargs)    
    else:  # clear the plot title
        plot.setTitle(title=" ")

def setPlotLabel(plotitem, plotlabel='', **kwargs):
    """
    Set the plotlabel of a plotitem. Basic HTML formatting is allowed, along
    with "size", "bold", "italic", etc..
    If plotlabel is not defined, then a blank label is used
    A plotlabel is a text label that appears the upper left corner of the
    QGridLayout (position 0,0) of the plotitem.
    params
    -------
    :param plotitem: The plot item to label
    :param plotlabel: The text string to use for the label
    :kwargs: keywords to pass to the pg.LabelItem
    :return: None
    
    """
    
    plotitem.LabelItem = pg.LabelItem(plotlabel, **kwargs)
    plotitem.LabelItem.setMaximumHeight(30)
    plotitem.layout.setRowFixedHeight(0, 30)
    plotitem.layout.addItem(plotitem.LabelItem, 0, 0)
    plotitem.LabelItem.setVisible(True)


class LayoutMaker():
    def __init__(self, win=None, cols=1, rows=1, letters=True, titles=False, labelEdges=True, margins=4, spacing=4, ticks='default'):
        self.sequential_letters = string.ascii_uppercase
        self.cols = cols
        self.rows = rows
        self.letters = letters
        self.titles = titles
        self.edges = labelEdges
        self.margins = margins
        self.spacing = spacing
        self.rcmap = [None]*cols*rows
        self.plots = None
        self.win = win
        self.ticks = ticks
        self._makeLayout(letters=letters, titles=titles, margins=margins, spacing=spacing)
        #self.addLayout(win)

    # def addLayout(self, win=None):
    #     if win is not None:
    #         win.setLayout(self.gridLayout)

    def getCols(self):
        return self.cols

    def getRows(self):
        return self.rows

    def mapFromIndex(self, index):
        """
        for a given index, return the row, col tuple associated with the index
        """
        return self.rcmap[index]

    def getPlot(self, index):
        """
        return the plot item in the list corresponding to the index n
        """
        if isinstance(index, tuple):
            r, c = index
        elif isinstance(index, int):
            r, c = self.rcmap[index]
        else:
            raise ValueError ('pyqtgraphPlotHelpers, LayoutMaker plot: index must be int or tuple(r,c)')
        return self.plots[r][c]

    def plot(self, index, *args, **kwargs):
        p = self.getPlot(index).plot(*args, **kwargs)
        if self.ticks == 'talbot':
            talbotTicks(self.getPlot(index))
        return p

    def _makeLayout(self, letters=True, titles=True, margins=4, spacing=4):
        """
        Create a multipanel plot.
        The pyptgraph elements (widget, gridlayout, plots) are stored as class variables.
        The layout is always a rectangular grid with shape (cols, rows)
        if letters is true, then the plot is labeled "A, B, C..." Indices move horizontally first, then vertically
        margins sets the margins around the outside of the plot
        spacing sets the spacing between the elements of the grid
        """
        import string
        self.gridLayout = self.win.ci.layout  # the window's 'central item' is the main gridlayout.
        self.gridLayout.setContentsMargins(margins, margins, margins, margins)
        self.gridLayout.setSpacing(spacing)
        self.plots = [[0 for x in xrange(self.cols)] for x in xrange(self.rows)]
        i = 0
        for r in range(self.rows):
            for c in range(self.cols):
                self.plots[r][c] = self.win.addPlot(row=r, col=c) # pg.PlotWidget()
                if letters:
                    labelPanels(self.plots[r][c], label=self.sequential_letters[i], size='14pt', bold=True)
                if titles:
                    labelTitles(self.plots[r][c], title=self.sequential_letters[i], size='14pt', bold=False)

                self.rcmap[i] = (r, c)
                i += 1
                if i > 25:
                    i = 0
        self.labelAxes('T(s)', 'Y', edgeOnly=self.edges)

    def labelAxes(self, xlabel='T(s)', ylabel='Y', edgeOnly=True, **kwargs):
        """
        label the axes on the outer edges of the gridlayout, leaving the interior axes clean
        """
        (lastrow, lastcol) = self.rcmap[-1]
        i = 0
        for (r,c) in self.rcmap:
            if c == 0 or not edgeOnly:
                ylab = ylabel
            else:
                ylab = ''
            if r == self.rows-1 or not edgeOnly:
                xlab = xlabel
            else:
                xlab = ''
            labelAxes(self.plots[r][c], xlab, ylab, **kwargs)
            i += 1

    def columnAutoScale(self, col, axis='left'): 
        """
        autoscale the columns according to the max value in the column.
        Finds outside range of column data, then sets the scale of all plots
        in the column to that range
        """
        atmax = None
        atmin = None
        for (r, c) in self.rcmap:
            if c != col:
                continue
            ax = self.getPlot((r,c))
            thisaxis = ax.getAxis(axis)
            amin, amax = thisaxis.range
            if atmax is None:
                atmax = amax
            else:
                if amax > atmax:
                    atmax = amax
            if atmin is None:
                atmin = amin
            else:
                if amin > atmin:
                    atmin = amin
            
        self.columnSetScale(col, axis=axis, range=(atmin, atmax))
        return(atmin, atmax)
                
    def columnSetScale(self, col, axis='left', range=(0., 1.)):
        """
        Set the column scale
        """
        for (r, c) in self.rcmap:
            if c != col:
                continue
            ax = self.getPlot((r,c))
            if axis == 'left':
                ax.setYRange(range[0], range[1])
            elif axis == 'bottom':
                ax.setXRange(range[0], range[1])

            if self.ticks == 'talbot':
                talbotTicks(ax)

        
    def title(self, index, title='', **kwargs):
        """
        add a title to a specific plot (specified by index) in the layout
        """
        labelTitles(self.getPlot(index), title=title, **kwargs)

def figure(title = None, background='w'):
    if background == 'w':
        pg.setConfigOption('background', 'w')  # set background to white
        pg.setConfigOption('foreground', 'k')
    pg.mkQApp()
    win = pg.GraphicsWindow(title=title)
    return win

def show():
    QtGui.QApplication.instance().exec_()

def test_layout(win):
    layout = LayoutMaker(cols=4,rows=2, win=win, labelEdges=True, ticks='talbot')
    x=np.arange(0, 10., 0.1)
    y = np.sin(x*3.)
    r = np.random.random(10)
    theta = np.linspace(0, 2.*np.pi, 10, endpoint=False)
    for n in range(4*2):
        if n not in [1,2]:
            layout.plot(n, x, y)
        p = layout.getPlot(n)
        if n == 0:
            crossAxes(p, xyzero=[0., 0.], density=(0.75, 1.5), tickPlacesAdd=(1, 0), pointSize=12)
        if n in [1,2]:
            if n == 1:
                polar(p, r, theta, pen=pg.mkPen('r'))
            if n == 2: 
                polar(p, r, theta, vectors=True, pen=pg.mkPen('k', width=2.0))
    layout.title(0, 'this title')
    #talbotTicks(layout.getPlot(1))
    layout.columnAutoScale(col=3, axis='left')
    show()

def test_crossAxes(win):
    layout = LayoutMaker(cols=1,rows=1, win=win, labelEdges=True)
    x=np.arange(-1, 1., 0.01)
    y = np.sin(x*10.)
    layout.plot(0, x, y)
    p = layout.getPlot(0)
    crossAxes(p, xyzero=[0., 0.], limits=[None, None, None, None], density=1.5, tickPlacesAdd=1, pointSize=12)
    show()
    

if __name__ == '__main__':
    win = figure(title='testing')
    test_layout(win)
    #test_crossAxes(win)
    