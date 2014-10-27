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

import sys
import os
import string

stdFont = 'Arial'

from scipy.stats import gaussian_kde
import numpy as np
try:
    import pyqtgraph as pg
    from PyQt4 import QtCore, QtGui
except:
    pass

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
        print 'PlotHelpers: cannot use list as source to set Y axis'
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

def crossAxes(axl, xyzero=[0., 0.], limits=[None, None, None, None]):
    """
    Make the plot(s) have crossed axes at the data points set by xyzero, and optionally
    set axes limits
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
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


def labelUp(plot, xtext, ytext, title=None, label=None, **kwargs):
    """
        helper to label up the plot
        Inputs: plot item
                text for x axis
                text for yaxis
                plot title (on top)
                plot panel label (for example, "A", "A1")
    """


    plot.setLabel('bottom', xtext)
    plot.setLabel('left', ytext)
    if title is not None:
        plot.setTitle(title="<b><large>%s</large></b>" % title, visible=True)
        
    else:
        plot.setTitle(title=" ")
    if label is not None:
        setPlotLabel(plot, plotlabel="<b>%s</b>" % label, **kwargs)
    else:
        setPlotLabel(plot, plotlabel="")

def setPlotLabel(plotitem, plotlabel='', **kwargs):
    """
    Set the plotlabel of a plotitem. Basic HTML formatting is allowed.
    If plotlabel is None, then a blank label is used
    A plotlabel is a text label that appears the upper left corner of the
    QGridLayout (position 0,0) of the plotitem.
    """
    
    plotitem.LabelItem = pg.LabelItem(plotlabel, **kwargs)
    plotitem.LabelItem.setMaximumHeight(30)
    plotitem.layout.setRowFixedHeight(0, 30)
    plotitem.layout.addItem(plotitem.LabelItem, 0, 0)
    plotitem.LabelItem.setVisible(True)


class LayoutMaker():
    def __init__(self, win=None, cols=1, rows=1, letters=True, margins=4, spacing=4):
        self.sequential_letters = string.ascii_uppercase
        self.cols=cols
        self.rows=rows
        self.letters=letters
        self.margins=margins
        self.spacing=spacing
        self.rcmap = [None]*cols*rows
        self.plots = None
        self.win = win
        self._makeLayout(letters=letters, margins=margins, spacing=spacing)
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
        self.getPlot(index).plot(*args, **kwargs)

    def _makeLayout(self, letters=True, margins=4, spacing=4):
        """
        Create a multipanel plot.
        The pyptgraph elements (widget, gridlayout, plots) are stored as class variables.
        The layout is always a rectangular grid with shape (cols, rows)
        if letters is true, then the plot is labeled "A, B, C..." Indices move horizontally first, then vertically
        margins sets the margins around the outside of the plot
        spacing sets the spacing between the elements of the grid
        """
        import string
       # self.widget = QtGui.QWidget()
       # self.gridLayout = QtGui.QGridLayout()
        self.gridLayout = self.win.ci.layout  # the window's 'central item' is the main gridlayout.
        # self.widget.setLayout(self.gridLayout)  # don't need to put into an additional widget then
        self.gridLayout.setContentsMargins(margins, margins, margins, margins)
        self.gridLayout.setSpacing(spacing)
        self.plots = [[0 for x in xrange(self.cols)] for x in xrange(self.rows)]
        i = 0
        for r in range(self.rows):
            for c in range(self.cols):
                self.plots[r][c] = self.win.addPlot(row=r, col=c) # pg.PlotWidget()
                #self.gridLayout.addWidget(self.plots[r][c], r, c)
                # if i == 0:
                #     print dir(self.plots[r][c])
                if letters:
                    labelUp(self.plots[r][c], 'T(s)', 'Y', title=self.sequential_letters[i], 
                        label=self.sequential_letters[i], size='24pt', bold=True)
                else:
                    labelUp(self.plots[r][c], 'T(s)', 'Y', label=None, size='12pt', title=self.sequential_letters[i])

                self.rcmap[i] = (r, c)
                i += 1
                if i > 25:
                    i = 0

    def labelEdges(self, xlabel='T(s)', ylabel='Y'):
        """
        label the axes on the outer edges of the gridlayout, leaving the interior axes clean
        """
        (lastrow, lastcol) = self.rcmap[-1]
        i = 0
        for (r,c) in self.rcmap:
            if c == 0:
                ylab = ylabel
            else:
                ylab = ''
            if r == self.rows-1:
                xlab = xlabel
            else:
                xlab = ''
            labelUp(self.plots[r][c], xlab, ylab, label=self.sequential_letters[i])
            i += 1


def figure(title = None, background='w'):
    if background == 'w':
        pg.setConfigOption('background', 'w')  # set background to white
        pg.setConfigOption('foreground', 'k')
    pg.mkQApp()
    win = pg.GraphicsWindow(title=title)
    return win

def show():
    QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    win = figure(title='testing')
    layout = LayoutMaker(cols=4,rows=2, win=win)
    x=np.arange(0, 10., 0.1)
    y = np.sin(x*3.)
    for n in range(4*2):
        layout.plot(n, x, y)
    # win.setLayout(layout.gridLayout)
    layout.labelEdges()
    show()
