__author__ = 'pbmanis'
"""
Routines to export pyqtgraph window or graphicsLayout items to a matplotlib window.
Basic call is:
matplotlibExport(object=[QWindow | QGraphicsLayout], title='text', show=True)

See main routine for how to run examples with either a QWindow or a QGraphicsLayout

Copyright 2014  Paul Manis and Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infofmation.
"""

import re
try:
    from PyQt4 import QtGui, QtCore
except:
    try:
        from PyQt5 import QtGui, QtCore
    except:
        raise ImportError()

import pyqtgraph as pg
import numpy as np

try:
    import matplotlib as MP
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.pyplot as pylab
    import matplotlib.gridspec as gridspec
    import matplotlib.gridspec as GS
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

if HAVE_MPL:
    #MP.use('TKAgg')
    # Do not modify the following code
    # sets up matplotlib with sans-serif plotting...
    pylab.rcParams['text.usetex'] = True
    pylab.rcParams['interactive'] = False
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = 'Arial'
    pylab.rcParams['mathtext.default'] = 'sf'
    pylab.rcParams['figure.facecolor'] = 'white'
    # next setting allows pdf font to be readable in Adobe Illustrator
    pylab.rcParams['pdf.fonttype'] = 42
    #pylab.rcParams['text.dvipnghack'] = True
    # to here (matplotlib stuff - touchy!)

stdFont = 'Arial'

def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    for c in s:
            if c == '<' and not quote:
                tag = True
            elif c == '>' and not quote:
                tag = False
            elif (c == '"' or c == "'") and tag:
                quote = not quote
            elif not tag:
                out = out + c
    return out

def cleanRepl(matchobj):
    """
        Clean up a directory name so that it can be written to a
        matplotlib title without encountering LaTeX escape sequences
        Replace backslashes with forward slashes
        replace underscores (subscript) with escaped underscores
    """
    if matchobj.group(0) == '\\':
        return '/'
    if matchobj.group(0) == '_':
        return '\_'
    if matchobj.group(0) == '/':
        return '/'
    else:
        return ''

def matplotlibExport(object=None, title=None, show=True, size=None):
    """
    Constructs a matplotlib window that shows the current plots laid out in the same
    format as the pyqtgraph window
    You might use this for publication purposes, since matplotlib allows export
    of the window to a variety of formats, and will contain proper fonts (not "outlined").
    Also can be used for automatic generation of PDF files with savefig.

    :param: object is either a QWindow or a QtGridLayout object that specifies
            how the grid was built
            The layout will contain pyqtgraph widgets added with .addLayout
    :return: nothing

    """
    if not HAVE_MPL:
        raise Exception("Method matplotlibExport requires matplotlib; not importable.")
    if object.__class__ == pg.graphicsWindows.GraphicsWindow:
        gridlayout = object.ci.layout
        if gridlayout is None or gridlayout.__class__ != QtGui.QGraphicsGridLayout().__class__:
            raise Exception("Method matplotlibExport: With graphicsWindow, requires a central item of type QGraphicsGridLayout")
        ltype='QGGL'
    elif object.__class__ == QtGui.QGridLayout().__class__:
            gridlayout = object
            ltype = 'QGL'
    else:
        raise Exception("Method matplotlibExport requires Window or gridlayout as first argument (object=)")
    if size is not None:
        pylab.rcParams['figure.figsize'] = size[0], size[1]
    fig = pylab.figure()  # create the matplotlib figure
    pylab.rcParams['text.usetex'] = False
    # escape filename information so it can be rendered by removing
    # common characters that trip up latex...:
    escs = re.compile('[\\\/_]')
    #print title
    if title is not None:
        title = remove_html_markup(title)
        tiname = '%r' % title
        tiname = re.sub(escs, cleanRepl, tiname)[1:-1]
        fig.suptitle(r''+tiname)
    pylab.autoscale(enable=True, axis='both', tight=None)
    # build the plot based on the grid layout
    gs = gridspec.GridSpec(gridlayout.rowCount(), gridlayout.columnCount())  # build matplotlib gridspec
    if ltype == 'QGGL':  # object of type QGraphicsGridLayout
        for r in range(gridlayout.rowCount()):
            for c in range(gridlayout.columnCount()):
                panel = gridlayout.itemAt(r, c) #.widget()  # retrieve the plot widget...
                if panel is not None:
                    mplax = pylab.subplot(gs[r, c])  # map to mpl subplot geometry
                    export_panel(panel, mplax)  # now fill the plot
    elif ltype == 'QGL': # object of type QGridLayout
        for i in range(gridlayout.count()):
            w = gridlayout.itemAt(i).widget()  # retrieve the plot widget...
            (x, y, c, r) = gridlayout.getItemPosition(i)  # and gridspecs paramters
            mplax = pylab.subplot(gs[x:(c+x), y:(r+y)])  # map to mpl subplot geometry
            export_panel(w.getPlotItem(), mplax)  # now fill the ploti += 1
    else:
        raise ValueError ('Object must be a QWindow with a QGraphicsGridLayout, or a QGridLayout; neither received')
    gs.update(wspace=0.25, hspace=0.5)  # adjust spacing
# hook to save figure - not used here, but could be added as a flag.
#       pylab.savefig(os.path.join(outputfile))
    if show:
        pylab.show()

def export_panel(plitem, ax):
    """
    export_panel recreates the contents of one pyqtgraph PlotItem into a specified
    matplotlib axis item.
    Handles PlotItem types of PlotCurveItem, PlotDataItem, BarGraphItem, and ScatterPlotItem
    :param plitem: The PlotItem holding a single plot
    :param ax: the matplotlib axis to put the result into
    :return: Nothing
    """
    # get labels from the pyqtgraph PlotItems
    xlabel = plitem.axes['bottom']['item'].label.toPlainText()
    ylabel = plitem.axes['left']['item'].label.toPlainText()
    title = remove_html_markup(plitem.titleLabel.text)
    label = remove_html_markup(plitem.plotLabel.text)
    fontsize = 12
    fn = pg.functions
    ax.clear()
    cleanAxes(ax)  # make a "nice" plot
    ax.set_title(title)  # add the plot title here
    for item in plitem.items:  # was .curves, but let's handle all items
        # dispatch do different routines depending on what we need to plot
        if isinstance(item, pg.graphicsItems.PlotCurveItem.PlotCurveItem):
            export_curve(fn, ax, item)
        elif isinstance(item, pg.graphicsItems.PlotDataItem.PlotDataItem):
            export_curve(fn, ax, item)
        elif isinstance(item, pg.graphicsItems.BarGraphItem.BarGraphItem):
            export_bargraph(fn, ax, item)
        elif isinstance(item, pg.graphicsItems.ScatterPlotItem.ScatterPlotItem):
            export_scatterplot(fn, ax, item)
        else:
            print ('unknown item encountered : ', item)
            continue
    xr, yr = plitem.viewRange()
    # now clean up the matplotlib/pylab plot and annotations
    ax.set_xbound(*xr)
    ax.set_ybound(*yr)
    at = TextArea(label, textprops=dict(color="k", verticalalignment='bottom',
        weight='bold', horizontalalignment='right', fontsize=fontsize, family='sans-serif'))
    box = HPacker(children=[at], align="left", pad=0, sep=2)
    ab = AnchoredOffsetbox(loc=3, child=box, pad=0., frameon=False, bbox_to_anchor=(-0.08, 1.1),
        bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(ab)
    ax.set_xlabel(xlabel)  # place the axis labels.
    ax.set_ylabel(ylabel)

def export_curve(fn, ax, item):
    """
    Export a single curve into a plot, retaining color, linewidth, symbols, and linetype
    :param fn: pg.functions - how to make the pens, etc.
    :param ax: The matplotlib axis to plot into
    :param item: The pyqtgraph PlotItem to replicate
    :return: None
    """
    x, y = item.getData()
    opts = item.opts
    pen = fn.mkPen(opts['pen'])
    dashPattern = None
    linestyles={QtCore.Qt.NoPen: '',         QtCore.Qt.SolidLine: '-', 
                QtCore.Qt.DashLine: '--',    QtCore.Qt.DotLine: ':', 
                QtCore.Qt.DashDotLine: '-.', QtCore.Qt.DashDotDotLine: '_.',
                QtCore.Qt.CustomDashLine: '--'}
    if pen.style() in linestyles.keys():
        linestyle = linestyles[pen.style()]
    else:
        linestyle = '-'
    if len(pen.dashPattern()) > 0:  # get the dash pattern
        dashPattern = pen.dashPattern()
    color = tuple([c/255. for c in fn.colorTuple(pen.color())])
    if 'symbol' in opts:  # handle symbols
        symbol = opts['symbol']
        if symbol == 't':
            symbol = '^'
        symbolPen = fn.mkPen(opts['symbolPen'])
        symbolBrush = fn.mkBrush(opts['symbolBrush'])
        markeredgecolor = tuple([c/255. for c in fn.colorTuple(symbolPen.color())])
        markerfacecolor = tuple([c/255. for c in fn.colorTuple(symbolBrush.color())])
        markersize = opts['symbolSize']
    else:
        symbol = None
        markeredgecolor = None
        markerfacecolor = None
        markersize = 1.0
    if 'filllevel' in opts:  # fill between level and data.
        if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
            fillBrush = fn.mkBrush(opts['fillBrush'])
            fillcolor = tuple([c/255. for c in fn.colorTuple(fillBrush.color())])
            ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
    if 'stepMode' in opts and opts['stepMode']: # handle step mode (bar graph/histogram)
        if 'fillBrush' in opts:
            fillBrush = fn.mkBrush(opts['fillBrush'])
            fillcolor = tuple([c/255. for c in fn.colorTuple(fillBrush.color())])
        else:
            fillcolor = 'k'
        edgecolor = fillcolor
        if len(y) > len(x):
            y = y[:len(x)]
        if len(x) > len(y):
            x = x[:len(y)]
        print ('len x,y: ', len(x), len(y))
        pl = ax.bar(x, y, width=1.0, color=fillcolor, edgecolor=edgecolor, linewidth=pen.width())
        #ax.bar(left, height, width=1.0, bottom=None, hold=None, **kwargs)
    else: # any but step mode gets a regular plot
        line = MP.lines.Line2D(x, y, marker=symbol, color=color, linewidth=pen.width(),
                 linestyle=linestyle, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor,
                 markersize=markersize)
        if dashPattern is not None:  # add dashpattern if ther was one set...
            line.set_dashes(dashPattern)
        pl = ax.add_line(line)
    return pl  # incase we need the plot that was created.

def export_scatterplot(fn, ax, item):
    """
    Export a scatter plot into matplotlib plot, retaining color and symbols
    :param fn: pg.functions - how to make the pens, etc.
    :param ax: The matplotlib axis to plot into
    :param item: The pyqtgraph PlotItem to replicate
    :return: None
    """
    x, y = item.getData()
    opts = item.opts
    pen = fn.mkPen(opts['pen'])
    brush = fn.mkBrush(opts['brush'])
    if pen.style() == QtCore.Qt.NoPen:
        linestyle = ''
    else:
        linestyle = '-'
    color = tuple([c/255. for c in fn.colorTuple(brush.color())])
    if 'symbol' in opts:
        symbol = opts['symbol']
        if symbol == 't':  # remap the symbols - just one to change.
            symbol = '^'
        symbolPen = fn.mkPen(opts['pen'])
        symbolBrush = fn.mkBrush(opts['brush'])
        markeredgecolor = tuple([c/255. for c in fn.colorTuple(symbolPen.color())])
        markerfacecolor = tuple([c/255. for c in fn.colorTuple(symbolBrush.color())])
        markersize = np.pi*(opts['size']/2.)**2  # matplotlib expresses size as area
    else:
        symbol = None
        markeredgecolor = None
        markerfacecolor = None
        markersize = 1.0

    pl = ax.scatter(x, y, marker=symbol, c=color, linewidth=pen.width(),
                 linestyle=linestyle, s=markersize, edgecolors=markeredgecolor)
    return pl  # in case we need the plot that was created.

def export_bargraph(fn, ax, item):
    """
    Export a bargraph into matplotlib, retaining color and width
    :param fn: pg.functions - how to make the pens, etc.
    :param ax: The matplotlib axis to plot into
    :param item: The pyqtgraph PlotItem to replicate
    :return: None
    """
    if item.opts['x'] is not None:
        x = item.opts['x']
        xalign = 0
    else:
        x = item.opts['x0']
        xalign = 1
    if item.opts['height'] is not None:
        y = item.opts['height']
        yalign = 0
    else:
        y = item.opts['y']
    barwidth = item.opts['width']
    opts = item.opts
    pen = fn.mkPen(opts['pen'])
    brush = fn.mkBrush(opts['brush'])
#    print 'bargraph pen: ', pen
#    print pen.style()
    if pen.style() == QtCore.Qt.NoPen or pen.style() == 0:
        edgecolor = 'None'
    else:
        edgecolor = [tuple([c/255. for c in fn.colorTuple(pen.color())])]
    color = [tuple([c/255. for c in fn.colorTuple(brush.color())])]
    symcolor = color
    if 'symbol' in opts:
        symbol = opts['symbol']
        if symbol == 't':
            symbol = '^'
        symbolPen = fn.mkPen(opts['symbolPen'])
        symbolBrush = fn.mkBrush(opts['symbolBrush'])
        markeredgecolor = tuple([c/255. for c in fn.colorTuple(symbolPen.color())])
        markerfacecolor = tuple([c/255. for c in fn.colorTuple(symbolBrush.color())])
        markersize = opts['symbolSize']

    else:
        symbol = None
        markeredgecolor = None
        markerfacecolor = None
        markersize = 1.0
    if 'filllevel' in opts:
        if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
            fillBrush = fn.mkBrush(opts['fillBrush'])
            fillcolor = tuple([c/255. for c in fn.colorTuple(fillBrush.color())])
            ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
    pl = ax.bar(x, y, barwidth, color=symcolor, edgecolor=edgecolor)


# for matplotlib cleanup:
# These were borrowed from Manis' "PlotHelpers.py"
#
def cleanAxes(axl):
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        for loc, spine in ax.spines.iteritems():
            if loc in ['left', 'bottom']:
                pass
            elif loc in ['right', 'top']:
                spine.set_color('none')  # do not draw the spine
            else:
                raise ValueError('Unknown spine location: %s' % loc)
             # turn off ticks when there is no spine
            ax.xaxis.set_ticks_position('bottom')
            # stopped working in matplotlib 1.10
            ax.yaxis.set_ticks_position('left')
        update_font(ax)

def update_font(axl, size=6, font=stdFont):
    if type(axl) is not list:
        axl = [axl]
    fontProperties = {'family': 'sans-serif', 'sans-serif': [font],
                      'weight': 'normal', 'font-size': size}
    for ax in axl:
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_family('sans-serif')
            tick.label1.set_fontname(stdFont)
            tick.label1.set_size(size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_family('sans-serif')
            tick.label1.set_fontname(stdFont)
            tick.label1.set_size(size)
        ax.xaxis.set_smart_bounds(True)
        ax.yaxis.set_smart_bounds(True)
        ax.tick_params(axis='both', labelsize=9)

def formatTicks(axl, axis='xy', fmt='%d', font='Arial'):
    """
    Convert tick labels to intergers
    to do just one axis, set axis = 'x' or 'y'
    control the format with the formatting string
    """
    if type(axl) is not list:
        axl = [axl]
    majorFormatter = FormatStrFormatter(fmt)
    for ax in axl:
        if 'x' in axis:
            ax.xaxis.set_major_formatter(majorFormatter)
        if 'y' in axis:
            ax.yaxis.set_major_formatter(majorFormatter)


if __name__ == "__main__":
    """
    run demonstration/examples either construnction from a GraphicsWindow or from a QGridLayout
    inside a GraphicsWindow
    """
    import numpy as np
    import sys
    if len(sys.argv) == 1:
        arg = 'win'
    else:
        arg = sys.argv[1]
    if arg == 'win':
        win = pg.GraphicsWindow()
        gl = win.ci.layout # this is a QGraphicsGridLayout()
        xh = np.arange(0, 10, 0.5)
        yh = np.random.randint(0, 12, 20)
        cu = pg.PlotCurveItem(xh, yh, pen=pg.mkPen('r'))
        bg = pg.BarGraphItem(x0=xh, height=yh, width=0.5, brush='c', pen=None)
        sp = pg.ScatterPlotItem(xh, 10*np.random.random(20), symbol='o', brush=pg.mkBrush('c'))
        #pl = pg.PlotWidget()
        pl1 = win.addPlot(row=0, col=0)
        pl1.addItem(cu)
        pl3 = win.addPlot(row=0, col=1)
        pl3.addItem(sp)
        pl2 = win.addPlot(row=1, col=1)
        pl2.addItem(bg)
        pg.show()
        matplotlibExport(object=win, title=None, show=True)

    if arg == 'gl':
        win = pg.GraphicsWindow()
        gl = QtGui.QGridLayout()
     #   wid = QtGui.QWidget()
        win.setLayout(gl)
        xh = np.arange(0, 10, 0.5)
        yh = np.random.randint(0, 12, 20)
        cu = pg.PlotCurveItem(xh, yh, pen=pg.mkPen('r'))
        bg = pg.BarGraphItem(x0=xh, height=yh, width=0.25, brush='c')
        sp = pg.ScatterPlotItem(xh, 10*np.random.random(20), symbol='o', brush=pg.mkBrush('c'))
        pl = pg.PlotWidget()
        gl.addWidget(pl, 0, 0)
        pl.addItem(cu)
        pl2=pg.PlotWidget()
        gl.addWidget(pl2, 0, 1)
        pl2.addItem(bg)
        pl3=pg.PlotWidget()
        gl.addWidget(pl3, 1, 1)
        pl3.addItem(sp)
        pg.show()
        matplotlibExport(object=gl, title=None, show=True)
