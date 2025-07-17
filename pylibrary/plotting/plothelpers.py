import string
from collections import OrderedDict
from decimal import Decimal
from typing import List, Literal, Tuple, Union

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as mpl
import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axes_grid1.inset_locator as INSET
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties

# from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
#                                   TextArea)
from matplotlib.patches import Circle, Rectangle

# import seaborn  # a bit dangerous because it changes defaults, but it has wider capabiities also
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

from pylibrary.plotting import talbotetalticks as ticks  # logical tick formatting...

# matplotlib.rc("text", usetex=False)  # if true, you get computer modern fonts ALWAYS
# # if false, symbols a rendered in deja vu sans...regardless
# matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial", "Helvetica"]})
# # matplotlib.rcParams['pdf.fonttype'] = 42  # doesn't seem to do anything with Ill 2019.
# # matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use("Qt5Agg")

"""
PlotHelpers.py

Routines to help use matplotlib and make cleaner plots
as well as get plots ready for publication.

Modified to allow us to use a list of axes, and operate on all of those,
or to use just one axis if that's all that is passed.
Therefore, the first argument to these calls can either be an axes object,
or a list of axes objects.  2/10/2012 pbm.

Plotter class: a simple class for managing figures with multiple plots.
Uses gridspec to build sets of axes.

        Also allow quickaccess to things we like to do for publication plots, including:
        using a calbar instead of an axes: calbar = [x0, y0, xs, ys]
        inserting a reference line (grey, 3pt dashed, 0.5pt, at refline = y position)

Created by Paul Manis on 2010-03-09.
Copyright 2010-2016  Paul Manis
Distributed under MIT/X11 license. See license.txt for more infofmation.

Others:
rectangles and circles use the following license:

    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)

"""

class Plotter:
    """
    The Plotter class provides a simple convenience for plotting data in
    an row x column array.
    """

    def __init__(
        self,
        rcshape=None,
        axmap=None,
        arrangement=None,
        title: Union[str, None] = None,
        label=False,
        order: str = "rowsfirst",
        units: str = "page",
        refline=None,
        figsize: Union[List, Tuple, None] = None,
        margins: Union[dict, None] = None,
        labelalignment: str = "left",
        labelposition: Union[List, Tuple] = [0.0, 0.0],  # global settings
        font: str="Arial",
        labelsize: Union[float, None] = None,
        fontsize: Union[float, None] = None,
        fontweight: str = "normal",
        position=0,
        parent_figure=None,
        num=None,
    ):
        """
        Create an instance of the plotter. Generates a new matplotlib figure,
        and sets up an array of subplots as defined, initializes the counters

        Examples
        --------
        Ex. 1:
        One way to generate plots on a standard grid, uses gridspec to specify an axis map:
        labels = ['A', 'B1', 'B2', 'C1', 'C2', 'D', 'E', 'F']
        gr = [(0, 4, 0, 1), (0, 3, 1, 2), (3, 4, 1, 2), (0, 3, 2, 3), (3, 4, 2, 3), (5, 8, 0, 1), (5, 8, 1, 2), (5, 8, 2, 3)]
        axmap = OrderedDict(zip(labels, gr))
        P = PH.Plotter((8, 1), axmap=axmap, label=True, figsize=(8., 6.))
        PH.show_figure_grid(P.figure_handle)

        Ex. 2:
        Place plots on defined locations on the page - no messing with gridspec or subplots.
        For this version, we just generate N subplots with labels (used to tag each plot)
        The "sizer" array then maps the tags to specific panel locations
        # define positions for each panel in Figure coordinages (0, 1, 0, 1)
        # you don't have to use an ordered dict for this, I just prefer it when debugging
        sizer = {'A': {'pos': [0.08, 0.22, 0.50, 0.4], 'labelpos': (x,y), 'noaxes': True}, 'B1': {'pos': [0.40, 0.25, 0.60, 0.3], 'labelpos': (x,y)},
        'B2': {'pos': [0.40, 0.25, 0.5, 0.1],, 'labelpos': (x,y), 'noaxes': False},
        'C1': {'pos': [0.72, 0.25, 0.60, 0.3], 'labelpos': (x,y)}, 'C2': {'pos': [0.72, 0.25, 0.5, 0.1], 'labelpos': (x,y)},
        'D': {'pos': [0.08, 0.25, 0.1, 0.3], 'labelpos': (x,y)},
        'E': {'pos': [0.40, 0.25, 0.1, 0.3], 'labelpos': (x,y)}, 'F': {'pos': [0.72, 0.25, 0.1, 0.3],, 'labelpos': (x,y)}
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [(a, a+1, 0, 1) for a in range(0, 8)]   # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((8, 1), axmap=axmap, label=True, figsize=(8., 6.))
        PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic
        P.axdict['B1'] access the plot associated with panel B1

        Parameters
        ----------
        rcshape : a list or tuple: 2x1 (no default)
                  rcshape is an array [row, col] telling us how many rows and columns to build.
                  default defines a rectangular array r x c of plots
                  a dict :  None: expect axmap to provide the input...

        axmap :
            list of gridspec slices (default : None)
            define slices for the axes of a gridspec, allowing for non-rectangular arrangements
            The list is defined as:
            [(r1t, r1b, c1l, c1r), slice(r2, c2)]
            where r1t is the top for row 1 in the grid, r1b is the bottom, etc...
            When using this mode, the axarr returned is a 1-D list, as if r is all plots indexed,
            and the number of columns is 1. The results match in order the list entered in axmap

        arrangement: Ordered Dict (default: None)
            Arrangement allows the data to be plotted according to a logical arrangement
            The dict keys are the names ("groups") for each column, and the elements are
            string names for the entities in the groups

        title : string (default: None)
            Provide a title for the entire plot

        label : Boolean (default: False)
            If True, sets default labels on panels

        labelalignment : string (default: 'left')
            Horizontaalignment of label ('center', 'left', 'right')

        order : string (default: "rowsfirst)
            Define whether labels run in row order first ("rowsfirst")
             or column order first ("columnsfirst")
        units : units for margin setup (default: page)
            if "page", the margins are all expressed as fractions of the page size
                for width and height. This is the default and "natural" matplotlib
                unit.
            if "inches", the margins and spacing are expressed in inches
            if "cm", the margins and spacings are expresssed in centimeters
            Note that for cm, the figsize is also interpreted in centimteres.

        refline : float (default: None)
            Define the position of a reference line to be used in all panels

        figsize : tuple (default : (11, 8.5))
            Figure size in inches. Default is for a landscape figure

        font: str (default : 'Arial')
            Font to use for all text in the figure

        fontsize : points (default : 10) OR dict {'tick', 'label', 'panel'}
            Defines the size of the font to use for panel labels

        fontweight : weights (str) dict {'tick', 'label', 'panel'}
            Defines the weight of the font to use for labels: 'normal', 'bold', etc.

        position : position of spines (0 means close, 0.05 means break out)
            x, y spines..

        parent_figure: instance of an existing plotter figure to add plots to

        num : number or string name for figure

        Returns
        -------
        Nothing

        """
        self.arrangement = arrangement
        self.referenceLines = {}
        self.fontsize = {}
        self.parent = parent_figure
        self.panel_labels = label
        self.font = font
        matplotlib.rcParams["font.family"] = self.font
        self.sizer = None
        assert order in ["rowsfirst", "columnsfirst"]
        self.order = order
        figsize, margins, verticalspacing, horizontalspacing = figure_scaling(
            units=units,
            figsize=figsize,
            margins=margins,
            verticalspacing=None,
            horizontalspacing=None,
        )

        if self.parent is None:  # just create a new figure
            if figsize is None:
                figsize = (11.5, 8)  # landscape
            self.figure_handle = mpl.figure(
                figsize=figsize, num=num
            )  # create the figure
            self.figure_handle.set_size_inches(figsize[0], figsize[1], forward=True)
            self.figsize = figsize
            if title is not None:
                self.figure_handle.canvas.set_window_title(title)
                self.figure_handle.suptitle(title)

        else:  # place into an existing figure - but it must have the same figsize
            self.figure_handle = self.parent.figure_handle
            self.figure_handle.get_size_inches()  # get original figure size
            self.figsize = self.parent.figsize

        self.labelalignment = labelalignment
        self.label_flag = label
        self.axlabels = []
        self.axdict = OrderedDict()

        # the following overrides fonts
        if isinstance(fontsize, int):
            self.fontsize = {"tick": fontsize, "label": fontsize, "panel": fontsize}
        elif isinstance(fontsize, dict):
            self.fontsize = fontsize
        elif fontsize is None:
            self.fontsize = {"tick": None, "label": fontsize, "panel": fontsize}
        else:
            raise ValueError("Plotter: Font size must be int or dict")
        if isinstance(fontweight, str):
            self.fontweight = {"tick": fontweight, "label": fontweight, "panel": "bold"}
        elif isinstance(fontweight, dict):
            self.fontweight = fontweight
        elif fontweight is None:
            self.fontweight = {"tick": None, "label": "normal", "panel": "bold"}
        else:
            raise ValueError("Plotter: Font size must be int or dict")
        # otherwise we assume it is a dict and the sizes are set in the dict.
        gridbuilt = False
        # compute label offsets - global
        label_position = [0.0, 0.0]
        if self.label_flag:
            if isinstance(labelposition, int):
                label_position = [labelposition, labelposition]
            elif isinstance(labelposition, dict):
                label_position = [position["left"], position["bottom"]]
            elif isinstance(labelposition, (list, tuple)):
                label_position = labelposition
            else:
                raise ValueError(
                    "Label flag requests position of unknown type: ", labelposition
                )

        # build axes arrays
        # 1. nxm grid
        if isinstance(rcshape, list) or isinstance(rcshape, tuple):
            rc = rcshape
            self.GS = gridspec.GridSpec(rc[0], rc[1])  # define a grid using gridspec
            if margins is not None:
                self.GS.update(
                    top=1.0 - margins["topmargin"],
                    bottom=margins["bottommargin"],
                    left=margins["leftmargin"],
                    right=1.0 - margins["rightmargin"],
                )
            # assign to axarr
            self.axarr = np.empty(
                shape=(
                    rc[0],
                    rc[1],
                ),
                dtype=object,
            )  # use a numpy object array, indexing features
            ix = 0
            for r in range(rc[0]):
                for c in range(rc[1]):
                    self.axarr[r, c] = mpl.subplot(self.GS[ix])
                    ix += 1
            gridbuilt = True
        # 2. specified values - starts with Nx1 subplots, then reorganizes according to shape boxes
        elif isinstance(rcshape, dict):  # true for OrderedDict also
            nplots = len(rcshape.keys())
            self.GS = gridspec.GridSpec(nplots, 1)
            if margins is not None:
                self.GS.update(
                    top=1.0 - margins["topmargin"],
                    bottom=margins["bottommargin"],
                    left=margins["leftmargin"],
                    right=1.0 - margins["rightmargin"],
                )

            rc = (nplots, 1)
            self.axarr = np.empty(
                shape=(
                    rc[0],
                    rc[1],
                ),
                dtype=object,
            )  # use a numpy object array, indexing features
            ix = 0
            for r in range(rc[0]):  # rows
                for c in range(rc[1]):  # columns
                    self.axarr[r, c] = mpl.subplot(self.GS[ix])
                    ix += 1
            gridbuilt = True
            for k, pk in enumerate(rcshape.keys()):
                self.axdict[pk] = self.axarr[k, 0]

            # Label the plots

            self.axlabels = labelPanels(
                self.axarr.tolist(),
                axlist=rcshape.keys(),
                rcshape=rcshape,
                order=self.order,
                xy=(-0.095 + labelposition[0], 0.95 + labelposition[1]),
                fontsize=self.fontsize["panel"],
                weight="bold",
                horizontalalignment=self.labelalignment,
            )
            self.resize(rcshape)
        else:
            raise ValueError("Input rcshape must be list/tuple or dict")

        # create sublots
        if axmap is not None:
            if isinstance(axmap, list) and not gridbuilt:
                self.axarr = np.empty(shape=(len(axmap), 1), dtype=object)
                for k, g in enumerate(axmap):
                    self.axarr[k,] = mpl.subplot(self.GS[g[0] : g[1], g[2] : g[3]])
            elif isinstance(axmap, dict) or isinstance(
                axmap, OrderedDict
            ):  # keys are panel labels
                if not gridbuilt:
                    self.axarr = np.empty(shape=(len(axmap.keys()), 1), dtype=object)
                for k, pk in enumerate(axmap.keys()):
                    g = axmap[pk]  # get the gridspec info
                    if not gridbuilt:
                        self.axarr[k,] = mpl.subplot(self.GS[g[0] : g[1], g[2] : g[3]])
                    self.axdict[pk] = self.axarr.ravel()[k]
            else:
                raise TypeError("Plotter in PlotHelpers: axmap must be a list or dict")

        if len(self.axdict) == 0:
            for i, a in enumerate(self.axarr.flatten()):
                label = string.ascii_uppercase[i]
                self.axdict[label] = a

        self.nrows = self.axarr.shape[0]
        if len(self.axarr.shape) > 1:
            self.ncolumns = self.axarr.shape[1]
        else:
            self.ncolumns = 1
        self.reset_axis_counters()
        for i in range(self.nrows):
            for j in range(self.ncolumns):
                self.axarr[i, j].spines["top"].set_visible(False)
                self.axarr[i, j].get_xaxis().set_tick_params(
                    direction="out", width=0.8, length=4.0
                )
                self.axarr[i, j].get_yaxis().set_tick_params(
                    direction="out", width=0.8, length=4.0
                )
                if self.fontsize["tick"] is not None:
                    self.axarr[i, j].tick_params(
                        axis="both", which="major", labelsize=self.fontsize["tick"]
                    )
                #                if i < self.nrows-1:
                #                    self.axarr[i, j].xaxis.set_major_formatter(mpl.NullFormatter())
                nice_plot(self.axarr[i, j], position=position)
                if refline is not None:
                    self.referenceLines[self.axarr[i, j]] = referenceline(
                        self.axarr[i, j], reference=refline
                    )

        if label:
            if isinstance(axmap, dict) or isinstance(
                axmap, OrderedDict
            ):  # in case predefined...
                self.axlabels = labelPanels(
                    self.axarr.ravel().tolist(),
                    order=self.order,
                    axlist=axmap.keys(),
                    xy=(-0.095 + label_position[0], 0.95 + label_position[1]),
                    horizontalalignment=self.labelalignment,
                    fontsize=self.fontsize["panel"],
                    weight=self.fontweight["panel"],
                )
                return
            self.axlist = []
            if self.order == "rowsfirst":  # straight down rows in sequence
                for i in range(self.nrows):
                    for j in range(self.ncolumns):
                        self.axlist.append(self.axarr[i, j])
            else:  # go across in columns (zig zag)
                for i in range(self.ncolumns):
                    for j in range(self.nrows):
                        self.axlist.append(self.axarr[j, i])

            if self.nrows * self.ncolumns > 26:  # handle large plot using "A1..."
                ctxt = string.ascii_uppercase[0 : self.ncolumns]  # columns are lettered
                rtxt = [
                    str(x + 1) for x in range(self.nrows)
                ]  # rows are numbered, starting at 1
                axl = []
                for i in range(self.nrows):
                    for j in range(self.ncolumns):
                        axl.append(ctxt[j] + rtxt[i])
                self.axlabels = labelPanels(
                    self.axlist,
                    axlist=axl,
                    order=self.order,
                    xy=(-0.35 + label_position[0], 0.75),
                    fontsize=self.fontsize["panel"],
                    weight=self.fontweight["panel"],
                    horizontalalignment=self.labelalignment,
                )

            else:
                self.axlabels = labelPanels(
                    self.axlist,
                    order=self.order,
                    xy=(-0.095 + label_position[0], 0.95 + label_position[1]),
                    fontsize=self.fontsize["panel"],
                    weight=self.fontweight["panel"],
                    horizontalalignment=self.labelalignment,
                )

    def _next(self):
        """
        Private function
        _next gets the axis pointer to the next row, column index that is available
        Only sets internal variables
        """
        # if self.order == 'rowsfirst':
        self.row_counter += 1
        if self.row_counter >= self.nrows:
            self.column_counter += 1
            self.row_counter = 0
            if self.column_counter > self.ncolumns:
                raise ValueError(
                    "Call to get next axis exceeds the number of columns requested initially: %d"
                    % self.columns
                )
        # else:
        #     self.column_counter += 1
        #     if self.column_counter >= self.ncolumns:
        #         self.row_counter += 1
        #         self.column_counter = 0
        #         if self.row_counter >= self.nrows:
        #             raise ValueError('Call to get next axis exceeds the number of rows requested initially: %d' % self.nrows)

    def reset_axis_counters(self):
        """
        Set the column and row counters back to zero
        """

        self.column_counter = 0
        self.row_counter = 0

    def getaxis(self, group=None):
        """
        getaxis gets the current row, column counter, and calls _next to increment the counter
        (so that the next getaxis returns the next available axis pointer)

        Parameters
        ----------
        group : string (default: None)
            forces the current axis to be selected from text name of a "group"

        Returns
        -------
        the current axis or the axis associated with a group
        """

        if group is None:
            currentaxis = self.axarr[self.row_counter, self.column_counter]
            self._next()  # prepare for next call
        else:
            currentaxis = self.getRC(group)

        return currentaxis

    def adjust_panel_labels(
        self,
        fontsize: Union[None, str, float] = None,
        fontweight: Union[None, str, int] = None,
        fontstyle: Union[None, str] = None,
        fontname: Union[None, str] = None,
    ):
        """Change the label size for all of the panel labels

        Parameters
        ----------
        fontsize : float, optional
            font size, points, by default None (no change)
        fontweight : str, optional
            font weight, by default None (no change)
        fontstyle : str, optional
            font style, by default None (no change)
        fontname : str, optional
            font name, by default None (no change)
        """
        for ax_text in self.axlabels:  # text objects
            if fontsize is not None:
                ax_text.set_fontsize(fontsize)
            if fontweight is not None:
                ax_text.set_fontweight(fontweight)
            if fontstyle is not None:
                ax_text.set_fontstyle(fontstyle)
            if fontname is not None:
                ax_text.set_fontname(fontname)

    def getaxis_fromlabel(self, label):
        """
        Parameters
        ----------
        label : str
            Find the plot instance axis object that has the
            specified label

        Returns
        -------
        axisobject or None if the labe is not found
        """

        axobj = self.axdict[label]
        for i in range(self.nrows):
            for j in range(self.ncolumns):
                if axobj == self.axarr[i, j]:
                    return axobj
        return None  # not found

    def getRC(self, group):
        """
        Get the axis associated with a group

        Parameters
        ----------
        group : string (default: None)
            returns the matplotlib axis associated with a text name of a "group"

        Returns
        -------
        The matplotlib axis associated with the group name, or None if no group by
        that name exists in the arrangement
        """

        if self.arrangement is None:
            raise ValueError("specifying a group requires an arrangment dictionary")
        # look for the group label in the arrangement dicts
        for c, colname in enumerate(self.arrangement.keys()):
            if group in self.arrangement[colname]:
                # print ('column name, column: ', colname, self.arrangement[colname])
                # print ('group: ', group)
                r = self.arrangement[colname].index(
                    group
                )  # get the row position this way
                return self.axarr[r, c]
        print("Group {:s} not in the arrangement".format(group))
        return None

        # sizer = {'A': {'pos': [0.08, 0.22, 0.50, 0.4]}, 'B1': {'pos': [0.40, 0.25, 0.60, 0.3]}, 'B2': {'pos': [0.40, 0.25, 0.5, 0.1]},
        #         'C1': {'pos': [0.72, 0.25, 0.60, 0.3]}, 'C2': {'pos': [0.72, 0.25, 0.5, 0.1]},
        #         'D': {'pos': [0.08, 0.25, 0.1, 0.3]}, 'E': {'pos': [0.40, 0.25, 0.1, 0.3]}, 'F': {'pos': [0.72, 0.25, 0.1, 0.3]},
        # }

    def resize(self, sizer):
        """
        Resize the graphs in the array.

        Parameters
        ----------
        sizer : dict (no default)
            A dictionary with keys corresponding to the plot labels.
            The values for each key are a list (or tuple) of [left, width, bottom, height]
            for each panel in units of the graph [0, 1, 0, 1].

        sizer = {'A': {'pos': [0.08, 0.22, 0.50, 0.4], 'labelpos': (x,y), 'noaxes': True}, 'B1': {'pos': [0.40, 0.25, 0.60, 0.3], 'labelpos': (x,y)},
                 'B2': {'pos': [0.40, 0.25, 0.5, 0.1],, 'labelpos': (x,y), 'noaxes': False},
                'C1': {'pos': [0.72, 0.25, 0.60, 0.3], 'labelpos': (x,y)}, 'C2': {'pos': [0.72, 0.25, 0.5, 0.1], 'labelpos': (x,y)},
                'D': {'pos': [0.08, 0.25, 0.1, 0.3], 'labelpos': (x,y)},
                'E': {'pos': [0.40, 0.25, 0.1, 0.3], 'labelpos': (x,y)}, 'F': {'pos': [0.72, 0.25, 0.1, 0.3],, 'labelpos': (x,y)}
                }
        Returns
        -------
        Nothing
        """

        for i, s in enumerate(sizer.keys()):
            ax = self.axdict[s]
            bbox = ax.get_position()
            bbox.x0 = sizer[s]["pos"][0]
            bbox.x1 = sizer[s]["pos"][1] + sizer[s]["pos"][0]
            bbox.y0 = sizer[s]["pos"][2]
            bbox.y1 = (
                sizer[s]["pos"][3] + sizer[s]["pos"][2]
            )  # offsets are in figure fractions
            ax.set_position(bbox)
            if (
                self.panel_labels
                and "labelpos" in sizer[s].keys()
                and len(sizer[s]["labelpos"]) == 2
            ):
                x, y = sizer[s]["labelpos"]
                self.axlabels[i].set_x(x)
                self.axlabels[i].set_y(y)
            if "noaxes" in sizer[s] and sizer[s]["noaxes"] is True:
                noaxes(ax)



def _ax_tolist(ax: object) -> list:
    """
    Private

    Convert axis of different format to a list
    Lists are just returned
    Dictionaries are converted to a list without the keys
    numpy arrays are just converted to a list
    simple instances are made a list.
    """
    if isinstance(ax, list):
        pass
    elif isinstance(ax, dict):
        axlist = ax.keys()
        ax = [ax for ax in ax[axlist]]
    elif isinstance(ax, np.ndarray):
        ax = ax.tolist()
    elif isinstance(ax, matplotlib.axes.Axes):  # a single axes from a subplot
        ax = [ax]
    return [a for a in ax if a is not None]


def nice_plot(
    axl: object,
    spines: list = ["left", "bottom"],
    position: dict = {'left': 0.0, 'bottom': 0.0},
    direction: str = "inward",
    ticklength: float = 5.0,
    axesoff: bool = False,
):
    """Adjust a plot so that it looks nicer than the default matplotlib plot.

    Parameters
    ----------
    axl : list of axes objects
        If a single axis object is present, it will be converted to a list here.

    spines : list of strings (default : ['left', 'bottom'])
        Sets whether spines will occur on particular axes. Choices are 'left', 'right',
        'bottom', and 'top'. Chosen spines will be displayed, others are not

    position : float (default : 10)
        Determines position of spines in points, typically outward by x points. The
        spines are the main axes lines, not the tick marks
        if the position is dict, then interpret as such.

    direction : string (default : 'inward')
        Sets the direction of spines. Choices are 'inward' and 'outward'

    axesoff : boolean (default : False)
        If true, forces the axes to be turned completely off.

    Returns
    -------
        Nothing.
    """
    axl = _ax_tolist(axl)
    for ax in axl:
        if ax is None:
            continue
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_color("k")
                if isinstance(position, int) or isinstance(position, float):
                    spine.set_position(("axes", float(position)))
                elif isinstance(position, dict):
                    if loc in list(position.keys()):
                        spine.set_position(("axes", position[loc]))
                else:
                    raise ValueError(
                        "position must be int, float or dict [ex: ]{'left': -0.05, 'bottom': -0.05}]"
                    )
            else:
                spine.set_color("none")
        if axesoff is True:
            noaxes(ax)

        # turn off ticks where there is no spine, if there are axes
        if "left" in spines and not axesoff:
            ax.yaxis.set_ticks_position("left")
            ax.yaxis.set_tick_params(color="k")
            ax.yaxis.set_tick_params(length=ticklength)
        else:
            ax.yaxis.set_ticks([])  # no yaxis ticks
        if "bottom" in spines and not axesoff:
            ax.xaxis.set_ticks_position("bottom")
            ax.xaxis.set_tick_params(color="k")
            ax.xaxis.set_tick_params(length=ticklength)
        else:
            ax.xaxis.set_ticks([])  # no xaxis ticks

        if direction in ["inward", "in"]:
            ax.tick_params(axis="y", direction="in")
            ax.tick_params(axis="x", direction="in")
        elif direction in ["outward", "out"]:
            ax.tick_params(axis="y", direction="out")
            ax.tick_params(axis="x", direction="out")
        else:
            pass
        # or call adjust_spines?


def set_axes_ticks(
    ax: mpl.Axes,
    xticks: Union[List, None] = None,
    xticks_str: Union[List, None] = None,
    xticks_pad: Union[List, None] = None,
    x_minor: Union[List, None] = None,
    x_rotation: float = 0.0,
    major_length: float = 3.0,
    minor_length: float = 1.5,
    yticks: Union[List, None] = None,
    yticks_str: Union[List, None] = None,
    yticks_pad: Union[List, None] = None,
    y_minor: Union[List, None] = None,
    y_rotation: float = 0.0,
    fontsize: int = 8,
):
    """Manually set axes ticks on a plot. This works on both
    x and y axes.

    Parameters
    ----------
    ax: Axis object to work on
    xticks : Union[List, None], optional
        xtick positions, by default None
    xticks_pad: Union[List, None], Optional
    xticks_str : Union[List, None], optional
        xtick labels, by default None
    x_minor : Union[List, None], optional
        minor tick positions, by default None
    yticks : Union[List, None], optional
        ytick positions, by default None
    yticks_pad : Union[List, None], optional
    yticks_str : Union[List, None], optional
        ytick labels, by default None
    y_minor : Union[List, None], optional
        minor tick positions, by default None
    major_length: float = 3,
        major tick length
    minor_length: float = 1.5
        minor tick length
    fontsize : int, optional
        tick label font size, default 8 pt
    """

    if yticks is not None:
        ax.set_yticks(yticks, yticks_str, fontsize=fontsize)
        if yticks_pad is not None:
            ylabels = ax.get_yticklabels()
            for i, tick in enumerate(ax.get_yaxis().get_major_ticks()):
                tick.set_pad(yticks_pad[i])
                if tick.get_pad() < 0:
                    ax.get_yaxis().set_tick_params(direction="out")
                tick.label1 = ylabels[i]
        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=major_length,
            labelrotation=y_rotation,
        )
    if y_minor is not None:
        ax.set_yticks(y_minor, minor=True)
        ax.tick_params(axis="y", which="minor", direction="out", length=minor_length)
    if xticks is not None:
        ax.set_xticks(xticks, xticks_str, fontsize=8)
        if xticks_pad is not None:
            xlabels = ax.get_xticklabels()
            for i, tick in enumerate(ax.get_xaxis().get_major_ticks()):
                tick.set_pad(xticks_pad[i])
                tick.label1 = xlabels[i]
        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            length=major_length,
            labelrotation=x_rotation,
        )
    if x_minor is not None:
        ax.set_xticks(x_minor, minor=True)
        ax.tick_params(axis="x", which="minor", direction="out", length=minor_length)


def noaxes(axl: Union[object, List], whichaxes: str = "xytr"):
    """
    Take away all the axis ticks and the lines.

    Parameters
    ----------

    axl : list of axes objects
        If a single axis object is present, it will be converted to a list here.

    whichaxes : string (default : 'xy')
        Sets which axes are turned off. The presence of an 'x' in
        the string turns off x, the presence of 'y' turns off y.

    Returns
    -------
        Nothing
    """

    axl = _ax_tolist(axl)
    for ax in axl:
        if ax is None:
            continue
        if "x" in whichaxes:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
        if "t" in whichaxes:
            ax.spines["top"].set_visible(False)
            ax.tick_params(top=False, labeltop=False)
        if "y" in whichaxes:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
        if "r" in whichaxes:
            ax.spines["right"].set_visible(False)
            ax.tick_params(right=False, labelright=False)


def showaxes(axl: Union[object, List], whichaxes: str = "xy"):
    """
    Show all the axis ticks and the lines.

    Parameters
    ----------

    axl : list of axes objects
        If a single axis object is present, it will be converted to a list here.

    whichaxes : string (default : 'xy')
        Sets which axes are turned on. The presence of an 'x' in
        the string turns on x, the presence of 'y' turns on y.

    Returns
    -------
        Nothing
    """

    axl = _ax_tolist(axl)
    for ax in axl:
        if ax is None:
            continue
        if "x" in whichaxes:
            ax.spines["bottom"].set_visible(True)
            ax.tick_params(bottom=True, labelbottom=True)
        if "y" in whichaxes:
            ax.spines["left"].set_visible(True)
            ax.tick_params(left=True, labelleft=True)


def noaxeslabels(axl: Union[object, List], whichaxes: str = "xy"):
    """
    Remove the axes labels without removing the tick marks

    Parameters
    ----------
    whichaxes : string (default: 'xy')
        string indicating which axes will be unlabeled
        'xy', 'x', 'y'

    Returns
    -------
        Nothing
    """
    axl = _ax_tolist(axl)
    for ax in axl:
        if "x" in whichaxes:
            ax.tick_params(labelbottom=False)
        if "y" in whichaxes:
            ax.tick_params(labelleft=False)
            # ax.set_yticklabels([])


def setY(ax1: mpl.axes, ax2: Union[mpl.axes, List]):
    """
    Set the Y limits for an axes from a source axes to
    the target axes.

    Parameters
    ----------

    ax1 : axis object
        The source axis object
    ax2 : list of axes objects
        If a single axis object is present, it will be converted to a list here.
        These are the target axes objects that will take on the limits of the source.

    Returns
    -------
        Nothing

    """
    if type(ax1) is list:
        print("PlotHelpers: cannot use list as source to set Y axis")
        return
    ax2 = _ax_tolist(ax2)
    # if type(ax2) is not list:
    #     ax2 = [ax2]
    refy = ax1.get_ylim()
    for ax in ax2:
        ax.set_ylim(refy)


def setX(ax1: mpl.axes, ax2: Union[mpl.axes, List]):
    """
    Set the X limits for an axes from a source axes to
    the target axes.

    Parameters
    ----------

    ax1 : axis object
        The source axis object
    ax2 : list of axes objects
        If a single axis object is present, it will be converted to a list here.
        These are the target axes objects that will take on the limits of the source.

    Returns
    -------
        Nothing

    """
    if type(ax1) is list:
        print("PlotHelpers: cannot use list as source to set Y axis")
        return
    ax2 = _ax_tolist(ax2)
    # if type(ax2) is not list:
    #     ax2 = [ax2]
    refx = ax1.get_xlim()
    for ax in ax2:
        ax.set_xlim(refx)


def tickStrings(
    values: Union[list, np.ndarray],
    scale: float = 1,
    spacing: Union[float, None] = None,
    tickPlacesAdd: int = 1,
    floatAdd: Union[int, None] = None,
) -> list:
    """Return the strings that should be placed next to ticks. This method is called
    when redrawing the axis and is a good method to override in subclasses.

    Parameters
    ----------
    values : array or list
         An array or list of tick values
    scale : float, optional
        a scaling factor (see below), defaults to 1
    spacing : float, optional
        spaceing between ticks (this is required since, in some instances, there may be only
    one tick and thus no other way to determine the tick spacing). Defaults to None
    tickPlacesToAdd : int, optional
        the number of decimal places to add to the ticks, default is 1

    Returns
    -------
    list : a list containing the tick strings

    The scale argument is used when the axis label is displaying units which may have an SI scaling prefix.
    When determining the text to display, use value*scale to correctly account for this prefix.
    For example, if the axis label's units are set to 'V', then a tick value of 0.001 might
    be accompanied by a scale value of 1000. This indicates that the label is displaying 'mV', and
    thus the tick should display 0.001 * 1000 = 1.
    Copied rom pyqtgraph; we needed it here.
    """
    #    print ('tickplacesadd: ', tickPlacesAdd)

    if spacing is None:
        spacing = np.mean(np.diff(values))
    places = tickPlacesAdd  # int(np.max((0, np.ceil(-np.log10(spacing*scale)))) + tickPlacesAdd)
    if tickPlacesAdd == 0 and floatAdd in [0, None]:
        places = 0
    strings = []
    for v in values:
        vs = v * scale
        if np.fabs(vs) < 1e-3 or np.fabs(vs) >= 1e4:
            vstr = "%g" % vs
        else:
            if floatAdd in [None, 0]:
                vstr = ("%%0.%df" % places) % vs
            else:  # check and reformat if not a match...
                vstr = ("%%0.%df" % floatAdd) % vs
        strings.append(vstr)
    return strings


def talbotTicks(axl: Union[object, List], **kwds):
    r"""
    Adjust the tick marks using the talbot et al algorithm, on an existing plot.

    Parameters
    ----------
    axl : axis instance, list, etc

    \**kwds : keywords passed to do_talbotTicks

    Returns
    -------
    Nothing
    """
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        do_talbotTicks(ax, **kwds)


def do_talbotTicks(
    ax,
    axes="xy",
    density=(1.0, 1.0),
    insideMargin=0.05,
    pointSize=None,
    tickPlacesAdd={"x": 0, "y": 0},
    floatAdd={"x": 0, "y": 0},
    axrange={"x": None, "y": None},
    colorbar=None,
):
    """
    Change the axis ticks to use the talbot algorithm for ONE axis
    Paramerters control the ticks

    Parameters
    ----------
    ax : matplotlib axis instance
        the axis to change the ticks on
    axes : str
        'xy' for both x and y
        'x' for just x, 'y' for just y
    density : tuple
        tick density (for talbotTicks), defaults to (1.0, 1.0)
    insideMargin : float
        Inside margin space for plot, defaults to 0.05 (5%)
    pointSize : int
         point size for tick text, defaults to 12
    tickPlacesAdd : dict
        number of decimal places to add in tickstrings for the ticks, pair for x and y axes, defaults to (0,0)
    floatAdd : dict
        if tickplaces is 0, but the number would be better represented by a float, how many plances?
    axrange : dict. Default: {'x': None, 'y': None}
        override the standard axis limits for the labeling
        values can be list or tuple (0, 1), or can be (0, None) to use lower bound
        as 0.
    colorbar: a colorbar instance or None
        set to true if this axis is a colorbar axis, so we use the right call.
    Returns
    -------
    Nothing
    """

    # get axis limits
    # aleft = ax.getAxis('left')
    # abottom = ax.getAxis('bottom')

    if "y" in axes:
        yRange = list(ax.get_ylim())
        if axrange["y"] is not None:  # any overrides
            for ra in range(0, 2):
                if axrange["y"][ra] is not None:
                    yRange[ra] = axrange["y"][ra]

        yr = np.diff(yRange)[0]
        ymin, ymax = (
            np.min(yRange) - yr * insideMargin,
            np.max(yRange) + yr * insideMargin,
        )
        ytick = ticks.Extended(
            density=density[1], figure=None, range=(ymin, ymax), axis="y"
        )
        yt = ytick()
        yts = tickStrings(
            yt,
            scale=1,
            spacing=None,
            tickPlacesAdd=tickPlacesAdd["y"],
            floatAdd=floatAdd["y"],
        )
        #        ytickl = [[(y, yts[i]) for i, y in enumerate(yt)] , []]  # no minor ticks here
        if colorbar is not None:
            colorbar.set_ticks(yt)
            colorbar.set_ticklabels(yts)
        else:
            ax.set_yticks(yt)
            ax.set_yticklabels(yts)  # , rotation='horizontal', fontsize=pointSize)
            #        print ('yt, yts: ', yt, yts)

            # ytxt = ax.get_yticklabels()
            ytxt = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ytxt))
            # ax.set_yticklabels([label_format.format(x) for x in ytxt], fontsize=pointSIze,
            #         rotation='horizontal')
            xRange = list(ax.get_xlim())
            if axrange["x"] is not None:  # any overrides
                for ra in range(0, 2):
                    if axrange["x"][ra] is not None:
                        xRange[ra] = axrange["x"][ra]
            # now create substitue tick marks and labels, using Talbot et al algorithm
            xr = np.diff(xRange)[0]
            xmin, xmax = (
                np.min(xRange) - xr * insideMargin,
                np.max(xRange) + xr * insideMargin,
            )
            xtick = ticks.Extended(
                density=density[0], figure=None, range=(xmin, xmax), axis="x"
            )
            if pointSize is not None:
                ytxt = ax.get_yticklabels()
                ax.set_yticklabels(ytxt, fontsize=pointSize, rotation="horizontal")

    if "x" in axes:
        xRange = list(ax.get_xlim())
        if axrange["x"] is not None:  # any overrides
            for ra in range(0, 2):
                if axrange["x"][ra] is not None:
                    xRange[ra] = axrange["x"][ra]
        # now create substitue tick marks and labels, using Talbot et al algorithm
        xr = np.diff(xRange)[0]
        xmin, xmax = (
            np.min(xRange) - xr * insideMargin,
            np.max(xRange) + xr * insideMargin,
        )
        xtick = ticks.Extended(
            density=density[0], figure=None, range=(xmin, xmax), axis="x"
        )
        xt = xtick()
        xts = tickStrings(
            xt,
            scale=1,
            spacing=None,
            tickPlacesAdd=tickPlacesAdd["x"],
            floatAdd=floatAdd["x"],
        )
        #        xtickl = [[(x, xts[i]) for i, x in enumerate(xt)] , []]  # no minor ticks here
        # x_ticks_labels = ax.set_xticks(xt)
        # ax.set_xticklabels(xts)  # , rotation='horizontal', fontsize=pointSize)
        if pointSize is not None:
            xtxt = ax.get_xticklabels()
            # ax.set_xticklabels(xtxt, fontsize = pointSize, rotation ="horizontal")


def labelPanels(
    axl: Union[list, dict],
    axlist=None,
    rcshape=None,
    order="rowsfirst",
    font="Arial",
    fontsize=18,
    weight="normal",
    xy=(-0.05, 1.05),
    horizontalalignment="right",
    verticalalignment="bottom",
    rotation=0.0,
):
    """
    Provide labeling of panels in a figure with multiple subplots (axes)

    Parameters
    ----------
    axl : list  or dict of axes objects
        If a single axis object is present, it will be converted to a list here.
        if the array is a multidimensional numpy array (ndim = 2),
        then the grid method id sued
        if axl is a dictionary, then the dict is parsed by keys and possibly
        the labelpos

    axlist : list of string labels (default : None)
        Contains a list of the string labels. If the default value  of None is provided,
        the axes will be lettered in alphabetical sequence,
        or taken from the keys of axl

    order : str (default "rowfirst")
        A string describing the labeling order when axlist is None.
        Must be "rowfirst" or "columnfirst".

    font : string (default : 'Arial')
        Name of a valid font to use for the panel labels

    fontsize : float (default : 18, in points)
        Font size to use for axis labeling

    weight : string (default : 'normal')
        Font weight to use for labels. 'Bold', 'Italic', and 'Normal' are options

    xy : tuple (default : (-0.05, 1.05))
        A tuple (x,y) indicating where the label should go relative to the axis frame.
        Values are normalized as a fraction of the frame size.

    Returns
    -------
        list of the annotations

    """
    if isinstance(axl, dict):
        axlist = list(axl.keys())  # replace the list with the dictionary keys
        print("is dict: ", axl)
    if isinstance(axl, np.ndarray):
        rc = axl.shape  # get row and column sizes before converting to list

    axl = _ax_tolist(axl)

    if axlist is None:
        if order == "rowsfirst":
            axlist = string.ascii_uppercase[0 : len(axl)]
        elif order == "columnsfirst":
            nl = np.array([i for i in string.ascii_uppercase[0 : len(axl)]])
            nl = nl.reshape(rc[1], rc[0]).T.ravel().tolist()  # changes order
    else:
        axlist = list(axlist)
    # assume we wish to go in sequence
    if len(axlist) > len(axl):
        raise ValueError(
            "axl must have more entries than axlist: got axl=%d and axlist=%d for axlist:"
            % (len(axl), len(axlist)),
            axlist,
        )
    labels = []
    for i, ax in enumerate(axl):
        if i >= len(axlist):
            continue
        if ax is None:
            continue
        if isinstance(ax, list):
            ax = ax[0]
        xy_label = xy
        fsize = fontsize
        # possibly replace the xy position from the rcshape dictionary
        if rcshape is not None:
            if axlist[i] in rcshape.keys():
                if "labelpos" in rcshape[axlist[i]].keys():
                    xy_label = rcshape[axlist[i]]["labelpos"]
                if "fontsize" in rcshape[axlist[i]].keys():
                    fsize = rcshape[axlist[i]]["fontsize"]

        ann = ax.text(
            xy_label[0],
            xy_label[1],
            axlist[i],
            transform=ax.transAxes,
            fontdict={
                "fontsize": fsize,
                "weight": weight,
                "family": "sans-serif",
                "verticalalignment": verticalalignment,
                "horizontalalignment": horizontalalignment,
                "rotation": rotation,
            },
        )
        labels.append(ann)

    return labels


def list_axes(axd):  # convienence
    listAxes(axd)


def listAxes(axd):
    """
    make a list of the axes from the dictionary

    Parameters
    ----------
    axd : axes (list, dict)

    Returns
    -------
    list of axes
    """
    if type(axd) is not dict:
        if type(axd) is list:
            return axd
        else:
            print("listAxes expects dictionary or list; type not known (fix the code)")
            raise
    axl = [axd[x] for x in axd]
    return axl


def clean_axes(axl):
    """
    Remove top and right spines and ticks from the axes to make
    a clean plot without the junk normally defaulted in matplotlib
    (because I always forget which way I coded it)
    Parameters
    ----------
    axl : list, dict, or single axis

    Returns
    -------
    Nothing
    """
    cleanAxes(axl)


def cleanAxes(axl):
    """
    Remove top and right spines and ticks from the axes to make
    a clean plot without the junk normally defaulted in matplotlib

    Parameters
    ----------
    axl : list, dict, or single axis

    Returns
    -------
    Nothing
    """

    axl = _ax_tolist(axl)
    for ax in axl:
        if ax is None:
            continue
        for loc, spine in ax.spines.items():
            if loc in ["left", "bottom"]:
                spine.set_visible(True)
            elif loc in ["right", "top"]:
                spine.set_visible(False)
                # spine.set_color('none')
                # do not draw the spine
            else:
                raise ValueError("Unknown spine location: %s" % loc)
            # turn off ticks when there is no spine
            ax.xaxis.set_ticks_position("bottom")
            # pdb.set_trace()
            ax.yaxis.set_ticks_position("left")  # stopped working in matplotlib 1.10
        update_font(ax)


def setTicks(axl: object, axis: str = "x", ticks: object = np.arange(0, 1.1, 1.0)):
    axl = _ax_tolist(axl)
    # if type(axl) is dict:
    #     axl = [axl[x] for x in axl.keys()]
    # if type(axl) is not list:
    #     axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if axis == "x":
            ax.set_xticks(ticks)
        if axis == "y":
            ax.set_yticks(ticks)


def formatTicks(axl: object, axis: str = "xy", fmt: str = "%d", font: str = "Arial"):
    """
    Convert tick labels to integers
    To do just one axis, set axis = 'x' or 'y'
    Control the format with the formatting string
    """
    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    majorFormatter = FormatStrFormatter(fmt)
    for ax in axl:
        if ax is None:
            continue
        if "x" in axis:
            ax.xaxis.set_major_formatter(majorFormatter)
        if "y" in axis:
            ax.yaxis.set_major_formatter(majorFormatter)


def autoFormatTicks(axl: object, axis: str = "xy", font: str = "Arial"):
    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if "x" in axis:
            #    print ax.get_xlim()
            x0, x1 = ax.get_xlim()
            setFormatter(ax, x0, x1, axis="x")
        if "y" in axis:
            y0, y1 = ax.get_xlim
            setFormatter(ax, y0, y1, axis="y")


def setFormatter(axl: object, x0: float, x1: float, axis: str = "x"):
    axl = _ax_tolist(axl)
    datarange = np.abs(x0 - x1)
    mdata = np.ceil(np.log10(datarange))
    if mdata > 0 and mdata <= 4:
        majorFormatter = FormatStrFormatter("%d")
    elif mdata > 4:
        majorFormatter = FormatStrFormatter("%e")
    elif mdata <= 0 and mdata > -1:
        majorFormatter = FormatStrFormatter("%5.1f")
    elif mdata < -1 and mdata > -3:
        majorFormatter = FormatStrFormatter("%6.3f")
    else:
        majorFormatter = FormatStrFormatter("%e")
    for ax in axl:
        if axis == "x":
            ax.xaxis.set_major_formatter(majorFormatter)
        elif axis == "y":
            ax.yaxis.set_major_formatter(majorFormatter)


def update_font(axl: object, size: float = 8.0, font: str = "Arial"):
    """
    Change the font on a axis

    Parameters
    ----------
    size : float (default: 8)
        New font size
    font : str (default: stdFont)
        New font (default is 'Arial')

    """
    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    fontProperties = {
        "family": "sans-serif",
        "weight": "normal",
        "size": size,
    }
    for ax in axl:
        if ax is None:
            continue

        ax.tick_params(axis="both", labelsize=size)


def lockPlot(axl, lims, ticks=None):
    """
    This routine forces the plot of invisible data to force the axes to take certain
    limits and to force the tick marks to appear.

    Parameters
    ----------
    axl : axis or list of axes (no default)
        a Matplotlib axis instance, or a list fo axes
    lims : list : no default
        List of 4 values for axis limits: [x0, x1, y0, y1]
    ticks: bool (default None)
        Not implemented
    """
    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    plist = []
    for ax in axl:
        if ax is None:
            continue
        lpl = ax.plot(
            [lims[0], lims[0], lims[1], lims[1]],
            [lims[2], lims[3], lims[2], lims[3]],
            color="none",
            marker="",
            linestyle="None",
        )
        plist.extend(lpl)
        ax.axis(lims)
    return plist  # just in case you want to modify these plots later.


def adjust_spines(axl, spines=["left", "bottom"], direction="outward", length=5):
    """
    Change spine size, location and direction

    Parameters
    ----------
    axl : axis or list of axes (no default)
        a Matplotlib axis instance, or a list fo axes

    spines : list (default ['left', 'bottom])
        List of which spines to adjust

    direction: str (default: 'outward')
        Direction spines should point

    length : float (default: 5)
        Length of spines

    """
    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        # turn off ticks where there is no spine
        if "left" in spines:
            ax.yaxis.set_ticks_position("left")
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if "bottom" in spines:
            ax.xaxis.set_ticks_position("bottom")
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])
        for loc, spine in ax.spines:  # .iteritems():
            if loc in spines:
                spine.set_position((direction, length))  # outward by 10 points
                # if smart is True:
            #      spine.set_bounds(True)
            #  else:
            #      spine.set_bounds(False)
            else:
                spine.set_color("none")  # don't draw spine


class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.


    Pulled from a stack overflow answer:
    https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python

    Usage:
        fig, ax = plt.subplots(1)

        ax.plot(np.arange(0, 9)**2, label='$y=x^2$')
        ax.legend()

        ax.set_yscale('squareroot')
        ax.set_yticks(np.arange(0,9,2)**2)
        ax.set_yticks(np.arange(0,8.5,0.5)**2, minor=True)

        plt.show()

    Because this registers the class in the matplotlib.scale module, it should be
    just "usable" by specifying the 'squareroot' scale.

    """

    name = "squareroot"

    def __init__(self, **kwargs):
        mscale.ScaleBase.__init__(self)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


def getLayoutDimensions(n, pref="height"):
    """
    Return a tuple of optimized layout dimensions for n axes

    Parameters
    ----------
    n : int (no default):
        Number of plots needed

    pref : string (default : 'height')
        prefered way to organized the plots (height, or width)

    Returns
    -------
    (h, w) : tuple
        height (rows) and width (columns)

    """
    nopt = np.sqrt(n)
    inoptw = int(nopt)
    inopth = int(nopt)
    while inoptw * inopth < n:
        if pref == "width":
            inoptw += 1
            if inoptw * inopth > (n - inopth):
                inoptw -= 1
                inopth += 1
        else:
            inopth += 1
            if inoptw * inopth > (n - inoptw):
                inopth -= 1
                inoptw += 1

    return (inopth, inoptw)


def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    if isinstance(exponent, str):
        return 4
    return len(digits) + exponent - 1


def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()


def nextup(x, steps=[1, 2, 5, 10]):
    """
    Find the next value up in 1,2,5 sequence
    """
    x_exp = fexp(x)
    x_man = fman(x)
    for i, s in enumerate(steps):
        sd = Decimal(str(s))
        if sd > x_man:
            break
    return float(s) * np.power(10.0, x_exp)


def calbar(
    axl,
    calbar: Union[list, None] = None,
    scale: list = [1.0, 1.0],
    xyoffset: list = [0.05, 0.1],
    axesoff: bool = True,
    orient: Literal["left", "right"] = "left",
    unitNames: Union[dict, None] = None,
    fontsize: Union[float, None] = None,
    linewidth: float = 1.5,
    weight: Literal["normal", "bold"] = "normal",
    color: "str" = "k",
    font: str = "Arial",
):
    """
    draw a calibration bar and label it. T

    Parameters
    ----------
    axl : axis or list of axes

    calbar :  cal position and size array [x0, y0, xlen, ylen] (default: None)

    scale : (default [1.0, 1.0])
        scale factor to apply to the calibration bar in x and y axes
        The scale is applied to the printed value, not the actual calbar
        xylh values

    xyoffset: (default[0.05, 0.1])
        cal bar offset, x, y. Make larger to space text further from the bar
        (for example, [0.08, 0.15])

    axesoff : boolean (default: True)
        If true, turns of the standard axes so we just sidplay the calibration bar

    orient : string (default: 'left')
        Orientation ['left', 'right'], whether the vertical part is to the left or right

    unitNames : dict with 'x' and 'y' (default: None)
        Names of the units for each of the axes, e.g., {'x': 'ms', 'y': 'pA'}

    fontsize : float (default: 11)
        Text font size

    weight : string (default: 'normal')
        Weight for the text

    color : string or matplotlib color (default: 'k')
        Color for the bar and text

    font : string or matplotlib font (default: 'Arial')
        The font to use for the text and labels

    Returns
    -------
    Nothing

    """
    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        if axesoff:
            noaxes(ax)
        Hfmt = r"{:.0f}"
        if calbar[2] * scale[0] < 1.0:
            Hfmt = r"{:.1f}"
        Vfmt = r" {:.0f}"
        if calbar[3] * scale[1] < 1.0:
            Vfmt = r" {:.1f}"
        if unitNames is not None:
            Vfmt = Vfmt + r" " + r"{:s}".format(unitNames["y"])
            Hfmt = Hfmt + r" " + r"{:s}".format(unitNames["x"])
        # print(Vfmt, unitNames['y'])
        # print(Vfmt.format(calbar[3]))
        font = FontProperties()
        font.set_family("sans-serif")
        font.set_weight = weight
        font.set_size = fontsize
        font.set_style("normal")

        if calbar is not None:
            if orient == "left":  # vertical part is on the left
                ax.plot(
                    [calbar[0], calbar[0], calbar[0] + calbar[2]],
                    [calbar[1] + calbar[3], calbar[1], calbar[1]],
                    color=color,
                    linestyle="-",
                    linewidth=linewidth,
                    clip_on=False,
                )
                if calbar[3] != 0.0:
                    ax.text(
                        calbar[0] + xyoffset[0] * calbar[2],
                        calbar[1] + 0.5 * calbar[3],
                        Vfmt.format(calbar[3] * scale[1]),
                        horizontalalignment="left",
                        verticalalignment="center",
                        color=color,
                        fontsize=fontsize,
                        weight=weight,
                        family="sans-serif",
                        clip_on=False,
                    )
            elif orient == "right":  # vertical part goes on the right
                ax.plot(
                    [calbar[0] + calbar[2], calbar[0] + calbar[2], calbar[0]],
                    [calbar[1] + calbar[3], calbar[1], calbar[1]],
                    color=color,
                    linestyle="-",
                    linewidth=linewidth,
                    clip_on=False,
                )
                if calbar[3] != 0.0:
                    ax.text(
                        calbar[0] + calbar[2] - xyoffset[0] * calbar[2],
                        calbar[1] + 0.5 * calbar[3],  # center
                        Vfmt.format(calbar[3] * scale[1]),
                        horizontalalignment="right",
                        verticalalignment="center",
                        color=color,
                        fontsize=fontsize,
                        weight=weight,
                        family="sans-serif",
                        clip_on=False,
                    )
            else:
                print("PlotHelpers.py: I did not understand orientation: %s" % (orient))
                print("plotting as if set to left... ")
                ax.plot(
                    [calbar[0], calbar[0], calbar[0] + calbar[2]],
                    [calbar[1] + calbar[3], calbar[1], calbar[1]],
                    color=color,
                    linestyle="-",
                    linewidth=linewidth,
                    clip_on=False,
                )

                ax.text(
                    calbar[0] + xyoffset[0] * calbar[2],
                    calbar[1] + 0.5 * calbar[3] * scale[1],  # center
                    Vfmt.format(calbar[3] * scale[1]),
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=color,
                    fontsize=fontsize,
                    weight=weight,
                    family="sans-serif",
                    clip_on=False,
                )
            if calbar[2] != 0.0:
                ax.text(
                    calbar[0] + calbar[2] * 0.5,
                    calbar[1] - xyoffset[1] * calbar[3],
                    Hfmt.format(calbar[2] * scale[0]),
                    horizontalalignment="center",
                    verticalalignment="top",
                    color=color,
                    fontsize=fontsize,
                    weight=weight,
                    family="sans-serif",
                    clip_on=False,
                )


def referenceline(
    axl: object,
    reference: Union[float, None] = None,
    limits: Union[list, tuple, None] = None,
    color: list = [0.33, 0.33, 0.33, 1],
    linestyle: str = "--",
    linewidth: float = 0.5,
    dashes: Union[str, None] = None,
):
    """
    draw a reference line at a particular level of the data on the y axis
    returns the line object.

    Parameters
    ----------
    axl : axes object (object, list of objects, etc)

    reference : float (default: None)
        The value for the reference line. If None, we plot at y=0.

    limits : list or tuple of (xmin, xmax) (default: None)
        limits over which the reference line will be drawn

    color : float or matplotlibcolor (default: 0.33)
        line color

    linestyle : string (default: '--')
        The style for the reference line

    linewidth : float (default: 0.5)
        line width in points

    dashes : mpl dashes value (default: None)
        Type of special dashes to use in line

    Returns
    -------
    The reference line instance
    """

    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    if reference is None:
        reference = 0.0
    for ax in axl:
        if ax is None:
            continue
        if limits is None or type(limits) is not list or len(limits) != 2:
            xlims = ax.get_xlim()
        else:
            xlims = limits
        (rl,) = ax.plot(
            [xlims[0], xlims[1]],
            [reference, reference],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        if dashes is not None:
            rl.set_dashes(dashes)
    return rl


def crossAxes(
    axl,
    xyzero=[0.0, 0.0],
    limits=[None, None, None, None],
    labels: Union[list, None] = ["nA", "mV"],
):
    """
    Make plot(s) with crossed axes at the data points set by xyzero, and optionally
    set axes limits

    Parameters
    ----------
    axl : axes objects (single, list, etc)
        list or similar of axes that will be converted to crossed format

    xyzero : list or tuple of floats (length=2)
        the position used for the zero in x and y where
        the axes will cross

    limits: list or tuple of floats (length=4)
        min and max for x, y in order of: [xmin, ymin, xmax, ymax]

    Returns
    -------
    Nothing
    """

    axl = _ax_tolist(axl)
    # if type(axl) is not list:
    #     axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        #        ax.set_title('spines at data (1,2)')
        #        ax.plot(x,y)
        ax.spines["left"].set_position(("data", xyzero[0]))
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position(("data", xyzero[1]))
        ax.spines["top"].set_color("none")
        # ax.spines['left'].set_smart_bounds(True)
        # ax.spines['bottom'].set_smart_bounds(True)  # deprecated, not sure what to do
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        if limits[0] is not None:
            ax.set_xlim(left=limits[0], right=limits[2])
            ax.set_ylim(bottom=limits[1], top=limits[3])


def violin_plot(ax, data, pos, bp=False, median=False):
    """
    create violin plots on an axis

    Parameters
    ----------
    ax : axis object for the plot

    data : float array
        The data to be plotted

    pos : float array
        The x position to be plotted

    bp : boolean (default: False)
        If True, also plot a boxplot

    median : boolean (default: False)
        If True, draw a line at the median of the data

    Returns
    -------
    Nothing
    """

    dist = max(pos) - min(pos)
    w = min(0.15 * max(dist, 1.0), 0.5)
    for d, p in zip(data, pos):
        k = gaussian_kde(d)  # calculates the kernel density
        m = k.dataset.min()  # lower bound of violin
        M = k.dataset.max()  # upper bound of violin
        x = np.arange(m, M, (M - m) / 100.0)  # support for violin
        v = k.evaluate(x)  # violin profile (density curve)
        v = v / v.max() * w  # scaling the violin to the available space
        ax.fill_betweenx(x, p, v + p, facecolor="y", alpha=0.3)
        ax.fill_betweenx(x, p, -v + p, facecolor="y", alpha=0.3)
        if median:
            ax.plot([p - 0.5, p + 0.5], [np.median(d), np.median(d)], "-")
    if bp:
        bpf = ax.boxplot(data, notch=0, positions=pos, vert=1)
        mpl.setp(bpf["boxes"], color="black")
        mpl.setp(bpf["whiskers"], color="black", linestyle="-")


def make_colorbar(
    ax,
    pos: Union[list, tuple] = [0.0, 0.0, 1.0, 1.0],
    vmin=0,
    vmax=1.0,
    nticks=4,
    bbox=[0.1, 0.80, 0.8, 0.15],
    palette: str = "hls",
):
    """
    Make a floating colorbar with its own axes that can be placed "anywhere"
    relative to an existing axis (ax).
    pos feeds the rectangles in the colorbar (the defaults should be ok)
    bbox is x0, y0, xl, yh (relative to the parent axis)
    vmin, vmax refer to the values assigned along the colorbar scale from 0 to 1
    nticks is the number of tick marks between vmin and vmax.

    """
    # build color patches
    cmx = sns.color_palette(palette, as_cmap=True)
    xpos = pos[0] + np.linspace(0, 1, cmx.N)
    pc = [
        matplotlib.patches.Rectangle(
            (xpos[i], pos[1]),
            width=pos[2],
            height=pos[3],
            facecolor=c,
            color=c,
            edgecolor=c,
        )
        for i, c in enumerate(cmx.colors)
    ]
    p = matplotlib.collections.PatchCollection(pc, match_original=True)

    axins = INSET.inset_axes(
        ax,
        width="100%",
        height="20%",
        bbox_to_anchor=bbox,
        bbox_transform=ax.transAxes,
        loc=3,
    )
    axins.set_yticks([])
    ticklabel_f = np.linspace(vmin, vmax, nticks)
    ticklabels = [f"{int(x):d}" for x in ticklabel_f]
    axins.set_xticks(ticks=np.linspace(0, 1.0, nticks), labels=ticklabels)
    axins.add_collection(p)
    # ax.set_clip_on(False)
    return axins


mscale.register_scale(SquareRootScale)

# # from somewhere on the web:


class NiceScale:
    """
    Class to select what I condisider to be better scaling range
    choices than the default
    """

    def __init__(self, minv, maxv):
        """
        Parameters
        ----------
        minv : float (no default)
            min value for axis
        maxv : float (no default)
            max value for axis

        Returns
        -------
        Nothing. all values are class variables.
        self.maxTicks = 6
        self.tickSpacing = 0
        self.lst = 10
        self.niceMin = 0
        self.niceMax = 0
        self.minPoint = minv
        self.maxPoint = maxv

        """
        self.maxTicks = 6
        self.tickSpacing = 0
        self.lst = 10
        self.niceMin = 0
        self.niceMax = 0
        self.minPoint = minv
        self.maxPoint = maxv
        self.calculate()

    def calculate(self):
        """
        Perform the calculation based on the data we have been given
        """
        self.lst = self._niceNum(self.maxPoint - self.minPoint, False)
        self.tickSpacing = self._niceNum(self.lst / (self.maxTicks - 1), True)
        self.niceMin = np.floor(self.minPoint / self.tickSpacing) * self.tickSpacing
        self.niceMax = np.ceil(self.maxPoint / self.tickSpacing) * self.tickSpacing

    def _niceNum(self, lst, rround):
        """
        Private: compute nice numbers for the axes
        """
        self.lst = lst
        exponent = 0  # exponent of range */
        fraction = 0  # fractional part of range */
        niceFraction = 0  # nice, rounded fraction */

        exponent = np.floor(np.log10(self.lst))
        fraction = self.lst / np.power(10, exponent)

        if self.lst:
            if fraction < 1.5:
                niceFraction = 1
            elif fraction < 3:
                niceFraction = 2
            elif fraction < 7:
                niceFraction = 5
            else:
                niceFraction = 10
        else:
            if fraction <= 1:
                niceFraction = 1
            elif fraction <= 2:
                niceFraction = 2
            elif fraction <= 5:
                niceFraction = 5
            else:
                niceFraction = 10

        return niceFraction * np.power(10, exponent)

    def setMinMaxPoints(self, minPoint, maxPoint):
        """
        Adjust the maximum values and recalculate
        """
        self.minPoint = minPoint
        self.maxPoint = maxPoint
        self.calculate()

    def setMaxTicks(self, maxTicks):
        """
        Reset the max number of tick marks and
        recalculate
        """
        self.maxTicks = maxTicks
        self.calculate()


def circles(x, y, s, c="b", ax=None, vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data scale (ie. in data unit)
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or
        RGBA sequence because that is indistinguishable from an array of
        values to be colormapped.  `c` can be a 2-D array in which the
        rows are RGB or RGBA, however.
    ax : Axes object, optional, default: None
        Parent axes of the plot. It uses gca() if not specified.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.  (Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.)

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Other parameters
    ----------------
    kwargs : `~matplotlib.collections.Collection` properties
        eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')

    """

    if ax is None:
        ax = mpl.gca()

    if np.isscalar(x):
        patches = [
            Circle((x, y), s),
        ]
    elif np.isscalar(s):
        patches = [Circle((x_, y_), s) for x_, y_ in zip(x, y)]
    else:
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zip(x, y, s)]
    collection = PatchCollection(patches, **kwargs)

    if isinstance(c, (str, bytes)):
        color = c  # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
        kwargs.update(color=color)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def rectangles(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    sw: Union[None, List, Tuple] = None,
    sh: Union[None, List, Tuple] = None,
    c: str = "b",
    ax: Union[mpl.axes, List] = None,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    **kwargs,
) -> object:
    """
    Make a scatter of squares plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of sqares are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        side of square in data scale (ie. in data unit)
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or
        RGBA sequence because that is indistinguishable from an array of
        values to be colormapped.  `c` can be a 2-D array in which the
        rows are RGB or RGBA, however.
    ax : Axes object, optional, default: None
        Parent axes of the plot. It uses gca() if not specified.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.  (Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.)

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Other parameters
    ----------------
    kwargs : `~matplotlib.collections.Collection` properties
        eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap

    Examples
    --------
    a = np.arange(11)
    squaress(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')


    """
    # import matplotlib.colors as colors

    if ax is None:
        ax = mpl.gca()

    if isinstance(c, (str, bytes)):
        color = c  # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
    kwargs.update(color=color)
    if sh is None:
        sh = sw
    if not isinstance(x, (List, np.ndarray)):
        x = [x]
    if not isinstance(y, (List, np.ndarray)):
        y = [y]
    x = [p - sw / 2.0 for p in x]  # offset as position specified is "lower left corner"
    y = [p - sh / 2.0 for p in y]
    if np.isscalar(x):
        patches = [
            Rectangle((x, y), sw, sh),
        ]
    elif np.isscalar(sw):
        patches = [Rectangle((x_, y_), sw, sh) for x_, y_ in zip(x, y)]
    else:
        patches = [
            Rectangle((x_, y_), sw_, sh_) for x_, y_, sw_, sh_ in zip(x, y, sw, sh)
        ]
    collection = PatchCollection(patches, **kwargs)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def show_figure_grid(fig: mpl.figure, figx: int = 10, figy: int = 10, alpha=1.0) -> object:
    """
    Create a background grid with major and minor lines like graph paper
    if using default figx and figy, the grid will be in units of the
    overall figure on a [0,1,0,1] grid
    if figx and figy are in units of inches or cm, then the grid
    will be on that scale.

    Figure grid is useful when building figures and placing labels
    at absolute locations on the figure.

    Parameters
    ----------

    fig : Matplotlib figure handle (no default):
        The figure to which the grid will be applied

    figx : float (default: 10.)
        # of major lines along the X dimension

    figy : float (default: 10.)
        # of major lines along the Y dimension

    """

    backGrid = fig.figure_handle.add_axes([0, 0, 1, 1], frameon=False)
    backGrid.set_ylim(0.0, figy)
    backGrid.set_xlim(0.0, figx)
    backGrid.grid(True)

    backGrid.set_yticks(np.arange(0.0, figy + 0.01, 1.0))
    backGrid.set_yticks(np.arange(0.0, figy + 0.01, 0.1), minor=True)
    backGrid.set_xticks(np.arange(0.0, figx + 0.01, 1.0))
    backGrid.set_xticks(np.arange(0.0, figx + 0.01, 0.1), minor=True)
    #   backGrid.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    #   backGrid.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    backGrid.grid(visible=True, which="major", color="g", alpha=alpha, linewidth=0.8)
    backGrid.grid(visible=True, which="minor", color="g", alpha=alpha*0.6, linewidth=0.2)
    return backGrid


def hide_figure_grid(fig: mpl.figure, grid: mpl.grid) -> None:
    """
    Hide the figure grid in the figure
    """
    grid.grid(False)


def delete_figure_grid(fig: mpl.figure, grid: mpl.grid) -> None:
    """
    Remove the figure grid from the figure
    """

    mpl.delete(grid)


def figure_scaling(
    units: str,
    figsize: Union[list, tuple],
    margins: dict,
    verticalspacing: Union[float, None] = None,
    horizontalspacing: Union[float, None] = None,
):
    # perform all unit conversions now
    # this converts all measures to the "unit measures" on a page
    # e.g. page is 0..1 x 0..1
    assert units in ["page", "inches", "in", "cm", "centimeters"]
    if units in ["cm", "centimeters", "inches", "in"]:
        margins["leftmargin"] = margins["leftmargin"] / figsize[0]
        margins["rightmargin"] = margins["rightmargin"] / figsize[0]
        margins["topmargin"] = margins["topmargin"] / figsize[1]
        margins["bottommargin"] = margins["bottommargin"] / figsize[1]
    if verticalspacing is not None and units != "page":
        verticalspacing = verticalspacing / figsize[1]
    if horizontalspacing is not None and units != "page":
        horizontalspacing = horizontalspacing / figsize[0]
    if units in ["cm", "centimeters"] and figsize is not None:
        figsize = [f / 2.54 for f in figsize]  # rescale figure page size
    return figsize, margins, verticalspacing, horizontalspacing


def regular_grid(
    rows: int,
    cols: int,
    order: str = "columnsfirst",
    units: str = "page",
    figsize: tuple = (8.0, 10),
    showgrid: bool = False,
    verticalspacing: float = 0.08,
    horizontalspacing: float = 0.08,
    margins: dict = {
        "leftmargin": 0.07,
        "rightmargin": 0.05,
        "topmargin": 0.03,
        "bottommargin": 0.1,
    },
    labelposition: tuple = (0.0, 0.0),
    parent_figure: Union[mpl.figure, None] = None,
    panel_labels: Union[str, List, None] = None,
    font: str="Arial",
    **kwds,
) -> Plotter:
    """
    make a regular layout grid for plotters

    Parameters
    ----------

    rows : int (no default):
        number of rows in figure
    cols : int (nodefault)
        number of columns in figure
    order : str (default: 'columns')
        lettering order 'rows' | 'columns'
    figsize : tuple floats (default: (8., 10.))
        figure size (width, height) in inches
    showgrid : bool (default: False)
        If true, plots a pale green grid on the page to help with alignment
    verticalspacing : float (default: 0.08)
        fractional width of spacing between columns
    horizontalspacing : float (default: 0.08)
        fractional height spacing between rows
    margins : dict (default: 'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.03, 'bottommargin': 0.1})
        fraxtional spacing around borders of graphs relative to edge of "paper"
        You can also use 'width' and 'height' to use absolute settings where
        rightmargin = width of the plot after allowing for left margin
        topmargin = height of plot after allowing for bottom margin
    labelposition : tuple of floats (default: (-0.12, 0.95))
        panel label offset from axes. Axes are from 0 to 1, so default places label to left
        and just below the top of the left axis.
    parent_figure : Figure object
        The object of a parent figure into which this plot will be inserted
    panel_labels : None or list
        The labels for the panels to be created in this grid.
        If None, and there is a parent figure, we continue labeling in order from that figure A, B, C, etc.
        Otherwise, the labels should be a list, and must be unique (no checking)
        to the figure and should match the rowsxcols
        panel_labels sets up a convience dictionary as well, in which the labels point to the associated axes.

    **kwds:
        inciudes:
        parent_figure
        prior_label : last label of previous grid, so start labeling with next label in list
        num : number (or string name) to attach to the figure.
    """

    figsize, margins, verticalspacing, horizontalspacing = figure_scaling(
        units=units,
        figsize=figsize,
        margins=margins,
        verticalspacing=verticalspacing,
        horizontalspacing=horizontalspacing,
    )
    original_units = units
    units = "page"  # we have already converted, so no further action
    lmar = margins["leftmargin"]
    mkeys = list(margins.keys())
    assert ("rightmargin" in mkeys) or ("width" in mkeys)
    if "rightmargin" in mkeys:
        rmar = margins["rightmargin"]
    else:
        rmar = 1.0 - (lmar + margins["width"])
        margins["rightmargin"] = rmar
    assert "bottommargin" in mkeys or "height" in mkeys
    bmar = margins["bottommargin"]
    if "topmargin" in mkeys:
        tmar = margins["topmargin"]
    else:
        tmar = 1.0 - (bmar + margins["height"])
        margins["topmargin"] = tmar
    if original_units == "page":
        assert (tmar + bmar) < 1.0
        assert (lmar + rmar) < 1.0
    else:
        assert (tmar + bmar) < figsize[1]
        assert (lmar + rmar) < figsize[0]

    hs = horizontalspacing
    # tmar = margins["topmargin"]
    # bmar = margins["bottommargin"]
    vs = verticalspacing

    xw = ((1.0 - lmar - rmar) - (cols - 1.0) * hs) / cols
    xl = [lmar + (xw + hs) * i for i in range(0, cols)]
    yh = ((1.0 - tmar - bmar) - (rows - 1.0) * vs) / rows
    yb = [1.0 - tmar - (yh * (i + 1)) - vs * i for i in range(0, rows)]
    if panel_labels is None:  # just generate a list of the labels in alphabetical order
        plabels = list(string.ascii_uppercase)
        a2 = ["%c%c" % (plabels[i], b) for i in range(len(plabels)) for b in plabels]
        plabels.extend(a2)

    # auto generate sizer dict based on this

    sizer = OrderedDict()
    if (
        panel_labels is None and parent_figure is not None
    ):  # generate labels continuing from parent figure
        lastlabel = list(parent_figure.axdict.keys())[-1]
        if lastlabel in plabels:
            istart = plabels.index(lastlabel) + 1
    elif panel_labels is not None:
        istart = 0
        plabels = panel_labels  # use the labels provided in the call
    else:
        istart = 0  # panel_labels is none and start at first index

    if order == "rowsfirst":
        i = 0
        for r in range(rows):
            for c in range(cols):
                pos = [xl[c], xw, yb[r], yh]
                sizer[plabels[i + istart]] = {
                    "pos": pos,
                    "labelpos": labelposition,
                    "noaxes": False,
                }
                i = i + 1
    else:
        i = 0
        for c in range(cols):
            for r in range(rows):
                pos = [xl[c], xw, yb[r], yh]
                sizer[plabels[i + istart]] = {
                    "pos": pos,
                    "labelpos": labelposition,
                    "noaxes": False,
                }
                i = i + 1
    gr = [
        (a, a + 1, 0, 1) for a in range(0, rows * cols)
    ]  # just generate subplots - shape does not matter
    axmap = OrderedDict(zip(sizer.keys(), gr))
    if "label" not in kwds.keys():  # keep label in kwds
        if panel_labels is None:
            kwds["label"] = False
        else:
            kwds["label"] = True
    P = Plotter(
        (rows, cols),
        axmap=axmap,
        units="page",
        figsize=figsize,
        margins=margins,
        labelposition=labelposition,
        font = font,
        parent_figure=parent_figure,
        order=order,
        **kwds,
    )
    if showgrid:
        show_figure_grid(P)
    P.resize(sizer)  # perform positioning magic
    P.sizer = sizer
    return P


def test_sizergrid():
    """
    Just display a regular grid in a test mode.
    Grid should be 8x3
    """
    regular_grid(8, 3)
    mpl.show()


def arbitrary_grid(sizer, units="page", figsize=(8, 10), showgrid=False, **kwds):
    """
    Helper function
    Takes a plot definition array dict:
        sizer = {'A': {'pos': [0.08, 0.22, 0.50, 0.4], 'labelpos': (x,y), 'noaxes': True}, 'B1': {'pos': [0.40, 0.25, 0.60, 0.3], 'labelpos': (x,y)},
        'B2': {'pos': [0.40, 0.25, 0.5, 0.1],, 'labelpos': (x,y), 'noaxes': False},
        'C1': {'pos': [0.72, 0.25, 0.60, 0.3], 'labelpos': (x,y)}, 'C2': {'pos': [0.72, 0.25, 0.5, 0.1], 'labelpos': (x,y)},
        'D': {'pos': [0.08, 0.25, 0.1, 0.3], 'labelpos': (x,y)},
        'E': {'pos': [0.40, 0.25, 0.1, 0.3], 'labelpos': (x,y)}, 'F': {'pos': [0.72, 0.25, 0.1, 0.3],, 'labelpos': (x,y)}
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
    A minimum dict would just have the panel labels as keys, and the position information
    Sets up figure panels, returning the plotter instance
    All keywords are passed to plotter
    """
    nplots = len(sizer)
    gr = [
        (a, a + 1, 0, 1) for a in range(0, nplots)
    ]  # just generate subplots - shape does not matter
    cm_scale = 1.0
    if units == "cm":
        cm_scale = 2.54
        figsize[0] = figsize[0] / cm_scale
        figsize[1] = figsize[1] / cm_scale
    if units != "page":  # use figsize to convert positions to page
        for ax in sizer:
            p = sizer[ax]["pos"]
            p[0] = p[0] / figsize[0]
            p[1] = p[1] / figsize[0]
            p[2] = p[2] / figsize[1]
            p[3] = p[3] / figsize[1]

    axmap = OrderedDict(zip(sizer.keys(), gr))
    P = Plotter((nplots, 1), axmap=axmap, figsize=figsize, units="page", **kwds)
    # PH.show_figure_grid(P.figure_handle)
    P.resize(sizer)  # perform positioning magic
    if showgrid:
        show_figure_grid(P, figsize[0], figsize[1])
    return P




def main(sf="P5"):
    #    P = Plotter((3,3), axmap=[(0, 1, 0, 3), (1, 2, 0, 2), (2, 1, 2, 3), (2, 3, 0, 1), (2, 3, 1, 2)])
    #    test_sizergrid()
    #    exit(1)
    # labels = ["A", "B"]  # , 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    # left = [(a, a + 2, 0, 1) for a in range(0, 6, 2)]
    # right = [(a, a + 1, 1, 2) for a in range(0, 6)]
    # axmap = OrderedDict(zip(labels, left + right))
    # P = Plotter((1, 2), axmap=axmap, figsize=(6.0, 6.0), label=True)
    # #    P = Plotter((2,3), label=True)  # create a figure with plots
    # for a in P.axarr.ravel():
    #     a.plot(np.random.random(10) * 3, np.random.random(10) * 72)
    # nice_plot(P.axdict["A"])
    # talbotTicks(
    #     P.axdict["A"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 1, "y": 1}
    # )
    # show_figure_grid(P.figure_handle)
    # P.figure_handle.suptitle("random")

    # make a more complex plot
    if sf == "P1":
        P1 = Plotter((3, 2), units="page", figsize=(8.0, 10.0))  # a default
        show_figure_grid(P1)
        P1.figure_handle.suptitle("Defaults, units=page")
        mpl.show()
    elif sf == "P2":
        P2 = Plotter(
            (3, 2),
            units="in",
            margins={
                "leftmargin": 1.0,
                "rightmargin": 1.0,
                "topmargin": 1.5,
                "bottommargin": 1.0,
            },
            figsize=(8.0, 10.0),
        )
        show_figure_grid(P2)
        P2.figure_handle.suptitle("defaults, units=inches")
        mpl.show()
        return
    elif sf == "P3":
        P3 = Plotter(
            (3, 2),
            units="cm",
            margins={
                "leftmargin": 1.0,
                "rightmargin": 1.0,
                "topmargin": 1.5,
                "bottommargin": 1.0,
            },
            figsize=(8.0, 10.0),
        )
        P3.figure_handle.suptitle("Defaults, units=cm")
        mpl.show()
        return
    elif sf == "P4":
        P4 = regular_grid(
            rows=3,
            cols=2,
            units="cm",
            margins={
                "leftmargin": 1.0,
                "rightmargin": 1.0,
                "topmargin": 2.0,
                "bottommargin": 1.0,
            },
            figsize=(12.0, 16.0),
        )
        P4.figure_handle.suptitle("regulargrid, units=cm")
        show_figure_grid(P4)
        mpl.show()
        return

    elif sf == "P5":
        x = -0.05
        y = 1.05
        sizer = {
            "A": {"pos": [0.5, 1.4, 4, 1.5], "labelpos": (x, y)},
            "B": {"pos": [2.2, 1.4, 4, 1.53], "labelpos": (x, y)},
            "C": {"pos": [0.5, 3, 2, 1.5], "labelpos": (x, y)},
            "D": {"pos": [0.5, 2.5, 0.5, 1.25], "labelpos": (x, y)},
            "E": {"pos": [3.25, 0.5, 0.5, 1.00], "labelpos": (x, y)},
        }
        P5 = arbitrary_grid(sizer, units="in", figsize=(4, 6))
        P5.figure_handle.suptitle("arbitrary grid, units=in")
        show_figure_grid(P5, int(P5.figsize[0]), int(P5.figsize[1]))

        mpl.show()
    return


if __name__ == "__main__":
    main(sf="P5")
