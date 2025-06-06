"""
Styler.py - based on awseomizer.py
Removed much of the code in the geometry section, as this is handled by plothelpers
The style routines are useful for us however.
pbmanis 7/2018

How to use:

call
ST.styler(journal, figuresize, etc)
ST.setxyz to adjust some factors.
create figure (e.g., with PlotHelpers, such as rectangular grid,
and use figsize=style.Figure['figsize'], etc during initial call.

ST.apply()
ST.geometry_adjust()


-----------------------------------------------------------------------------
Distributed under the GNU General Public License.

Contributors: Andrei Maksimov (maksimov.andrei7@gmail.com)
-----------------------------------------------------------------------------
File description:

Contains functions used to improve appearance of figures
-----------------------------------------------------------------------------
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import string


# --------------------------------------------------------------


class styler:
    """
    Rules defining figure style
    Creates an object with the following additional attributes:

    Parameters
    ----------
        -Font     = dict
        -Lines    = dict
        -Axes     = dict
        -Ticks    = dict
        -Grid     = dict
        -Legend   = dict
        -Patch    = dict
        -Text     = dict
        -Savefig  = dict
        -Figure   = dict
        -Misc     = dict
        -Colors   = dict
    """

    def __init__(self, journal=None, figuresize="small", height_factor=1.0, font="Arial"):
        if not journal in ["PLoS", "JNeurophys", "JNeurosci", "CerebralCortex", "Generic"]:
            raise ValueError("got journal: %s" % journal)
        self.journal = journal
        self.golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        self.font = font
        self.font_pt = 8  # [pt] font size
        self.figuresize = figuresize
        self.height_factor = height_factor
        self.set_style()  # init with a default - set later

    def set_base_font_pt(self, font_pt):
        self.font_pt = font_pt
        self.set_style()

    def set_panel_fontweight(self, fontweight):
        self.Panel["fontweight"] = fontweight
        self.set_style()

    def set_style(self):
        # set default figure properties
        self.Font = {
            "weight": "normal",
            "size": self.font_pt,
            "family": "sans-serif",
            "sans-serif": self.font,
        }

        self.Lines = {
            "linewidth": 1.0,  # [pt] line width
            "markersize": 3.0,  # [pt] markersize
            "markeredgewidth": 1.0,  # width of marker border
        }

        self.Axes = {
            "grid": False,
            "titlesize": self.font_pt * 1.2,
            "labelweight": "normal",
            "linewidth": 1.0,  # edge linewidth
        }

        self.Ticks = {
            "labelsize": self.font_pt,
            "major.size": self.font_pt / 8.0,  # [pt] major tick size
            "minor.size": self.font_pt / 12.0,  # [pt] minor tick size
            "major.pad": self.font_pt / 4.0,  # [pt] distance to major tick label
            "minor.pad": self.font_pt / 4.0,  # [pt] distance to the minor tick label
            "direction": "out",  # direction of ticks relative to axis
        }

        self.Grid = {"linewidth": 0.5}

        self.Legend = {
            "fontsize": self.font_pt,
            "fancybox": True,
            "markerscale": 1.0,
            "shadow": True,
        }

        self.Label = {
            "fontsize": self.font_pt,
            "fontweight": "normal",
        }

        self.Panel = {
            "fontsize": self.font_pt * 1.25,
            "fontweight": "bold",
        }

        self.Patch = {"linewidth": 1.0}

        self.Text = {"usetex": True}  # use latex for text handling

        self.Savefig = {"dpi": 1200}  # resolution in dots per inch

        self.Figure = {"figsize": self.set_figure()}

        self.Misc = {
            "label_pad": 0,  # [pt] shift axis labels relative to ticks
            "text_ratio": 2,  # average ratio of y to x dimensions of letters
            "title_shift": 0.5 * self.font_pt,  # [pt] add this space to subplot
            # title's position
            "wspace": 1.0 * self.font_pt,  # [pt] extra horizontal space between subplots
            "hspace": 1.0 * self.font_pt,  # [pt] extra vertical space between subplots
            "cb_width": 1.0 * self.font_pt,  # [pt] thickness of color bars
            "cb_pad": 1.0 * self.font_pt,  # [pt] distance between plot and color bar
        }

        self.Colors = {
            "magenta": "#882255",
            "red": "#CC6677",
            "yellow": "#DDCC77",
            "green": "#117733",
            "blue": "#88CCEE",
            "white": "#FFFFFF",
            "black": "#000000",
        }

        for key in self.Colors.keys():
            matplotlib.colors.cnames[key] = self.Colors[key]

        self.style_apply()

    # style dict is rc params format
    # return {'font': Font, 'lines': Lines, 'axes': Axes,
    #         'xtick': Ticks, 'ytick': Ticks, 'grid': Grid, 'legend': Legend,
    #         'patch': Patch, 'text': Text, 'savefig': Savefig,
    #         'figure': Figure, 'misc': Misc, 'colors': Colors, 'label': Label, 'panel': Panel}

    def set_figure(self, width=8.5):
        """
        compute figure size from a dict (table) of sizes
        Also set the main font size to correspond
        """
        self.fig_widths = {
            "PLoS": {"small": 3.1, "medium": 3.1 * 1.5, "large": 3.1 * 2, "special": width},
            "CerebralCortex": {"single": 3.38582, "double": 7.0866, "special": width},
            "JNeurophys": {"single": 3.5, "double": 4.5, "full": 7.5, "special": width},
            "JNeurosci": {"single": 3.346, "double": 4.57, "full": 6.929, "special": width},
            "JPhysiol": {"single": 3.3, "double": 4.33, "full": 6.69, "special": width},
            "Generic": {"single": 3.25, "double": 5.0, "full": 7.0, "special": width},
        }

        if not self.figuresize in self.fig_widths[self.journal]:
            raise ValueError(
                "{0:s} fig_width should be float or one of {1:s}".format(
                    self.journal, str(self.fig_widths[self.journal].keys())
                )
            )

        fig_width = self.fig_widths[self.journal][self.figuresize]
        fig_height = self.height_factor * fig_width / self.golden_ratio
        return (fig_width, fig_height)

    def get_fontsizes(self):
        """
        make a font size dict with tick, label and panel font sizes from the style
        for PlotHelpers plotter
        """
        fontsizes = {
            "tick": self.Ticks["labelsize"],
            "label": self.Label["fontsize"],
            "panel": self.Panel["fontsize"],
        }
        return fontsizes

    def get_fontweights(self):
        """
        make a font size dict with tick, label and panel font sizes from the style
        for PlotHelpers plotter
        """
        fontweights = {
            "tick": "normal",  # ticks don't have a weight
            "label": self.Label["fontweight"],
            "panel": self.Panel["fontweight"],
            "axes": self.Axes["labelweight"],
        }
        return fontweights

    # ----------------------------------------------------------------------
    def style_apply(self):
        """
        Applies figure parameters defined in style as default for matplotlib

        Parameters
        ----------
        -style: list - list of dictionaries: Font, Lines, Axes, Ticks, Grid, Legend,
                       Patch, Text, Savefig, Figure, Misc. For details see example
                       style function,e.g. style_Plos()
        Returns
        -------
        -None
        """

        matplotlib.rc("font", **self.Font)  # **Font)
        matplotlib.rc("lines", **self.Lines)  # **Lines)
        matplotlib.rc("axes", **self.Axes)  # **Axes)
        matplotlib.rc("grid", **self.Grid)  # **Grid)
        matplotlib.rc("patch", **self.Patch)  # **Patch)
        matplotlib.rc("xtick", **self.Ticks)  # **Ticks)
        matplotlib.rc("ytick", **self.Ticks)  # **Ticks)
        matplotlib.rc("figure", **self.Figure)  # **Figure)
        matplotlib.rc("legend", **self.Legend)  # **Legend)
        matplotlib.rc("text", **self.Text)  # **Text)
        matplotlib.rc("savefig", **self.Savefig)  # **Savefig)

        return None


# -------------------------------------------------------------------------


def create_inset_axes(dim, ax, label):
    """
    Creates new sub-axes of a given size in the given position of
    existing axes

    Parameters
    ----------
    -dim: list/array - [x, y, size_x, size_y].
            x, y - coordinates of the left bottom corner of the sub-axes relative
            to existing axes (in units of axes 'ax' dimensions )
            size_x, size_y - dimensions of created sub-axes in units of
            axes 'ax' dimensions
    -ax:   axes - new axes are created relative to these axes

    label: a unique label for this inset axis (required).

    Returns
    -------
    -ax1  = new axes
    """

    fig = ax.get_figure()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(dim[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= dim[2]
    height *= dim[3]

    return fig.add_axes(
        [x, y, width, height], label=label
    )  # , axisbg='w')  # keyword removed in 3.2?


# -------------------------------------------------------------------------


def tight_layout(axes, style, filename, fig_width="small", label_order=[]):
    """
    Function for optimizing figure layout

    Parameters
    ----------
    -axes: list of matplotlib major axes instances
    -style: list - list of dictionaries: Font, Lines, Axes, Ticks, Grid, Legend,
            Patch, Text, Savefig, Figure, Misc.
            For details see example style function, e.g., style_Plos()
    -fig_width: str or float - 'small', 'medium', 'large' to adjust width to
             1, 1.5, 2 columns; use float to set width
             as a fraction of A4 width
    -filename: str - path + filename to which to save the figure. The file name
             should include the extension.
    -label_order: defines order of labels across subplots. Possible values:
            []: default - use standard left-to-right labeling
            ['a', 'b', 'c', 'd']: use this template as left-to-right sequence.
            Number of elements should be equal to number of axes

    Returns
    -------
    -None
    """

    # print ('---Preparing awesomization engine---')
    fig = axes[0]["ax"].get_figure()
    fig.savefig(filename)  # necessary for this function to work properly

    geometry_adjust(axes, style, fig_width, label_order)
    fig.savefig(filename)

    # print ('---Congratulations! Now your figure is AWESOME!!!---')

    return None


def geometry_adjust(axes, style, fig_width, label_order=[]):
    """
    Adjusts figure appearance:

    For each axis the following steps are performed:
    1) Compute geometrical position of corresponding subplot in the original
    figure
    2) Compute size of the figure region for the present subplot
    3) Compute spacing [left, right, bottom, top] from the axes reserved for
    labels and tick labels
    4) Adjust size of axes to fit in a given subplot region
    5) In case of color bar in a subplot, perform steps 3-4 separately for
    image and color bar

    Parameters
    ----------
    axes: list of matplotlib major axes instances.
           axes[i]={'ax':axes,'cb':None, 'twin':None} where axes=axes instance,
           cb=colorbar instance, twin=twin axes instance
    style: list - list of dictionaries:
                   Font, Lines, Axes, Ticks, Grid,
                   Legend, Patch,Text,Savefig,Figure,Misc.
                   For details see example style function, e.g., style_Plos()
    fig_width: str or float - 'small', 'medium', 'large' to adjust width to 1,
                1.5, 2 columns; use float to set width as a fraction of A4 width
    label_order: defines order of labels across subplots. Possible values:
              []: default - use standard left-to-right labeling
              ['a', 'b', 'c', 'd']: use this template as left-to-right sequence.
              Number of elements should be equal to number of axes
    Notes
    -----
    The present implementation is valid only for color bars used with option
    "use_gridspec=False".
    When creating color bars, do not specify 'fraction' or 'pad' arguments.
    To use mathematical symbols in labels:
    First, create a random label with a shape roughly similar to
    the expected one.
    Second, after geometry optimization, assign the mathematical
    expression to the label.
    3D plots still require manual adjustment.

    Returns
    -------
    None
    """

    # reconstruct the figure object

    figure = axes[0]["ax"].get_figure()

    # reconstruct dimensions related to texts

    # (Font, Lines, Axes, Ticks, Grid, Legend, Patch, Text, Savefig, Figure, Misc,
    #  Colors) = style

    font_pt = style["font"]["size"]  # size of standard font in pt
    font_inch = font_pt / 72.0  # standard font size transformed to inches
    tick_length = style["ticks"]["major.size"]  # length of ticks in pt
    tick_pad = style["ticks"]["major.pad"]  # padding for tick labels relative to
    # ticks in pt
    label_pad = style["misc"]["label_pad"]  # padding of labels relative to tick
    # labels in pt
    text_ratio = style["misc"]["text_ratio"]  # average ratio of y to x dimensions
    # of letters
    title_shift = style["misc"]["title_shift"]  # add this space in pt to subplot
    # title's position

    # define figure geometry

    # # number of subplots in y,x directions
    # ny, nx = fig.get_axes()[0].get_geometry()[0:2]
    #
    # # golden ratio of x to y subplot dimensions
    # x_to_y_ratio = (1. + np.sqrt(5)) / 2.
    #
    #
    #
    # dx = fig_width / nx       # size of single subplot in x direction
    # dy = dx / x_to_y_ratio    # size of single subplot in y direction
    # fig_height = dy * ny      # figure y size in inches

    figure.set_figwidth(figure["figsize"][0])
    figure.set_figheight(figure["figsize"][1])

    # for each subplot adjust dimensions
    for ax in axes:

        # check if axes are 2d or 3d
        ax_type = str(type(ax["ax"]))
        if ax_type == "<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
            flag_3d = True
        elif ax_type == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
            flag_3d = False
        else:
            raise Exception(
                "axes type should be                            \
                             matplotlib.axes._subplots.Axes3DSubplot or     \
                             matplotlib.axes._subplots.AxesSubplot"
            )

            # check if color bar is present
        flag_cb = ax["cb"] != None  # True if color bar exists

        # check the orientation of color bar
        if flag_cb:
            # can be 'horizontal' or 'vertical'
            cb_orientation = ax["cb"].orientation
        else:
            cb_orientation = None

        ###########################################
        #########  Analyze subplot dimension   ####
        ###########################################

        # define geometry of present subplot
        ax_id = ax["ax"].get_geometry()[2]  # ID of subplot in the figure

        # position (indices) of present subplot in x, y directions
        # from top left corner

        # ax_pos_x = np.mod(ax_id - 1, nx)
        # ax_pos_y = (ax_id - 1 - ax_pos_x) / nx
        #
        # # reconstruct dimensions of current subplot (case of complex subplots)
        #
        # box = ax['ax'].get_position()  # matplotlib object
        #
        # # subplot contains this amount of standart subplots in x direction
        # dim_x_ind = int(np.ceil(box.width / (dx / fig_width)))
        #
        # # subplot contains this amount of standart subplots in y direction
        # dim_y_ind = int(np.ceil(box.height / (dy / fig_height)))
        #
        # # size of subplot in x, y directions in units of figure size
        # dim_x = 1. * dim_x_ind / nx
        # dim_y = 1. * dim_y_ind / ny
        #
        # # left bottom corner of present subplot in units of figure
        # subplot_start = [1. * ax_pos_x / nx, 1. *
        #                  (ny - (ax_pos_y + dim_y_ind)) / ny]

        # adjust position of labels and ticks
        if not flag_3d:
            ax["ax"].xaxis.labelpad = label_pad
            ax["ax"].yaxis.labelpad = label_pad
            if "twin" in ax.keys():
                ax["twin"].xaxis.labelpad = label_pad
                ax["twin"].yaxis.labelpad = label_pad

        else:
            ax["ax"].xaxis._axinfo["label"]["space_factor"] = 2.8
            ax["ax"].yaxis._axinfo["label"]["space_factor"] = 2.8
            ax["ax"].zaxis._axinfo["label"]["space_factor"] = 2.8
            ax["ax"].xaxis._axinfo["label"]["va"] = "top"
            ax["ax"].yaxis._axinfo["label"]["va"] = "top"
            ax["ax"].zaxis._axinfo["label"]["va"] = "top"

        # make top and right axes invisible
        if not flag_3d:
            ax["ax"].spines["right"].set_visible(False)
            ax["ax"].spines["top"].set_visible(False)
            ax["ax"].yaxis.set_ticks_position("left")
            ax["ax"].xaxis.set_ticks_position("bottom")

        # find size of area required for text

        # for present axes contains dimensions (in units of font) required for
        # labels and ticks at [left, right, bottom, top] of the subplot
        Buffer = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}

        # Left size: horizontal dimension of label
        text = ax["ax"].get_ylabel().split("\n")
        if text != [""]:
            Buffer["left"] += len(text) + 1.0 * label_pad / font_pt

        # Left size: horizontal size related to max number of
        # letters in tick labels
        ticks = ax["ax"].get_yticklabels()
        longest = 0
        for tick in ticks:
            text = tick.get_text()
            if len(text) > 0:
                if text[0] == "$":
                    text = text[1:]
                if text[-1] == "$":
                    text = text[:-1]

            if len(text) > longest:
                longest = len(text)

        if longest > 0:
            Buffer["left"] += 1.0 * longest / text_ratio + 1.0 * tick_pad / font_pt

        # Left size: consider also ticks and wspace
        Buffer["left"] += 1.0 * (tick_length + Misc["wspace"] / 2.0) / font_pt

        if "twin" in ax.keys():
            # Right size:  horizontal dimension of label in case of twin axes
            text = ax["twin"].get_ylabel().split("\n")
            if text != [""]:
                Buffer["right"] += len(text) + 1.0 * label_pad / font_pt

            # Right size: horizontal size related to max number of letters in
            # ticklabels in case of twin axes
            ticks = ax["twin"].get_yticklabels()
            longest = 0
            for tick in ticks:
                text = tick.get_text()
                if len(text) > 0:
                    if text[0] == "$":
                        text = text[1:]
                    if text[-1] == "$":
                        text = text[:-1]

                if len(text) > longest:
                    longest = len(text)

            if longest > 0:
                Buffer["right"] += 1.0 * longest / text_ratio + 1.0 * tick_pad / font_pt

            # Right size: consider also ticks
            Buffer["right"] += 1.0 * (tick_length) / font_pt

        # Right size: consider the size of last ticklabel - can go beyond
        # the plot if too long
        if "twin" not in ax.keys():
            tick = ax["ax"].get_xticklabels()[-1]
            text = tick.get_text()
            if text != "":
                if text[0] == "$":
                    text = text[1:]
                if text[-1] == "$":
                    text = text[:-1]
            Buffer["right"] += len(text) / text_ratio / 2.0

        # Right size: consider also wspace if no vertical color bar
        if cb_orientation != "vertical":
            Buffer["right"] += 1.0 * (Misc["wspace"] / 2.0) / font_pt

        # Bottom size: vertical dimension of label
        text = ax["ax"].get_xlabel().split("\n")
        if text != [""]:
            Buffer["bottom"] += len(text) + 1.0 * label_pad / font_pt

        # Bottom size: vertical dimensions of tick labels
        ticks = ax["ax"].get_xticklabels()
        longest = 0
        for tick in ticks:
            text = tick.get_text()
            if len(text.split("\n")) > longest:
                longest = len(text.split("\n"))

        if longest > 0:
            Buffer["bottom"] += longest + 1.0 * tick_pad / font_pt

        # Bottom size: consider also tick_pad;  hspace is added if no color bar
        Buffer["bottom"] += 1.0 * (tick_length) / font_pt
        if cb_orientation != "horizontal":
            Buffer["bottom"] += 1.0 * (Misc["hspace"] / 2.0) / font_pt

        # Top size: vertical dimension of title and its shift
        text = ax["ax"].get_title().split("\n")
        if text != [""]:
            Buffer["top"] += len(text) + 1.0 * title_shift / font_pt

        # Top size: consider also hspace
        Buffer["top"] += 0.5
        Buffer["top"] += 1.0 * (Misc["hspace"] / 2.0) / font_pt  # account for tick_length
        # ,tick_pad and label_pad

        ############################################
        #########  Analyze color bar dimension  ####
        ############################################

        if flag_cb:

            # adjust position of labels
            ax["cb"].ax.xaxis.labelpad = label_pad
            ax["cb"].ax.yaxis.labelpad = label_pad

            # extract geometry of color bar.
            # position of color bar axes [left, bottom, right, top] in figure
            # units
            cb_pos = ax["cb"].ax.get_position()

            # find size of area required for text.
            # For present color bar contains dimensions (in units of font)
            # required for labels and ticks at [left, right, bottom, top] of
            # the color bar
            Buffer_cb = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}

            # Left size: horizontal, account for distance between plot and
            # color bar
            if cb_orientation == "vertical":
                Buffer_cb["left"] += 1.0 * Misc["cb_pad"] / font_pt

            # Bottom size: vertical
            if cb_orientation == "horizontal":

                # Bottom size: vertical dimension of label
                text = ax["cb"].ax.get_xlabel().split("\n")
                if text != [""]:
                    Buffer_cb["bottom"] += len(text) + 1.0 * label_pad / font_pt

                # Bottom size: vertical dimensions of tick labels
                ticks = ax["cb"].ax.get_xticklabels()
                longest = 0
                for tick in ticks:
                    text = tick.get_text()
                    if len(text.split("\n")) > longest:
                        longest = len(text.split("\n"))

                if longest > 0:
                    Buffer_cb["bottom"] += 1.0 * longest + 1.0 * tick_pad / font_pt

                # Bottom size: add hspace
                Buffer_cb["bottom"] += 1.0 * (Misc["hspace"] / 2.0) / font_pt

            # Right size: horizontal
            if cb_orientation == "vertical":
                # Right size: horizontal dimension of label
                text = ax["cb"].ax.get_ylabel().split("\n")
                if text != [""]:
                    Buffer_cb["right"] += len(text) + 1.0 * label_pad / font_pt

                # Right size: horizontal dimensions of tick labels
                ticks = ax["cb"].ax.get_yticklabels()
                longest = 0
                for tick in ticks:
                    text = tick.get_text()
                    if len(text) > 0:
                        if text[0] == "$":
                            text = text[1:]
                        if text[-1] == "$":
                            text = text[:-1]

                    if len(text) > longest:
                        longest = len(text)

                if longest > 0:
                    Buffer_cb["right"] += 1.0 * longest / text_ratio + 1.0 * tick_pad / font_pt

                # Right size: add wspace
                Buffer_cb["right"] += 1.0 * (Misc["wspace"] / 2.0) / font_pt

            # Top size: vertical => assume no text
            if cb_orientation == "horizontal":
                Buffer_cb["top"] += 1.0 * Misc["cb_pad"] / font_pt

        ####################################################
        # Adjust size of axes according to information on ##
        # text dimensions from buffer                     ##
        ####################################################

        # initial point of plot in case of no color bar
        # (left bottom corner)

        # start_x_ax = subplot_start[0] + Buffer['left'] * font_inch / fig_width
        # start_y_ax = subplot_start[1] + Buffer['bottom'] * font_inch / fig_height
        #
        # # correction of initial point for horizontal axes
        # if cb_orientation == 'horizontal':
        #     start_y_ax += (Buffer_cb['bottom'] + Buffer_cb['top'] +
        #                    Misc['cb_width'] / font_pt) * font_inch / fig_height
        #
        # # width of plot in case of no color bar
        # width_x_ax = dim_x - \
        #     (Buffer['left'] + Buffer['right']) * font_inch / fig_width
        # width_y_ax = dim_y - \
        #     (Buffer['top'] + Buffer['bottom']) * font_inch / fig_height
        #
        # # correction of plot dimensions for color bar
        # if cb_orientation == 'vertical':
        #     width_x_ax -= (Buffer_cb['left'] + Buffer_cb['right'] +
        #                    Misc['cb_width'] / font_pt) * font_inch / fig_width
        #
        # if cb_orientation == 'horizontal':
        #     width_y_ax -= (Buffer_cb['bottom'] + Buffer_cb['top'] +
        #                    Misc['cb_width'] / font_pt) * font_inch / fig_height
        #
        # # set dimensions of the plot and twin plot
        # ax['ax'].set_position(
        #     pos=[start_x_ax, start_y_ax, width_x_ax, width_y_ax])
        # if 'twin' in ax.keys():
        #     ax['twin'].set_position(pos=[start_x_ax, start_y_ax,
        #                                  width_x_ax, width_y_ax])
        #
        # # initial point of color bar
        # if cb_orientation == 'vertical':
        #     # actual position of right plot border
        #     stop_x_ax = ax['ax'].get_position().x1
        #     start_x_cb = stop_x_ax + (Buffer['right'] + Buffer_cb['left']) * \
        #                                                 font_inch / fig_width
        #     # actual position of bottom plot border
        #     start_y_cb = ax['ax'].get_position().y0
        #     # color bar width is fixed
        #     width_x_cb = Misc['cb_width'] / font_pt * font_inch / fig_width
        #     # color bar height is equal to height of plot
        #     width_y_cb = ax['ax'].get_position().height
        #
        # if cb_orientation == 'horizontal':
        #     # actual position of left plot border
        #     start_x_cb = ax['ax'].get_position().x0
        #     start_y_cb = subplot_start[1] + Buffer_cb['bottom'] * font_inch / \
        #                                                           fig_height
        #     # color bar width is fixed
        #     width_y_cb = Misc['cb_width'] / font_pt * font_inch / fig_height
        #     # color bar width is equal to width of plot
        #     width_x_cb = ax['ax'].get_position().width
        #
        # # set dimensions of the color bar
        # if flag_cb:
        #
        #     ax['cb'].ax.set_aspect('auto')
        #     ax['cb'].ax.set_position(pos=[start_x_cb, start_y_cb,
        #                                   width_x_cb, width_y_cb])

    ##############################
    ###  iterate over subplots  ##
    ##############################

    # define list of indices
    if label_order == []:
        list_labels = string.ascii_uppercase
    else:
        list_labels = label_order

    # extract indices of existing subplots. Label is assigned to subplot
    # according to its index.

    List_indices = []  # list with unique indices of existing subplots
    # Used later for subplot enumeration
    for ax in axes:

        ax_id = ax["ax"].get_geometry()[2]
        List_indices += [ax_id]

    # sort indices => define the sequence of labeling
    List_indices.sort()

    # assign label to subplot according to its index
    # for ax in axes:
    #
    #     # define geometry of present subplot
    #
    #     ax_id = ax['ax'].get_geometry()[2]  # ID of subplot in the figure
    #
    #     # position (indices) of present subplot
    #     # in x, y directions from top left corner
    #     ax_pos_x = np.mod(ax_id - 1, nx)
    #     ax_pos_y = (ax_id - 1 - ax_pos_x) / nx
    #
    #     # find position of top left corner of the subplot sector relative
    #     # to bottom left figure corner
    #     y = 1. - 1. * ax_pos_y / ny - 0.1 * font_inch / fig_height
    #     x = 1. * ax_pos_x / nx
    #
    #     # define the label
    #     label = list_labels[List_indices.index(ax_id)]
    #
    #     # place label to this coordinate
    #     if label != '':
    #         fig.text(x, y, r'\bfseries{}' + label,
    #                  weight='bold', va='top', ha='left')

    return None


# -----------------------------------------------------------------
