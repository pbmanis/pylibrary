#!/usr/bin/env python


# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 6, '')
__version__ = '%d.%d.%d%s' % version_info

listimports = False
import bootstrap
if listimports:
    print 'imported bootstrap'
import Fitting
if listimports:
    print 'imported Fitting'
import PlotHelpers
if listimports:
    print 'imported PlotHelpers'
import Utility
if listimports:
    print 'imported Utility'
import Params
if listimports:
    print 'imported Params'
# import RStats
# if listimports:
#     print 'imported RStats'
import pyqtgraphPlotHelpers
if listimports:
    print 'imported pyqtgraphPlotHelpers'
import permutation
if listimports:
    print 'imported permutation'
import tau_lmfit
if listimports:
    print 'imported tau_lmfit'
import simplex
if listimports:
    print 'imported simplex'
import TiffInfo
if listimports:
    print 'imported TiffInfo'
import tifffile
if listimports:
    print 'imported tifffile'
import titlecase
if listimports:
    print 'imported titlecase'
import matplotlibexporter
if listimports:
    print 'import matplotlibexporter'
# try:
#     import RStats
#     if listimports:
#         print 'imported RStats'
# except:
#     raise ImportWarning("pylibrary: failed to import RStats")
