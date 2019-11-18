import pyqtgraph as pg
from . import pyqtgraphPlotHelpers as PH
from pyqtgraph.Qt import QtCore, QtGui


def main():
    import sys
    pa = pg.plot([0,2,3,5,8], [2,9,0,1,3])
    p = pa.getPlotItem()
    PH.nice_plot(p, axesoff=True)
    PH.refline(p, refline = 0.)
    PH.calbar(pa, [1,1,1,1])
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()
