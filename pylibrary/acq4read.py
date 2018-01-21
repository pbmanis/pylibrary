#!/usr/bin/python

"""
Class to read acq4 data blocks in simple manner. Does not requre
acq4 link; bypasses DataManager and PatchEPhys

Requires pyqtgraph to read the .ma files and the .index file
Requires Python 3
"""
import os
from pyqtgraph import metaarray
from pyqtgraph import configfile
import numpy as np
import matplotlib.pyplot as mpl

class Acq4Read():
    def __init__(self, protocolFilename=None, dataname=None):
        self.protocol = None
        if protocolFilename is not None:
            self.setProtocol(protocolFilename)
        if dataname is None:
            self.dataname = 'MultiClamp1.ma'
        else:
            self.dataname = dataname
        self.clampInfo = {}
        
    def setProtocol(self, pathtoprotocol):
        self.protocol = pathtoprotocol
    
    def setDataName(self, dataname):
        """
        Define the data metaarray name that will be read
        """
        self.dataname = dataname

    def subDirs(self, p):
        dirs = filter(os.path.isdir, [os.path.join(p, d) for d in os.listdir(p)])
        return dirs

    def getData(self, pos=1):  # non threaded
        dirs = self.subDirs(self.protocol)
        index = self._readIndex()
        self.clampInfo['dirs'] = dirs
        self.clampInfo['missingData'] = []
        self.traces = []
        self.cmd = []
        self.cmd_wave = []
        self.time_base = []
        self.values = []
        self.trace_StartTimes = np.zeros(0)
        self.sample_rate = []
        sequence_values = None
        for i, d in enumerate(dirs):
            fn = os.path.join(d, self.dataname)
            if not os.path.isfile(fn):
                continue
            tr = metaarray.MetaArray(file=fn)
            self.traces.append(tr.view(np.ndarray)[pos])
            self.time_base.append(tr.xvals('Time'))
            info = tr[0].infoCopy()
            sr = info[1]['DAQ']['primary']['rate']
            self.sample_rate.append(sr)
        self.traces = np.array(self.traces)
        self.time_base = np.array(self.time_base[0])
        self.repetitions = index['.']['sequenceParams'][('protocol', 'repetitions')][0] + 1
        

    def _readIndex(self, currdir=''):
        indexFile = os.path.join(self.protocol, currdir, '.index')
#        print self.protocol, currdir, indexFile
        if not os.path.isfile(indexFile):
            raise Exception("Directory '%s' is not managed!" % (self.dataname))
        self._index = configfile.readConfigFile(indexFile)
        return self._index

    def getScannerPositions(self):
        dataname = 'Laser-Blue-raw.ma'
        dirs = self.subDirs(self.protocol)
        self.scannerpositions = np.zeros((len(dirs), 2))
        self.targets = [[]]*len(dirs)
        self.spotsize = 0.
        rep = 0
        tar = 0
        supindex = self._readIndex(self.protocol)
        ntargets = len(supindex['.']['sequenceParams'][('Scanner', 'targets')])
        pars={}
        pars['sequence1'] = {}
        pars['sequence2'] = {}
        reps = supindex['.']['sequenceParams'][('protocol', 'repetitions')]
        pars['sequence1']['index'] = reps
        pars['sequence2']['index'] = ntargets
        self.sequenceparams = pars
        self.scannerinfo = {}
        for i, d in enumerate(dirs):
            index = self._readIndex(d)
            if 'Scanner' in index['.'].keys():
                self.scannerpositions[i] = index['.']['Scanner']['position']
                self.targets[i] = index['.'][('Scanner', 'targets')]
                self.spotsize = index['.']['Scanner']['spotSize']
                self.scannerinfo[(rep, tar)] = {'directory': d, 'rep': rep, 'pos': self.scannerpositions[i]}
            else:
                self.scannerpositions[i] = [0., 0.]
                self.targets[i] = None
                self.spotsize = None
                self.scannerinfo[(rep, tar)] = {'directory': d, 'rep': rep, 'pos': self.scannerpositions[i]}
            tar = tar + 1
            if tar > ntargets:
                tar = 0
                rep = rep + 1

    def plotClampData(self, all=True):
        mpl.figure()
        if all:
            for i in range(len(self.traces)):
                mpl.plot(self.time_base, self.traces[i])
        else:
            mpl.plot(self.time_base, np.array(self.traces).mean(axis=0))
        mpl.show()
        
if __name__ == '__main__':
    # test on a big file
    a = Acq4Read()
    a.setProtocol('/Volumes/Pegasus/ManisLab_Data3/Kasten, Michael/2017.10.11_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003')
#    a.setProtocol('/Volumes/Pegasus/ManisLab_Data3/Kasten, Michael/2017.11.20_000/slice_000/cell_000/CCIV_4nA_max_000')
    a.getScannerPositions()
    # print a.scannerpositions
    # print (a.spotsize)
#    mpl.plot(a.scannerpositions[:,0], a.scannerpositions[:,1], 'ro')
    a.getData()
    a.plotClampData(all=True)
    #print a.clampInfo
   # print a.traces[0]
    mpl.show()
            

