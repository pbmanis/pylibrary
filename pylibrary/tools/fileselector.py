"""
Standalone file/directory selector dialog
Provides Qt5 based, system independent file selection

pbmanis 2019
"""

import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

class FileSelector(pg.QtWidgets.QMainWindow):

    def __init__(self, title='', dialogtype='dir', extensions=None, startingdir='.', useNative=True, standalone=False):
        """
        File Selector

        Parameters
        ----------
        title : str (no default)
            Title for file selector window
        dialogtype : str (default: dir)
            Type of dialog: [file, files, save, dir]
        extensions : str (default: None)
            File extension string for the selector (not fully implemented yet)
        startingdir : str (default '.')
            Path to directory to start in

        Usage:
        FS = FileSelector(title, dialogtype)
        FS.fileName holds the filename
        Results are string (file, save, dir), list of strings (files)
        FS.fileName will be None if the dialog is cancelled
        """
        # super(FileSelector, self).__init__()

        self.title = title
        self.left = 110
        self.top = 110
        self.width = 400
        self.height = 300
        self.fileName = None
        self.startingdir = str(startingdir) # remove path stuff... 
        self.useNative = useNative
        self.dialogs = {'file': self.openFileNameDialog,
                   'files': self.openFileNamesDialog,
                   'save': self.saveFileDialog,
                   'dir': self.openDirNameDialog,
               }
        if dialogtype not in self.dialogs.keys():
            raise ValueError('Dialog type %s is not knowns to us ' % dialogtype)
        self.dialogtype = dialogtype
        self.standalone = standalone
        self.initUI()

    def initUI(self):
        if self.standalone:
            self.app = QtGui.QApplication(sys.argv)
        self.win = QtGui.QDialog() # top level
        # self.win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.win.setWindowTitle(self.title)
        self.win.setGeometry(self.left, self.top, self.width, self.height)
        self.active_dialog = self.dialogs[self.dialogtype]()
        if self.active_dialog is not None:
            self.active_dialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.active_dialog.setOptions(QtGui.QFileDialog.DontUseNativeDialog)

    def openFileNameDialog(self):
        self.dialog = pg.QtWidgets.QFileDialog()
        windows_options = self.dialog.windowFlags()
        if not self.useNative:
            options |= QtGui.QFileDialog.DontUseNativeDialog
        fileName = self.dialog.getOpenFileName(self.win, self.title, self.startingdir,
                "All Files (*);;", options=QtGui.QFileDialog.Option.DontResolveSymlinks)
        self.savefilename(fileName)

    def openDirNameDialog(self):
        self.dialog = pg.QtWidgets.QFileDialog()
        windows_options = self.dialog.windowFlags()
        if not self.useNative:
            options |= QtGui.QFileDialog.DontUseNativeDialog
        dirName = self.dialog.getExistingDirectory(self.win, self.title,
                self.startingdir, QtGui.QFileDialog.Option.ShowDirsOnly
                                                | QtGui.QFileDialog.Option.DontResolveSymlinks)
        self.savefilename(dirName)

    def openFileNamesDialog(self):
        self.dialog = pg.QtWidgets.QFileDialog()
        windows_options = self.dialog.windowFlags()
        if not self.useNative:
            options |= QtGui.QFileDialog.DontUseNativeDialog
        fileNames = self.dialog.getOpenFileName(self.win, self.title, self.startingdir,
                self.startingdir,"All Files (*);;Python Files (*.py)",
                options=QtGui.QFileDialog.Option.DontResolveSymlinks)
        self.savefilename(fileNames)

    def saveFileDialog(self):
        self.dialog = pg.QtWidgets.QFileDialog()
        windows_options = QFileDialog.windowFlags()
        if not self.useNative:
            ptions |= QFileDialog.DontUseNativeDialog
        fileName, _ = self.dialog.getSaveFileName(self.win, self.title, self.startingdir,
            "All Files (*);;Text Files (*.txt)", options=QtGui.QFileDialog.Option.DontResolveSymlinks)
        self.savefilename(fileName)

    def savefilename(self, fileName):
        if fileName == '' or len(fileName) == 0:
            fileName = None
        self.fileName = fileName
        # self.app.flush()  # make sure no hanging events
        self.win.close()  # supposed to close the window?
        # self.app.quit()  # close out the app we init'd


def main():
    ex = FileSelector(dialogtype='files')
    
    print('exfile: ', ex.fileName)
    
    
if __name__ == '__main__':
    main()

