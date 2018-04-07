#!/usr/bin/env python
from __future__ import print_function
"""
Standalone file/directory selector dialog
Provides Qt5 based, system independent file selection

"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
 
class FileSelector(QWidget):
 
    def __init__(self, title='', dialogtype='dir', extensions=None, startingdir='.'):
        """
        File Selector
        
        Parameters
        ----------
        title : str (no default)
            Title for file selector window
        dialogtype : str (default: dir)
            Type of dialog: [file, files, save, dir]
        extensions : str (default: None)
            FIle extension string for the selector (not fully implemented yet)
        startingdir : str (default '.')
            Path to directory to start in
        
        Usage:
        FS = FileSelector(title, dialogtype)
        FS.fileName holds the filename
        Results are string (file, save, dir), list of strings (files)
        FS.fileName will be None if the dialog is cancelled
        """
        
        app = QApplication(sys.argv)
        super(FileSelector, self).__init__()
        self.title = title
        self.left = 110
        self.top = 110
        self.width = 400
        self.height = 300
        self.fileName = None
        self.startingdir = startingdir
        self.dialogs = {'file': self.openFileNameDialog,
                   'files': self.openFileNamesDialog,
                   'save': self.saveFileDialog,
                   'dir': self.openDirNameDialog,
               }
        if dialogtype not in self.dialogs.keys():
            raise ValueError('Dialog type %s is not knowns to us ' % dialogtype)
        self.dialogtype = dialogtype
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        a = self.dialogs[self.dialogtype]()
        self.show()
 
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, self.title, self.startingdir,
                "All Files (*);;Python Files (*.py)", options=options)
        self.savefilename(fileName)

    def openDirNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DirectoryOnly
        #options |= QFileDialog.ShowDirsOnly
        fileName = QFileDialog.getExistingDirectory(self, self.title,
                self.startingdir, options=options)
        self.savefilename(fileName)
 
    def openFileNamesDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, self.title, 
                self.startingdir,"All Files (*);;Python Files (*.py)", options=options)
        self.savefilename(files)
 
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, self.title, self.startingdir,
            "All Files (*);;Text Files (*.txt)", options=options)
        self.savefilename(fileName)

    def savefilename(self, fileName):
        if fileName == '' or len(fileName) == 0:
            fileName = None
        self.fileName = fileName
               
        
if __name__ == '__main__':
    ex = FileSelector(dialogtype='dir')
    print('exfile: ', ex.fileName)

    


    
