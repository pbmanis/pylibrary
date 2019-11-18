#!/usr/bin/env python

import os

def tell(command):
    cmds = "osascript -e 'tell application \"Igor Pro\"' "
    
    cmde = " -e 'end tell' END"
    cmd = cmds + "-e " + "'" + command + "'" + cmde
    print(cmd)
    os.system(cmd)
    
    

