#!/usr/bin/env python

import os

def tellIgor(command):
    cmd = """osascript<<END
    tell application "Igor"
    """
    
    cmde = """end tell
    END"""

    os.system(cmd+command+cmde)
    
    
def startIgor():
    cmd = """osascript<<END
    tell application "Igor Pro"
    activate
    end tell
    END"""
    os.system(cmd)
    