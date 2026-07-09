import tkinter
try:
    print(f"Tcl Version: {tkinter.Tcl().eval('info patchlevel')}")
    print(f"Tk Version: {tkinter.Tk().eval('info patchlevel')}")
except Exception as e:
    print(f"Error checking Tcl/Tk: {e}")

import sys
print("Sys Path:", sys.path)
import _tkinter
print("_tkinter file:", _tkinter.__file__)
