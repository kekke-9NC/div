"""Small lifecycle guards for Tkinter interpreter shutdown."""

import tkinter as tk
import threading


_original_variable_delete = getattr(
    tk.Variable, "_meteor_original_delete", tk.Variable.__del__
)
tk.Variable._meteor_original_delete = _original_variable_delete


def _safe_variable_delete(variable):
    """Silence only teardown errors after the Tcl interpreter has stopped.

    CPython 3.11's ``Variable.__del__`` asks Tcl whether the variable exists.
    During interpreter shutdown that call can return ``None`` instead of a Tcl
    boolean and ``getboolean`` raises from a destructor.  At that point there
    is no live Tcl resource left to release, so detaching the dead interpreter
    is the correct cleanup.
    """
    # Tk is strictly owned by the GUI thread.  A dialog can become collectible
    # while a video worker happens to trigger cyclic GC; calling even
    # ``info exists`` from that worker can crash macOS in Tcl with SIGBUS before
    # Python has an exception to catch.  Tcl will reclaim the variable with its
    # interpreter, so detach without making any Tcl call off the main thread.
    if threading.current_thread() is not threading.main_thread():
        variable._tk = None
        variable._tclCommands = None
        return
    try:
        _original_variable_delete(variable)
    except (tk.TclError, RuntimeError, TypeError):
        variable._tk = None
        variable._tclCommands = None


def install_tkinter_shutdown_guard():
    current = tk.Variable.__del__
    if getattr(current, "_meteor_shutdown_guard", False):
        return
    _safe_variable_delete._meteor_shutdown_guard = True
    tk.Variable.__del__ = _safe_variable_delete


install_tkinter_shutdown_guard()
