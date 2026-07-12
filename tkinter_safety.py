"""Small lifecycle guards for Tkinter interpreter shutdown."""

import tkinter as tk


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
