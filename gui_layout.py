"""Shared, scoped scrolling and navigation for the legacy workspaces."""
import sys
import tkinter as tk
from tkinter import ttk

import ui_theme


class ScrollPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0,
                                bg=ui_theme.COLORS['content_raised'])
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.body = ttk.Frame(self.canvas, padding=(4, 6, 8, 12))
        window = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', self._update_body_layout)
        self.bind('<Map>', self._update_body_layout)
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(window, width=e.width))
        root = self.winfo_toplevel()
        if not getattr(root, '_workspace_wheel_installed', False):
            root._workspace_wheel_installed = True
            for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                root.bind_all(sequence, scroll_under_pointer, add='+')

    def _update_body_layout(self, _event=None):
        # Standalone explanations follow the available page width. Inline field
        # labels keep their intrinsic width so forms retain their alignment.
        pending = list(self.body.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            if isinstance(widget, (tk.Label, ttk.Label)) and widget.winfo_manager() == 'pack':
                if widget.pack_info().get('side') == 'top':
                    available = widget.master.winfo_width() - 32
                    if available > 48 and str(widget.cget('wraplength')) != str(available):
                        widget.configure(wraplength=available)
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))


def scroll_under_pointer(event):
    widget = event.widget
    # Text/list controls already implement their own wheel behavior.
    if isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview, ttk.Combobox, ttk.Spinbox)):
        return
    canvas = None
    while widget is not None:
        if isinstance(widget, tk.Canvas) and widget.cget('yscrollcommand'):
            canvas = widget
            break
        widget = getattr(widget, 'master', None)
    if canvas is None or not canvas.winfo_viewable():
        return
    delta = getattr(event, 'delta', 0)
    amount = (-int(delta) if sys.platform == 'darwin' else -int(delta / 120)) if delta else (
        -1 if getattr(event, 'num', None) == 4 else 1)
    if amount:
        canvas.yview_scroll(amount, 'units')
    return 'break'


def section_pages(parent, sections):
    notebook = ttk.Notebook(parent)
    notebook.pack(fill=tk.BOTH, expand=True)
    pages = {}
    for key, label in sections:
        page = ScrollPage(notebook)
        notebook.add(page, text=label)
        pages[key] = page
    return notebook, pages


def reveal_widget(widget, top_margin=16):
    """Select all containing tabs, then scroll the nearest page to the control."""
    ancestors = []
    current = widget
    while current is not None:
        ancestors.append(current)
        current = getattr(current, 'master', None)
    for child in reversed(ancestors):
        if isinstance(getattr(child, 'master', None), ttk.Notebook):
            child.master.select(child)
    widget.update_idletasks()
    for page in ancestors:
        if isinstance(page, ScrollPage):
            canvas = page.canvas
            bbox = canvas.bbox('all')
            if bbox:
                height = max(1, bbox[3] - bbox[1])
                y = widget.winfo_rooty() - page.body.winfo_rooty()
                canvas.yview_moveto(max(0, y - top_margin) / height)
            break
