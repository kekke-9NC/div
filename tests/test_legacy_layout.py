"""Exercise real Tk tab selection and scroll routing without detection workers."""
import sys
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

import gui_layout


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    window.geometry('700x400')
    yield window
    window.destroy()


def test_reveal_opens_containing_tabs_and_scrolls_target_into_view(root):
    outer = ttk.Notebook(root)
    outer.pack(fill='both', expand=True)
    first = ttk.Frame(outer)
    second = ttk.Frame(outer)
    outer.add(first, text='Other workspace')
    outer.add(second, text='Settings')
    inner, pages = gui_layout.section_pages(second, [('output', 'Output'), ('camera', 'Camera')])
    for i in range(40):
        ttk.Label(pages['camera'].body, text=f'Setting {i}').pack()
    target = ttk.Button(pages['camera'].body, text='Create mask')
    target.pack()
    root.update()
    gui_layout.reveal_widget(target)
    root.update_idletasks()
    assert outer.select() == str(second)
    assert inner.select() == str(pages['camera'])
    page = pages['camera']
    assert page.canvas.winfo_rooty() <= target.winfo_rooty()
    assert target.winfo_rooty() + target.winfo_height() <= page.canvas.winfo_rooty() + page.canvas.winfo_height()


def test_wheel_scrolls_only_page_containing_event_target(root):
    notebook, pages = gui_layout.section_pages(root, [('one', 'One'), ('two', 'Two')])
    for page in pages.values():
        for i in range(60):
            ttk.Label(page.body, text=str(i)).pack()
    root.update()
    label = pages['one'].body.winfo_children()[0]
    result = gui_layout.scroll_under_pointer(SimpleNamespace(widget=label, delta=-120))
    assert result == 'break'
    assert pages['one'].canvas.yview()[0] > 0
    assert pages['two'].canvas.yview()[0] == 0
    notebook.select(pages['two'])
    root.update()
    before = pages['one'].canvas.yview()
    gui_layout.scroll_under_pointer(SimpleNamespace(widget=label, delta=-120))
    assert pages['one'].canvas.yview() == before


def test_wheel_preserves_inner_list_and_text_scrolling(root):
    page = gui_layout.ScrollPage(root)
    page.pack(fill='both', expand=True)
    canvas = tk.Canvas(page.body, height=80, yscrollcommand=lambda *args: None)
    canvas.pack()
    canvas.configure(scrollregion=(0, 0, 200, 2000))
    text = tk.Text(page.body, height=3)
    text.pack()
    root.update()
    gui_layout.scroll_under_pointer(SimpleNamespace(widget=canvas, delta=-120))
    assert canvas.yview()[0] > 0
    assert page.canvas.yview()[0] == 0
    assert gui_layout.scroll_under_pointer(SimpleNamespace(widget=text, delta=-120)) is None


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS trackpad deltas')
def test_small_trackpad_delta_is_not_discarded(root):
    page = gui_layout.ScrollPage(root)
    page.pack(fill='both', expand=True)
    for i in range(60):
        ttk.Label(page.body, text=str(i)).pack()
    root.update()
    gui_layout.scroll_under_pointer(SimpleNamespace(widget=page.body, delta=-1))
    assert page.canvas.yview()[0] > 0


def test_standalone_explanations_follow_page_width_after_resize(root):
    page = gui_layout.ScrollPage(root)
    page.pack(fill='both', expand=True)
    label = ttk.Label(page.body, text='A long explanation ' * 25, wraplength=1120)
    label.pack(anchor='w')
    root.update()
    original_height = label.winfo_height()
    root.geometry('360x400')
    root.update()
    assert label.winfo_reqwidth() <= page.body.winfo_width()
    assert label.winfo_height() > original_height
