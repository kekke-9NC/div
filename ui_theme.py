"""Visual system and reusable controls for the desktop UI.

Tk does not expose SwiftUI's Liquid Glass API.  This module keeps the same
layering model, though: content stays quiet and opaque while navigation and
actions sit on elevated, lightly tinted surfaces with adaptive-looking edges.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Dict, Optional


COLORS: Dict[str, str] = {
    "window": "#070A11",
    "content": "#0B0F18",
    "content_raised": "#0F1521",
    "glass": "#141C2A",
    "glass_strong": "#192436",
    "glass_hover": "#202D42",
    "glass_selected": "#263955",
    "field": "#111A28",
    "field_focus": "#17243A",
    "border": "#2B3A51",
    "border_bright": "#415875",
    "text": "#F4F7FC",
    "text_secondary": "#A8B3C5",
    "text_tertiary": "#718096",
    "accent": "#70A7FF",
    "accent_hover": "#8BB8FF",
    "accent_pressed": "#568FEF",
    "cyan": "#66D9EF",
    "success": "#5DE2A5",
    "warning": "#F5C76B",
    "danger": "#FF7185",
    "danger_hover": "#FF8A9A",
    "selection": "#315581",
    "shadow": "#05070C",
}


PAGE_META = (
    {
        "key": "usage",
        "glyph": "⌂",
        "label": "はじめる",
        "eyebrow": "OVERVIEW",
        "title": "観測ワークスペース",
        "subtitle": "流星検出までの流れと、よく使う操作を確認できます。",
    },
    {
        "key": "source",
        "glyph": "⌁",
        "label": "入力ソース",
        "eyebrow": "CAPTURE",
        "title": "入力ソース",
        "subtitle": "動画・フォルダ・RTSPカメラをひとつの場所で管理します。",
    },
    {
        "key": "settings",
        "glyph": "⇩",
        "label": "保存と出力",
        "eyebrow": "OUTPUT",
        "title": "保存と出力",
        "subtitle": "生成するデータ、保存先、解析モデルを設定します。",
    },
    {
        "key": "analysis",
        "glyph": "◇",
        "label": "解析ツール",
        "eyebrow": "ANALYZE",
        "title": "解析ツール",
        "subtitle": "観測データの可視化、合成、補正、動画作成を行います。",
    },
    {
        "key": "chat",
        "glyph": "✦",
        "label": "AIアシスタント",
        "eyebrow": "ASSIST",
        "title": "AIアシスタント",
        "subtitle": "操作方法や設定について、アプリ内で質問できます。",
    },
    {
        "key": "advanced",
        "glyph": "⚙",
        "label": "詳細設定",
        "eyebrow": "SYSTEM",
        "title": "詳細設定",
        "subtitle": "検出モデルと高度な処理パラメータを調整します。",
    },
)


def install_named_fonts(root: tk.Misc) -> None:
    """Use the system UI family while preserving Tk's named-font behavior."""
    preferred = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"
    heading = "SF Pro Display" if sys.platform == "darwin" else preferred
    specs = {
        "TkDefaultFont": (preferred, 12, "normal"),
        "TkTextFont": (preferred, 12, "normal"),
        "TkMenuFont": (preferred, 12, "normal"),
        "TkHeadingFont": (heading, 12, "bold"),
        "TkCaptionFont": (preferred, 11, "normal"),
        "TkSmallCaptionFont": (preferred, 10, "normal"),
        "TkFixedFont": ("SF Mono" if sys.platform == "darwin" else "Menlo", 10, "normal"),
    }
    for name, (family, size, weight) in specs.items():
        try:
            font = tkfont.nametofont(name, root=root)
            font.configure(family=family, size=size, weight=weight)
        except tk.TclError:
            pass


def configure_macos_window(root: tk.Tk) -> None:
    """Apply safe native-window hints when running on macOS."""
    if sys.platform != "darwin":
        return
    try:
        root.tk.call(
            "tk::unsupported::MacWindowStyle",
            "style",
            root._w,
            "document",
            ("standardDocument", "closeBox", "collapseBox", "resizable", "liveResize"),
        )
    except tk.TclError:
        pass


class SidebarButton(tk.Frame):
    """A full-row navigation control with hover and selected glass states."""

    def __init__(
        self,
        parent: tk.Misc,
        glyph: str,
        text: str,
        command: Callable[[], None],
    ) -> None:
        super().__init__(
            parent,
            bg=COLORS["glass"],
            height=44,
            cursor="hand2",
            highlightthickness=0,
        )
        self.pack_propagate(False)
        self._command = command
        self._selected = False

        self._indicator = tk.Frame(self, bg=COLORS["glass"], width=3)
        self._indicator.pack(side=tk.LEFT, fill=tk.Y)
        self._glyph = tk.Label(
            self,
            text=glyph,
            width=3,
            anchor=tk.CENTER,
            bg=COLORS["glass"],
            fg=COLORS["text_secondary"],
            font=("SF Pro Text" if sys.platform == "darwin" else "Segoe UI", 15),
        )
        self._glyph.pack(side=tk.LEFT, padx=(7, 1))
        self._label = tk.Label(
            self,
            text=text,
            anchor=tk.W,
            bg=COLORS["glass"],
            fg=COLORS["text_secondary"],
            font=("SF Pro Text" if sys.platform == "darwin" else "Segoe UI", 12, "normal"),
        )
        self._label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 12))

        for widget in (self, self._indicator, self._glyph, self._label):
            widget.bind("<Button-1>", self._invoke, add="+")
            widget.bind("<Enter>", self._on_enter, add="+")
            widget.bind("<Leave>", self._on_leave, add="+")

    def _contains_pointer(self) -> bool:
        try:
            pointer = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            while pointer is not None:
                if pointer == self:
                    return True
                pointer = pointer.master
        except (AttributeError, tk.TclError):
            pass
        return False

    def _invoke(self, _event=None) -> None:
        self._command()

    def _on_enter(self, _event=None) -> None:
        if not self._selected:
            self._paint(COLORS["glass_hover"], COLORS["text"])

    def _on_leave(self, _event=None) -> None:
        if not self._selected and not self._contains_pointer():
            self._paint(COLORS["glass"], COLORS["text_secondary"])

    def _paint(self, background: str, foreground: str) -> None:
        self.configure(bg=background)
        self._glyph.configure(bg=background, fg=foreground)
        self._label.configure(bg=background, fg=foreground)

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        family = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"
        if self._selected:
            self._paint(COLORS["glass_selected"], COLORS["text"])
            self._indicator.configure(bg=COLORS["accent"])
            self._label.configure(font=(family, 12, "bold"))
        else:
            self._paint(COLORS["glass"], COLORS["text_secondary"])
            self._indicator.configure(bg=COLORS["glass"])
            self._label.configure(font=(family, 12, "normal"))


def make_badge(
    parent: tk.Misc,
    text: str,
    *,
    foreground: Optional[str] = None,
    background: Optional[str] = None,
) -> tk.Label:
    return tk.Label(
        parent,
        text=text,
        fg=foreground or COLORS["success"],
        bg=background or COLORS["glass_strong"],
        padx=10,
        pady=4,
        font=("SF Pro Text" if sys.platform == "darwin" else "Segoe UI", 10, "bold"),
    )
