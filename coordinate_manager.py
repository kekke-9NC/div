"""
coordinate_manager.py

Manages custom coordinate points for meteor analysis.
Provides UI for adding, deleting, and persisting coordinate points.
"""
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
from typing import List, Tuple, Optional, Callable


class CoordinatePoint:
    """Represents a single coordinate point."""
    def __init__(self, name: str, ra: float, dec: float):
        self.name = name
        self.ra = ra
        self.dec = dec
    
    def to_dict(self):
        return {'name': self.name, 'ra': self.ra, 'dec': self.dec}
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['name'], data['ra'], data['dec'])
    
    def __repr__(self):
        return f"{self.name} (RA={self.ra:.3f}°, Dec={self.dec:.3f}°)"


class CoordinateManager:
    """Manages a collection of coordinate points with persistence."""
    
    def __init__(self, storage_file: str = "custom_coordinates.json"):
        self.storage_file = storage_file
        self.points: List[CoordinatePoint] = []
        self.on_change_callback: Optional[Callable] = None
        self.load()
    
    def add_point(self, name: str, ra: float, dec: float) -> CoordinatePoint:
        """Add a new coordinate point."""
        point = CoordinatePoint(name, ra, dec)
        self.points.append(point)
        self.save()
        if self.on_change_callback:
            self.on_change_callback()
        return point
    
    def remove_point(self, index: int) -> bool:
        """Remove a coordinate point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.save()
            if self.on_change_callback:
                self.on_change_callback()
            return True
        return False
    
    def clear_all(self) -> None:
        """Remove all coordinate points."""
        self.points.clear()
        self.save()
        if self.on_change_callback:
            self.on_change_callback()
    
    def get_points(self) -> List[Tuple[str, float, float]]:
        """Get all points as tuples (name, ra, dec)."""
        return [(p.name, p.ra, p.dec) for p in self.points]
    
    def save(self) -> None:
        """Save points to JSON file."""
        try:
            data = [p.to_dict() for p in self.points]
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save coordinates: {e}")
    
    def load(self) -> None:
        """Load points from JSON file."""
        if not os.path.exists(self.storage_file):
            return
        
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.points = [CoordinatePoint.from_dict(item) for item in data]
        except Exception as e:
            print(f"Failed to load coordinates: {e}")
            self.points = []
    
    def set_change_callback(self, callback: Callable) -> None:
        """Set a callback function to be called when points change."""
        self.on_change_callback = callback


class CoordinateDialog:
    """Dialog for adding a new coordinate point."""
    
    def __init__(self, parent, on_add_callback: Callable[[str, float, float], None]):
        self.parent = parent
        self.on_add_callback = on_add_callback
        self.dialog = None
    
    def show(self):
        """Show the dialog for adding a coordinate point."""
        self.dialog = Toplevel(self.parent)
        self.dialog.title("座標点を追加")
        self.dialog.geometry("350x180")
        self.dialog.grab_set()
        self.dialog.transient(self.parent)
        
        # Apply dark theme colors to match the app
        self.dialog.configure(background="#2E3F5B")

        ttk.Label(self.dialog, text="名前:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        name_entry = ttk.Entry(self.dialog, width=25)
        name_entry.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(self.dialog, text="RA (度):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ra_entry = ttk.Entry(self.dialog, width=25)
        ra_entry.grid(row=1, column=1, padx=10, pady=10)

        ttk.Label(self.dialog, text="Dec (度):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        dec_entry = ttk.Entry(self.dialog, width=25)
        dec_entry.grid(row=2, column=1, padx=10, pady=10)

        def on_add():
            name = name_entry.get().strip()
            try:
                ra = float(ra_entry.get())
                dec = float(dec_entry.get())
            except ValueError:
                messagebox.showerror("入力エラー", "RA と Dec は数値で入力してください。")
                return

            if not name:
                messagebox.showerror("入力エラー", "名前を入力してください。")
                return

            self.on_add_callback(name, ra, dec)
            messagebox.showinfo("成功", f"座標点 '{name}' (RA={ra}°, Dec={dec}°) を追加しました。")
            self.dialog.destroy()

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=15)
        ttk.Button(btn_frame, text="追加", command=on_add).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)


class CoordinateListDialog:
    """Dialog for viewing and managing coordinate points."""
    
    def __init__(self, parent, manager: CoordinateManager):
        self.parent = parent
        self.manager = manager
        self.dialog = None
        self.listbox = None
    
    def show(self):
        """Show the dialog for managing coordinate points."""
        self.dialog = Toplevel(self.parent)
        self.dialog.title("座標点の管理")
        self.dialog.geometry("500x400")
        self.dialog.grab_set()
        self.dialog.transient(self.parent)
        
        # Apply dark theme
        self.dialog.configure(background="#2E3F5B")
        
        # Label
        ttk.Label(self.dialog, text="登録済み座標点:").pack(pady=5, padx=10, anchor='w')
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(self.dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.listbox = tk.Listbox(
            list_frame, 
            selectmode=tk.EXTENDED, 
            bg="#3A4D6B", 
            fg="#EAEAEA", 
            relief=tk.FLAT, 
            highlightthickness=0
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        # Populate listbox
        self.refresh_list()
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="選択項目を削除", command=self.remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="すべて削除", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="閉じる", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def refresh_list(self):
        """Refresh the listbox with current points."""
        if not self.listbox:
            return
        
        self.listbox.delete(0, tk.END)
        for point in self.manager.points:
            self.listbox.insert(tk.END, str(point))
    
    def remove_selected(self):
        """Remove selected points from the list."""
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("情報", "削除する項目を選択してください。")
            return
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.manager.remove_point(index)
        
        self.refresh_list()
    
    def clear_all(self):
        """Clear all coordinate points."""
        if not self.manager.points:
            messagebox.showwarning("情報", "削除する座標点がありません。")
            return
        
        if messagebox.askyesno("確認", "すべての座標点を削除しますか？"):
            self.manager.clear_all()
            self.refresh_list()
