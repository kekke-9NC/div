from gui_common import *


class NavigationMixin:
    def navigate_to_source_drop_area(self):
        """Navigate to Source Selection tab and highlight the drop area for a few seconds."""
        self.notebook.select(self.tab_source)

        style = ttk.Style()
        style.configure("Highlight.TLabel", background="#FFD700", foreground="#000000")
        self.source_drop_label.configure(style="Highlight.TLabel")

        def flash_highlight(count=0):
            if count >= 6:
                self.source_drop_label.configure(style="TLabel")
                return
            if count % 2 == 0:
                style.configure("Highlight.TLabel", background="#FFD700", foreground="#000000")
            else:
                style.configure("Highlight.TLabel", background="#4A6A9B", foreground="#EAEAEA")
            self.after(400, lambda: flash_highlight(count + 1))

        flash_highlight()

    def navigate_to_start_button(self):
        self._flash_button(self.start_button)

    def navigate_to_rtsp_entry(self):
        """Navigate to Source tab and highlight RTSP URL entry."""
        self.notebook.select(self.tab_source)
        self._flash_entry(self.rtsp_url_entry)
        try:
            self.rtsp_url_entry.focus_set()
        except Exception:
            pass

    def navigate_to_rtsp_add_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_add_rtsp"):
            self._flash_button(self.btn_add_rtsp)

    def navigate_to_rtsp_plate_solve_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_rtsp_plate_solve"):
            self._flash_button(self.btn_rtsp_plate_solve)

    def navigate_to_rtsp_mask_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_rtsp_mask"):
            self._flash_button(self.btn_rtsp_mask)

    def navigate_to_periodic_scan_section(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_select_periodic_dir"):
            self._flash_button(self.btn_select_periodic_dir)

    def navigate_to_periodic_dir_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_select_periodic_dir"):
            self._flash_button(self.btn_select_periodic_dir)

    def navigate_to_periodic_auto_time_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_periodic_auto_time"):
            self._flash_button(self.btn_periodic_auto_time)

    def navigate_to_settings_tab(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_detection_mask"):
            self._flash_button(self.btn_detection_mask)

    def navigate_to_mask_download_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_download_mask"):
            self._flash_button(self.btn_download_mask)

    def navigate_to_plate_solve_select_video_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_select_plate_solve_video"):
            self._scroll_settings_to_widget(self.btn_select_plate_solve_video, top_margin=20)
            self.after(160, lambda: self._flash_button(self.btn_select_plate_solve_video))

    def navigate_to_plate_solve_run_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_run_plate_solve"):
            self._scroll_settings_to_widget(self.btn_run_plate_solve, top_margin=20)
            self.after(160, lambda: self._flash_button(self.btn_run_plate_solve))

    def navigate_to_api_key_entry(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "api_key_entry"):
            self._scroll_settings_to_widget(self.api_key_entry, top_margin=20)
            self._flash_entry(self.api_key_entry)
            try:
                self.api_key_entry.focus_set()
            except Exception:
                pass

    def navigate_to_summary_settings_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_summary_settings"):
            self._flash_button(self.btn_summary_settings)

    def navigate_to_analysis_tab(self):
        self.notebook.select(self.tab_analysis)

    def navigate_to_model_training_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_model_training"):
            self._flash_button(self.btn_model_training)

    def navigate_to_model_selector(self):
        self.notebook.select(self.tab_advanced_settings)
        if hasattr(self, "btn_model_refresh"):
            self._flash_button(self.btn_model_refresh)

    def navigate_to_plate_solve_fov_settings(self):
        self.notebook.select(self.tab_advanced_settings)
        if hasattr(self, "btn_apply_plate_solve_fov"):
            self._flash_button(self.btn_apply_plate_solve_fov)

    def navigate_to_analysis_start_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_analysis_start"):
            self._flash_button(self.btn_analysis_start)

    def navigate_to_blend_image_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_blend_image"):
            self._flash_button(self.btn_blend_image)

    def navigate_to_blend_video_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_blend_video"):
            self._flash_button(self.btn_blend_video)

    def navigate_to_timelapse_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_timelapse"):
            self._flash_button(self.btn_timelapse)

    def navigate_to_long_exposure_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_long_exposure"):
            self._flash_button(self.btn_long_exposure)

    def navigate_to_distortion_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_distortion"):
            self._flash_button(self.btn_distortion)

    def navigate_to_angle_analysis_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_angle_analysis"):
            self._flash_button(self.btn_angle_analysis)

    def navigate_to_video_concat_start_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_video_concat_start"):
            self._flash_button(self.btn_video_concat_start)

    def navigate_to_chat_tab(self):
        self.notebook.select(self.tab_chat)

    def navigate_to_advanced_tab(self):
        self.notebook.select(self.tab_advanced_settings)
        if hasattr(self, "btn_reset_advanced"):
            self._flash_button(self.btn_reset_advanced)

    def navigate_to_log_tab(self):
        if hasattr(self, "status_panel") and hasattr(self.status_panel, "notebook"):
            try:
                self.status_panel.notebook.select(self.status_panel.log_frame)
            except Exception:
                pass

    def navigate_to_processing_status_tab(self):
        if hasattr(self, "status_panel") and hasattr(self.status_panel, "notebook"):
            try:
                self.status_panel.notebook.select(self.status_panel.status_frame)
            except Exception:
                pass

    def navigate_to_detection_mask_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_detection_mask"):
            self._scroll_settings_to_widget(self.btn_detection_mask, top_margin=20)
            self.after(160, lambda: self._flash_button(self.btn_detection_mask))

    def navigate_to_ps_mask_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_ps_mask"):
            self._scroll_settings_to_widget(self.btn_ps_mask, top_margin=20)
            self.after(160, lambda: self._flash_button(self.btn_ps_mask))

    def _scroll_settings_to_widget(self, widget, top_margin=16):
        """Scroll settings tab canvas so target widget becomes visible near top."""
        def do_scroll():
            try:
                if widget is None or not widget.winfo_exists():
                    return
                if not hasattr(self, "settings_canvas") or not hasattr(self, "settings_scrollable_frame"):
                    return
                canvas = self.settings_canvas
                scrollable_frame = self.settings_scrollable_frame
                if not canvas.winfo_exists() or not scrollable_frame.winfo_exists():
                    return

                self.update_idletasks()
                bbox = canvas.bbox("all")
                if not bbox:
                    return

                total_h = max(1, bbox[3] - bbox[1])
                view_h = max(1, canvas.winfo_height())
                max_scroll = max(1, total_h - view_h)

                # y position inside scrollable frame (independent from current scroll)
                y_in_frame = widget.winfo_rooty() - scrollable_frame.winfo_rooty()
                target_y = max(0, y_in_frame - top_margin)
                frac = min(1.0, max(0.0, target_y / max_scroll))
                canvas.yview_moveto(frac)
            except Exception:
                pass

        # Run twice to stabilize position after tab switch/layout refresh.
        self.after(20, do_scroll)
        self.after(140, do_scroll)

    def _flash_entry(self, entry):
        if entry is None:
            return
        try:
            if not entry.winfo_exists():
                return
            base_style = entry.cget("style") or "TEntry"
            style = ttk.Style()
            highlight_style = f"Highlight{entry.winfo_id()}.TEntry"
            style.configure(highlight_style, fieldbackground="#FFD700", foreground="#000000")

            def flash(count=0):
                if not entry.winfo_exists():
                    return
                if count >= 6:
                    entry.configure(style=base_style)
                    return
                entry.configure(style=highlight_style if count % 2 == 0 else base_style)
                self.after(400, lambda: flash(count + 1))

            flash()
        except Exception:
            pass

    def _flash_button(self, button):
        if button is None:
            return
        try:
            if not button.winfo_exists():
                return
            base_style = button.cget("style") or "TButton"
            style = ttk.Style()
            highlight_style = f"Highlight{button.winfo_id()}.TButton"
            style.configure(highlight_style, background="#FFD700", foreground="#000000")

            def flash(count=0):
                if not button.winfo_exists():
                    return
                if count >= 6:
                    button.configure(style=base_style)
                    return
                button.configure(style=highlight_style if count % 2 == 0 else base_style)
                self.after(400, lambda: flash(count + 1))

            flash()
        except Exception:
            pass

    def navigate_to_analysis_actions(self):
        """Navigate to Analysis tab and highlight the blend/timelapse buttons."""
        self.notebook.select(self.tab_analysis)
        self.navigate_to_blend_image_button()
        self.navigate_to_blend_video_button()
        self.navigate_to_timelapse_button()

    def _ensure_date_prefix(self, path: str) -> str:
        """Ensure the filename starts with YYYYMMDD_. If not, prepend today's date.

        Returns possibly-updated full path.
        """
        try:
            if not path:
                return path
            dirpath = os.path.dirname(path)
            basename = os.path.basename(path)
            # If already starts with YYYYMMDD_ then leave as-is
            if len(basename) >= 9 and basename[:9].isdigit() and basename[8] == '_':
                return path
            date_prefix = datetime.now().strftime("%Y%m%d_")
            new_name = date_prefix + basename
            return os.path.join(dirpath, new_name)
        except Exception:
            return path

