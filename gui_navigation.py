from gui_common import *
import gui_layout


class NavigationMixin:
    def navigate_to_source_drop_area(self):
        """Navigate to Source Selection tab and highlight the drop area for a few seconds."""
        self.notebook.select(self.tab_source)

        gui_layout.reveal_widget(self.source_drop_label)
        style = ttk.Style()
        style.configure(
            "Highlight.TLabel",
            background=ui_theme.COLORS["accent"],
            foreground=ui_theme.COLORS["window"],
        )
        self.source_drop_label.configure(style="Highlight.TLabel")

        def flash_highlight(count=0):
            if count >= 6:
                self.source_drop_label.configure(style="DropZone.TLabel")
                return
            if count % 2 == 0:
                style.configure(
                    "Highlight.TLabel",
                    background=ui_theme.COLORS["accent"],
                    foreground=ui_theme.COLORS["window"],
                )
            else:
                style.configure(
                    "Highlight.TLabel",
                    background=ui_theme.COLORS["glass_selected"],
                    foreground=ui_theme.COLORS["text"],
                )
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
        self.settings_sections.select(0)

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
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_model_training"):
            self._scroll_analysis_to_widget(self.btn_model_training)
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
        if hasattr(self, "btn_radiant_analysis"):
            self._scroll_analysis_to_widget(self.btn_radiant_analysis)
            self._flash_button(self.btn_radiant_analysis)

    def navigate_to_blend_image_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_blend_image"):
            self._scroll_analysis_to_widget(self.btn_blend_image)
            self._flash_button(self.btn_blend_image)

    def navigate_to_blend_video_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_blend_video"):
            self._scroll_analysis_to_widget(self.btn_blend_video)
            self._flash_button(self.btn_blend_video)

    def navigate_to_timelapse_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_timelapse"):
            self._scroll_analysis_to_widget(self.btn_timelapse)
            self._flash_button(self.btn_timelapse)

    def navigate_to_long_exposure_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_long_exposure"):
            self._scroll_analysis_to_widget(self.btn_long_exposure)
            self._flash_button(self.btn_long_exposure)

    def navigate_to_distortion_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_distortion"):
            self._scroll_analysis_to_widget(self.btn_distortion)
            self._flash_button(self.btn_distortion)

    def navigate_to_angle_analysis_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_angle_analysis"):
            self._scroll_analysis_to_widget(self.btn_angle_analysis)
            self._flash_button(self.btn_angle_analysis)

    def navigate_to_video_concat_start_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_video_concat_start"):
            self._scroll_analysis_to_widget(self.btn_video_concat_start)
            self._flash_button(self.btn_video_concat_start)

    def _scroll_analysis_to_widget(self, widget, top_margin=16):
        gui_layout.reveal_widget(widget, top_margin)

    def navigate_to_chat_tab(self):
        self.notebook.select(self.tab_chat)

    def navigate_to_advanced_tab(self):
        self.notebook.select(self.tab_advanced_settings)
        if hasattr(self, "btn_reset_advanced"):
            self._flash_button(self.btn_reset_advanced)

    def navigate_to_log_tab(self):
        self._set_activity_visible(True)
        if hasattr(self, "status_panel") and hasattr(self.status_panel, "notebook"):
            try:
                self.status_panel.notebook.select(self.status_panel.log_frame)
            except Exception:
                pass

    def navigate_to_processing_status_tab(self):
        self._set_activity_visible(True)
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
        gui_layout.reveal_widget(widget, top_margin)

    def _flash_entry(self, entry):
        if entry is None:
            return
        gui_layout.reveal_widget(entry)
        try:
            if not entry.winfo_exists():
                return
            base_style = entry.cget("style") or "TEntry"
            style = ttk.Style()
            highlight_style = f"Highlight{entry.winfo_id()}.TEntry"
            style.configure(
                highlight_style,
                fieldbackground=ui_theme.COLORS["accent"],
                foreground=ui_theme.COLORS["window"],
            )

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
        gui_layout.reveal_widget(button)
        try:
            if not button.winfo_exists():
                return
            base_style = button.cget("style") or "TButton"
            style = ttk.Style()
            highlight_style = f"Highlight{button.winfo_id()}.TButton"
            style.configure(
                highlight_style,
                background=ui_theme.COLORS["accent"],
                foreground=ui_theme.COLORS["window"],
            )

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
            # YYYYMMDD_... and YYYYMMDDHHMMSS... are already date-prefixed.
            if re.match(r"^\d{8}(?:_|\d{6})", basename):
                return path
            date_prefix = datetime.now().strftime("%Y%m%d_")
            new_name = date_prefix + basename
            return os.path.join(dirpath, new_name)
        except Exception:
            return path

