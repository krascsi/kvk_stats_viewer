# src/gui.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import json

# --- Matplotlib Imports ---
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.dates import DateFormatter
from datetime import datetime # Needed for date parsing

# Import necessary functions from other modules
try:
    from . import data_handler
    from . import utils
except ImportError:
    import data_handler
    import utils

logger = logging.getLogger(__name__)

# --- Matplotlib Style Configuration ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    "figure.facecolor": "#2b2b2b", "axes.facecolor": "#343638",
    "axes.edgecolor": "#DCE4EE", "axes.labelcolor": "#DCE4EE",
    "text.color": "#DCE4EE", "xtick.color": "#DCE4EE",
    "ytick.color": "#DCE4EE", "grid.color": "#565B5E",
    "figure.autolayout": True, "figure.figsize": (6, 4)
})


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("KovaaK's Stats Viewer")
        self.geometry("1150x800")
        self.minsize(850, 680)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Data Variables ---
        self.stats_folder_path = None
        self.stats_df = pd.DataFrame()
        self.filtered_stats_df = pd.DataFrame()
        self.current_scenario = None
        self.current_time_series = pd.DataFrame()
        self.all_scenarios_after_date_filter = []
        # Playlist Data Structures
        self.playlist_map = {} # Stores {playlist_name: [ordered_list_of_scenarios]} <-- Now a list!
        self.playlist_scenarios = set() # Still a set for quick "is playlist active?" check
        self.active_playlist_names = []

        # --- Filter Variables ---
        self.start_date_filter = None
        self.start_date_var = tk.StringVar(value="")

        # --- Plotting Variables ---
        self.fig = None; self.ax_score = None; self.ax_acc = None
        self.canvas = None; self.toolbar = None

        # --- Plot Options Variables ---
        self.ma_window_var = tk.StringVar(value="10")
        self.show_raw_score_var = tk.BooleanVar(value=True)
        self.show_raw_acc_var = tk.BooleanVar(value=True)
        self.show_ma_score_var = tk.BooleanVar(value=True)
        self.show_ma_acc_var = tk.BooleanVar(value=True)
        self.show_pb_var = tk.BooleanVar(value=False)

        # --- Debounce Timers ---
        self._redraw_job = None; self._search_job = None; self._date_filter_job = None

        # Add listeners
        trace_callback = self._redraw_plot_options_changed
        self.show_raw_score_var.trace_add("write", trace_callback)
        self.show_raw_acc_var.trace_add("write", trace_callback)
        self.show_ma_score_var.trace_add("write", trace_callback)
        self.show_ma_acc_var.trace_add("write", trace_callback)
        self.show_pb_var.trace_add("write", trace_callback)
        self.ma_window_var.trace_add("write", trace_callback)
        self.start_date_var.trace_add("write", self._on_date_filter_change)

        # --- Configure Main Grid Layout ---
        self.grid_rowconfigure(0, weight=0); self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Create Widgets ---
        self._create_widgets()

        # --- Initial State ---
        self.path_label.configure(text="Click 'Browse' to select a stats folder.")
        self._clear_plot("No folder selected")


    def _create_widgets(self):
        logger.debug("Creating widgets...")
        # --- Top Frame ---
        self.top_frame = ctk.CTkFrame(self, corner_radius=5)
        self.top_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.top_frame.grid_columnconfigure(1, weight=1); self.top_frame.grid_columnconfigure(2, weight=0)
        self.browse_button = ctk.CTkButton(self.top_frame, text="Browse Stats Folder", command=self._browse_folder)
        self.browse_button.grid(row=0, column=0, padx=10, pady=10)
        self.path_label = ctk.CTkLabel(self.top_frame, text="No folder selected.", anchor="w", justify="left")
        self.path_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.load_playlist_button = ctk.CTkButton(self.top_frame, text="Load Playlist(s)", command=self._load_playlists)
        self.load_playlist_button.grid(row=0, column=2, padx=10, pady=10)

        # --- Main Content Frame ---
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_content_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=0, minsize=250)
        self.main_content_frame.grid_columnconfigure(1, weight=1)

        # --- Left Frame (Scenario List & Search) ---
        self.left_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.left_frame.grid(row=0, column=0, padx=(10, 5), pady=(5, 10), sticky="nsew")
        self.left_frame.grid_rowconfigure(0, weight=0); self.left_frame.grid_rowconfigure(1, weight=0)
        self.left_frame.grid_rowconfigure(2, weight=0); self.left_frame.grid_rowconfigure(3, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.scenario_label = ctk.CTkLabel(self.left_frame, text="Scenarios", font=ctk.CTkFont(weight="bold"))
        self.scenario_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(self.left_frame, textvariable=self.search_var, placeholder_text="Search scenarios...")
        self.search_entry.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._on_search_key_release)
        self.playlist_filter_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.playlist_filter_frame.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.playlist_filter_frame.grid_columnconfigure(0, weight=1)
        self.playlist_filter_label = ctk.CTkLabel(self.playlist_filter_frame, text="Filter: None", anchor="w", text_color="gray", font=ctk.CTkFont(size=11))
        self.playlist_filter_label.grid(row=0, column=0, sticky="ew")
        self.clear_filter_button = ctk.CTkButton(self.playlist_filter_frame, text="Clear", width=50, command=self._clear_playlist_filter, state="disabled")
        self.clear_filter_button.grid(row=0, column=1, padx=(5,0))
        self.scenario_scroll_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="", fg_color="transparent")
        self.scenario_scroll_frame.grid(row=3, column=0, padx=5, pady=(0, 5), sticky="nsew")
        self.scenario_buttons = {} # Holds {scenario_name: button_widget}
        self.playlist_header_labels = {} # Holds {playlist_name: label_widget}
        self.no_matches_label = ctk.CTkLabel(self.scenario_scroll_frame, text=" No matching scenarios", anchor="w")
        self.no_matches_label.pack_forget()

        # --- Right Frame (Stats Display & Plot Area) ---
        self.right_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.right_frame.grid(row=0, column=1, padx=(5, 10), pady=(5, 10), sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=0); self.right_frame.grid_rowconfigure(1, weight=0)
        self.right_frame.grid_rowconfigure(2, weight=0); self.right_frame.grid_rowconfigure(3, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        # Stats Display Frame
        self.stats_display_frame = ctk.CTkFrame(self.right_frame)
        self.stats_display_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        self.stats_display_frame.grid_columnconfigure(0, weight=0); self.stats_display_frame.grid_columnconfigure(1, weight=1)
        self.stats_display_frame.grid_columnconfigure(2, weight=0); self.stats_display_frame.grid_columnconfigure(3, weight=1)
        self.stats_label = ctk.CTkLabel(self.stats_display_frame, text="Statistics", font=ctk.CTkFont(weight="bold"))
        self.stats_label.grid(row=0, column=0, columnspan=4, padx=10, pady=(5, 10), sticky="w")
        row_num = 1
        self.pb_label = ctk.CTkLabel(self.stats_display_frame, text="PB Score:", anchor="w"); self.pb_label.grid(row=row_num, column=0, padx=(10,2), pady=2, sticky="w")
        self.pb_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w"); self.pb_value.grid(row=row_num, column=1, padx=(0,10), pady=2, sticky="ew")
        self.runs_label = ctk.CTkLabel(self.stats_display_frame, text="Runs:", anchor="w"); self.runs_label.grid(row=row_num, column=2, padx=(10,2), pady=2, sticky="w")
        self.runs_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w"); self.runs_value.grid(row=row_num, column=3, padx=(0,10), pady=2, sticky="ew")
        row_num += 1
        self.avg_score_label = ctk.CTkLabel(self.stats_display_frame, text="Avg Score:", anchor="w"); self.avg_score_label.grid(row=row_num, column=0, padx=(10,2), pady=2, sticky="w")
        self.avg_score_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w"); self.avg_score_value.grid(row=row_num, column=1, padx=(0,10), pady=2, sticky="ew")
        self.last_played_label = ctk.CTkLabel(self.stats_display_frame, text="Last Played:", anchor="w"); self.last_played_label.grid(row=row_num, column=2, padx=(10,2), pady=2, sticky="w")
        self.last_played_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w"); self.last_played_value.grid(row=row_num, column=3, padx=(0,10), pady=2, sticky="ew")
        row_num += 1
        self.avg_acc_label = ctk.CTkLabel(self.stats_display_frame, text="Avg Accuracy:", anchor="w"); self.avg_acc_label.grid(row=row_num, column=0, padx=(10,2), pady=2, sticky="w")
        self.avg_acc_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w"); self.avg_acc_value.grid(row=row_num, column=1, padx=(0,10), pady=2, sticky="ew")
        self.first_played_label = ctk.CTkLabel(self.stats_display_frame, text="First Played:", anchor="w"); self.first_played_label.grid(row=row_num, column=2, padx=(10,2), pady=2, sticky="w")
        self.first_played_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w"); self.first_played_value.grid(row=row_num, column=3, padx=(0,10), pady=2, sticky="ew")
        row_num += 1
        # Plot Options Frame
        self.plot_options_frame = ctk.CTkFrame(self.right_frame)
        self.plot_options_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.plot_options_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=0)
        self.ma_label = ctk.CTkLabel(self.plot_options_frame, text="MA Window:")
        self.ma_label.grid(row=0, column=0, padx=(10, 2), pady=(5, 2), sticky="w")
        self.ma_entry = ctk.CTkEntry(self.plot_options_frame, textvariable=self.ma_window_var, width=40)
        self.ma_entry.grid(row=0, column=1, padx=(0, 10), pady=(5, 2), sticky="w")
        self.start_date_label = ctk.CTkLabel(self.plot_options_frame, text="Start Date (YYYY-MM-DD):")
        self.start_date_label.grid(row=0, column=2, padx=(20, 2), pady=(5, 2), sticky="w")
        self.start_date_entry = ctk.CTkEntry(self.plot_options_frame, textvariable=self.start_date_var, placeholder_text="Optional", width=100)
        self.start_date_entry.grid(row=0, column=3, padx=(0, 10), pady=(5, 2), sticky="w")
        self.raw_score_check = ctk.CTkCheckBox(self.plot_options_frame, text="Raw Score", variable=self.show_raw_score_var)
        self.raw_score_check.grid(row=1, column=0, padx=10, pady=(2, 5), sticky="w")
        self.ma_score_check = ctk.CTkCheckBox(self.plot_options_frame, text="Score MA", variable=self.show_ma_score_var)
        self.ma_score_check.grid(row=1, column=1, padx=10, pady=(2, 5), sticky="w")
        self.pb_check = ctk.CTkCheckBox(self.plot_options_frame, text="PB", variable=self.show_pb_var)
        self.pb_check.grid(row=1, column=2, padx=10, pady=(2, 5), sticky="w")
        self.raw_acc_check = ctk.CTkCheckBox(self.plot_options_frame, text="Raw Accuracy", variable=self.show_raw_acc_var)
        self.raw_acc_check.grid(row=2, column=0, padx=10, pady=(2, 5), sticky="w")
        self.ma_acc_check = ctk.CTkCheckBox(self.plot_options_frame, text="Accuracy MA", variable=self.show_ma_acc_var)
        self.ma_acc_check.grid(row=2, column=1, padx=10, pady=(2, 5), sticky="w")
        # Plot Area
        self.plot_container_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.plot_container_frame.grid(row=2, column=0, rowspan=2, padx=10, pady=(0, 10), sticky="nsew")
        self.plot_container_frame.grid_rowconfigure(1, weight=1); self.plot_container_frame.grid_columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_score = self.fig.add_subplot(111); self.ax_acc = self.ax_score.twinx()
        self.ax_score.set_xlabel("Date"); self.ax_score.set_ylabel("Score", color='cyan'); self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_acc.set_ylabel("Accuracy (%)", color='lime'); self.ax_acc.tick_params(axis='y', labelcolor='lime')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container_frame)
        self.canvas_widget = self.canvas.get_tk_widget(); self.canvas_widget.grid(row=1, column=0, sticky="nsew")
        self.toolbar_frame = ctk.CTkFrame(self.plot_container_frame, fg_color="#2b2b2b", corner_radius=0)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame); self.toolbar.update()
        try:
             self.toolbar.config(background="#2b2b2b")
             for button in self.toolbar.winfo_children(): button.config(background="#2b2b2b", relief=tk.FLAT)
        except Exception: logger.warning("Could not fully style Matplotlib toolbar.")
        logger.debug("Widgets created.")

    # --- Methods (_find_and_load_default_path, _browse_folder) ---
    def _find_and_load_default_path(self):
        """Tries to find the default KovaaK's path and load data."""
        logger.info("Attempting to find and load default KovaaK's path...")
        default_path = utils.find_default_kovaaks_path()
        if default_path and default_path.is_dir():
            logger.info(f"Default path found: {default_path}")
            self.stats_folder_path = default_path
            self.path_label.configure(text=str(self.stats_folder_path))
            self._load_data()
        else:
            logger.warning("Default KovaaK's path not found or invalid.")
            self.path_label.configure(text="Default path not found. Please browse.")

    def _browse_folder(self):
        """Opens a dialog to select the stats folder."""
        logger.debug("Browse button clicked.")
        initial_dir = str(self.stats_folder_path) if self.stats_folder_path else "/"
        selected_path = filedialog.askdirectory(title="Select KovaaK's Stats Folder", initialdir=initial_dir)
        if selected_path:
            new_path = Path(selected_path)
            if new_path == self.stats_folder_path: return
            self.stats_folder_path = new_path
            logger.info(f"User selected folder: {self.stats_folder_path}")
            self.path_label.configure(text=str(self.stats_folder_path))
            self.stats_df = pd.DataFrame(); self.filtered_stats_df = pd.DataFrame()
            self.current_scenario = None; self.current_time_series = pd.DataFrame()
            self.search_var.set("")
            self._clear_playlist_filter(update_list=False)
            self._clear_scenario_list_ui()
            self._clear_stats_display()
            self._clear_plot("Loading data...")
            self._load_data()
        else: logger.debug("Folder selection cancelled.")

    def _load_data(self):
        """Loads data, creates buttons, and applies initial filters."""
        if not self.stats_folder_path or not self.stats_folder_path.is_dir():
            logger.warning("Load data called with invalid path.")
            self.path_label.configure(text="Invalid folder selected. Please browse.")
            self._clear_plot("Invalid folder"); return
        logger.info(f"Loading data from: {self.stats_folder_path}")
        self.browse_button.configure(state="disabled", text="Loading...")
        self.update_idletasks()
        try:
            self.stats_df = data_handler.load_stats_data(self.stats_folder_path)
            self.all_scenarios_after_date_filter = []
            self._clear_scenario_list_ui()

            if self.stats_df.empty:
                logger.warning("Loaded data is empty.")
                messagebox.showwarning("No Data", "No valid KovaaK's stats files found or parsed.")
                self.filtered_stats_df = pd.DataFrame()
                self._clear_plot("No data found")
                self._filter_and_display_scenarios() # Show "No scenarios" message
            else:
                logger.info(f"Data loaded successfully. Shape: {self.stats_df.shape}")
                self._apply_date_filter_and_refresh_list(clear_selection=True)
                self._clear_plot("Select a scenario")
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            messagebox.showerror("Loading Error", f"An error occurred:\n{e}")
            self.stats_df = pd.DataFrame(); self.filtered_stats_df = pd.DataFrame()
            self.all_scenarios_after_date_filter = []
            self._filter_and_display_scenarios(); self._clear_plot("Error loading data")
        finally:
             self.browse_button.configure(state="normal", text="Browse Stats Folder")

    def _clear_scenario_list_ui(self):
         """Clears scenario buttons and playlist headers."""
         logger.debug("Clearing scenario list UI elements.")
         for widget in self.scenario_scroll_frame.winfo_children():
              if widget != self.no_matches_label: widget.destroy()
         self.scenario_buttons = {}
         self.playlist_header_labels = {}

    def _create_all_scenario_buttons(self):
        """Creates all CTkButton widgets based on self.all_scenarios_after_date_filter."""
        logger.debug(f"Creating {len(self.all_scenarios_after_date_filter)} scenario buttons in memory.")
        self.scenario_buttons = {}
        for scenario in self.all_scenarios_after_date_filter:
            button = ctk.CTkButton(
                self.scenario_scroll_frame, text=scenario,
                command=lambda s=scenario: self._on_scenario_select(s),
                anchor="w", fg_color="transparent",
                text_color=("gray10", "gray90"), hover_color=("gray75", "gray25")
            )
            self.scenario_buttons[scenario] = button

    def _on_search_key_release(self, event=None):
        """Schedules a scenario list display update after a short delay."""
        if self._search_job: self.after_cancel(self._search_job)
        self._search_job = self.after(300, self._filter_and_display_scenarios)
        logger.debug("Search display update scheduled.")

    def _on_date_filter_change(self, *args):
        """Schedules a full filter update when the date entry changes."""
        if self._date_filter_job: self.after_cancel(self._date_filter_job)
        self._date_filter_job = self.after(400, lambda: self._apply_date_filter_and_refresh_list(clear_selection=True))
        logger.debug("Date filter update scheduled.")

    def _apply_date_filter_and_refresh_list(self, clear_selection=False):
        """Applies date filter to main DF, updates scenario list UI."""
        logger.info("Applying date filter and refreshing list...")
        self._date_filter_job = None

        if self.stats_df.empty:
            logger.debug("Original DataFrame is empty, nothing to filter.")
            self.filtered_stats_df = pd.DataFrame()
            self.all_scenarios_after_date_filter = []
            self._clear_scenario_list_ui()
            self._filter_and_display_scenarios()
            if clear_selection: self._clear_selection()
            return

        # Apply Date Filter
        start_date_str = self.start_date_var.get().strip()
        temp_start_date = None
        if start_date_str:
            try:
                temp_start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info(f"Applying start date filter: >= {temp_start_date.date()}")
                self.filtered_stats_df = self.stats_df[self.stats_df['Timestamp'] >= temp_start_date].copy()
            except ValueError:
                logger.warning(f"Invalid start date format: '{start_date_str}'. Ignoring.")
                messagebox.showwarning("Invalid Date", f"Ignoring invalid start date: '{start_date_str}'. Use YYYY-MM-DD.")
                self.filtered_stats_df = self.stats_df.copy()
                temp_start_date = None
        else:
            logger.info("No start date filter applied.")
            self.filtered_stats_df = self.stats_df.copy()
            temp_start_date = None

        filter_state_changed = (self.start_date_filter != temp_start_date)
        self.start_date_filter = temp_start_date

        # Update Scenario List based on Filtered Data
        new_scenario_list = data_handler.get_unique_scenarios(self.filtered_stats_df) if not self.filtered_stats_df.empty else []

        if set(new_scenario_list) != set(self.all_scenarios_after_date_filter):
             logger.info(f"Scenario list changed due to date filter. Found {len(new_scenario_list)} scenarios.")
             self.all_scenarios_after_date_filter = new_scenario_list
             self._clear_scenario_list_ui()
             self._create_all_scenario_buttons()
             self._filter_and_display_scenarios()
             if clear_selection: self._clear_selection()
        elif filter_state_changed:
             self._filter_and_display_scenarios()
             if clear_selection: self._clear_selection()


    def _clear_selection(self):
        """Clears the current scenario selection, stats, and plot."""
        logger.debug("Clearing current selection.")
        self.current_scenario = None
        self.current_time_series = pd.DataFrame()
        self._clear_stats_display()
        self._clear_plot("Select a scenario")


    def _filter_and_display_scenarios(self):
        """Filters the current scenario list (based on date/playlist) and displays."""
        search_term = self.search_var.get().lower()
        base_scenario_list = self.all_scenarios_after_date_filter # Already date-filtered
        logger.debug(f"Filtering {len(base_scenario_list)} available scenarios with term: '{search_term}' and playlist filter: {bool(self.playlist_map)}")
        self._search_job = None

        # Determine which scenarios match the search term *within the date-filtered list*
        scenarios_matching_search = {
            s for s in base_scenario_list if search_term in s.lower()
        }

        # Now display based on whether playlist filter is active
        self._display_scenario_list(scenarios_matching_search)


    def _display_scenario_list(self, scenarios_matching_search: set):
        """Shows/Hides scenario buttons based on current filters (date, playlist, search)."""
        logger.debug(f"Updating displayed scenario list. Search matches: {len(scenarios_matching_search)}. Playlist filter active: {bool(self.playlist_map)}")

        # Hide all widgets currently in the frame first
        all_widgets = list(self.playlist_header_labels.values()) + list(self.scenario_buttons.values()) + [self.no_matches_label]
        for widget in all_widgets:
            widget.pack_forget()

        found_any_match = False

        if self.playlist_map: # --- Grouped by Playlist ---
            logger.debug(f"Displaying grouped by {len(self.playlist_map)} playlists.")
            # Ensure header labels exist for loaded playlists
            for pl_name in self.playlist_map:
                 if pl_name not in self.playlist_header_labels:
                      label = ctk.CTkLabel(self.scenario_scroll_frame, text=pl_name, font=ctk.CTkFont(weight="bold"), anchor="w")
                      self.playlist_header_labels[pl_name] = label

            # Iterate through playlists *in the order they were loaded*
            for pl_name in self.active_playlist_names: # Use the ordered list of names
                 # Get the ordered list of scenarios for this playlist
                 scenarios_in_playlist_ordered = self.playlist_map.get(pl_name, [])
                 header_label = self.playlist_header_labels.get(pl_name)
                 if not header_label: continue # Should exist, but safety check

                 buttons_to_show_for_this_playlist = []
                 # Find buttons for this playlist that match filters, maintaining order
                 for scenario in scenarios_in_playlist_ordered: # Iterate through ordered list
                      if scenario in self.scenario_buttons and scenario in scenarios_matching_search:
                           buttons_to_show_for_this_playlist.append(self.scenario_buttons[scenario])

                 # If any buttons should be shown for this playlist, pack header then buttons
                 if buttons_to_show_for_this_playlist:
                      header_label.pack(pady=(5,1), padx=5, fill="x") # Pack header first
                      for button in buttons_to_show_for_this_playlist: # Pack buttons in order
                           button.pack(pady=(1,0), padx=15, fill="x")
                      found_any_match = True

        else: # --- Ungrouped (No Playlist Filter) ---
            logger.debug("Displaying ungrouped list.")
            # Show/hide based only on search term, maintaining original sort order
            for scenario in self.all_scenarios_after_date_filter:
                 button = self.scenario_buttons.get(scenario)
                 if button:
                      if scenario in scenarios_matching_search:
                           button.pack(pady=(1,0), padx=5, fill="x")
                           found_any_match = True

        # Show "No matches" label if applicable
        if not found_any_match:
            if not self.all_scenarios_after_date_filter:
                 self.no_matches_label.configure(text=" No scenarios found (check date filter)")
            else:
                 self.no_matches_label.configure(text=" No matching scenarios")
            self.no_matches_label.pack(pady=2, padx=5, fill="x")


    def _on_scenario_select(self, scenario_name: str):
        """Handles scenario selection using the filtered DataFrame."""
        if not scenario_name or self.filtered_stats_df.empty: return
        logger.info(f"Scenario selected: {scenario_name}")
        self.current_scenario = scenario_name
        try:
            summary = data_handler.get_scenario_summary(self.filtered_stats_df, scenario_name)
            self._update_stats_display(summary)
            self.current_time_series = data_handler.get_scenario_time_series(self.filtered_stats_df, scenario_name)
            self._update_plot()
        except Exception as e:
            logger.error(f"Error handling selection for '{scenario_name}': {e}", exc_info=True)
            messagebox.showerror("Error", f"Error processing scenario '{scenario_name}':\n{e}")
            self._clear_stats_display(); self._clear_plot(f"Error displaying {scenario_name}")

    def _redraw_plot_options_changed(self, *args):
        """Callback function when a plot option changes."""
        if self.current_scenario and not self.current_time_series.empty:
             try:
                 if self._redraw_job: self.after_cancel(self._redraw_job)
             except AttributeError: self._redraw_job = None
             self._redraw_job = self.after(100, self._update_plot)
             logger.debug("Plot redraw scheduled.")

    def _update_stats_display(self, summary: dict | None):
        """Updates the labels in the stats display area."""
        logger.debug(f"Updating stats display with: {summary}")
        if summary:
            self.pb_value.configure(text=str(summary.get('PB Score', 'N/A')))
            self.avg_score_value.configure(text=str(summary.get('Average Score', 'N/A')))
            self.avg_acc_value.configure(text=str(summary.get('Average Accuracy', 'N/A')))
            self.runs_value.configure(text=str(summary.get('Number of Runs', 'N/A')))
            self.last_played_value.configure(text=str(summary.get('Date Last Played', 'N/A')))
            self.first_played_value.configure(text=str(summary.get('First Played', 'N/A')))
        else: self._clear_stats_display()

    def _clear_stats_display(self):
        """Resets the stats display labels to N/A."""
        logger.debug("Clearing stats display.")
        self.pb_value.configure(text="N/A"); self.avg_score_value.configure(text="N/A")
        self.avg_acc_value.configure(text="N/A"); self.runs_value.configure(text="N/A")
        self.last_played_value.configure(text="N/A"); self.first_played_value.configure(text="N/A")

    def _clear_plot(self, message="No data to display"):
        """Clears the plot area and displays a message."""
        logger.debug(f"Clearing plot. Message: {message}")
        self.ax_score.clear(); self.ax_acc.clear()
        self.ax_score.set_xlabel("Date"); self.ax_score.set_ylabel("Score", color='cyan')
        self.ax_acc.set_ylabel("Accuracy (%)", color='lime')
        self.ax_acc.tick_params(axis='y', labelcolor='lime')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_score.text(0.5, 0.5, message, ha='center', va='center', transform=self.ax_score.transAxes, fontsize=12, color='gray')
        self.ax_score.grid(False); self.ax_acc.grid(False)
        self.ax_acc.set_yticks([]); self.ax_acc.set_ylim(0, 1)
        try: self.canvas.draw()
        except Exception as e: logger.error(f"Error drawing cleared canvas: {e}")

    def _update_plot(self):
        """Updates the Matplotlib plot based on current data and options."""
        # --- Plotting logic remains the same ---
        # ... (Include the full _update_plot method from the previous version here) ...
        time_series_df = self.current_time_series
        scenario_name = self.current_scenario
        logger.info(f"Updating plot for scenario: {scenario_name} with current options.")
        self._redraw_job = None

        # Pre-checks
        if time_series_df is None or time_series_df.empty:
            self._clear_plot("No run data for this scenario" if scenario_name else "Select a scenario"); return
        required_cols = ['Timestamp', 'Score']
        if not all(col in time_series_df.columns for col in required_cols):
             self._clear_plot("Data format error"); return
        if time_series_df['Timestamp'].isnull().all() or time_series_df['Score'].isnull().all():
            self._clear_plot("Invalid run data"); return

        # Get Plot Options
        try:
            ma_window = int(self.ma_window_var.get()); ma_window = max(2, ma_window)
        except ValueError: ma_window = 10; logger.warning("Invalid MA window, using 10.")
        show_raw_score = self.show_raw_score_var.get(); show_raw_acc = self.show_raw_acc_var.get()
        show_ma_score = self.show_ma_score_var.get(); show_ma_acc = self.show_ma_acc_var.get()
        show_pb = self.show_pb_var.get()

        # Clear axes
        self.ax_score.clear(); self.ax_acc.clear()

        # Calculate Statistics
        min_periods = max(1, min(len(time_series_df), ma_window))
        score_ma = time_series_df['Score'].rolling(window=ma_window, min_periods=min_periods).mean() if show_ma_score else None
        score_pb = time_series_df['Score'].cummax() if show_pb else None
        acc_ma = None
        if 'Avg Accuracy' in time_series_df.columns and show_ma_acc:
             acc_ma = time_series_df['Avg Accuracy'].rolling(window=ma_window, min_periods=min_periods).mean()

        # Plot Score Axis Data
        lines_score_axis, labels_score_axis, score_axis_data_list = [], [], []
        if show_raw_score:
            line, = self.ax_score.plot(time_series_df['Timestamp'], time_series_df['Score'], marker='o', ls='-', ms=3, c='cyan', label='Score', alpha=0.7)
            lines_score_axis.append(line); labels_score_axis.append('Score'); score_axis_data_list.append(time_series_df['Score'])
        if show_ma_score and score_ma is not None:
            line, = self.ax_score.plot(time_series_df['Timestamp'], score_ma, ls='-', lw=1.5, c='orange', label=f'Score MA({ma_window})')
            lines_score_axis.append(line); labels_score_axis.append(f'Score MA({ma_window})'); score_axis_data_list.append(score_ma)
        if show_pb and score_pb is not None:
             line, = self.ax_score.plot(time_series_df['Timestamp'], score_pb, ls='--', drawstyle='steps-post', lw=1.5, c='#FF6B6B', label='PB')
             lines_score_axis.append(line); labels_score_axis.append('PB'); score_axis_data_list.append(score_pb)
        self.ax_score.set_ylabel("Score / PB", color='cyan'); self.ax_score.tick_params(axis='y', labelcolor='cyan')

        # Plot Accuracy Axis Data
        plot_accuracy_axis = 'Avg Accuracy' in time_series_df.columns and time_series_df['Avg Accuracy'].notna().any()
        acc_axis_lines_plotted = False
        lines_acc_axis, labels_acc_axis, acc_axis_data_list = [], [], []
        if plot_accuracy_axis and show_raw_acc:
            acc_df_filtered = time_series_df[(time_series_df['Avg Accuracy'].notna()) & (time_series_df['Avg Accuracy'] > 0)]
            if not acc_df_filtered.empty:
                line, = self.ax_acc.plot(acc_df_filtered['Timestamp'], acc_df_filtered['Avg Accuracy'], marker='x', ls=':', ms=4, c='lime', label='Accuracy (%)', alpha=0.7)
                lines_acc_axis.append(line); labels_acc_axis.append('Accuracy (%)'); acc_axis_data_list.append(acc_df_filtered['Avg Accuracy']); acc_axis_lines_plotted = True
        if plot_accuracy_axis and show_ma_acc and acc_ma is not None:
             line, = self.ax_acc.plot(time_series_df['Timestamp'], acc_ma, ls='--', lw=1.5, c='#B2FF59', label=f'Acc MA({ma_window})')
             lines_acc_axis.append(line); labels_acc_axis.append(f'Acc MA({ma_window})'); acc_axis_data_list.append(acc_ma); acc_axis_lines_plotted = True

        # Configure Secondary Axis
        if acc_axis_lines_plotted:
            self.ax_acc.set_ylabel("Accuracy (%)", color='lime'); self.ax_acc.tick_params(axis='y', labelcolor='lime'); self.ax_acc.grid(False)
            if acc_axis_data_list: # Set dynamic Y limits for secondary axis
                full_acc_axis_data = pd.concat(acc_axis_data_list)
                if full_acc_axis_data.notna().any():
                     min_val_a, max_val_a = full_acc_axis_data.min(), full_acc_axis_data.max()
                     min_lim_a = max(0, min_val_a); max_lim_a = max(max_val_a, 100) # Ensure scale includes 0-100
                     padding_a = (max_lim_a - min_lim_a) * 0.05 if max_lim_a > min_lim_a else 1
                     self.ax_acc.set_ylim(min_lim_a - padding_a, max_lim_a + padding_a)
                else: self.ax_acc.set_ylim(0, 105) # Default if no valid data
            else: self.ax_acc.set_ylim(0, 105) # Default if no data plotted
        else: self.ax_acc.set_ylabel(""); self.ax_acc.set_yticks([])

        # Formatting
        self.ax_score.set_title(f"Progress: {scenario_name}", color="#DCE4EE"); self.ax_score.set_xlabel("Date")
        self.ax_score.grid(True, linestyle='--', linewidth=0.5)
        date_form = DateFormatter("%Y-%m-%d"); self.ax_score.xaxis.set_major_formatter(date_form); self.fig.autofmt_xdate()

        # Legends
        all_lines = lines_score_axis + lines_acc_axis; all_labels = labels_score_axis + labels_acc_axis
        if all_labels: self.ax_score.legend(all_lines, all_labels, loc='best', fontsize='small')

        # Set Y-Limits for Score Axis
        if score_axis_data_list:
            full_score_axis_data = pd.concat(score_axis_data_list)
            if full_score_axis_data.notna().any():
                 min_val, max_val = full_score_axis_data.min(), full_score_axis_data.max()
                 padding = (max_val - min_val) * 0.05 if max_val > min_val else 1
                 self.ax_score.set_ylim(min_val - padding, max_val + padding)
            else: self.ax_score.set_ylim(0, 1)
        else: self.ax_score.set_ylim(0, 1)

        # Redraw
        try: self.canvas.draw_idle(); logger.debug("Canvas redraw requested.")
        except Exception as e: logger.error(f"Error drawing canvas: {e}", exc_info=True)

    # --- Playlist Methods ---
    def _load_playlists(self):
        """Opens file dialog to select JSON playlist files."""
        logger.info("Opening playlist load dialog.")
        filepaths = filedialog.askopenfilenames(
            title="Select KovaaK's Playlist Files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepaths: logger.debug("Playlist selection cancelled."); return

        new_playlist_map = {}
        new_playlist_scenarios_union = set()
        loaded_playlist_names = [] # Keep track of names in load order
        errors_occurred = False

        for fp_str in filepaths:
            fp = Path(fp_str)
            playlist_name = fp.stem
            logger.debug(f"Parsing playlist file: {fp.name}")
            try:
                # Returns an ORDERED LIST or None
                scenarios_ordered = data_handler.parse_playlist_json(fp)
                if scenarios_ordered is not None:
                    if scenarios_ordered:
                         # Store the ordered list in the map
                         new_playlist_map[playlist_name] = scenarios_ordered
                         # Add to the union set for quick checking
                         new_playlist_scenarios_union.update(scenarios_ordered)
                         # Add name to ordered list of loaded playlists
                         loaded_playlist_names.append(playlist_name)
                    else: logger.warning(f"Playlist file was empty or contained no scenarios: {fp.name}")
                else: errors_occurred = True
            except Exception as e:
                logger.error(f"Error calling parse_playlist_json for {fp.name}: {e}", exc_info=True)
                errors_occurred = True

        if errors_occurred: messagebox.showerror("Playlist Error", "Error parsing one or more playlist files. Check logs.")
        if not new_playlist_map:
             logger.warning("No scenarios found in any selected valid playlists.")
             if not errors_occurred: self._clear_playlist_filter()
             return

        self.playlist_map = new_playlist_map # Store map {name: [ordered_list]}
        self.playlist_scenarios = new_playlist_scenarios_union # Store union set
        self.active_playlist_names = loaded_playlist_names # Store ordered names
        logger.info(f"Loaded {len(self.playlist_scenarios)} unique scenarios from {len(self.active_playlist_names)} playlists.")

        self._update_playlist_filter_label()
        self.clear_filter_button.configure(state="normal")
        self._filter_and_display_scenarios()
        self._clear_selection()


    def _clear_playlist_filter(self, update_list=True):
        """Clears the active playlist filter."""
        logger.info("Clearing playlist filter.")
        playlist_was_active = bool(self.playlist_map)
        self.playlist_map = {}; self.playlist_scenarios = set(); self.active_playlist_names = []
        self._update_playlist_filter_label()
        self.clear_filter_button.configure(state="disabled")
        if update_list and playlist_was_active:
             self._filter_and_display_scenarios()
             self._clear_selection()

    def _update_playlist_filter_label(self):
        """Updates the label showing the active playlist filter."""
        if self.active_playlist_names:
            max_len = 40
            names_str = ", ".join(self.active_playlist_names)
            if len(names_str) > max_len: names_str = names_str[:max_len-3] + "..."
            self.playlist_filter_label.configure(text=f"Filter: {names_str}")
        else:
            self.playlist_filter_label.configure(text="Filter: None")


# --- Main Execution (in main.py) ---
# if __name__ == "__main__":
#     app = App()
#     app.mainloop()
