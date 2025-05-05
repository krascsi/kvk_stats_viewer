# src/gui.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import json
from collections import deque # For efficient tracking of packed widgets (optional optimization)

# --- Matplotlib Imports ---
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.dates import DateFormatter, AutoDateLocator, AutoDateFormatter
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
# Using a style that generally works well with dark themes
plt.style.use('seaborn-v0_8-darkgrid') # Or 'dark_background'
plt.rcParams.update({
    "figure.facecolor": "#2b2b2b",
    "axes.facecolor": "#343638",
    "axes.edgecolor": "#DCE4EE",
    "axes.labelcolor": "#DCE4EE",
    "text.color": "#DCE4EE",
    "xtick.color": "#DCE4EE",
    "ytick.color": "#DCE4EE",
    "grid.color": "#565B5E",
    "figure.autolayout": True, # Helps prevent labels overlapping
    "figure.figsize": (6, 4), # Default figure size
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "patch.edgecolor": "#DCE4EE", # For legend box edge, etc.
    "legend.facecolor": "#343638",
    "legend.edgecolor": "#565B5E",
    "legend.frameon": True,
    "legend.loc": "best",
    "legend.fontsize": "small"
})


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("KovaaK's Stats Viewer")
        self.geometry("1150x800")
        self.minsize(850, 680)

        # Try setting theme before creating widgets
        try:
            ctk.set_appearance_mode("Dark") # Or "System", "Light"
            ctk.set_default_color_theme("blue")
        except Exception as e:
            logger.warning(f"Could not set CTk theme: {e}")


        # --- Data Variables ---
        self.stats_folder_path = None
        self.stats_df = pd.DataFrame()
        self.filtered_stats_df = pd.DataFrame()
        self.current_scenario = None
        self.current_time_series = pd.DataFrame()
        self.all_scenarios_after_date_filter = []
        # Playlist Data Structures
        self.playlist_map = {} # Stores {playlist_name: [ordered_list_of_scenarios]}
        self.playlist_scenarios = set() # Quick "is playlist active?" check
        self.active_playlist_names = [] # Order playlists were loaded

        # --- Filter Variables ---
        self.start_date_filter = None
        self.start_date_var = tk.StringVar(value="")

        # --- Plotting Variables ---
        self.fig = None
        self.ax_score = None # Primary Y-axis (Score)
        self.ax_acc = None   # Secondary Y-axis (Accuracy)
        self.canvas = None
        self.toolbar = None
        # --- References to plotted lines for efficient updates ---
        self.line_raw_score = None
        self.line_ma_score = None
        self.line_pb_score = None
        self.line_raw_acc = None
        self.line_ma_acc = None
        self.plot_message_text = None # Reference to the "Select scenario" text

        # --- Plot Options Variables ---
        self.ma_window_var = tk.StringVar(value="10")
        self.show_raw_score_var = tk.BooleanVar(value=True)
        self.show_raw_acc_var = tk.BooleanVar(value=True)
        self.show_ma_score_var = tk.BooleanVar(value=True)
        self.show_ma_acc_var = tk.BooleanVar(value=True)
        self.show_pb_var = tk.BooleanVar(value=False)

        # --- Debounce Timers ---
        self._redraw_job = None
        self._search_job = None
        self._date_filter_job = None

        # Add listeners for plot option changes
        trace_callback = self._schedule_plot_redraw # Use a dedicated scheduler
        self.show_raw_score_var.trace_add("write", trace_callback)
        self.show_raw_acc_var.trace_add("write", trace_callback)
        self.show_ma_score_var.trace_add("write", trace_callback)
        self.show_ma_acc_var.trace_add("write", trace_callback)
        self.show_pb_var.trace_add("write", trace_callback)
        self.ma_window_var.trace_add("write", trace_callback)
        # Listener for date filter
        self.start_date_var.trace_add("write", self._on_date_filter_change)

        # --- Configure Main Grid Layout ---
        # Allow content area (row 1) to expand
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        # Allow right frame (column 1) to expand more than left
        self.grid_columnconfigure(0, weight=1) # Changed to weight=1 for main frame
        # self.grid_columnconfigure(1, weight=3) # Example: make plot area wider

        # --- Create Widgets ---
        self._create_widgets()

        # --- Initial State ---
        self.path_label.configure(text="Click 'Browse' to select a stats folder.")
        self._initialize_plot_area() # Initialize plot axes etc.
        self._show_plot_message("No folder selected") # Show initial message

        # Try to load default path on startup
        self._find_and_load_default_path()


    def _create_widgets(self):
        """Creates and grids all the main GUI widgets."""
        logger.debug("Creating widgets...")

        # --- Top Frame (Folder Selection & Playlist) ---
        self.top_frame = ctk.CTkFrame(self, corner_radius=5)
        self.top_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.top_frame.grid_columnconfigure(1, weight=1) # Allow path label to expand
        self.top_frame.grid_columnconfigure(2, weight=0) # Playlist button fixed size

        self.browse_button = ctk.CTkButton(self.top_frame, text="Browse Stats Folder", command=self._browse_folder)
        self.browse_button.grid(row=0, column=0, padx=10, pady=10)

        self.path_label = ctk.CTkLabel(self.top_frame, text="No folder selected.", anchor="w", justify="left")
        self.path_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.load_playlist_button = ctk.CTkButton(self.top_frame, text="Load Playlist(s)", command=self._load_playlists)
        self.load_playlist_button.grid(row=0, column=2, padx=10, pady=10)

        # --- Main Content Frame (Splits Left/Right) ---
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_content_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1) # Allow frames to expand vertically
        # Adjust column weights for desired split
        self.main_content_frame.grid_columnconfigure(0, weight=1, minsize=250) # Scenario list
        self.main_content_frame.grid_columnconfigure(1, weight=3) # Stats & Plot area (wider)

        # --- Left Frame (Scenario List & Search) ---
        self.left_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.left_frame.grid(row=0, column=0, padx=(10, 5), pady=(5, 10), sticky="nsew")
        # Configure rows: Title, Search, Filter, List (expands)
        self.left_frame.grid_rowconfigure(0, weight=0)
        self.left_frame.grid_rowconfigure(1, weight=0)
        self.left_frame.grid_rowconfigure(2, weight=0)
        self.left_frame.grid_rowconfigure(3, weight=1) # Scrollable frame expands
        self.left_frame.grid_columnconfigure(0, weight=1) # Allow content to expand horizontally

        self.scenario_label = ctk.CTkLabel(self.left_frame, text="Scenarios", font=ctk.CTkFont(weight="bold"))
        self.scenario_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(self.left_frame, textvariable=self.search_var, placeholder_text="Search scenarios...")
        self.search_entry.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._on_search_key_release) # Debounced search

        # Playlist Filter Display Area
        self.playlist_filter_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.playlist_filter_frame.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.playlist_filter_frame.grid_columnconfigure(0, weight=1) # Label expands

        self.playlist_filter_label = ctk.CTkLabel(self.playlist_filter_frame, text="Filter: None", anchor="w", text_color="gray", font=ctk.CTkFont(size=11))
        self.playlist_filter_label.grid(row=0, column=0, sticky="ew")

        self.clear_filter_button = ctk.CTkButton(self.playlist_filter_frame, text="Clear", width=50, command=self._clear_playlist_filter, state="disabled")
        self.clear_filter_button.grid(row=0, column=1, padx=(5,0))

        # Scenario List Scrollable Frame
        self.scenario_scroll_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="", fg_color="transparent")
        self.scenario_scroll_frame.grid(row=3, column=0, padx=5, pady=(0, 5), sticky="nsew")
        self.scenario_buttons = {} # Holds {scenario_name: button_widget} - Buttons created on demand
        self.playlist_header_labels = {} # Holds {playlist_name: label_widget}
        self._packed_widgets_in_scroll = deque() # Track packed widgets for faster clearing (optional opt.)

        # Label for "No matches" or "No scenarios"
        self.no_matches_label = ctk.CTkLabel(self.scenario_scroll_frame, text="", anchor="w")
        # Don't pack initially

        # --- Right Frame (Stats Display & Plot Area) ---
        self.right_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.right_frame.grid(row=0, column=1, padx=(5, 10), pady=(5, 10), sticky="nsew")
        # Configure rows: Stats, Options, Plot (expands), Toolbar (fixed)
        self.right_frame.grid_rowconfigure(0, weight=0) # Stats display
        self.right_frame.grid_rowconfigure(1, weight=0) # Plot options
        self.right_frame.grid_rowconfigure(2, weight=1) # Plot canvas expands
        self.right_frame.grid_rowconfigure(3, weight=0) # Toolbar
        self.right_frame.grid_columnconfigure(0, weight=1) # Allow content to expand horizontally

        # Stats Display Frame
        self._create_stats_display_widgets(self.right_frame)

        # Plot Options Frame
        self._create_plot_options_widgets(self.right_frame)

        # Plot Area (Canvas and Toolbar)
        self._create_plot_area_widgets(self.right_frame)

        logger.debug("Widgets created.")

    def _create_stats_display_widgets(self, parent_frame):
        """Creates the widgets for the statistics display area."""
        self.stats_display_frame = ctk.CTkFrame(parent_frame)
        self.stats_display_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        # Configure columns for label-value pairs
        self.stats_display_frame.grid_columnconfigure(0, weight=0) # Label (fixed)
        self.stats_display_frame.grid_columnconfigure(1, weight=1) # Value (expands slightly)
        self.stats_display_frame.grid_columnconfigure(2, weight=0) # Label (fixed)
        self.stats_display_frame.grid_columnconfigure(3, weight=1) # Value (expands slightly)

        self.stats_label = ctk.CTkLabel(self.stats_display_frame, text="Statistics", font=ctk.CTkFont(weight="bold"))
        self.stats_label.grid(row=0, column=0, columnspan=4, padx=10, pady=(5, 10), sticky="w")

        # Create labels and value holders in a loop for cleaner code
        stat_items = [
            ("PB Score:", "pb_value"), ("Runs:", "runs_value"),
            ("Avg Score:", "avg_score_value"), ("Last Played:", "last_played_value"),
            ("Avg Accuracy:", "avg_acc_value"), ("First Played:", "first_played_value")
        ]

        row_num = 1
        col_num = 0
        for label_text, value_attr_name in stat_items:
            label = ctk.CTkLabel(self.stats_display_frame, text=label_text, anchor="w")
            label.grid(row=row_num, column=col_num, padx=(10,2), pady=2, sticky="w")

            value_label = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
            value_label.grid(row=row_num, column=col_num + 1, padx=(0,10), pady=2, sticky="ew")
            setattr(self, value_attr_name, value_label) # Store reference to value label

            col_num += 2
            if col_num >= 4:
                col_num = 0
                row_num += 1

    def _create_plot_options_widgets(self, parent_frame):
        """Creates the widgets for plot customization options."""
        self.plot_options_frame = ctk.CTkFrame(parent_frame)
        self.plot_options_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        # Configure columns - adjust weights as needed
        self.plot_options_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=0) # Keep compact

        # MA Window
        self.ma_label = ctk.CTkLabel(self.plot_options_frame, text="MA Window:")
        self.ma_label.grid(row=0, column=0, padx=(10, 2), pady=(5, 2), sticky="w")
        self.ma_entry = ctk.CTkEntry(self.plot_options_frame, textvariable=self.ma_window_var, width=40)
        self.ma_entry.grid(row=0, column=1, padx=(0, 10), pady=(5, 2), sticky="w")

        # Start Date Filter
        self.start_date_label = ctk.CTkLabel(self.plot_options_frame, text="Start Date (YYYY-MM-DD):")
        self.start_date_label.grid(row=0, column=2, padx=(20, 2), pady=(5, 2), sticky="w")
        self.start_date_entry = ctk.CTkEntry(self.plot_options_frame, textvariable=self.start_date_var, placeholder_text="Optional", width=100)
        self.start_date_entry.grid(row=0, column=3, padx=(0, 10), pady=(5, 2), sticky="w")

        # Checkboxes (arranged slightly differently for space)
        self.raw_score_check = ctk.CTkCheckBox(self.plot_options_frame, text="Raw Score", variable=self.show_raw_score_var)
        self.raw_score_check.grid(row=1, column=0, padx=10, pady=(2, 5), sticky="w")
        self.ma_score_check = ctk.CTkCheckBox(self.plot_options_frame, text="Score MA", variable=self.show_ma_score_var)
        self.ma_score_check.grid(row=1, column=1, padx=10, pady=(2, 5), sticky="w")

        self.raw_acc_check = ctk.CTkCheckBox(self.plot_options_frame, text="Raw Acc", variable=self.show_raw_acc_var) # Shortened label
        self.raw_acc_check.grid(row=1, column=2, padx=10, pady=(2, 5), sticky="w")
        self.ma_acc_check = ctk.CTkCheckBox(self.plot_options_frame, text="Acc MA", variable=self.show_ma_acc_var) # Shortened label
        self.ma_acc_check.grid(row=1, column=3, padx=10, pady=(2, 5), sticky="w")

        self.pb_check = ctk.CTkCheckBox(self.plot_options_frame, text="Show PB", variable=self.show_pb_var)
        self.pb_check.grid(row=1, column=4, padx=10, pady=(2, 5), sticky="w")

    def _create_plot_area_widgets(self, parent_frame):
        """Creates the Matplotlib canvas and toolbar."""
        # Frame to contain the plot canvas
        self.plot_canvas_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        self.plot_canvas_frame.grid(row=2, column=0, padx=10, pady=(0, 0), sticky="nsew")
        self.plot_canvas_frame.grid_rowconfigure(0, weight=1)
        self.plot_canvas_frame.grid_columnconfigure(0, weight=1)

        # Create Matplotlib Figure and Axes
        self.fig = Figure(figsize=(5, 4), dpi=100) # Use rcParams for facecolor etc.
        self.ax_score = self.fig.add_subplot(111)
        self.ax_acc = self.ax_score.twinx() # Create secondary axis

        # Create Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Create Toolbar Frame (below canvas)
        self.toolbar_frame = ctk.CTkFrame(parent_frame, fg_color="#2b2b2b", corner_radius=0)
        self.toolbar_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Create Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Style the toolbar (best effort)
        try:
             self.toolbar.config(background="#2b2b2b")
             # Style toolbar buttons (might not work perfectly on all systems/themes)
             for button in self.toolbar.winfo_children():
                 try:
                     button.config(background="#444", foreground="#DCE4EE", relief=tk.FLAT, borderwidth=0)
                 except tk.TclError:
                     pass # Ignore widgets that don't support these options
        except Exception as e:
            logger.warning(f"Could not fully style Matplotlib toolbar: {e}")

    def _initialize_plot_area(self):
        """Sets up the initial state of the plot axes (labels, colors, etc.)."""
        logger.debug("Initializing plot area.")
        # Clear any existing lines (just in case)
        self.ax_score.clear()
        self.ax_acc.clear()

        # --- Configure Score Axis (Primary) ---
        self.ax_score.set_xlabel("Date")
        self.ax_score.set_ylabel("Score / PB", color='cyan')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_score.grid(True, linestyle='--', linewidth=0.5, axis='both', color=plt.rcParams['grid.color']) # Grid on primary

        # --- Configure Accuracy Axis (Secondary) ---
        self.ax_acc.set_ylabel("Accuracy (%)", color='lime')
        self.ax_acc.tick_params(axis='y', labelcolor='lime')
        self.ax_acc.grid(False) # Turn off grid for secondary axis to avoid clutter
        self.ax_acc.set_ylim(0, 105) # Default accuracy range

        # --- Configure X Axis (Date) ---
        # Use AutoDateLocator and AutoDateFormatter for better date ticking
        locator = AutoDateLocator(minticks=3, maxticks=7) # Adjust density as needed
        formatter = AutoDateFormatter(locator)
        self.ax_score.xaxis.set_major_locator(locator)
        self.ax_score.xaxis.set_major_formatter(formatter)
        self.fig.autofmt_xdate() # Rotate date labels

        # --- Initialize Line Objects (set to None initially) ---
        # Create empty lines with labels, initially invisible
        self.line_raw_score, = self.ax_score.plot([], [], marker='o', ls='-', ms=3, c='cyan', label='Score', alpha=0.7, visible=False)
        self.line_ma_score, = self.ax_score.plot([], [], ls='-', lw=1.5, c='orange', label='Score MA', visible=False)
        self.line_pb_score, = self.ax_score.plot([], [], ls='--', drawstyle='steps-post', lw=1.5, c='#FF6B6B', label='PB', visible=False) # Pinkish-red
        self.line_raw_acc, = self.ax_acc.plot([], [], marker='x', ls=':', ms=4, c='lime', label='Accuracy (%)', alpha=0.7, visible=False)
        self.line_ma_acc, = self.ax_acc.plot([], [], ls='--', lw=1.5, c='#B2FF59', label='Acc MA', visible=False) # Light green

        # Initialize placeholder text
        self.plot_message_text = self.ax_score.text(0.5, 0.5, "", ha='center', va='center',
                                                     transform=self.ax_score.transAxes,
                                                     fontsize=12, color='gray', visible=False)

        self.fig.tight_layout() # Adjust layout after setting labels
        self.canvas.draw_idle()

    def _show_plot_message(self, message):
        """Displays a message in the center of the plot area."""
        logger.debug(f"Showing plot message: {message}")
        # Hide all actual data lines
        for line in [self.line_raw_score, self.line_ma_score, self.line_pb_score, self.line_raw_acc, self.line_ma_acc]:
            if line: line.set_visible(False)

        # Clear title and axes labels temporarily
        self.ax_score.set_title("")
        self.ax_score.set_ylabel("")
        self.ax_acc.set_ylabel("")
        self.ax_score.set_xlabel("")
        self.ax_score.tick_params(axis='y', labelcolor=plt.rcParams['axes.facecolor']) # Hide labels
        self.ax_acc.tick_params(axis='y', labelcolor=plt.rcParams['axes.facecolor'])
        self.ax_score.tick_params(axis='x', labelcolor=plt.rcParams['axes.facecolor'])
        self.ax_score.grid(False) # Hide grid

        # Show the message
        if self.plot_message_text:
            self.plot_message_text.set_text(message)
            self.plot_message_text.set_visible(True)

        # Hide legend if it exists
        legend = self.ax_score.get_legend()
        if legend: legend.set_visible(False)

        self.canvas.draw_idle()

    # --- Methods (_find_and_load_default_path, _browse_folder) ---
    def _find_and_load_default_path(self):
        """Tries to find the default KovaaK's path and load data."""
        logger.info("Attempting to find and load default KovaaK's path...")
        default_path = utils.find_default_kovaaks_path()
        if default_path and default_path.is_dir():
            logger.info(f"Default path found: {default_path}")
            self.stats_folder_path = default_path
            self.path_label.configure(text=str(self.stats_folder_path))
            self._load_data() # Trigger data loading
        else:
            logger.warning("Default KovaaK's path not found or invalid.")
            self.path_label.configure(text="Default path not found. Please browse.")
            self._show_plot_message("Default path not found")

    def _browse_folder(self):
        """Opens a dialog to select the stats folder."""
        logger.debug("Browse button clicked.")
        initial_dir = str(self.stats_folder_path) if self.stats_folder_path else "/"
        selected_path = filedialog.askdirectory(title="Select KovaaK's Stats Folder", initialdir=initial_dir)

        if selected_path:
            new_path = Path(selected_path)
            if new_path == self.stats_folder_path:
                logger.debug("Selected the same folder again.")
                return # No change needed

            self.stats_folder_path = new_path
            logger.info(f"User selected folder: {self.stats_folder_path}")
            self.path_label.configure(text=str(self.stats_folder_path))

            # Reset application state
            self.stats_df = pd.DataFrame()
            self.filtered_stats_df = pd.DataFrame()
            self.current_scenario = None
            self.current_time_series = pd.DataFrame()
            self.search_var.set("")
            self._clear_playlist_filter(update_list=False) # Clear playlist filter internally
            self._clear_scenario_list_ui() # Clear buttons
            self._clear_stats_display() # Clear stat labels
            self._show_plot_message("Loading data...") # Show loading message on plot

            # Load data for the new path
            self._load_data()
        else:
            logger.debug("Folder selection cancelled.")

    def _load_data(self):
        """Loads data using data_handler, handles UI updates and errors."""
        if not self.stats_folder_path or not self.stats_folder_path.is_dir():
            logger.warning("Load data called with invalid path.")
            self.path_label.configure(text="Invalid folder selected. Please browse.")
            self._show_plot_message("Invalid folder"); return

        logger.info(f"Loading data from: {self.stats_folder_path}")
        self.browse_button.configure(state="disabled", text="Loading...")
        self.update_idletasks() # Ensure UI updates before blocking call

        try:
            # --- Call the optimized data loader ---
            # Pass max_workers=None to use default (usually os.cpu_count())
            # Or specify a number, e.g., max_workers=os.cpu_count() - 1
            self.stats_df = data_handler.load_stats_data(self.stats_folder_path, max_workers=None)
            # --------------------------------------

            self.all_scenarios_after_date_filter = [] # Reset this list
            self._clear_scenario_list_ui() # Clear old buttons

            if self.stats_df.empty:
                logger.warning("Loaded data is empty or failed to load.")
                messagebox.showwarning("No Data", "No valid KovaaK's stats files found or parsed in the selected folder.")
                self.filtered_stats_df = pd.DataFrame()
                self._show_plot_message("No data found")
                self._filter_and_display_scenarios() # Update list to show "No scenarios"
            else:
                logger.info(f"Data loaded successfully. Shape: {self.stats_df.shape}")
                # Apply initial date filter (if any) and populate scenario list
                self._apply_date_filter_and_refresh_list(clear_selection=True)
                self._show_plot_message("Select a scenario") # Update plot message
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            messagebox.showerror("Loading Error", f"An error occurred while loading data:\n{e}")
            self.stats_df = pd.DataFrame(); self.filtered_stats_df = pd.DataFrame()
            self.all_scenarios_after_date_filter = []
            self._filter_and_display_scenarios() # Update list
            self._show_plot_message("Error loading data") # Show error on plot
        finally:
             # Ensure button is re-enabled regardless of success/failure
             self.browse_button.configure(state="normal", text="Browse Stats Folder")

    def _clear_scenario_list_ui(self):
         """Clears scenario buttons and playlist headers from the scroll frame."""
         logger.debug("Clearing scenario list UI elements.")
         # More efficient way to clear: iterate and destroy
         for widget in self.scenario_scroll_frame.winfo_children():
             widget.destroy()
         # Re-add the persistent "no matches" label (initially hidden)
         self.no_matches_label = ctk.CTkLabel(self.scenario_scroll_frame, text="", anchor="w")
         # Reset internal tracking
         self.scenario_buttons = {}
         self.playlist_header_labels = {}
         self._packed_widgets_in_scroll.clear() # Clear packed widget tracker

    def _create_scenario_button(self, scenario_name):
        """Creates a single scenario button widget (used on demand)."""
        # Check if button already exists (e.g., due to filter changes)
        if scenario_name in self.scenario_buttons:
            return self.scenario_buttons[scenario_name]

        button = ctk.CTkButton(
            self.scenario_scroll_frame, text=scenario_name,
            command=lambda s=scenario_name: self._on_scenario_select(s),
            anchor="w", fg_color="transparent",
            text_color=("gray10", "gray90"), # Dark/Light mode text
            hover_color=("gray75", "gray25") # Dark/Light mode hover
        )
        self.scenario_buttons[scenario_name] = button
        return button

    def _create_playlist_header(self, playlist_name):
        """Creates a single playlist header label widget."""
        if playlist_name in self.playlist_header_labels:
            return self.playlist_header_labels[playlist_name]

        label = ctk.CTkLabel(self.scenario_scroll_frame, text=playlist_name,
                             font=ctk.CTkFont(weight="bold"), anchor="w")
        self.playlist_header_labels[playlist_name] = label
        return label

    def _on_search_key_release(self, event=None):
        """Schedules a scenario list display update after a short delay (debouncing)."""
        if self._search_job:
            self.after_cancel(self._search_job)
        self._search_job = self.after(300, self._filter_and_display_scenarios) # 300ms delay
        logger.debug("Search display update scheduled.")

    def _on_date_filter_change(self, *args):
        """Schedules a full filter update when the date entry changes (debouncing)."""
        if self._date_filter_job:
            self.after_cancel(self._date_filter_job)
        # Use lambda to pass argument, ensuring selection is cleared
        self._date_filter_job = self.after(400, lambda: self._apply_date_filter_and_refresh_list(clear_selection=True))
        logger.debug("Date filter update scheduled.")

    def _apply_date_filter_and_refresh_list(self, clear_selection=False):
        """
        Applies the date filter to the main DataFrame, updates the list of
        available scenarios, and refreshes the displayed list.
        """
        logger.info("Applying date filter and refreshing list...")
        self._date_filter_job = None # Mark job as done

        if self.stats_df.empty:
            logger.debug("Original DataFrame is empty, nothing to filter.")
            self.filtered_stats_df = pd.DataFrame()
            self.all_scenarios_after_date_filter = []
            self._clear_scenario_list_ui() # Ensure UI is cleared
            self._filter_and_display_scenarios() # Display "No scenarios"
            if clear_selection: self._clear_selection()
            return

        # --- Apply Date Filter ---
        start_date_str = self.start_date_var.get().strip()
        new_start_date_filter = None # Store the parsed date filter

        if start_date_str:
            try:
                # Parse and normalize to start of the day
                parsed_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                new_start_date_filter = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info(f"Applying start date filter: >= {new_start_date_filter.date()}")
                # Filter the original DataFrame
                self.filtered_stats_df = self.stats_df[self.stats_df['Timestamp'] >= new_start_date_filter].copy()
            except ValueError:
                logger.warning(f"Invalid start date format: '{start_date_str}'. Ignoring filter.")
                messagebox.showwarning("Invalid Date", f"Ignoring invalid start date: '{start_date_str}'. Use YYYY-MM-DD.")
                self.filtered_stats_df = self.stats_df.copy() # Use unfiltered data
                new_start_date_filter = None # Reset filter state
        else:
            # No date string provided, use the full original DataFrame
            logger.info("No start date filter applied.")
            self.filtered_stats_df = self.stats_df.copy()
            new_start_date_filter = None

        # Check if the actual filter state changed
        filter_state_changed = (self.start_date_filter != new_start_date_filter)
        self.start_date_filter = new_start_date_filter # Update the stored filter state

        # --- Update Scenario List based on Filtered Data ---
        # Get unique scenarios from the *newly filtered* DataFrame
        new_scenario_list = data_handler.get_unique_scenarios(self.filtered_stats_df) if not self.filtered_stats_df.empty else []

        # Check if the list of available scenarios actually changed
        if set(new_scenario_list) != set(self.all_scenarios_after_date_filter):
             logger.info(f"Scenario list changed due to date filter. Found {len(new_scenario_list)} scenarios.")
             self.all_scenarios_after_date_filter = new_scenario_list
             # No need to create all buttons here anymore
             self._filter_and_display_scenarios() # Refresh the displayed list
             if clear_selection: self._clear_selection() # Clear selection if list changed
        elif filter_state_changed:
             # Filter date changed, but scenario list didn't (e.g., removed runs from current scenario)
             logger.info("Date filter changed, but the set of unique scenarios remained the same.")
             self._filter_and_display_scenarios() # Still need to refresh display potentially
             if clear_selection: self._clear_selection() # Clear selection if filter changed
             # If a scenario was selected, re-trigger its selection to update plot/stats
             elif self.current_scenario in self.all_scenarios_after_date_filter:
                 logger.debug(f"Re-selecting current scenario '{self.current_scenario}' after date filter change.")
                 self._on_scenario_select(self.current_scenario)


    def _clear_selection(self):
        """Clears the current scenario selection, stats display, and plot."""
        logger.debug("Clearing current selection.")
        self.current_scenario = None
        self.current_time_series = pd.DataFrame()
        self._clear_stats_display()
        self._show_plot_message("Select a scenario") # Reset plot to initial message

    def _filter_and_display_scenarios(self):
        """
        Filters the available scenarios based on search term and active playlist,
        then updates the widgets displayed in the scrollable frame.
        """
        search_term = self.search_var.get().lower()
        # Use the date-filtered list as the base
        base_scenario_list = self.all_scenarios_after_date_filter
        logger.debug(f"Filtering {len(base_scenario_list)} available scenarios. Search: '{search_term}'. Playlist filter active: {bool(self.playlist_map)}")
        self._search_job = None # Mark debounced job as done

        # Determine which scenarios (from the date-filtered list) match the search term
        scenarios_matching_search = {
            s for s in base_scenario_list if search_term in s.lower()
        }

        # --- Update the UI ---
        # 1. Forget all currently packed widgets efficiently
        while self._packed_widgets_in_scroll:
            widget = self._packed_widgets_in_scroll.popleft()
            widget.pack_forget()

        found_any_match = False

        # 2. Decide which widgets to pack based on filters
        if self.playlist_map: # --- Grouped by Playlist ---
            logger.debug(f"Displaying grouped by {len(self.playlist_map)} playlists.")
            # Iterate through playlists *in the order they were loaded*
            for pl_name in self.active_playlist_names:
                 header_label = self._create_playlist_header(pl_name) # Get or create header
                 scenarios_in_playlist_ordered = self.playlist_map.get(pl_name, [])

                 buttons_to_show_for_this_playlist = []
                 # Check each scenario in the ordered playlist against filters
                 for scenario in scenarios_in_playlist_ordered:
                      if scenario in scenarios_matching_search: # Check if it matches search AND exists in date-filtered list
                           button = self._create_scenario_button(scenario) # Get or create button
                           buttons_to_show_for_this_playlist.append(button)

                 # If any buttons match for this playlist, pack header then buttons
                 if buttons_to_show_for_this_playlist:
                      header_label.pack(pady=(5,1), padx=5, fill="x")
                      self._packed_widgets_in_scroll.append(header_label) # Track packed widget
                      for button in buttons_to_show_for_this_playlist:
                           button.pack(pady=(1,0), padx=15, fill="x")
                           self._packed_widgets_in_scroll.append(button) # Track packed widget
                      found_any_match = True

        else: # --- Ungrouped (No Playlist Filter) ---
            logger.debug("Displaying ungrouped list.")
            # Iterate through the date-filtered list (already sorted)
            for scenario in self.all_scenarios_after_date_filter:
                 if scenario in scenarios_matching_search: # Check if it matches search
                      button = self._create_scenario_button(scenario) # Get or create button
                      button.pack(pady=(1,0), padx=5, fill="x")
                      self._packed_widgets_in_scroll.append(button) # Track packed widget
                      found_any_match = True

        # 3. Show "No matches" label if applicable
        if not found_any_match:
            if not self.all_scenarios_after_date_filter:
                 self.no_matches_label.configure(text=" No scenarios found (check date filter)")
            else:
                 self.no_matches_label.configure(text=" No matching scenarios")
            self.no_matches_label.pack(pady=2, padx=5, fill="x")
            self._packed_widgets_in_scroll.append(self.no_matches_label) # Track packed widget


    def _on_scenario_select(self, scenario_name: str):
        """Handles scenario selection: updates stats display and plot."""
        if not scenario_name or self.filtered_stats_df.empty:
            logger.warning("Scenario selection ignored (no name or empty filtered data).")
            return

        # Prevent re-selecting the same scenario if already current
        # Although re-selecting might be desired if the date filter changed
        # if scenario_name == self.current_scenario:
        #     logger.debug(f"Scenario '{scenario_name}' already selected.")
        #     return

        logger.info(f"Scenario selected: {scenario_name}")
        self.current_scenario = scenario_name

        try:
            # Get summary and time series data from the *filtered* DataFrame
            summary = data_handler.get_scenario_summary(self.filtered_stats_df, scenario_name)
            self.current_time_series = data_handler.get_scenario_time_series(self.filtered_stats_df, scenario_name)

            self._update_stats_display(summary)
            self._update_plot() # Trigger the optimized plot update

        except Exception as e:
            logger.error(f"Error handling selection for '{scenario_name}': {e}", exc_info=True)
            messagebox.showerror("Error", f"Error processing scenario '{scenario_name}':\n{e}")
            self._clear_stats_display()
            self._show_plot_message(f"Error displaying {scenario_name}")

    def _schedule_plot_redraw(self, *args):
        """Schedules a plot redraw using `after` for debouncing option changes."""
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        # Schedule _update_plot only if a scenario is actually selected
        if self.current_scenario and not self.current_time_series.empty:
            self._redraw_job = self.after(100, self._update_plot) # 100ms delay
            logger.debug("Plot redraw scheduled due to option change.")
        else:
            self._redraw_job = None # No need to redraw if no scenario selected

    def _update_stats_display(self, summary: dict | None):
        """Updates the labels in the stats display area."""
        logger.debug(f"Updating stats display with: {summary}")
        if summary:
            # Use .get with default 'N/A' for safety
            self.pb_value.configure(text=str(summary.get('PB Score', 'N/A')))
            self.avg_score_value.configure(text=str(summary.get('Average Score', 'N/A')))
            self.avg_acc_value.configure(text=str(summary.get('Average Accuracy', 'N/A')))
            self.runs_value.configure(text=str(summary.get('Number of Runs', 'N/A')))
            self.last_played_value.configure(text=str(summary.get('Date Last Played', 'N/A')))
            self.first_played_value.configure(text=str(summary.get('First Played', 'N/A')))
        else:
            self._clear_stats_display()

    def _clear_stats_display(self):
        """Resets the stats display labels to N/A."""
        logger.debug("Clearing stats display.")
        self.pb_value.configure(text="N/A")
        self.avg_score_value.configure(text="N/A")
        self.avg_acc_value.configure(text="N/A")
        self.runs_value.configure(text="N/A")
        self.last_played_value.configure(text="N/A")
        self.first_played_value.configure(text="N/A")

    def _update_plot(self):
        """
        Optimized plot update function. Updates line data instead of clearing axes.
        """
        time_series_df = self.current_time_series
        scenario_name = self.current_scenario
        logger.info(f"Updating plot for scenario: {scenario_name} using optimized method.")
        self._redraw_job = None # Mark debounced job as done

        # --- Pre-checks ---
        if time_series_df is None or time_series_df.empty:
            self._show_plot_message("No run data for this scenario"); return
        required_cols = ['Timestamp', 'Score']
        if not all(col in time_series_df.columns for col in required_cols):
             logger.error("Time series data missing required columns.")
             self._show_plot_message("Data format error"); return
        # Ensure Timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(time_series_df['Timestamp']):
             time_series_df['Timestamp'] = pd.to_datetime(time_series_df['Timestamp'], errors='coerce')
        # Drop rows where essential data is missing after conversion
        time_series_df = time_series_df.dropna(subset=['Timestamp', 'Score'])
        if time_series_df.empty:
             self._show_plot_message("Invalid run data (missing dates/scores)"); return

        # --- Get Plot Options ---
        try:
            ma_window = int(self.ma_window_var.get()); ma_window = max(2, ma_window)
        except ValueError: ma_window = 10; logger.warning("Invalid MA window, using 10.")
        show_raw_score = self.show_raw_score_var.get()
        show_raw_acc = self.show_raw_acc_var.get()
        show_ma_score = self.show_ma_score_var.get()
        show_ma_acc = self.show_ma_acc_var.get()
        show_pb = self.show_pb_var.get()

        # Hide placeholder message
        if self.plot_message_text: self.plot_message_text.set_visible(False)

        # --- Prepare Data ---
        x_data = time_series_df['Timestamp']
        y_score = time_series_df['Score']
        y_acc = time_series_df['Avg Accuracy'] if 'Avg Accuracy' in time_series_df.columns else pd.Series(dtype=float)
        y_acc = pd.to_numeric(y_acc, errors='coerce') # Ensure numeric, handle potential errors

        min_periods = max(1, min(len(time_series_df), ma_window)) # For rolling calculation

        # Calculate MAs and PB only if needed
        score_ma = y_score.rolling(window=ma_window, min_periods=min_periods).mean() if show_ma_score else None
        score_pb = y_score.cummax() if show_pb else None
        acc_ma = y_acc.rolling(window=ma_window, min_periods=min_periods).mean() if show_ma_acc and y_acc.notna().any() else None

        # --- Update Score Axis Lines ---
        score_axis_visible_data = []
        self.line_raw_score.set_data(x_data, y_score)
        self.line_raw_score.set_visible(show_raw_score)
        if show_raw_score: score_axis_visible_data.append(y_score)

        if score_ma is not None:
            self.line_ma_score.set_data(x_data, score_ma)
            self.line_ma_score.set_label(f'Score MA({ma_window})') # Update label if window changes
            self.line_ma_score.set_visible(show_ma_score)
            if show_ma_score: score_axis_visible_data.append(score_ma)
        else:
            self.line_ma_score.set_visible(False)

        if score_pb is not None:
            self.line_pb_score.set_data(x_data, score_pb)
            self.line_pb_score.set_visible(show_pb)
            if show_pb: score_axis_visible_data.append(score_pb)
        else:
            self.line_pb_score.set_visible(False)

        # --- Update Accuracy Axis Lines ---
        acc_axis_visible_data = []
        plot_accuracy_axis = y_acc.notna().any() # Check if there's any valid accuracy data

        if plot_accuracy_axis:
            # Filter out potential NaNs for raw plot to avoid gaps if desired, or plot directly
            # self.line_raw_acc.set_data(x_data[y_acc.notna()], y_acc.dropna()) # Option 1: Plot only non-NaN
            self.line_raw_acc.set_data(x_data, y_acc) # Option 2: Plot with NaNs (may show gaps)
            self.line_raw_acc.set_visible(show_raw_acc)
            if show_raw_acc: acc_axis_visible_data.append(y_acc) # Use original for limits

            if acc_ma is not None:
                self.line_ma_acc.set_data(x_data, acc_ma)
                self.line_ma_acc.set_label(f'Acc MA({ma_window})') # Update label
                self.line_ma_acc.set_visible(show_ma_acc)
                if show_ma_acc: acc_axis_visible_data.append(acc_ma)
            else:
                self.line_ma_acc.set_visible(False)
        else:
            # No valid accuracy data, hide accuracy lines
            self.line_raw_acc.set_visible(False)
            self.line_ma_acc.set_visible(False)

        # --- Update Axes Limits and Appearance ---
        # Score Axis Limits
        if score_axis_visible_data:
            all_score_data = pd.concat(score_axis_visible_data).dropna()
            if not all_score_data.empty:
                min_val_s, max_val_s = all_score_data.min(), all_score_data.max()
                padding_s = (max_val_s - min_val_s) * 0.05 if max_val_s > min_val_s else 1
                self.ax_score.set_ylim(min_val_s - padding_s, max_val_s + padding_s)
            else: self.ax_score.set_ylim(0, 1) # Default if no valid data
        else: self.ax_score.set_ylim(0, 1) # Default if nothing visible

        # Accuracy Axis Limits & Visibility
        if acc_axis_visible_data:
            self.ax_acc.set_visible(True)
            self.ax_acc.set_ylabel("Accuracy (%)", color='lime')
            self.ax_acc.tick_params(axis='y', labelcolor='lime')
            all_acc_data = pd.concat(acc_axis_visible_data).dropna()
            if not all_acc_data.empty:
                 min_val_a, max_val_a = all_acc_data.min(), all_acc_data.max()
                 # Ensure range includes at least 0-100 if data is within it
                 min_lim_a = max(0, min_val_a - 5) # Add some padding below 0 if needed
                 max_lim_a = max(max_val_a + 5, 100) # Ensure max is at least 100
                 padding_a = (max_lim_a - min_lim_a) * 0.05 if max_lim_a > min_lim_a else 1
                 # self.ax_acc.set_ylim(min_lim_a - padding_a, max_lim_a + padding_a)
                 self.ax_acc.set_ylim(max(0, min_lim_a - padding_a), max_lim_a + padding_a) # Ensure min limit is not negative

            else: self.ax_acc.set_ylim(0, 105) # Default if no valid data
        else:
            # No accuracy data visible, hide the axis
            self.ax_acc.set_visible(False)

        # X Axis Limits (Date)
        self.ax_score.set_xlim(x_data.min(), x_data.max())
        # Re-apply date locator/formatter as limits change
        locator = AutoDateLocator(minticks=3, maxticks=7)
        formatter = AutoDateFormatter(locator)
        self.ax_score.xaxis.set_major_locator(locator)
        self.ax_score.xaxis.set_major_formatter(formatter)

        # Restore labels and title
        self.ax_score.set_title(f"Progress: {scenario_name}", color="#DCE4EE")
        self.ax_score.set_xlabel("Date")
        self.ax_score.set_ylabel("Score / PB", color='cyan')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_score.tick_params(axis='x', labelcolor=plt.rcParams['xtick.color']) # Restore x ticks
        self.ax_score.grid(True, linestyle='--', linewidth=0.5, axis='both', color=plt.rcParams['grid.color'])

        # Update Legend
        handles, labels = self.ax_score.get_legend_handles_labels()
        acc_handles, acc_labels = self.ax_acc.get_legend_handles_labels()
        # Filter out invisible lines from legend items
        visible_handles = [h for h in handles + acc_handles if h.get_visible()]
        visible_labels = [l for h, l in zip(handles + acc_handles, labels + acc_labels) if h.get_visible()]

        # Get existing legend or create if needed
        legend = self.ax_score.get_legend()
        if visible_handles:
            if legend:
                legend.remove() # Remove old legend before creating new one
            self.ax_score.legend(visible_handles, visible_labels, loc='best', fontsize='small')
            # legend.set_visible(True)
        elif legend:
            legend.set_visible(False) # Hide legend if no lines are visible

        # --- Redraw Canvas ---
        try:
            self.fig.tight_layout() # Adjust layout before drawing
            self.canvas.draw_idle() # Efficiently schedule redraw
            logger.debug("Optimized plot update complete, redraw scheduled.")
        except Exception as e:
            logger.error(f"Error during optimized plot draw: {e}", exc_info=True)

    # --- Playlist Methods ---
    def _load_playlists(self):
        """Opens file dialog to select JSON playlist files."""
        logger.info("Opening playlist load dialog.")
        # Remember last directory if available
        initial_dir = str(self.stats_folder_path) if self.stats_folder_path else "/"
        filepaths = filedialog.askopenfilenames(
            title="Select KovaaK's Playlist Files (.json)",
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepaths:
            logger.debug("Playlist selection cancelled.")
            return

        new_playlist_map = {}
        new_playlist_scenarios_union = set()
        loaded_playlist_names_in_order = [] # Keep track of names in load order
        errors_occurred = False

        for fp_str in filepaths:
            fp = Path(fp_str)
            # Use filename stem as playlist name (can be customized later)
            playlist_name = fp.stem
            logger.debug(f"Parsing playlist file: {fp.name}")
            try:
                # Call data_handler function which returns an ORDERED LIST or None
                scenarios_ordered = data_handler.parse_playlist_json(fp)
                if scenarios_ordered is not None: # Parsing was successful (even if list is empty)
                    if scenarios_ordered: # Only add if it contains scenarios
                         # Store the ordered list in the map
                         new_playlist_map[playlist_name] = scenarios_ordered
                         # Add scenarios to the union set for quick checking
                         new_playlist_scenarios_union.update(scenarios_ordered)
                         # Add name to our ordered list of loaded playlists
                         if playlist_name not in loaded_playlist_names_in_order:
                             loaded_playlist_names_in_order.append(playlist_name)
                    else:
                        logger.warning(f"Playlist file was empty or contained no valid scenarios: {fp.name}")
                else:
                    # parse_playlist_json returned None, indicating a parsing error
                    errors_occurred = True
            except Exception as e:
                # Catch errors during the call itself
                logger.error(f"Unexpected error calling parse_playlist_json for {fp.name}: {e}", exc_info=True)
                errors_occurred = True

        if errors_occurred:
            messagebox.showerror("Playlist Error", "Error parsing one or more playlist files. Please check the file format and logs for details.")
        if not new_playlist_map:
             logger.warning("No scenarios found in any selected valid playlists.")
             # Don't clear filter if errors occurred, user might want to try again
             if not errors_occurred:
                 self._clear_playlist_filter() # Clear if selection was valid but empty
             return

        # Successfully loaded playlists, update internal state
        self.playlist_map = new_playlist_map
        self.playlist_scenarios = new_playlist_scenarios_union
        self.active_playlist_names = loaded_playlist_names_in_order # Store the order
        logger.info(f"Loaded {len(self.playlist_scenarios)} unique scenarios from {len(self.active_playlist_names)} playlists: {', '.join(self.active_playlist_names)}")

        # Update UI
        self._update_playlist_filter_label()
        self.clear_filter_button.configure(state="normal") # Enable clear button
        self._filter_and_display_scenarios() # Refresh scenario list based on new filter
        self._clear_selection() # Clear current plot/stats

    def _clear_playlist_filter(self, update_list=True):
        """Clears the active playlist filter and resets related UI."""
        logger.info("Clearing playlist filter.")
        playlist_was_active = bool(self.playlist_map) # Check if filter was actually active

        # Reset internal playlist data
        self.playlist_map = {}
        self.playlist_scenarios = set()
        self.active_playlist_names = []

        # Update UI elements
        self._update_playlist_filter_label()
        self.clear_filter_button.configure(state="disabled") # Disable clear button

        # Refresh scenario list only if the filter was active before clearing
        if update_list and playlist_was_active:
             self._filter_and_display_scenarios()
             self._clear_selection() # Clear plot/stats as filter changed

    def _update_playlist_filter_label(self):
        """Updates the label showing the active playlist filter names."""
        if self.active_playlist_names:
            max_len = 40 # Max characters for the label
            names_str = ", ".join(self.active_playlist_names)
            # Truncate if too long
            if len(names_str) > max_len:
                names_str = names_str[:max_len-3] + "..."
            self.playlist_filter_label.configure(text=f"Filter: {names_str}")
        else:
            self.playlist_filter_label.configure(text="Filter: None")


# --- Main Execution (Typically in a separate main.py) ---
# if __name__ == "__main__":
#     # Configure logging for the application if needed
#     # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] %(message)s')
#
#     app = App()
#     app.mainloop()
