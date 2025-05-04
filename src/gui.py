# src/gui.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# --- Matplotlib Imports ---
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.dates import DateFormatter

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
        self.geometry("1150x780")
        self.minsize(850, 650)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Data Variables ---
        self.stats_folder_path = None
        self.stats_df = pd.DataFrame()
        self.current_scenario = None
        self.current_time_series = pd.DataFrame()
        self.all_scenarios = []

        # --- Plotting Variables ---
        self.fig = None
        self.ax_score = None
        self.ax_acc = None
        self.canvas = None
        self.toolbar = None

        # --- Plot Options Variables ---
        self.ma_window_var = tk.StringVar(value="10")
        self.show_raw_score_var = tk.BooleanVar(value=True)
        self.show_raw_acc_var = tk.BooleanVar(value=True)
        self.show_ma_score_var = tk.BooleanVar(value=True)
        self.show_ma_acc_var = tk.BooleanVar(value=True)
        self.show_pb_var = tk.BooleanVar(value=False)
        self.show_std_dev_var = tk.BooleanVar(value=False)

        # --- Debounce Timers ---
        self._redraw_job = None # For debouncing plot redraw
        self._search_job = None # For debouncing search updates

        # Add listeners to redraw plot when options change
        trace_callback = self._redraw_plot_options_changed
        self.show_raw_score_var.trace_add("write", trace_callback)
        self.show_raw_acc_var.trace_add("write", trace_callback)
        self.show_ma_score_var.trace_add("write", trace_callback)
        self.show_ma_acc_var.trace_add("write", trace_callback)
        self.show_pb_var.trace_add("write", trace_callback)
        self.show_std_dev_var.trace_add("write", trace_callback)
        self.ma_window_var.trace_add("write", trace_callback)


        # --- Configure Main Grid Layout ---
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
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
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.browse_button = ctk.CTkButton(self.top_frame, text="Browse Stats Folder", command=self._browse_folder)
        self.browse_button.grid(row=0, column=0, padx=10, pady=10)
        self.path_label = ctk.CTkLabel(self.top_frame, text="No folder selected.", anchor="w", justify="left")
        self.path_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # --- Main Content Frame ---
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_content_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=0, minsize=250)
        self.main_content_frame.grid_columnconfigure(1, weight=1)

        # --- Left Frame (Scenario List & Search) ---
        self.left_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.left_frame.grid(row=0, column=0, padx=(10, 5), pady=(5, 10), sticky="nsew")
        self.left_frame.grid_rowconfigure(0, weight=0)
        self.left_frame.grid_rowconfigure(1, weight=0) # Search bar row
        self.left_frame.grid_rowconfigure(2, weight=1) # List expands
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.scenario_label = ctk.CTkLabel(self.left_frame, text="Scenarios", font=ctk.CTkFont(weight="bold"))
        self.scenario_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        # Search Entry
        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(self.left_frame, textvariable=self.search_var, placeholder_text="Search scenarios...")
        self.search_entry.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        # Bind key release event to DEBOUNCED filter function
        self.search_entry.bind("<KeyRelease>", self._on_search_key_release) # Changed binding target

        # Scrollable Frame for Scenario List
        self.scenario_scroll_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="", fg_color="transparent")
        self.scenario_scroll_frame.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="nsew")
        self.scenario_buttons = {}

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
        self.std_dev_check = ctk.CTkCheckBox(self.plot_options_frame, text="Score Std Dev", variable=self.show_std_dev_var)
        self.std_dev_check.grid(row=2, column=2, padx=10, pady=(2, 5), sticky="w")
        # Plot Area
        self.plot_container_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.plot_container_frame.grid(row=2, column=0, rowspan=2, padx=10, pady=(0, 10), sticky="nsew")
        self.plot_container_frame.grid_rowconfigure(1, weight=1); self.plot_container_frame.grid_columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_score = self.fig.add_subplot(111); self.ax_acc = self.ax_score.twinx()
        self.ax_score.set_xlabel("Date"); self.ax_score.set_ylabel("Score", color='cyan'); self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_acc.set_ylabel("Accuracy (%) / Std Dev", color='lime'); self.ax_acc.tick_params(axis='y', labelcolor='lime')
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
    # Keep these methods as they were in the "No Default Load" version
    # ... (Include the full code for these methods here) ...
    def _find_and_load_default_path(self):
        """Tries to find the default KovaaK's path and load data."""
        # This method is no longer called automatically on startup
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
            if new_path == self.stats_folder_path:
                logger.debug("Selected folder is the same as the current one.")
                return
            self.stats_folder_path = new_path
            logger.info(f"User selected folder: {self.stats_folder_path}")
            self.path_label.configure(text=str(self.stats_folder_path))
            self.stats_df = pd.DataFrame()
            self.current_scenario = None
            self.current_time_series = pd.DataFrame()
            self.search_var.set("") # Clear search bar on new folder load
            # self._update_scenario_list() # Will be called by _load_data
            self._clear_stats_display()
            self._clear_plot("Loading data...")
            self._load_data() # Load data from the newly selected folder
        else: logger.debug("Folder selection cancelled.")

    def _load_data(self):
        """Loads and processes stats data from the selected folder."""
        if not self.stats_folder_path or not self.stats_folder_path.is_dir():
            logger.warning("Load data called with invalid path.")
            self.path_label.configure(text="Invalid folder selected. Please browse.")
            self._clear_plot("Invalid folder")
            return
        logger.info(f"Loading data from: {self.stats_folder_path}")
        self.browse_button.configure(state="disabled", text="Loading...")
        self.update_idletasks()
        try:
            self.stats_df = data_handler.load_stats_data(self.stats_folder_path)
            self.all_scenarios = [] # Reset full list
            if self.stats_df.empty:
                logger.warning("Loaded data is empty.")
                messagebox.showwarning("No Data", "No valid KovaaK's stats files found or parsed in the selected folder.")
                self._clear_plot("No data found")
            else:
                logger.info(f"Data loaded successfully. Shape: {self.stats_df.shape}")
                # Get and store the full list of scenarios
                self.all_scenarios = data_handler.get_unique_scenarios(self.stats_df)
                self._clear_plot("Select a scenario")
            # Update the list (will apply current search filter, which should be empty)
            self._update_scenario_list()
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            messagebox.showerror("Loading Error", f"An error occurred while loading data:\n{e}")
            self.stats_df = pd.DataFrame()
            self.all_scenarios = []
            self._update_scenario_list()
            self._clear_plot("Error loading data")
        finally:
             self.browse_button.configure(state="normal", text="Browse Stats Folder")

    def _on_search_key_release(self, event=None):
        """Schedules a scenario list update after a short delay."""
        # Cancel the previous job, if any
        if self._search_job:
            self.after_cancel(self._search_job)

        # Schedule the update function to run after 300ms
        self._search_job = self.after(300, self._update_scenario_list)
        logger.debug("Search update scheduled.")


    def _update_scenario_list(self):
        """Populates the scenario list, filtering based on the search entry."""
        search_term = self.search_var.get().lower()
        logger.debug(f"Updating scenario list with filter: '{search_term}'")
        self._search_job = None # Reset the job ID now that the function is running

        # Clear existing buttons/labels
        for widget in self.scenario_scroll_frame.winfo_children():
            widget.destroy()
        self.scenario_buttons = {}

        if not self.all_scenarios:
            label_text = " Error loading scenarios" if not self.stats_df.empty else " No scenarios loaded"
            label = ctk.CTkLabel(self.scenario_scroll_frame, text=label_text, anchor="w")
            label.pack(pady=2, padx=5, fill="x")
            return

        # Filter the full list
        scenarios_to_display = [
            s for s in self.all_scenarios if search_term in s.lower()
        ]
        logger.info(f"Displaying {len(scenarios_to_display)} scenarios after filtering.")

        if scenarios_to_display:
            for scenario in scenarios_to_display:
                button = ctk.CTkButton(
                    self.scenario_scroll_frame, text=scenario,
                    command=lambda s=scenario: self._on_scenario_select(s),
                    anchor="w", fg_color="transparent",
                    text_color=("gray10", "gray90"), hover_color=("gray75", "gray25")
                )
                button.pack(pady=(1,0), padx=5, fill="x")
                self.scenario_buttons[scenario] = button
        else:
             label = ctk.CTkLabel(self.scenario_scroll_frame, text=" No matching scenarios", anchor="w")
             label.pack(pady=2, padx=5, fill="x")


    def _on_scenario_select(self, scenario_name: str):
        """Handles scenario selection."""
        if not scenario_name or self.stats_df.empty: return
        logger.info(f"Scenario selected: {scenario_name}")
        self.current_scenario = scenario_name
        try:
            summary = data_handler.get_scenario_summary(self.stats_df, scenario_name)
            self._update_stats_display(summary)
            self.current_time_series = data_handler.get_scenario_time_series(self.stats_df, scenario_name)
            self._update_plot() # Initial plot for the selected scenario
        except Exception as e:
            logger.error(f"Error handling selection for '{scenario_name}': {e}", exc_info=True)
            messagebox.showerror("Error", f"Error processing scenario '{scenario_name}':\n{e}")
            self._clear_stats_display()
            self._clear_plot(f"Error displaying {scenario_name}")

    def _redraw_plot_options_changed(self, *args):
        """Callback function when a plot option changes."""
        if self.current_scenario and not self.current_time_series.empty:
             try:
                 if self._redraw_job: self.after_cancel(self._redraw_job)
             except AttributeError: self._redraw_job = None
             # Use a slightly shorter delay for plot options vs search
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
        else:
            self._clear_stats_display()

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
        self.ax_acc.set_ylabel("Accuracy (%) / Std Dev", color='lime')
        self.ax_acc.tick_params(axis='y', labelcolor='lime')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_score.text(0.5, 0.5, message, ha='center', va='center', transform=self.ax_score.transAxes, fontsize=12, color='gray')
        self.ax_score.grid(False); self.ax_acc.grid(False)
        self.ax_acc.set_yticks([]); self.ax_acc.set_ylim(0, 1)
        try: self.canvas.draw()
        except Exception as e: logger.error(f"Error drawing cleared canvas: {e}")

    def _update_plot(self):
        """Updates the Matplotlib plot based on current data and options."""
        time_series_df = self.current_time_series
        scenario_name = self.current_scenario
        logger.info(f"Updating plot for scenario: {scenario_name} with current options.")
        self._redraw_job = None # Reset redraw job ID as it's running now

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
        show_pb = self.show_pb_var.get(); show_std_dev = self.show_std_dev_var.get()

        # Clear axes
        self.ax_score.clear(); self.ax_acc.clear()

        # Calculate Statistics
        min_periods = max(1, min(len(time_series_df), ma_window))
        score_ma = time_series_df['Score'].rolling(window=ma_window, min_periods=min_periods).mean() if show_ma_score else None
        score_std_dev = time_series_df['Score'].rolling(window=ma_window, min_periods=min_periods).std() if show_std_dev else None
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
        if show_std_dev and score_std_dev is not None:
             line, = self.ax_acc.plot(time_series_df['Timestamp'], score_std_dev, ls=':', lw=1.5, c='#FFEE58', label=f'Score StdDev({ma_window})')
             lines_acc_axis.append(line); labels_acc_axis.append(f'Score StdDev({ma_window})'); acc_axis_data_list.append(score_std_dev); acc_axis_lines_plotted = True

        # Configure Secondary Axis
        if acc_axis_lines_plotted:
            y2_label = ""; acc_label_present = any('Acc' in label for label in labels_acc_axis); std_label_present = any('StdDev' in label for label in labels_acc_axis)
            if acc_label_present: y2_label += "Accuracy (%)"
            if std_label_present: y2_label += (" / " if y2_label else "") + "Score Std Dev"
            self.ax_acc.set_ylabel(y2_label, color='lime'); self.ax_acc.tick_params(axis='y', labelcolor='lime'); self.ax_acc.grid(False)
            if acc_axis_data_list: # Set dynamic Y limits for secondary axis
                full_acc_axis_data = pd.concat(acc_axis_data_list)
                if full_acc_axis_data.notna().any():
                     min_val_a, max_val_a = full_acc_axis_data.min(), full_acc_axis_data.max()
                     min_lim_a = max(0, min_val_a); max_lim_a = max_val_a
                     if acc_label_present: max_lim_a = max(max_lim_a, 100)
                     padding_a = (max_lim_a - min_lim_a) * 0.05 if max_lim_a > min_lim_a else 1
                     self.ax_acc.set_ylim(min_lim_a - padding_a, max_lim_a + padding_a)
                else: self.ax_acc.set_ylim(0, 1)
            else: self.ax_acc.set_ylim(0, 1)
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

# --- Main Execution (in main.py) ---
# if __name__ == "__main__":
#     app = App()
#     app.mainloop()
