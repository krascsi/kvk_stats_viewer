# src/gui.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import logging
import pandas as pd
import numpy as np # Import numpy for NaN checking if needed

# --- Matplotlib Imports ---
import matplotlib
matplotlib.use('TkAgg') # Use Tkinter backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.dates import DateFormatter # For formatting date axis

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
        self.geometry("1100x700")
        self.minsize(800, 550)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Data Variables ---
        self.stats_folder_path = None
        self.stats_df = pd.DataFrame()

        # --- Plotting Variables ---
        self.fig = None
        self.ax_score = None
        self.ax_acc = None
        self.canvas = None
        self.toolbar = None

        # --- Configure Main Grid Layout ---
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Create Widgets ---
        self._create_widgets()
        self._find_and_load_default_path()

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

        # --- Left Frame (Scenario List) ---
        self.left_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.left_frame.grid(row=0, column=0, padx=(10, 5), pady=(5, 10), sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.scenario_label = ctk.CTkLabel(self.left_frame, text="Scenarios", font=ctk.CTkFont(weight="bold"))
        self.scenario_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.scenario_scroll_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="", fg_color="transparent")
        self.scenario_scroll_frame.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="nsew")
        self.scenario_buttons = {}

        # --- Right Frame (Stats Display & Plot Area) ---
        self.right_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=5)
        self.right_frame.grid(row=0, column=1, padx=(5, 10), pady=(5, 10), sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=0)
        self.right_frame.grid_rowconfigure(1, weight=0)
        self.right_frame.grid_rowconfigure(2, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # --- Stats Display Frame ---
        self.stats_display_frame = ctk.CTkFrame(self.right_frame)
        self.stats_display_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        self.stats_display_frame.grid_columnconfigure(0, weight=0)
        self.stats_display_frame.grid_columnconfigure(1, weight=1)
        self.stats_display_frame.grid_columnconfigure(2, weight=0)
        self.stats_display_frame.grid_columnconfigure(3, weight=1)
        self.stats_label = ctk.CTkLabel(self.stats_display_frame, text="Statistics", font=ctk.CTkFont(weight="bold"))
        self.stats_label.grid(row=0, column=0, columnspan=4, padx=10, pady=(5, 10), sticky="w")
        # Stat Labels
        row_num = 1
        self.pb_label = ctk.CTkLabel(self.stats_display_frame, text="PB Score:", anchor="w")
        self.pb_label.grid(row=row_num, column=0, padx=(10,2), pady=2, sticky="w")
        self.pb_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
        self.pb_value.grid(row=row_num, column=1, padx=(0,10), pady=2, sticky="ew")
        self.runs_label = ctk.CTkLabel(self.stats_display_frame, text="Runs:", anchor="w")
        self.runs_label.grid(row=row_num, column=2, padx=(10,2), pady=2, sticky="w")
        self.runs_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
        self.runs_value.grid(row=row_num, column=3, padx=(0,10), pady=2, sticky="ew")
        row_num += 1
        self.avg_score_label = ctk.CTkLabel(self.stats_display_frame, text="Avg Score:", anchor="w")
        self.avg_score_label.grid(row=row_num, column=0, padx=(10,2), pady=2, sticky="w")
        self.avg_score_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
        self.avg_score_value.grid(row=row_num, column=1, padx=(0,10), pady=2, sticky="ew")
        self.last_played_label = ctk.CTkLabel(self.stats_display_frame, text="Last Played:", anchor="w")
        self.last_played_label.grid(row=row_num, column=2, padx=(10,2), pady=2, sticky="w")
        self.last_played_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
        self.last_played_value.grid(row=row_num, column=3, padx=(0,10), pady=2, sticky="ew")
        row_num += 1
        self.avg_acc_label = ctk.CTkLabel(self.stats_display_frame, text="Avg Accuracy:", anchor="w")
        self.avg_acc_label.grid(row=row_num, column=0, padx=(10,2), pady=2, sticky="w")
        self.avg_acc_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
        self.avg_acc_value.grid(row=row_num, column=1, padx=(0,10), pady=2, sticky="ew")
        self.first_played_label = ctk.CTkLabel(self.stats_display_frame, text="First Played:", anchor="w")
        self.first_played_label.grid(row=row_num, column=2, padx=(10,2), pady=2, sticky="w")
        self.first_played_value = ctk.CTkLabel(self.stats_display_frame, text="N/A", anchor="w")
        self.first_played_value.grid(row=row_num, column=3, padx=(0,10), pady=2, sticky="ew")
        row_num += 1

        # --- Plot Area ---
        self.plot_container_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.plot_container_frame.grid(row=1, column=0, rowspan=2, padx=10, pady=(5, 10), sticky="nsew")
        self.plot_container_frame.grid_rowconfigure(1, weight=1)
        self.plot_container_frame.grid_columnconfigure(0, weight=1)
        # Matplotlib Figure and Axes
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_score = self.fig.add_subplot(111)
        self.ax_score.set_xlabel("Date")
        self.ax_score.set_ylabel("Score", color='cyan')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_acc = self.ax_score.twinx()
        self.ax_acc.set_ylabel("Accuracy (%)", color='lime')
        self.ax_acc.tick_params(axis='y', labelcolor='lime')
        self.ax_acc.set_ylim(0, 105)
        # Embed Figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")
        # Toolbar
        self.toolbar_frame = ctk.CTkFrame(self.plot_container_frame, fg_color="#2b2b2b", corner_radius=0)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        try: # Style toolbar
             self.toolbar.config(background="#2b2b2b")
             for button in self.toolbar.winfo_children(): button.config(background="#2b2b2b", relief=tk.FLAT)
        except Exception: logger.warning("Could not fully style Matplotlib toolbar.")
        # Initial plot state
        self._clear_plot("Select a scenario to view plot")
        logger.debug("Widgets created.")

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
            self.stats_df = pd.DataFrame()
            self._update_scenario_list()
            self._clear_stats_display()
            self._clear_plot("Loading data...")
            self._load_data()
        else: logger.debug("Folder selection cancelled.")

    def _load_data(self):
        """Loads and processes stats data from the selected folder."""
        if not self.stats_folder_path or not self.stats_folder_path.is_dir():
            logger.warning("Load data called with invalid path.")
            messagebox.showerror("Error", "Invalid or no stats folder selected.")
            self._clear_plot("Select a folder")
            return
        logger.info(f"Loading data from: {self.stats_folder_path}")
        self.browse_button.configure(state="disabled", text="Loading...")
        self.update_idletasks()
        try:
            self.stats_df = data_handler.load_stats_data(self.stats_folder_path)
            if self.stats_df.empty:
                logger.warning("Loaded data is empty.")
                messagebox.showwarning("No Data", "No valid KovaaK's stats files found or parsed.")
                self._clear_plot("No data found")
            else:
                logger.info(f"Data loaded successfully. Shape: {self.stats_df.shape}")
                self._clear_plot("Select a scenario")
            self._update_scenario_list()
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            messagebox.showerror("Loading Error", f"An error occurred:\n{e}")
            self.stats_df = pd.DataFrame()
            self._update_scenario_list()
            self._clear_plot("Error loading data")
        finally:
             self.browse_button.configure(state="normal", text="Browse Stats Folder")

    def _update_scenario_list(self):
        """Populates the scenario list using CTk buttons/labels."""
        logger.debug("Updating scenario list...")
        for widget in self.scenario_scroll_frame.winfo_children(): widget.destroy()
        self.scenario_buttons = {}
        if self.stats_df.empty:
            label = ctk.CTkLabel(self.scenario_scroll_frame, text=" No scenarios found", anchor="w")
            label.pack(pady=2, padx=5, fill="x")
            return
        try:
            scenarios = data_handler.get_unique_scenarios(self.stats_df)
            logger.info(f"Found {len(scenarios)} unique scenarios.")
            if scenarios:
                for scenario in scenarios:
                    button = ctk.CTkButton(
                        self.scenario_scroll_frame, text=scenario,
                        command=lambda s=scenario: self._on_scenario_select(s),
                        anchor="w", fg_color="transparent",
                        text_color=("gray10", "gray90"), hover_color=("gray75", "gray25")
                    )
                    button.pack(pady=(1,0), padx=5, fill="x")
                    self.scenario_buttons[scenario] = button
            else:
                 label = ctk.CTkLabel(self.scenario_scroll_frame, text=" No scenarios found", anchor="w")
                 label.pack(pady=2, padx=5, fill="x")
        except Exception as e:
            logger.error(f"Error getting unique scenarios: {e}", exc_info=True)
            label = ctk.CTkLabel(self.scenario_scroll_frame, text=" Error loading scenarios", anchor="w")
            label.pack(pady=2, padx=5, fill="x")

    def _on_scenario_select(self, scenario_name: str):
        """Handles scenario selection."""
        if not scenario_name or self.stats_df.empty: return
        logger.info(f"Scenario selected: {scenario_name}")
        try:
            summary = data_handler.get_scenario_summary(self.stats_df, scenario_name)
            self._update_stats_display(summary)
            time_series_df = data_handler.get_scenario_time_series(self.stats_df, scenario_name)
            self._update_plot(time_series_df, scenario_name)
        except Exception as e:
            logger.error(f"Error handling selection for '{scenario_name}': {e}", exc_info=True)
            messagebox.showerror("Error", f"Error processing scenario '{scenario_name}':\n{e}")
            self._clear_stats_display()
            self._clear_plot(f"Error displaying {scenario_name}")

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
        self.pb_value.configure(text="N/A")
        self.avg_score_value.configure(text="N/A")
        self.avg_acc_value.configure(text="N/A")
        self.runs_value.configure(text="N/A")
        self.last_played_value.configure(text="N/A")
        self.first_played_value.configure(text="N/A")

    def _clear_plot(self, message="No data to display"):
        """Clears the plot area and displays a message."""
        logger.debug(f"Clearing plot. Message: {message}")
        self.ax_score.clear()
        self.ax_acc.clear()
        self.ax_score.set_xlabel("Date")
        self.ax_score.set_ylabel("Score", color='cyan')
        self.ax_acc.set_ylabel("Accuracy (%)", color='lime')
        self.ax_acc.set_ylim(0, 105)
        self.ax_acc.tick_params(axis='y', labelcolor='lime')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')
        self.ax_score.text(0.5, 0.5, message, horizontalalignment='center',
                           verticalalignment='center', transform=self.ax_score.transAxes,
                           fontsize=12, color='gray')
        self.ax_score.grid(False)
        self.ax_acc.grid(False)
        # Ensure secondary y-axis ticks are off when cleared
        self.ax_acc.set_yticks([])
        self.canvas.draw()

    def _update_plot(self, time_series_df: pd.DataFrame, scenario_name: str):
        """Updates the Matplotlib plot with time series data, hiding 0% accuracy."""
        logger.info(f"Updating plot for scenario: {scenario_name}")

        # --- Pre-checks ---
        if time_series_df is None or time_series_df.empty:
            logger.warning("Time series data is empty, cannot plot.")
            self._clear_plot("No run data for this scenario")
            return
        required_cols = ['Timestamp', 'Score']
        if not all(col in time_series_df.columns for col in required_cols):
             logger.warning(f"Time series data missing required columns: {required_cols}")
             self._clear_plot("Data format error")
             return
        if time_series_df['Timestamp'].isnull().all() or time_series_df['Score'].isnull().all():
            logger.warning("Timestamp or Score column contains only null values.")
            self._clear_plot("Invalid run data")
            return

        # --- Clear previous plots ---
        self.ax_score.clear()
        self.ax_acc.clear()

        # --- Plot Score ---
        self.ax_score.plot(time_series_df['Timestamp'], time_series_df['Score'],
                           marker='o', linestyle='-', markersize=4, color='cyan', label='Score')
        self.ax_score.set_ylabel("Score", color='cyan')
        self.ax_score.tick_params(axis='y', labelcolor='cyan')

        # --- Plot Accuracy (if available AND > 0) ---
        plot_accuracy = ('Avg Accuracy' in time_series_df.columns and
                         time_series_df['Avg Accuracy'].notna().any())

        if plot_accuracy:
            # Filter out 0% accuracy values before plotting
            acc_df_filtered = time_series_df[
                (time_series_df['Avg Accuracy'].notna()) & (time_series_df['Avg Accuracy'] > 0)
            ]
            if not acc_df_filtered.empty:
                logger.debug(f"Plotting {len(acc_df_filtered)} non-zero accuracy points.")
                self.ax_acc.plot(acc_df_filtered['Timestamp'], acc_df_filtered['Avg Accuracy'],
                                 marker='x', linestyle='--', markersize=4, color='lime', label='Accuracy (%)')
                self.ax_acc.set_ylabel("Accuracy (%)", color='lime')
                self.ax_acc.tick_params(axis='y', labelcolor='lime')
                self.ax_acc.set_ylim(0, 105)
                self.ax_acc.grid(False)
                plot_accuracy = True # Mark that we actually plotted something
            else:
                logger.info("No accuracy data > 0% found for this scenario.")
                plot_accuracy = False # Set to false if filtering removed all points
        else:
             logger.info("No accuracy data available or all accuracy data is null.")
             plot_accuracy = False


        # If no accuracy was plotted (either not available or all were 0), hide the axis
        if not plot_accuracy:
            self.ax_acc.set_ylabel("")
            self.ax_acc.set_yticks([])


        # --- Formatting ---
        self.ax_score.set_title(f"Progress: {scenario_name}", color="#DCE4EE")
        self.ax_score.set_xlabel("Date")
        self.ax_score.grid(True, linestyle='--', linewidth=0.5)

        date_form = DateFormatter("%Y-%m-%d")
        self.ax_score.xaxis.set_major_formatter(date_form)
        self.fig.autofmt_xdate()

        # --- Legends ---
        # Only add legend if there's something to label
        lines_score, labels_score = self.ax_score.get_legend_handles_labels()
        if plot_accuracy:
            lines_acc, labels_acc = self.ax_acc.get_legend_handles_labels()
            # Place legend on the accuracy axis if it exists, otherwise on score axis
            self.ax_acc.legend(lines_score + lines_acc, labels_score + labels_acc, loc='best', fontsize='small')
        elif lines_score: # Only show score legend if accuracy isn't plotted
             self.ax_score.legend(lines_score, labels_score, loc='best', fontsize='small')
        # Else: no legend if neither score nor accuracy plotted (shouldn't happen with checks)


        # Redraw the canvas
        try:
            self.canvas.draw()
            logger.debug("Canvas redrawn successfully.")
        except Exception as e:
            logger.error(f"Error drawing canvas: {e}", exc_info=True)

# --- Main Execution (in main.py) ---
# if __name__ == "__main__":
#     app = App()
#     app.mainloop()
