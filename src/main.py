# src/main.py
import cProfile
import pstats
import io                 # For capturing pstats output to string/log
from pathlib import Path  # For creating profile directory
import customtkinter as ctk
import logging

# Import the App class from gui.py
try:
    from .gui import App
    # Ensure data_handler logger is configured (it should be when gui imports it)
    from . import data_handler
    # Ensure utils logger is configured (it should be when gui imports it)
    from . import utils
except ImportError:
    # Fallback for running script directly from src directory
    from gui import App
    import data_handler
    import utils

logger = logging.getLogger(__name__) # Get logger configured elsewhere (e.g., data_handler)

# --- Profiling Configuration ---
PROFILE_ENABLED = True # Set to False to disable profiling easily
PROFILE_DIR = Path(__file__).parent.parent / "profiles" # Place profiles dir alongside src/logs
PROFILE_FILENAME = "app_profile.prof"
# Sort options: 'calls', 'cumulative', 'filename', 'pcalls', 'line', 'name', 'nfl', 'stdname', 'time'
PROFILE_SORT_BY = "line"
PROFILE_PRINT_TOP = 30 # How many functions to show in the summary


def run_app_with_profiling():
    """Runs the main application loop wrapped in cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    logger.info(f"--- Profiling enabled. Output will be saved to {PROFILE_DIR / PROFILE_FILENAME} ---")

    app_instance = None # To ensure app reference exists in finally block if needed
    try:
        logger.info("Starting KovaaK's Stats Viewer Application (with profiling)...")
        app_instance = App()
        app_instance.mainloop()
        logger.info("Application closed normally.")

    except Exception as e:
        logger.critical(f"An unhandled exception occurred during app run: {e}", exc_info=True)
        # Optionally show a critical error message to the user here if GUI is still available
        # try:
        #     messagebox.showerror("Critical Error", f"A critical error occurred:\n{e}\nSee logs for details.")
        # except Exception: # Avoid errors if GUI is already gone
        #     pass

    finally:
        # --- Profiling Analysis ---
        profiler.disable()
        logger.info("Profiling finished.")

        try:
            # Create profile directory if needed
            PROFILE_DIR.mkdir(exist_ok=True)
            profile_path = PROFILE_DIR / PROFILE_FILENAME

            # Save the raw stats
            profiler.dump_stats(profile_path)
            logger.info(f"Raw profiling stats saved to: {profile_path}")

            # Print summary stats to log
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(PROFILE_SORT_BY)
            ps.print_stats(PROFILE_PRINT_TOP) # Print top N functions

            logger.info(f"\n--- Profiling Summary (Top {PROFILE_PRINT_TOP} by {PROFILE_SORT_BY}) ---")
            # Log the summary line by line
            for line in s.getvalue().splitlines():
                 logger.info(line.rstrip()) # Use rstrip to remove potential trailing whitespace
            logger.info("--- End Profiling Summary ---")
            print(f"\nProfiling complete. Detailed stats saved to: {profile_path}")
            print(f"Summary (Top {PROFILE_PRINT_TOP} by {PROFILE_SORT_BY}) also logged (check logs/{data_handler.log_file_path.name}).") # Refer to actual log file

        except Exception as profile_e:
             logger.error(f"Error occurred during profiling analysis: {profile_e}", exc_info=True)


if __name__ == "__main__":
    # Ensure logging is set up (it should be by data_handler import)
    if not logging.getLogger().hasHandlers():
         print("Warning: Root logger not configured. Setting up basic config.")
         # Basic fallback if data_handler wasn't imported somehow
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')


    if PROFILE_ENABLED:
        run_app_with_profiling()
    else:
        # Run normally without profiling
        logger.info("Starting KovaaK's Stats Viewer Application...")
        try:
            app = App()
            app.mainloop()
            logger.info("Application closed normally.")
        except Exception as e:
            logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)

