# src/main.py
import customtkinter as ctk
import logging

# Import the App class from gui.py
# Use absolute imports relative to the package root if running as a package later
try:
    from .gui import App # Use relative import if main.py is in the same package as gui.py
    # Ensure data_handler logger is configured (it should be when gui imports it)
    from . import data_handler
except ImportError:
    # Fallback for running script directly from src directory
    from gui import App
    import data_handler # Ensure logger is configured

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Basic check to ensure logging is set up (it should be by data_handler)
    if not logging.getLogger().hasHandlers():
         print("Warning: Root logger not configured. Setting up basic config.")
         logging.basicConfig(level=logging.INFO) # Basic fallback

    logger.info("Starting KovaaK's Stats Viewer Application...")
    try:
        app = App()
        app.mainloop()
        logger.info("Application closed normally.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        # Optionally show a critical error message to the user here
        # messagebox.showerror("Critical Error", f"A critical error occurred:\n{e}\nSee logs for details.")

