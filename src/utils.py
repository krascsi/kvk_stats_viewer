# src/utils.py
import sys
import os
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_default_kovaaks_path() -> Path | None:
    """
    Attempts to find the default KovaaK's stats directory based on OS.

    Returns:
        Path: The Path object to the stats directory if found, otherwise None.
    """
    logging.info("Attempting to find default KovaaK's stats path...")
    default_path = None

    try:
        if sys.platform == "win32":
            # Common Steam installation paths on Windows
            possible_steam_paths = [
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Steam",
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Steam"
            ]
            # Check default libraryfolders.vdf for other library locations
            for steam_path in possible_steam_paths:
                library_folders_vdf = steam_path / "steamapps" / "libraryfolders.vdf"
                possible_install_paths = [steam_path] # Include base steam path

                if library_folders_vdf.exists():
                    logging.info(f"Found libraryfolders.vdf at {library_folders_vdf}")
                    try:
                        with open(library_folders_vdf, 'r') as f:
                            # Very basic parsing: look for lines like '"path"\t\t"X:\\SteamLibrary"'
                            for line in f:
                                if '"path"' in line:
                                    parts = line.split('"')
                                    if len(parts) >= 4:
                                        lib_path = Path(parts[3].replace('\\\\', '\\'))
                                        if lib_path.is_dir():
                                            possible_install_paths.append(lib_path)
                                            logging.info(f"Found potential library path: {lib_path}")
                    except Exception as e:
                        logging.warning(f"Could not parse libraryfolders.vdf: {e}")

                # Check known relative path within each potential Steam library
                for install_path in possible_install_paths:
                    kovaaks_relative_path = Path("steamapps/common/FPSAimTrainer/FPSAimTrainer/stats")
                    potential_path = install_path / kovaaks_relative_path
                    logging.info(f"Checking potential path: {potential_path}")
                    if potential_path.is_dir():
                        default_path = potential_path
                        logging.info(f"Found KovaaK's stats path at: {default_path}")
                        return default_path

        elif sys.platform == "darwin": # macOS
            # Common Steam path on macOS
            home = Path.home()
            potential_path = home / "Library/Application Support/Steam/steamapps/common/FPSAimTrainer/FPSAimTrainer/stats"
            logging.info(f"Checking potential path: {potential_path}")
            if potential_path.is_dir():
                default_path = potential_path
                logging.info(f"Found KovaaK's stats path at: {default_path}")
                return default_path

        elif sys.platform.startswith("linux"):
            # Common Steam paths on Linux
            home = Path.home()
            possible_paths = [
                home / ".steam/steam/steamapps/common/FPSAimTrainer/FPSAimTrainer/stats",
                home / ".local/share/Steam/steamapps/common/FPSAimTrainer/FPSAimTrainer/stats"
            ]
            for potential_path in possible_paths:
                 logging.info(f"Checking potential path: {potential_path}")
                 if potential_path.is_dir():
                     default_path = potential_path
                     logging.info(f"Found KovaaK's stats path at: {default_path}")
                     return default_path

    except Exception as e:
        logging.error(f"Error finding default KovaaK's path: {e}")
        return None # Return None on any unexpected error

    if default_path is None:
        logging.warning("Could not automatically find KovaaK's stats path.")

    return default_path

# Example usage (for testing)
if __name__ == "__main__":
    found_path = find_default_kovaaks_path()
    if found_path:
        print(f"Default KovaaK's path found: {found_path}")
    else:
        print("Default KovaaK's path not found. Manual selection required.")
# src/utils.py
import sys
import os
from pathlib import Path
import logging

# Configure basic logging (can be overridden by main app)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Use the logger configured in data_handler or main app
logger = logging.getLogger(__name__)


def find_default_kovaaks_path() -> Path | None:
    """
    Attempts to find the default KovaaK's stats directory based on OS.
    Uses more robust Steam library folder parsing.

    Returns:
        Path: The Path object to the stats directory if found, otherwise None.
    """
    logger.info("Attempting to find default KovaaK's stats path...")
    default_path = None
    possible_install_paths = set() # Use a set to avoid duplicates

    try:
        if sys.platform == "win32":
            # Common Steam installation paths on Windows
            program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))

            possible_steam_roots = [
                program_files_x86 / "Steam",
                program_files / "Steam"
            ]

            for steam_root in possible_steam_roots:
                if steam_root.is_dir():
                    possible_install_paths.add(steam_root) # Add the root itself
                    # Check libraryfolders.vdf for other library locations
                    library_folders_vdf = steam_root / "steamapps" / "libraryfolders.vdf"
                    if library_folders_vdf.is_file():
                        logger.debug(f"Found libraryfolders.vdf at {library_folders_vdf}")
                        try:
                            with open(library_folders_vdf, 'r') as f:
                                # More robust parsing looking for lines like '"path"\t\t"X:\\SteamLibrary"'
                                for line in f:
                                    line = line.strip()
                                    if line.startswith('"path"'):
                                        parts = line.split('"')
                                        if len(parts) >= 4:
                                            # Path is usually the 4th element (index 3)
                                            lib_path_str = parts[3].replace('\\\\', '\\')
                                            lib_path = Path(lib_path_str)
                                            if lib_path.is_dir():
                                                possible_install_paths.add(lib_path)
                                                logger.debug(f"Found potential library path: {lib_path}")
                        except Exception as e:
                            logger.warning(f"Could not parse libraryfolders.vdf: {e}")

        elif sys.platform == "darwin": # macOS
            home = Path.home()
            # Common Steam library locations on macOS
            mac_lib_paths = [
                home / "Library/Application Support/Steam",
                # Add other common locations if known
            ]
            for lib_path in mac_lib_paths:
                 if lib_path.is_dir():
                      possible_install_paths.add(lib_path)

        elif sys.platform.startswith("linux"):
            home = Path.home()
            # Common Steam library locations on Linux
            linux_lib_paths = [
                home / ".steam/steam",
                home / ".local/share/Steam",
                home / ".var/app/com.valvesoftware.Steam/.local/share/Steam" # Flatpak
                # Add other common locations if known
            ]
            for lib_path in linux_lib_paths:
                 if lib_path.is_dir():
                      possible_install_paths.add(lib_path)
                      # Also check libraryfolders.vdf on Linux/Mac
                      library_folders_vdf = lib_path / "steamapps" / "libraryfolders.vdf"
                      if library_folders_vdf.is_file():
                          logger.debug(f"Found libraryfolders.vdf at {library_folders_vdf}")
                          try:
                              with open(library_folders_vdf, 'r') as f:
                                  for line in f:
                                      line = line.strip()
                                      if line.startswith('"path"'):
                                          parts = line.split('"')
                                          if len(parts) >= 4:
                                              lib_path_str = parts[3] # No need for replace on Linux/Mac usually
                                              lib_path_extra = Path(lib_path_str)
                                              if lib_path_extra.is_dir():
                                                  possible_install_paths.add(lib_path_extra)
                                                  logger.debug(f"Found potential library path: {lib_path_extra}")
                          except Exception as e:
                              logger.warning(f"Could not parse libraryfolders.vdf: {e}")


        # Check known relative path within each potential Steam library
        kovaaks_relative_path = Path("steamapps/common/FPSAimTrainer/FPSAimTrainer/stats")
        logger.debug(f"Checking relative path: {kovaaks_relative_path} in {len(possible_install_paths)} locations.")

        for install_path in possible_install_paths:
            potential_path = install_path / kovaaks_relative_path
            logger.debug(f"Checking potential path: {potential_path}")
            if potential_path.is_dir():
                default_path = potential_path
                logger.info(f"Found KovaaK's stats path at: {default_path}")
                return default_path # Return the first one found

    except Exception as e:
        logger.error(f"Error finding default KovaaK's path: {e}", exc_info=True)
        return None # Return None on any unexpected error

    if default_path is None:
        logger.warning("Could not automatically find KovaaK's stats path.")

    return default_path

# Example usage (for testing this file directly)
if __name__ == "__main__":
     # Setup basic logging just for this test run
     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
     found_path = find_default_kovaaks_path()
     if found_path:
         print(f"\nDefault KovaaK's path found: {found_path}")
     else:
         print("\nDefault KovaaK's path not found. Manual selection will be required.")
