# src/data_handler.py
import csv
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time # For parsing time strings like HH:MM:SS.fff
import os # For creating log directory
import json # For parsing playlist files

# --- Configure Logging ---
# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs" # Place logs dir outside src
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / "data_handler.log"

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Set root logger level to DEBUG

# Clear existing handlers (important if re-running in same session)
if logger.hasHandlers():
    logger.handlers.clear()

# Create console handler and set level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Create file handler and set level to DEBUG
fh = logging.FileHandler(log_file_path, mode='a') # Append mode
fh.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

# --- Filename Parser (User Provided - Corrected) ---
def parse_kovaaks_filename(filename):
    """
    Parses the KovaaK's CSV filename to extract scenario, mode, and timestamp.
    """
    pattern = re.compile(
        r"^(?P<scenario>.+?)"
        r"(?:\s*-\s*(?P<mode>[^-]+?))?"
        r"\s*-\s*"
        r"(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})"
        r"-"
        r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})"
        r"\s+Stats\.csv$",
        re.IGNORECASE
    )
    file_name_only = Path(filename).name
    match = pattern.match(file_name_only)

    if match:
        data = match.groupdict()
        try:
            timestamp = datetime(
                int(data['year']), int(data['month']), int(data['day']),
                int(data['hour']), int(data['minute']), int(data['second'])
            )
            scenario_name = data['scenario'].strip()
            mode = data.get('mode')
            mode = mode.strip() if mode else None
            logging.debug(f"Filename parsed (standard): Scenario='{scenario_name}', Mode='{mode}', Timestamp='{timestamp}'")
            return {'Scenario Name': scenario_name, 'Mode': mode, 'Timestamp': timestamp}
        except ValueError as e:
            logging.warning(f"Could not parse timestamp from filename '{file_name_only}': {e}")
            return None
    else:
        simple_pattern = re.compile(
            r"^(?P<scenario>.+?)"
            r"\s*-\s*"
            r"(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})"
            r"-"
            r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})"
            r"\s+Stats\.csv$",
            re.IGNORECASE
        )
        match = simple_pattern.match(file_name_only)
        if match:
            data = match.groupdict()
            try:
                timestamp = datetime(
                    int(data['year']), int(data['month']), int(data['day']),
                    int(data['hour']), int(data['minute']), int(data['second'])
                )
                scenario_name = data['scenario'].strip()
                logging.debug(f"Filename parsed (simple): Scenario='{scenario_name}', Mode='None', Timestamp='{timestamp}'")
                return {'Scenario Name': scenario_name, 'Mode': None, 'Timestamp': timestamp}
            except ValueError as e:
                logging.warning(f"Could not parse timestamp from filename '{file_name_only}' (simple pattern): {e}")
                return None
        else:
            logging.warning(f"Filename '{file_name_only}' did not match expected KovaaK's pattern(s).")
            return None

# --- Helper function to parse time string HH:MM:SS.fff (User Provided) ---
def _parse_challenge_start_time(time_str):
    """Parses time string like '14:18:41.799' into seconds."""
    logging.debug(f"Attempting to parse time string: '{time_str}' (Type: {type(time_str)})")
    if not isinstance(time_str, str) or not time_str.strip():
        logging.warning(f"Invalid input for time parsing: Input is not a non-empty string.")
        return None
    time_str = time_str.strip()
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h_str, m_str, s_full_str = parts[0], parts[1], parts[2]
            if not (h_str.isdigit() and m_str.isdigit()): raise ValueError("Hour or minute part is not purely digits.")
            h, m = int(h_str), int(m_str)
            s_parts = s_full_str.split('.')
            s_str = s_parts[0]; ms_str = s_parts[1] if len(s_parts) > 1 else '0'
            if not (s_str.isdigit() and ms_str.isdigit()): raise ValueError("Second or millisecond part is not purely digits.")
            s = int(s_str); ms = int(ms_str.ljust(3, '0')[:3])
            total_seconds = h * 3600 + m * 60 + s + ms / 1000.0
            logging.debug(f"Parsed '{time_str}' as {total_seconds} seconds.")
            return total_seconds
        else:
            logging.debug(f"Time string '{time_str}' is not HH:MM:SS format, attempting float conversion.")
            try:
                total_seconds = float(time_str)
                logging.debug(f"Parsed '{time_str}' directly as float: {total_seconds} seconds.")
                return total_seconds
            except (ValueError, TypeError) as float_e:
                 raise ValueError(f"Input is not HH:MM:SS format and also failed float conversion: {float_e}")
    except (ValueError, TypeError, IndexError, AttributeError) as e:
        logging.warning(f"Could not parse time string '{time_str}'. Reason: {e}")
        return None


# --- Content Parser (User Provided - KeyValue Only) ---
def parse_kovaaks_csv_content_kv_only(filepath):
    """
    Parses the content of a KovaaK's CSV file, extracting only specified
    key-value pairs.
    """
    filepath = Path(filepath)
    extracted_stats = {'File Path': str(filepath)}
    target_key_map = {
        'Score:': ('Score', float), 'Hit Count:': ('Hit Count', int),
        'Miss Count:': ('Miss Count', int), 'Kills:': ('Kills', int),
        'Deaths:': ('Deaths', int), 'Damage Done:': ('Damage Done', float),
        'Damage Taken:': ('Damage Taken', float), 'Reloads:': ('Reloads', int),
        'Accuracy:': ('Accuracy_Parsed', float), 'Avg Accuracy:': ('Avg Accuracy_Parsed', float),
        'Fight Time:': ('Fight Time', float), 'Time Remaining:': ('Time Remaining', float),
        'Avg TTK:': ('Avg TTK', float), 'Challenge Start:': ('Challenge Start Time', _parse_challenge_start_time),
        'Pause Count:': ('Pause Count', int), 'Pause Duration:': ('Pause Duration', float),
        'Distance Traveled:': ('Distance Traveled', float), 'Midairs:': ('Midairs', int),
        'Midaired:': ('Midaired', int), 'Directs:': ('Directs', int),
        'Directed:': ('Directed', int), 'Total Overshots:': ('Total Overshots', int),
        'Scenario:': ('Scenario_Content', str), 'Hash:': ('Hash', str),
        'Game Version:': ('Game Version', str), 'Avg Target Scale:': ('Avg Target Scale', float),
        'Avg Time Dilation:': ('Avg Time Dilation', float), 'Sens Scale:': ('Sens Scale', str),
        'Sens Increment:': ('Sens Increment', float), 'Horiz Sens:': ('Horiz Sens', float),
        'Vert Sens:': ('Vert Sens', float), 'DPI:': ('DPI', int), 'FOV:': ('FOV', float),
        'FOVScale:': ('FOV Scale', str), 'Hide Gun:': ('Hide Gun', str),
        'Crosshair:': ('Crosshair', str), 'Crosshair Scale:': ('Crosshair Scale', float),
        'Crosshair Color:': ('Crosshair Color', str), 'Resolution:': ('Resolution', str),
        'Resolution Scale:': ('Resolution Scale', float), 'Avg FPS:': ('Avg FPS', float),
        'Max FPS (config):': ('Max FPS Config', float), 'Input Lag:': ('Input Lag', float),
        'MBS Points:': ('MBS Points', float),
    }
    found_keys = set()

    try:
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                if not row or len(row) < 2: continue
                potential_key_part = row[0].strip()
                potential_value_part = row[1].strip()
                matched = False
                for csv_key, (output_key, convert_func) in target_key_map.items():
                    if output_key not in found_keys and potential_key_part == csv_key:
                        raw_value = potential_value_part
                        try:
                            converted_value = None
                            if convert_func == str: converted_value = raw_value
                            else:
                                if isinstance(raw_value, str) and '%' in raw_value and convert_func in (float, int):
                                    raw_value = raw_value.replace('%', '')
                                converted_value = convert_func(raw_value)

                            if converted_value is not None:
                                extracted_stats[output_key] = converted_value
                                found_keys.add(output_key)
                                logging.debug(f"KV Parse: Found '{output_key}'='{converted_value}' from key '{csv_key}' in '{filepath.name}'")
                            else:
                                logging.warning(f"KV Parse: Conversion function for key '{csv_key}' returned None for value '{raw_value}' in file '{filepath.name}'. Setting to None.")
                                extracted_stats[output_key] = None
                                found_keys.add(output_key)
                        except (ValueError, TypeError) as e:
                            logging.warning(f"KV Parse: Could not convert value '{raw_value}' for key '{csv_key}' in file '{filepath.name}'. Error: {e}. Setting to None.")
                            extracted_stats[output_key] = None
                            found_keys.add(output_key)
                        matched = True
                        break
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return {}
    except Exception as e:
        logging.error(f"Error reading or parsing key-values in file {filepath}: {e}", exc_info=True)
        return {}
    return extracted_stats

# --- Combined File Parser (User Provided) ---
def parse_kovaaks_stats_file(filepath):
    """Parses filename and content, combining results."""
    filepath = Path(filepath)
    if not filepath.is_file():
        logging.error(f"Provided path is not a file: {filepath}")
        return None
    filename_data = parse_kovaaks_filename(filepath)
    if not filename_data: return None
    content_data = parse_kovaaks_csv_content_kv_only(filepath)
    combined_data = {**content_data, **filename_data}
    if 'Scenario Name' in combined_data and 'Scenario_Content' in combined_data:
        if combined_data.get('Scenario Name') != combined_data.get('Scenario_Content'):
             logging.debug(f"Filename scenario differs from content scenario in {filepath.name}. Using filename version.")
        if 'Scenario_Content' in combined_data:
             del combined_data['Scenario_Content']
    if not content_data:
         logging.warning(f"Content parsing might have failed for: {filepath.name}.")
    return combined_data

# --- Data Loading and Aggregation Function (Integrated) ---
def load_stats_data(stats_dir: Path | str) -> pd.DataFrame:
    """
    Loads all KovaaK's stats CSVs, aggregates into a DataFrame, cleans types,
    and calculates average accuracy.
    """
    stats_dir = Path(stats_dir)
    if not stats_dir.is_dir():
        logging.error(f"Invalid stats directory provided: {stats_dir}")
        return pd.DataFrame()

    all_parsed_data = []
    logging.info(f"Scanning for CSV files in: {stats_dir}")
    csv_files = list(stats_dir.glob('*.csv'))
    logging.info(f"Found {len(csv_files)} CSV files.")

    processed_count = 0; skipped_count = 0
    for file_path in csv_files:
        logging.debug(f"Processing file: {file_path.name}")
        parsed_data = parse_kovaaks_stats_file(file_path)
        if parsed_data:
            all_parsed_data.append(parsed_data)
            processed_count += 1
        else: skipped_count += 1

    logging.info(f"Finished processing files. Successfully parsed: {processed_count}, Skipped: {skipped_count}")
    if not all_parsed_data:
        logging.warning("No valid stats data could be parsed from any files.")
        return pd.DataFrame()

    try:
        aggregated_df = pd.DataFrame(all_parsed_data)
        logging.info(f"Successfully created DataFrame with shape: {aggregated_df.shape}")
        logging.debug(f"DataFrame columns: {aggregated_df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to create DataFrame from parsed data: {e}", exc_info=True)
        return pd.DataFrame()
    if aggregated_df.empty:
        logging.warning("DataFrame is empty after creation.")
        return aggregated_df

    logging.debug(f"--- DataFrame BEFORE Cleaning ---")
    logging.debug(f"Dtypes BEFORE cleaning:\n{aggregated_df.dtypes}")

    logging.info("Starting DataFrame cleaning and type conversion...")
    # 1. Convert Timestamp
    if 'Timestamp' in aggregated_df.columns:
        aggregated_df['Timestamp'] = pd.to_datetime(aggregated_df['Timestamp'], errors='coerce')
        if aggregated_df['Timestamp'].isna().any():
            logging.warning(f"{aggregated_df['Timestamp'].isna().sum()} rows had Timestamp conversion errors.")
    else:
        logging.error("Timestamp column not found. Cannot proceed.")
        return pd.DataFrame()

    # 2. Convert potential numeric columns
    numeric_cols = [
        'Score', 'Hit Count', 'Miss Count', 'Kills', 'Deaths', 'Fight Time',
        'Time Remaining', 'Avg TTK', 'Damage Done', 'Damage Taken', 'Reloads',
        'Distance Traveled', 'Midairs', 'Midaired', 'Directs', 'Directed',
        'Total Overshots', 'MBS Points', 'Challenge Start Time', 'Pause Count',
        'Pause Duration', 'Avg Target Scale', 'Avg Time Dilation', 'Input Lag',
        'Max FPS Config', 'Sens Increment', 'Horiz Sens', 'Vert Sens', 'DPI', 'FOV',
        'Crosshair Scale', 'Avg FPS', 'Resolution Scale',
        'Accuracy_Parsed', 'Avg Accuracy_Parsed'
    ]
    for col in numeric_cols:
        if col in aggregated_df.columns:
            if not aggregated_df[col].isnull().all():
                logging.debug(f"Converting column '{col}' to numeric...")
                if aggregated_df[col].dtype == 'object':
                    str_col = aggregated_df[col].astype(str).str.replace(',', '', regex=False)
                    aggregated_df[col] = str_col
                aggregated_df[col] = pd.to_numeric(aggregated_df[col], errors='coerce')
                if aggregated_df[col].isna().any():
                     failed_count = aggregated_df[col].isna().sum() - aggregated_df[col].isnull().sum()
                     if failed_count > 0:
                          logging.warning(f"Approx {failed_count} values in column '{col}' failed numeric conversion.")
            else: aggregated_df[col] = pd.to_numeric(aggregated_df[col], errors='coerce')

    # 3. Calculate 'Avg Accuracy'
    if 'Hit Count' in aggregated_df.columns and 'Miss Count' in aggregated_df.columns:
        logging.info("Calculating 'Avg Accuracy' from 'Hit Count' and 'Miss Count'.")
        hits = aggregated_df['Hit Count'].fillna(0)
        misses = aggregated_df['Miss Count'].fillna(0)
        total_shots = hits + misses
        if pd.api.types.is_numeric_dtype(hits) and pd.api.types.is_numeric_dtype(misses):
            aggregated_df['Avg Accuracy'] = np.where(total_shots > 0, (hits / total_shots) * 100, 0.0)
            logging.debug(f"Sample calculated 'Avg Accuracy':\n{aggregated_df['Avg Accuracy'].head()}")
        else:
            logging.error("Cannot calculate 'Avg Accuracy': Hits/Misses not numeric.")
            aggregated_df['Avg Accuracy'] = np.nan
    else:
        logging.warning("Cannot calculate 'Avg Accuracy': Hits/Misses columns missing.")
        if 'Avg Accuracy_Parsed' in aggregated_df.columns and aggregated_df['Avg Accuracy_Parsed'].notna().any():
             logging.info("Using 'Avg Accuracy_Parsed' as fallback.")
             aggregated_df['Avg Accuracy'] = aggregated_df['Avg Accuracy_Parsed']
        elif 'Accuracy_Parsed' in aggregated_df.columns and aggregated_df['Accuracy_Parsed'].notna().any():
             logging.info("Using 'Accuracy_Parsed' as fallback.")
             aggregated_df['Avg Accuracy'] = aggregated_df['Accuracy_Parsed']
        else: aggregated_df['Avg Accuracy'] = np.nan

    aggregated_df.drop(columns=['Accuracy_Parsed', 'Avg Accuracy_Parsed'], errors='ignore', inplace=True)

    logging.debug(f"--- DataFrame AFTER Calculation & Conversion (BEFORE dropna) ---")

    # 4. Drop rows with missing essential Timestamp or Score
    essential_cols = ['Timestamp', 'Score']
    original_rows = len(aggregated_df)
    aggregated_df.dropna(subset=essential_cols, inplace=True)
    rows_dropped = original_rows - len(aggregated_df)
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} rows due to missing essential Timestamp or Score.")

    if aggregated_df.empty:
         logging.warning("DataFrame is empty after cleaning.")
         return aggregated_df

    # 5. Sort by timestamp
    aggregated_df.sort_values(by='Timestamp', inplace=True)

    logging.info(f"--- DataFrame AFTER Cleaning ---")
    logging.info(f"Cleaned & Aggregated DataFrame shape: {aggregated_df.shape}")
    logging.info(f"Final DataFrame columns: {aggregated_df.columns.tolist()}")
    return aggregated_df


# --- Helper Functions (Operating on the final DataFrame) ---
def get_scenario_summary(df: pd.DataFrame, scenario_name: str) -> dict | None:
    """Calculates summary statistics for a specific scenario."""
    if df.empty or 'Scenario Name' not in df.columns: return None
    scenario_df = df.loc[df['Scenario Name'] == scenario_name]
    if scenario_df.empty: return None

    avg_accuracy_val = scenario_df['Avg Accuracy'].mean() if 'Avg Accuracy' in scenario_df and scenario_df['Avg Accuracy'].notna().any() else None

    summary = {
        'Scenario Name': scenario_name,
        'PB Score': scenario_df['Score'].max() if 'Score' in scenario_df else None,
        'Average Score': scenario_df['Score'].mean() if 'Score' in scenario_df else None,
        'Average Accuracy': avg_accuracy_val,
        'Number of Runs': len(scenario_df),
        'Date Last Played': scenario_df['Timestamp'].max() if 'Timestamp' in scenario_df else None,
        'First Played': scenario_df['Timestamp'].min() if 'Timestamp' in scenario_df else None,
        'Avg Kills': scenario_df['Kills'].mean() if 'Kills' in scenario_df and scenario_df['Kills'].notna().any() else None,
        'Avg Damage Done': scenario_df['Damage Done'].mean() if 'Damage Done' in scenario_df and scenario_df['Damage Done'].notna().any() else None,
    }

    # Formatting
    summary['PB Score'] = f"{summary['PB Score']:.2f}" if pd.notna(summary.get('PB Score')) else 'N/A'
    summary['Average Score'] = f"{summary['Average Score']:.2f}" if pd.notna(summary.get('Average Score')) else 'N/A'
    summary['Average Accuracy'] = f"{summary['Average Accuracy']:.2f}%" if pd.notna(summary.get('Average Accuracy')) else 'N/A'
    summary['Date Last Played'] = summary['Date Last Played'].strftime('%Y-%m-%d %H:%M') if pd.notna(summary.get('Date Last Played')) else 'N/A'
    summary['First Played'] = summary['First Played'].strftime('%Y-%m-%d %H:%M') if pd.notna(summary.get('First Played')) else 'N/A'
    summary['Avg Kills'] = f"{summary['Avg Kills']:.1f}" if pd.notna(summary.get('Avg Kills')) else 'N/A'
    summary['Avg Damage Done'] = f"{summary['Avg Damage Done']:.0f}" if pd.notna(summary.get('Avg Damage Done')) else 'N/A'
    return summary

def get_scenario_time_series(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Extracts time-series data (Timestamp, Score, Avg Accuracy)."""
    required_cols = ['Timestamp', 'Score', 'Scenario Name']
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=['Timestamp', 'Score', 'Avg Accuracy'])
    cols_to_keep = ['Timestamp', 'Score']
    if 'Avg Accuracy' in df.columns: cols_to_keep.append('Avg Accuracy')
    try:
        time_series_df = df.loc[df['Scenario Name'] == scenario_name, cols_to_keep].copy()
        time_series_df.sort_values(by='Timestamp', inplace=True)
        return time_series_df
    except Exception as e:
        logging.error(f"Error getting time series for '{scenario_name}': {e}")
        return pd.DataFrame(columns=cols_to_keep)

def get_unique_scenarios(df: pd.DataFrame) -> list[str]:
    """Returns a sorted list of unique scenario names."""
    if df.empty or 'Scenario Name' not in df.columns: return []
    try: return sorted(list(df['Scenario Name'].dropna().unique()))
    except Exception as e:
        logging.error(f"Error getting unique scenarios: {e}")
        return []

# --- Playlist Parsing Function (Corrected Key) ---
def parse_playlist_json(filepath: Path) -> set[str] | None:
    """
    Parses a KovaaK's JSON playlist file to extract scenario names.
    Looks for "scenario_name" key within the scenarioList items.

    Args:
        filepath (Path): Path object for the JSON file.

    Returns:
        set[str] | None: A set of unique scenario names found in the playlist,
                         or None if parsing fails or the file is invalid.
    """
    logger.debug(f"Parsing playlist file: {filepath.name}")
    try:
         # Use utf-8-sig to handle potential BOM (Byte Order Mark)
         with open(filepath, 'r', encoding='utf-8-sig') as f:
              data = json.load(f)

         # Handle top-level list or dict containing "scenarioList"
         if isinstance(data, list):
             scenario_list = data
         elif isinstance(data, dict):
             scenario_list = data.get("scenarioList", [])
         else:
             logger.warning(f"Unexpected JSON structure (not list or dict) in {filepath.name}")
             return None

         if not isinstance(scenario_list, list):
              logger.warning(f"Invalid format: 'scenarioList' is not a list in {filepath.name}")
              return None

         names = set()
         for item in scenario_list:
              if isinstance(item, dict):
                   # --- Use the correct key: "scenario_name" ---
                   name = item.get("scenario_name") # Changed from "scenarioName"
                   # --- End Change ---
                   if isinstance(name, str) and name:
                        names.add(name.strip()) # Add non-empty names
              else:
                   logger.debug(f"Skipping non-dictionary item in scenarioList: {item}")

         logger.debug(f"Extracted {len(names)} unique scenario names from {filepath.name}")
         return names

    except FileNotFoundError:
         logger.error(f"Playlist file not found: {filepath}")
         return None
    except json.JSONDecodeError as e:
         logger.error(f"Invalid JSON format in {filepath.name}: {e}")
         return None
    except Exception as e:
         logger.error(f"Error reading playlist {filepath.name}: {e}", exc_info=True)
         return None


# --- Example Usage (Updated to use load_stats_data) ---
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    script_dir = Path(__file__).parent
    stats_path = script_dir / "test_stats"
    print(f"Looking for test stats in: {stats_path}")

    if stats_path and stats_path.is_dir():
        print(f"\n--- Loading and Processing Data ---")
        main_df = load_stats_data(stats_path)

        if not main_df.empty:
            print("\n--- Data Loading and Cleaning Successful ---")
            print(f"Final DataFrame Shape: {main_df.shape}")
            logging.info(f"Final Dtypes:\n{main_df.dtypes}")

            scenarios = get_unique_scenarios(main_df)
            print(f"\nFound {len(scenarios)} unique scenarios.")

            if scenarios:
                print("\n--- Summaries (Sample) ---")
                for test_scenario in scenarios[:5]:
                     print(f"\n--- Scenario: {test_scenario} ---")
                     summary = get_scenario_summary(main_df, test_scenario)
                     if summary:
                         for key, value in summary.items(): print(f"  {key}: {value}")
                     else: print(f"  Could not generate summary.")
                if len(scenarios) > 5: print("\n  ...")

                print(f"\n--- Time Series Data Sample (First Scenario) ---")
                if scenarios:
                    first_scenario = scenarios[0]
                    time_series = get_scenario_time_series(main_df, first_scenario)
                    if not time_series.empty:
                        print(f"Scenario: {first_scenario}")
                        print(f"Shape: {time_series.shape}")
                        print(f"Columns: {time_series.columns.tolist()}")
                        print(time_series.head())
                    else: print(f"Could not get time series data.")
                else: print("No scenarios available.")
            else: print("\nNo scenarios found in the final data.")
        else: print("\n--- Data Loading Failed or Resulted in Empty DataFrame ---")
    else: print(f"\nError: Could not find or access stats directory: {stats_path}")

    # --- Example Playlist Parsing ---
    print("\n--- Testing Playlist Parsing ---")
    # Create a dummy playlist file for testing if needed
    dummy_playlist_path = script_dir / "dummy_playlist_fixed.json" # New name
    dummy_data = {
        "playlistName": "Test Playlist Fixed",
        "scenarioList": [
            {"scenario_name": "1w2ts Angelic"}, # Using snake_case key
            {"scenario_name": "VT Popcorn Advanced S5"},
            {"scenario_name": "NonExistentScenario"},
            {"scenario_name": " cloverRawControl "}
        ]
    }
    try:
        with open(dummy_playlist_path, 'w') as f:
             json.dump(dummy_data, f, indent=2)
        print(f"Created dummy playlist: {dummy_playlist_path}")
        parsed_names = parse_playlist_json(dummy_playlist_path)
        if parsed_names:
             print(f"Parsed scenario names: {parsed_names}")
             # Check if expected names are present
             assert "1w2ts Angelic" in parsed_names
             assert "cloverRawControl" in parsed_names
             print("Parsing test successful.")
        else:
             print("Failed to parse dummy playlist.")
        # Clean up dummy file
        # dummy_playlist_path.unlink() # Uncomment to delete after test
    except Exception as e:
        print(f"Error creating/parsing dummy playlist: {e}")

