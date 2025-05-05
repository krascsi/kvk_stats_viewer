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
# --- Optimization Imports ---
import concurrent.futures
from functools import partial # To pass arguments to mapped function

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
# No changes needed here for performance based on profiler
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
# No changes needed here for performance based on profiler
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


# --- Content Parser (Optimized - KeyValue Only) ---
# Optimization: Stop reading file once all target keys are found.
def parse_kovaaks_csv_content_kv_only(filepath, target_key_map):
    """
    Parses the content of a KovaaK's CSV file, extracting only specified
    key-value pairs. Stops reading once all keys are found.

    Args:
        filepath (Path): Path to the CSV file.
        target_key_map (dict): Dictionary mapping CSV keys to (output_key, conversion_func).
    """
    filepath = Path(filepath)
    extracted_stats = {'File Path': str(filepath)}
    found_keys = set()
    keys_to_find = set(target_key_map.keys()) # The CSV keys we are looking for

    try:
        with open(filepath, mode='r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                # Optimization: Stop if all keys have been found
                if len(found_keys) == len(keys_to_find):
                    logging.debug(f"KV Parse: All target keys found in '{filepath.name}'. Stopping read early.")
                    break

                if not row or len(row) < 2: continue

                potential_key_part = row[0].strip()

                # Optimization: Only check keys we haven't found yet
                if potential_key_part in keys_to_find and potential_key_part not in found_keys:
                    potential_value_part = row[1].strip()
                    output_key, convert_func = target_key_map[potential_key_part]
                    raw_value = potential_value_part
                    try:
                        converted_value = None
                        if convert_func == str:
                            converted_value = raw_value
                        else:
                            if isinstance(raw_value, str) and '%' in raw_value and convert_func in (float, int):
                                raw_value = raw_value.replace('%', '')
                            converted_value = convert_func(raw_value)

                        if converted_value is not None:
                            extracted_stats[output_key] = converted_value
                            found_keys.add(potential_key_part) # Mark the CSV key as found
                            logging.debug(f"KV Parse: Found '{output_key}'='{converted_value}' from key '{potential_key_part}' in '{filepath.name}'")
                        else:
                            logging.warning(f"KV Parse: Conversion function for key '{potential_key_part}' returned None for value '{raw_value}' in file '{filepath.name}'. Setting to None.")
                            extracted_stats[output_key] = None
                            found_keys.add(potential_key_part)
                    except (ValueError, TypeError) as e:
                        logging.warning(f"KV Parse: Could not convert value '{raw_value}' for key '{potential_key_part}' in file '{filepath.name}'. Error: {e}. Setting to None.")
                        extracted_stats[output_key] = None
                        found_keys.add(potential_key_part)

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return {}
    except Exception as e:
        logging.error(f"Error reading or parsing key-values in file {filepath}: {e}", exc_info=True)
        return {}

    # Log if not all keys were found (optional, for debugging)
    if len(found_keys) < len(keys_to_find):
        missing_keys = keys_to_find - found_keys
        logging.debug(f"KV Parse: Did not find all target keys in '{filepath.name}'. Missing: {missing_keys}")

    return extracted_stats

# --- Combined File Parser (Modified to accept target_key_map) ---
# This function will be called by the parallel workers
def parse_kovaaks_stats_file(filepath, target_key_map):
    """
    Parses filename and content, combining results. Designed for parallel execution.

    Args:
        filepath (Path): Path to the CSV file.
        target_key_map (dict): Dictionary mapping CSV keys to (output_key, conversion_func).
    """
    filepath = Path(filepath)
    logging.debug(f"Worker processing file: {filepath.name}")
    if not filepath.is_file():
        logging.error(f"Provided path is not a file: {filepath}")
        return None

    filename_data = parse_kovaaks_filename(filepath)
    if not filename_data:
        logging.warning(f"Skipping file due to filename parsing error: {filepath.name}")
        return None # Skip this file if filename parsing fails

    # Pass the target_key_map to the content parser
    content_data = parse_kovaaks_csv_content_kv_only(filepath, target_key_map)

    # Combine data, preferring filename data if conflicts exist (like Scenario Name)
    combined_data = {**content_data, **filename_data}

    # Handle potential Scenario Name discrepancy (keep filename version)
    if 'Scenario Name' in combined_data and 'Scenario_Content' in combined_data:
        if combined_data.get('Scenario Name') != combined_data.get('Scenario_Content'):
             logging.debug(f"Filename scenario differs from content scenario in {filepath.name}. Using filename version.")
        # Remove the content version if it exists
        if 'Scenario_Content' in combined_data:
             del combined_data['Scenario_Content'] # Use pop for safety: combined_data.pop('Scenario_Content', None)

    if not content_data:
         logging.warning(f"Content parsing might have failed or returned empty for: {filepath.name}.")
         # Decide if you want to return filename_data only, or None
         # Returning filename_data allows tracking files even if content fails
         # return filename_data
         # Returning combined_data (which might just be filename_data) is also fine
         # Returning None might be too strict if you want partial data

    # Ensure essential keys from filename parsing are present before returning
    if 'Scenario Name' not in combined_data or 'Timestamp' not in combined_data:
        logging.warning(f"Essential filename data missing after parsing {filepath.name}. Skipping.")
        return None

    return combined_data

# --- Data Loading and Aggregation Function (Optimized with Parallel Processing) ---
def load_stats_data(stats_dir: Path | str, max_workers: int | None = None) -> pd.DataFrame:
    """
    Loads all KovaaK's stats CSVs in parallel, aggregates into a DataFrame,
    cleans types, and calculates average accuracy.

    Args:
        stats_dir (Path | str): Directory containing the KovaaK's CSV stat files.
        max_workers (int | None): Maximum number of worker processes to use.
                                   Defaults to the number of CPUs.
    """
    stats_dir = Path(stats_dir)
    if not stats_dir.is_dir():
        logging.error(f"Invalid stats directory provided: {stats_dir}")
        return pd.DataFrame()

    # Define the target keys and conversion functions *once*
    # This avoids redefining it in every call to the parser
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

    all_parsed_data = []
    logging.info(f"Scanning for CSV files in: {stats_dir}")
    csv_files = list(stats_dir.glob('*.csv'))
    logging.info(f"Found {len(csv_files)} CSV files. Starting parallel processing...")

    if not csv_files:
        logging.warning("No CSV files found in the directory.")
        return pd.DataFrame()

    # Use ProcessPoolExecutor for parallel processing
    # Wrap the target function with fixed arguments using partial
    parse_func_with_map = partial(parse_kovaaks_stats_file, target_key_map=target_key_map)

    processed_count = 0
    skipped_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to apply the function to each file path
        # map returns results in the order the tasks were submitted
        results = executor.map(parse_func_with_map, csv_files)

        # Process results as they complete
        for result in results:
            if result: # Check if parsing was successful (didn't return None)
                all_parsed_data.append(result)
                processed_count += 1
            else:
                skipped_count += 1
            # Optional: Log progress periodically
            # total_processed = processed_count + skipped_count
            # if total_processed % 100 == 0:
            #     logging.info(f"Processed {total_processed}/{len(csv_files)} files...")

    logging.info(f"Finished parallel processing. Successfully parsed: {processed_count}, Skipped: {skipped_count}")

    if not all_parsed_data:
        logging.warning("No valid stats data could be parsed from any files.")
        return pd.DataFrame()

    # --- DataFrame Creation and Cleaning (Same as before) ---
    try:
        start_df_time = time.time()
        aggregated_df = pd.DataFrame(all_parsed_data)
        logging.info(f"Successfully created DataFrame with shape: {aggregated_df.shape} (took {time.time() - start_df_time:.2f}s)")
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
    start_clean_time = time.time()
    # 1. Convert Timestamp
    if 'Timestamp' in aggregated_df.columns:
        aggregated_df['Timestamp'] = pd.to_datetime(aggregated_df['Timestamp'], errors='coerce')
        if aggregated_df['Timestamp'].isna().any():
            logging.warning(f"{aggregated_df['Timestamp'].isna().sum()} rows had Timestamp conversion errors.")
    else:
        logging.error("Timestamp column not found after parsing. Cannot proceed.")
        # Attempt to recover if possible, or return empty
        return pd.DataFrame() # Or handle differently

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
            # Check if column actually contains non-null data before converting
            if not aggregated_df[col].isnull().all():
                logging.debug(f"Converting column '{col}' to numeric...")
                # Ensure it's string type before replacing commas if object type
                if aggregated_df[col].dtype == 'object':
                    # Use .loc to avoid SettingWithCopyWarning
                    aggregated_df.loc[:, col] = aggregated_df[col].astype(str).str.replace(',', '', regex=False)

                # Convert to numeric, coercing errors
                aggregated_df.loc[:, col] = pd.to_numeric(aggregated_df[col], errors='coerce')

                # Check how many NaNs were introduced by coercion (optional logging)
                # initial_nas = aggregated_df[col].isnull().sum() # Count NaNs *before* coercion if needed
                final_nas = aggregated_df[col].isna().sum()
                # if final_nas > initial_nas: # Requires careful handling if column had NaNs initially
                #     logging.warning(f"Some values in column '{col}' failed numeric conversion (became NaN).")
            else:
                # If column is all null, ensure it's numeric type (e.g., float)
                 aggregated_df.loc[:, col] = pd.to_numeric(aggregated_df[col], errors='coerce')


    # 3. Calculate 'Avg Accuracy'
    if 'Hit Count' in aggregated_df.columns and 'Miss Count' in aggregated_df.columns:
        logging.info("Calculating 'Avg Accuracy' from 'Hit Count' and 'Miss Count'.")
        # Ensure columns are numeric before calculation
        hits = pd.to_numeric(aggregated_df['Hit Count'], errors='coerce').fillna(0)
        misses = pd.to_numeric(aggregated_df['Miss Count'], errors='coerce').fillna(0)
        total_shots = hits + misses

        # Use np.where for safe division
        aggregated_df['Avg Accuracy'] = np.where(total_shots > 0, (hits / total_shots) * 100.0, 0.0)
        logging.debug(f"Sample calculated 'Avg Accuracy':\n{aggregated_df['Avg Accuracy'].head()}")
    else:
        logging.warning("Cannot calculate 'Avg Accuracy': 'Hit Count' or 'Miss Count' columns missing or not numeric after conversion.")
        # Fallback logic
        if 'Avg Accuracy_Parsed' in aggregated_df.columns and aggregated_df['Avg Accuracy_Parsed'].notna().any():
             logging.info("Using 'Avg Accuracy_Parsed' as fallback.")
             aggregated_df['Avg Accuracy'] = aggregated_df['Avg Accuracy_Parsed']
        elif 'Accuracy_Parsed' in aggregated_df.columns and aggregated_df['Accuracy_Parsed'].notna().any():
             logging.info("Using 'Accuracy_Parsed' as fallback.")
             aggregated_df['Avg Accuracy'] = aggregated_df['Accuracy_Parsed']
        else:
            aggregated_df['Avg Accuracy'] = np.nan # Assign NaN if no source available

    # Drop the intermediate parsed accuracy columns
    aggregated_df.drop(columns=['Accuracy_Parsed', 'Avg Accuracy_Parsed'], errors='ignore', inplace=True)

    logging.debug(f"--- DataFrame AFTER Calculation & Conversion (BEFORE dropna) ---")

    # 4. Drop rows with missing essential Timestamp or Score
    essential_cols = ['Timestamp', 'Score']
    initial_rows = len(aggregated_df)
    # Ensure Score is numeric before checking for NaN
    if 'Score' in aggregated_df.columns:
         aggregated_df.loc[:, 'Score'] = pd.to_numeric(aggregated_df['Score'], errors='coerce')
    else:
         logging.error("'Score' column not found for dropna check.")
         # Handle error - maybe add a dummy NaN column or skip dropna for Score
         essential_cols = ['Timestamp'] # Only drop based on Timestamp

    # Check if essential columns exist before dropping
    cols_to_check = [col for col in essential_cols if col in aggregated_df.columns]
    if cols_to_check:
        aggregated_df.dropna(subset=cols_to_check, inplace=True)
        rows_dropped = initial_rows - len(aggregated_df)
        if rows_dropped > 0:
            logging.warning(f"Dropped {rows_dropped} rows due to missing essential data in columns: {cols_to_check}.")
    else:
        logging.warning("None of the essential columns found for dropna check.")


    if aggregated_df.empty:
         logging.warning("DataFrame is empty after cleaning.")
         return aggregated_df

    # 5. Sort by timestamp
    aggregated_df.sort_values(by='Timestamp', inplace=True)

    logging.info(f"Finished DataFrame cleaning and processing (took {time.time() - start_clean_time:.2f}s)")
    logging.info(f"--- DataFrame AFTER Cleaning ---")
    logging.info(f"Cleaned & Aggregated DataFrame shape: {aggregated_df.shape}")
    logging.info(f"Final DataFrame columns: {aggregated_df.columns.tolist()}")
    # logging.debug(f"Final Dtypes:\n{aggregated_df.dtypes}") # Optional: log final types
    # logging.debug(f"DataFrame head:\n{aggregated_df.head()}") # Optional: log head

    return aggregated_df


# --- Helper Functions (Operating on the final DataFrame) ---
# No changes needed here for performance
def get_scenario_summary(df: pd.DataFrame, scenario_name: str) -> dict | None:
    """Calculates summary statistics for a specific scenario."""
    if df.empty or 'Scenario Name' not in df.columns: return None
    # Use .loc for potentially better performance and clarity
    scenario_df = df.loc[df['Scenario Name'] == scenario_name]
    if scenario_df.empty: return None

    # Ensure Avg Accuracy is numeric before calculating mean
    avg_accuracy_val = None
    if 'Avg Accuracy' in scenario_df.columns:
        numeric_accuracy = pd.to_numeric(scenario_df['Avg Accuracy'], errors='coerce')
        if numeric_accuracy.notna().any():
            avg_accuracy_val = numeric_accuracy.mean()

    # Ensure Score is numeric before max/mean
    pb_score = None
    avg_score = None
    if 'Score' in scenario_df.columns:
        numeric_score = pd.to_numeric(scenario_df['Score'], errors='coerce')
        if numeric_score.notna().any():
            pb_score = numeric_score.max()
            avg_score = numeric_score.mean()

    # Ensure Kills/Damage are numeric
    avg_kills = None
    if 'Kills' in scenario_df.columns:
        numeric_kills = pd.to_numeric(scenario_df['Kills'], errors='coerce')
        if numeric_kills.notna().any():
            avg_kills = numeric_kills.mean()

    avg_damage = None
    if 'Damage Done' in scenario_df.columns:
        numeric_damage = pd.to_numeric(scenario_df['Damage Done'], errors='coerce')
        if numeric_damage.notna().any():
            avg_damage = numeric_damage.mean()

    summary = {
        'Scenario Name': scenario_name,
        'PB Score': pb_score,
        'Average Score': avg_score,
        'Average Accuracy': avg_accuracy_val,
        'Number of Runs': len(scenario_df),
        'Date Last Played': scenario_df['Timestamp'].max() if 'Timestamp' in scenario_df else None,
        'First Played': scenario_df['Timestamp'].min() if 'Timestamp' in scenario_df else None,
        'Avg Kills': avg_kills,
        'Avg Damage Done': avg_damage,
    }

    # Formatting (handle potential None values gracefully)
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
        # Return empty DataFrame with expected columns if input is invalid
        cols = ['Timestamp', 'Score'] + (['Avg Accuracy'] if 'Avg Accuracy' in df.columns else [])
        return pd.DataFrame(columns=cols)

    cols_to_keep = ['Timestamp', 'Score']
    if 'Avg Accuracy' in df.columns: cols_to_keep.append('Avg Accuracy')

    try:
        # Filter using .loc and select columns directly
        time_series_df = df.loc[df['Scenario Name'] == scenario_name, cols_to_keep].copy()
        # Ensure Timestamp is datetime type before sorting
        time_series_df['Timestamp'] = pd.to_datetime(time_series_df['Timestamp'], errors='coerce')
        time_series_df.sort_values(by='Timestamp', inplace=True)
        return time_series_df
    except Exception as e:
        logging.error(f"Error getting time series for '{scenario_name}': {e}")
        # Return empty DataFrame with expected columns on error
        return pd.DataFrame(columns=cols_to_keep)


def get_unique_scenarios(df: pd.DataFrame) -> list[str]:
    """Returns a sorted list of unique scenario names."""
    if df.empty or 'Scenario Name' not in df.columns: return []
    try:
        # Ensure NaN values are handled and result is sorted list of strings
        unique_names = df['Scenario Name'].dropna().unique()
        return sorted([str(name) for name in unique_names])
    except Exception as e:
        logging.error(f"Error getting unique scenarios: {e}")
        return []

# --- Playlist Parsing Function (Returns Ordered List) ---
# No changes needed here for performance
def parse_playlist_json(filepath: Path) -> list[str] | None:
    """
    Parses a KovaaK's JSON playlist file to extract scenario names,
    preserving the order found in the file.
    """
    logger.debug(f"Parsing playlist file: {filepath.name}")
    try:
         # Use utf-8-sig to handle potential BOM (Byte Order Mark)
         with open(filepath, 'r', encoding='utf-8-sig') as f:
              data = json.load(f)

         # Handle different possible top-level structures (list or dict)
         if isinstance(data, list):
             scenario_list = data
         elif isinstance(data, dict):
             scenario_list = data.get("scenarioList", []) # Default to empty list
         else:
             logger.warning(f"Unexpected JSON structure (not list or dict) in {filepath.name}")
             return None

         if not isinstance(scenario_list, list):
              logger.warning(f"Invalid format: 'scenarioList' key exists but is not a list in {filepath.name}")
              return None

         names = [] # Use a list to preserve order
         seen_names = set() # Use a set to track uniqueness efficiently
         for item in scenario_list:
              if isinstance(item, dict):
                   # Kovaak's JSON might use "scenario_name" or "Scenario Name"
                   name = item.get("scenario_name") or item.get("Scenario Name")
                   if isinstance(name, str) and name: # Check if name is a non-empty string
                        cleaned_name = name.strip()
                        if cleaned_name and cleaned_name not in seen_names: # Check uniqueness
                             names.append(cleaned_name)
                             seen_names.add(cleaned_name)
              else:
                  logger.debug(f"Skipping non-dictionary item in scenarioList: {item}")

         logger.debug(f"Extracted {len(names)} unique scenario names (ordered) from {filepath.name}")
         return names # Return the ordered list

    except FileNotFoundError:
         logger.error(f"Playlist file not found: {filepath}")
         return None
    except json.JSONDecodeError as e:
         logger.error(f"Invalid JSON format in {filepath.name}: {e}")
         return None
    except Exception as e:
         logger.error(f"Error reading playlist {filepath.name}: {e}", exc_info=True)
         return None


# --- Example Usage ---
if __name__ == "__main__":
    # Set logging level higher for example run if desired
    logging.getLogger().setLevel(logging.INFO) # Or logging.DEBUG for more detail

    script_dir = Path(__file__).parent
    # --- IMPORTANT: Point this to your ACTUAL KovaaK's stats folder ---
    # Example: stats_path = Path("C:/Program Files (x86)/Steam/steamapps/common/KovaaKs/FPSAimTrainer/stats")
    # Using a relative path for testing:
    stats_path = script_dir / "test_stats" # Make sure this directory exists and has CSVs

    print(f"Looking for test stats in: {stats_path}")

    if stats_path and stats_path.is_dir():
        print(f"\n--- Loading and Processing Data (Parallel) ---")
        start_time = time.time()
        # You can specify max_workers, e.g., max_workers=4
        # If None, it defaults to os.cpu_count()
        main_df = load_stats_data(stats_path, max_workers=None)
        end_time = time.time()
        print(f"Data loading and processing took: {end_time - start_time:.2f} seconds")

        if not main_df.empty:
            print(f"\n--- Sample Data ---")
            print(main_df.head())
            print(f"\nDataFrame Info:")
            main_df.info() # Shows dtypes and non-null counts

            print(f"\n--- Unique Scenarios ---")
            scenarios = get_unique_scenarios(main_df)
            print(f"Found {len(scenarios)} unique scenarios.")
            if scenarios:
                print(f"First 5: {scenarios[:5]}")

                # --- Example: Get Summary for the first scenario found ---
                test_scenario = scenarios[0]
                print(f"\n--- Summary for Scenario: {test_scenario} ---")
                summary = get_scenario_summary(main_df, test_scenario)
                if summary:
                    for key, value in summary.items():
                        print(f"{key}: {value}")
                else:
                    print("Could not generate summary.")

                # --- Example: Get Time Series for the first scenario ---
                print(f"\n--- Time Series for Scenario: {test_scenario} ---")
                time_series = get_scenario_time_series(main_df, test_scenario)
                if not time_series.empty:
                    print(time_series.head())
                else:
                    print("Could not generate time series.")
        else:
            print("\nFailed to load data or data is empty.")

    else:
        print(f"\nError: Could not find or access stats directory: {stats_path}")
        print("Please ensure the path is correct and the directory contains KovaaK's .csv files.")
        # Create dummy test_stats dir for basic execution if needed
        stats_path.mkdir(exist_ok=True)
        print(f"Created directory {stats_path} if it didn't exist. Place test CSVs there.")


    # --- Example Playlist Parsing ---
    print("\n--- Testing Playlist Parsing (Ordered) ---")
    dummy_playlist_path = script_dir / "dummy_playlist_ordered.json"
    # More realistic dummy data structure
    dummy_data = {
        "playlistName": "Test Playlist Ordered",
        "description": "A test playlist",
        "author": "OptimizerBot",
        "version": 1,
        "scenarioList": [
            {"scenario_name": "Smoothsphere", "custom_settings": None},
            {"scenario_name": "Air Invincible 7 Small", "custom_settings": {"timescale": 0.9}},
            {"scenario_name": "centering I", "custom_settings": None},
            {"scenario_name": "Smoothsphere", "custom_settings": None}, # Duplicate
            {"scenario_name": "Centering I 90 no strafes", "custom_settings": None}
        ]
    }
    try:
        with open(dummy_playlist_path, 'w', encoding='utf-8') as f: # Ensure utf-8 encoding
            json.dump(dummy_data, f, indent=2)
        print(f"Created/updated dummy playlist: {dummy_playlist_path}")
        parsed_names = parse_playlist_json(dummy_playlist_path)
        if parsed_names:
             print(f"Parsed scenario names (ordered): {parsed_names}")
             # Expected order should match the unique names in the list
             expected_order = ["Smoothsphere", "Air Invincible 7 Small", "centering I", "Centering I 90 no strafes"]
             # Simple assertion for testing
             if parsed_names == expected_order:
                 print("Parsing test successful (order preserved and duplicates handled).")
             else:
                 print(f"Order mismatch! Expected {expected_order}, got {parsed_names}")
        else:
             print("Failed to parse dummy playlist.")
        # Clean up the dummy file (optional)
        # dummy_playlist_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"Error creating/parsing dummy playlist: {e}")

