import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re
import argparse
import seaborn as sns
import json
import sys

warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default values.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file '{config_path}': {e}")
        sys.exit(1)

def get_default_config_path():
    """Get the default config file path based on the script name"""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    return os.path.join(script_dir, f"{script_name}.json")

parser = argparse.ArgumentParser(description="Calculates various statistics in the movement, angle, and position categories for locust arena behavioral experiments."
                                             " Based off of tracking output from DeepLabCut.")

# Required Arguments
parser.add_argument("data_directory", type=str, help="The path to the directory containing the .h5 files with tracking coordinates for behavioral videos from a DeepLabCut analysis.")
parser.add_argument("results_folder", type=str, help="The desired path for the result files", default="results/")

# Optional Arguments
parser.add_argument(
        "-s", "--stimulus",
        choices=["l", "r", "left", "right"],
        type=str.lower,  # Makes input case insensitive
        default="right",
        help="Which side of the arena the stimulus cage is on. Accepts one of 'l', 'r', 'left', or 'right' (case insensitive)"
    )
parser.add_argument("-r", "--regex", type=str,
                    help="The specifed regex pattern will be used to select trial numbers from your data names. For example:\n"
                    r"Trial\s+(\d+)"
                    r" Will find the string 'Trial' along with a unlimited number of whitespaces '\s+' and then match the following group of digits '(\d+)'"
                    "\nso Trial     68.mp4 would have the number 68 selected."
                    " That parentheses-selected group will be used to determine naming for your results.",
                    default=r"Trial\s+(\d+)"
)
parser.add_argument("-l", "--length", "--videolength", type=int,
                    help="The expected length for each trial of your video in seconds. The script will automatically trim the video down to the expected video length"
                    " videos under the specified length will be discarded. Default is 600 seconds (10 minutes)",
                    default=600)

# Advanced arguments (still optional)
parser.add_argument("-p", "--plot", type=str,
                    help="Instead of analyzing the files, plot the movement for the specified trial file in the data_directory (just the file, do not specify the directory again) "
                         " under 0.1 cm and -5 to 5 angles to help determine a good movement threshold. Plots will be display to the UI one at a time, to be edited and saved at the user's disgression")
parser.add_argument("-m", "--movethresh", type=float,
                    help="The threshold of movement in centimeters that is considered detected movement in the script. The default is 0.025 centimeters per frame, but may be adjusted here.",
                    default=0.025
                    )
parser.add_argument("-a", "--anglethresh", type=float,
                    help="The threshold of angular change in degrees that is considered a turn in the script. The default is 1 degree per frame, but may be adjusted here.",
                    default=1)
parser.add_argument("-i", "--likelihood", type=float,
                    help="Tracking likelihood from DLC is calculated on a scale from 0 to 1, with one being highest confidence that the tracked body part is authentic. Here, you can set"
                         " what tracking likelihood of the head and center should be the threshold for valid frames. Default is 0.5",
                    default=0.5)
parser.add_argument("-f", "--fps",
                    help="How many frames per second your videos were at when analyzed by DeepLabCut. Default is 2. Increasing this number may increase analysis time, especially"
                         " when generating the DeepLabCut coordinates.",
                    default=2)
parser.add_argument("-h5", "--saveh5",  action="store_true",
                    help="Save updated .H5 data with calculated movement on each frame, distances, and angles.")
parser.add_argument('-d', "--debug", action="store_true",
                    help="Output debug messages, graphs, and calculations.")
parser.add_argument('-t', "--trackstats", action="store_true",
                    help="Calculate and save information regarding tracking accuracy in the statistics")
parser.add_argument("-md", "--metadata", type=str,
                    help="A path to an excel or .csv spreadsheet containg metadata corresponding to the video trial numbers. If a `Stimulus Side` column is included, it will be used to dynamically adjust"
                         "the stimulus side used for related calculations. This will override the `-s` `--stimulus` parameter where necessary.")
parser.add_argument("-c", "--config", type=str,
                    help="Path to the configuration JSON file containing arena parameters. By default, looks for a config file with the same name as the script in the same directory.")



args = vars(parser.parse_args())

# ARGUMENT HANDLING LOGIC
data_directory = args["data_directory"]
results_folder = args["results_folder"]
LEFT_STIMWALL_SIDE = False if args["stimulus"] == "r" or args["stimulus"] =="right" else True
regexstring = args["regex"]
MOVEMENT_THRESHOLD_CM = args["movethresh"]
ANGLE_THRESHOLD = args["anglethresh"]
FRAMES_PER_SECOND = args["fps"]
EXPECTED_VIDEO_LENGTH_S = args["length"]
SAVE_HD5 = args["saveh5"]
DEBUG = args["debug"]
LIKELIHOOD_THRESHOLD = args["likelihood"]
TRACKSTATS = args["trackstats"]
METADATA_FILE_NAME = args["metadata"]
MAX_INVALID_FRAMES = 6


config_path = args["config"] if args["config"] else get_default_config_path()
config = load_config(config_path)

os.makedirs(results_folder, exist_ok=True)  # Create the folder if it doesn't exist


MAX_VIDEO_LENGTH_S = EXPECTED_VIDEO_LENGTH_S + 30
FRAMES_EXPECTED = FRAMES_PER_SECOND * EXPECTED_VIDEO_LENGTH_S
MAX_FRAMES = FRAMES_PER_SECOND * MAX_VIDEO_LENGTH_S
FRONT_TRIM_BUFFER = 30
FRONT_TRIM_BUFFER_FRAMES = FRAMES_PER_SECOND * FRONT_TRIM_BUFFER

X_LIM = config.get('X_LIM', 1280)
Y_LIM = config.get('Y_LIM', 1024)
LEFT_THIRD_BOUNDARY_X = config.get('LEFT_THIRD_BOUNDARY_X', 386)
RIGHT_THIRD_BOUNDARY_X = config.get('RIGHT_THIRD_BOUNDARY_X', 755)
LEFT_STIMWALL_BOUNDARY_X = config.get('LEFT_STIMWALL_BOUNDARY_X', 97)
RIGHT_STIMWALL_BOUNDARY_X = config.get('RIGHT_STIMWALL_BOUNDARY_X', 1050)

# For Y coordinates that depend on Y_LIM, calculate them after Y_LIM is loaded
TOP_WALL_BOUNDARY_Y = config.get('TOP_WALL_BOUNDARY_Y', Y_LIM - 110)
BOTTOM_WALL_BOUNDARY_Y = config.get('BOTTOM_WALL_BOUNDARY_Y', Y_LIM - 890)
TOP_THIRD_BOUNDARY_Y = config.get('TOP_THIRD_BOUNDARY_Y', Y_LIM - 347)
BOTTOM_THIRD_BOUNDARY_Y = config.get('BOTTOM_THIRD_BOUNDARY_Y', Y_LIM - 644)

HALFLINE_X = config.get('HALFLINE_X', 589)
HALFLINE_Y = config.get('HALFLINE_Y', 496)
BOUNDARY_BUFFER = config.get('BOUNDARY_BUFFER', 10)

# PIXELS_TO_MM calculations
ARENA_DIMENSION_CM_X = config.get('ARENA_DIMENSION_CM_X', 35)
ARENA_DIMENSION_CM_Y = config.get('ARENA_DIMENSION_CM_Y', 30)

X_PIXEL_RATIO = ARENA_DIMENSION_CM_X / abs(RIGHT_STIMWALL_BOUNDARY_X - LEFT_STIMWALL_BOUNDARY_X)
Y_PIXEL_RATIO =  ARENA_DIMENSION_CM_Y / abs(TOP_WALL_BOUNDARY_Y - BOTTOM_WALL_BOUNDARY_Y)
pd.options.display.float_format = '{:.2f}'.format

#MOVEMENT_THRESHOLD = 1.5 #How many pixels of movement per frame should be counted as movement from the animal
#ANGLE_THRESHOLD = 1 # How many degrees of movement is valid detected movement
#MOVEMENT_THRESHOLD_CM = 0.025


def validate_dlc_dataframe(df, expected_bodyparts):
    """
    Args:
        df: The .hdf5 dataframe from DLC
        expected_bodyparts: List of expected bodyparts.
    Raises: Value error if df fails validation
    """
    required_cols_per_bp = ['x', 'y', 'likelihood']
    for bp in expected_bodyparts:
        for col_prop in required_cols_per_bp:
            if (bp, col_prop) not in df.columns:
                raise ValueError(f"Missing expected column: { (bp, col_prop) }")
    # Add any other checks for data types, ranges, etc.
    print("DLC DataFrame integrity check passed.")

def load_metadata_spreadsheet(file_path: str):
    # -> pd.DataFrame or None
    """
    Reads an Excel (.xlsx) or CSV (.csv) file containing metadata information for the trials
    Returns the DataFrame or None if an error occurs or format is unsupported.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_path}. Please use .csv or .xlsx.")
            return None
        df = df.dropna()
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _find_actual_column_name(df_columns: list, primary_target_name: str, alternative_target_names: list = None):
    """
    Helper to find the actual column name in a list of df_columns, case-insensitively.
    Checks primary_target_name first, then any alternative_target_names.
    """
    primary_target_lower = primary_target_name.lower()
    for col in df_columns:
        if col.lower() == primary_target_lower:
            return col

    if alternative_target_names:
        for alt_name in alternative_target_names:
            alt_name_lower = alt_name.lower()
            for col in df_columns:
                if col.lower() == alt_name_lower:
                    return col
    return None

def extract_metadata_for_trial(metadata_df: pd.DataFrame, trial_number_to_find):
    """
    Extracts metadata for a specific trial number from the provided DataFrame.
    Args:
        metadata_df: Pandas DataFrame containing the metadata.
        trial_number_to_find: The trial number to search for (can be int or str).
    Returns:
        A dictionary with "Species", "Sex", "Treatment", and "StimulusSide"
        if the trial is found and columns exist. Otherwise, an empty dictionary.
    """
    if metadata_df is None or metadata_df.empty:
        return None

    extracted_info = {}
    df_columns_list = metadata_df.columns.tolist()
    actual_trial_col_name = _find_actual_column_name(df_columns_list, "trial", ["Trials"])

    if not actual_trial_col_name:
        return None

    # Convert Trial column to int first
    try:
        metadata_df[actual_trial_col_name] = metadata_df[actual_trial_col_name].astype(int)
    except (ValueError, TypeError) as e:
        print("Problem parsing trial numbers from Trial column in metadata: ", e)
        exit()

    try:
        condition = metadata_df[actual_trial_col_name].astype(str).str.lower() == str(trial_number_to_find).lower()
        matching_rows = metadata_df[condition]
    except AttributeError:
        condition = metadata_df[actual_trial_col_name].astype(str) == str(trial_number_to_find)
        matching_rows = metadata_df[condition]
    except Exception:
        return None

    if matching_rows.empty:
        return None

    trial_data_row = matching_rows.iloc[0]

    columns_to_extract_map = {
        "species": ("Species", []),
        "sex": ("Sex", []),
        "treatment": ("Treatment", ["treat"]),
        "stim_side": ("StimulusSide", ["Stimulus Side","stim_side"])
    }

    for output_key, (primary_name, alt_names) in columns_to_extract_map.items():
        actual_col_name = _find_actual_column_name(df_columns_list, primary_name, alt_names)
        if actual_col_name and actual_col_name in trial_data_row:
            value = trial_data_row[actual_col_name]
            extracted_info[output_key] = None if pd.isna(value) else value

    return extracted_info

def filter_and_smooth_predictions(videodata, likelihood_threshold=LIKELIHOOD_THRESHOLD, window_length=5, polyorder=2):
    """
    Filters low-likelihood predictions and applies Savitzky-Golay smoothing per body part.

    Parameters:
    videodata (pd.DataFrame): DataFrame with MultiIndex columns (bodyparts, coords).
    likelihood_threshold (float): Minimum likelihood to retain predictions.
    window_length (int): Window length for Savitzky-Golay filter (must be odd).
    polyorder (int): Polynomial order for Savitzky-Golay filter.

    Returns:
    pd.DataFrame: DataFrame with filtered and smoothed predictions.
    """
    # Create a copy of the DataFrame to store results
    result_df = videodata.copy()

    # Iterate through body parts
    for bodypart in videodata.columns.get_level_values('bodyparts').unique():
        # Extract relevant columns for the body part
        x_col = (bodypart, 'x')
        y_col = (bodypart, 'y')
        likelihood_col = (bodypart, 'likelihood')

        # Filter out low-likelihood points by setting x, y to NaN
        low_likelihood_mask = videodata[likelihood_col] < likelihood_threshold
        result_df.loc[low_likelihood_mask, [x_col, y_col]] = np.nan

    return result_df

def trim_video_frames(dataframe):
    """
    Trims the input DataFrame containing video frames based on length criteria.
    Parameters:
        dataframe (pd.DataFrame): DataFrame representing video frames with multi-level columns.
    Returns:
        pd.DataFrame: Trimmed DataFrame.
    Raises:
        ValueError: If the video length is less than 600 seconds.
    """
    total_frames = len(dataframe)
    total_seconds = total_frames / FRAMES_PER_SECOND

    if total_seconds < EXPECTED_VIDEO_LENGTH_S:
        raise ValueError(f"Video length for current trial is too short ({total_seconds} seconds). Minimum is {EXPECTED_VIDEO_LENGTH_S} seconds.")

    if total_seconds <= MAX_VIDEO_LENGTH_S:
        # Trim only from the beginning to fit 600 seconds
        start_frame = total_frames - FRAMES_EXPECTED
        trimmed_df = dataframe.iloc[start_frame:]
    else:
        # Trim 30 seconds from the beginning, then trim to fit 600 seconds
        start_frame = FRONT_TRIM_BUFFER_FRAMES
        end_frame = start_frame + FRAMES_EXPECTED
        trimmed_df = dataframe.iloc[start_frame:end_frame]

    return trimmed_df

def fill_start_NaNs(df):
# Fill NaNs at the start of the DataFrame
    valid_index = df[["x", "y"]].dropna().first_valid_index()
    if valid_index is not None:
        first_x, first_y = df.loc[valid_index, ["x", "y"]]
        df.loc[:valid_index, ["x", "y"]] = df.loc[:valid_index, ["x", "y"]].fillna({"x": first_x, "y": first_y})
    return df

def _calculate_max_consecutive_nans(series: pd.Series) -> int:
    """Calculates the length of the longest consecutive run of NaNs in a Series."""
    if series.empty:
        return 0
    is_na = series.isna()
    if not is_na.any():
        return 0
    group_ids = is_na.ne(is_na.shift()).cumsum()
    consecutive_nans_counts = is_na.groupby(group_ids).cumsum()
    return int(consecutive_nans_counts.max())

def get_indices_of_long_nan_gaps(series: pd.Series, max_gap_length: int) -> pd.Index:
    """
    Helper function for finding gaps longer than MAX_INVALID_FRAMES
    Args:
        series: pandas series of video coordinates
        max_gap_length: the allowed maximum gap length (int)

    Returns: pd.Index

    """
    is_nan = series.isna()
    if not is_nan.any():
        return pd.Index([])

    nan_group_ids = is_nan.ne(is_nan.shift()).cumsum()
    actual_nan_block_ids = nan_group_ids[is_nan]
    if actual_nan_block_ids.empty:
        return pd.Index([])

    nan_block_lengths = series[is_nan].groupby(actual_nan_block_ids).transform('size')
    long_gap_nan_indices = nan_block_lengths[nan_block_lengths > max_gap_length].index
    return long_gap_nan_indices

def get_gaze_direction(df):
    """
    Calculates whether the animal is looking at the left or right wall
    Args:
       df: The dataframe containing all the information on the animal's coordinates.

    Returns: a dictionary with the number of seconds the animal spent looking at the stimulus wall (or not)
    """
    # f indicates frames
    gaze_data_total = {
        "leftwall_gaze_f": 0,
        "rightwall_gaze_f": 0,
        "other_gaze_f": 0,
        "onwalls_f": 0,
        "unknown_gaze_f": 0
    }
    dotprods = []
    # TODO: See how often head is used instead of eyes
    #usedeyes = 0
    #usedhead = 0

    def check_arenavector_bounds(vector_to_check, arena_bound_vectors, gaze_data):
        if in_boundaries:
            dotprod = np.dot(eye_vector, body_vector)
            # In most situations on the arena floor, the eye vector should be perfectly perpendicular to the body vector
            if DEBUG:
                angle_error = 90 - abs(round(np.degrees(np.arccos(np.clip(dotprod, -1.0, 1.0))), 2))
                if angle_error > 15:
                    print(
                        f"Bad eye vs. body orientation vectors at min {int(idx / FRAMES_PER_SECOND // 60)}:{int(idx / FRAMES_PER_SECOND % 60)}. Angle deviation of {angle_error:.2f}°")
            dotprods.append(dotprod)
            if vector_to_check[0] > 0:  # Positive gaze vector for x cartesian coordinate, animal is looking right
                # Now check if y value of normalized vector is between the right stimwall bounds
                if arena_bound_vectors["right_wall_vector_top"][1] >= vector_to_check[1] >= arena_bound_vectors["right_wall_vector_bottom"][1]:
                    gaze_data["rightwall_gaze_f"] += 1
                else:
                    gaze_data["other_gaze_f"] += 1
            elif vector_to_check[0] < 0:  # Negative gaze vector for x cartesian coordinate, animal is looking left
                # Is it within left stimall wbounds?
                if arena_bound_vectors["left_wall_vector_top"][1] >= vector_to_check[1] >= arena_bound_vectors["left_wall_vector_bottom"][1]:
                    gaze_data["leftwall_gaze_f"] += 1
                else:
                    gaze_data["other_gaze_f"] += 1
            else:
                gaze_data["other_gaze_f"] += 1
        else:
            gaze_data["onwalls_f"] += 1

        return gaze_data

    for idx, row in df.iterrows():
        # Step 0: Look at likelihoods and figure out whether to use eyes or body vector
        le_likelihood = row["Left-eye", "likelihood"]
        re_likelihood = row["Right-eye", "likelihood"]
        head_likelihood = row["Head", "likelihood"]
        center_likelihood = row["Center", "likelihood"]

        le_x, le_y = row["Left-eye", "x"], row["Left-eye", "y"]
        re_x, re_y = row["Right-eye", "x"], row["Right-eye", "y"]
        h_x, h_y = row["Head", "x"], row["Head", "y"]  # Can be NaN
        c_x, c_y = row["Center", "x"], row["Center", "y"]  # Can be NaN

        useEyes = False
        useBodyvec = True
        if np.isnan(c_x):
            gaze_data_total["unknown_gaze_f"] += 1
            continue
        elif np.isnan(h_x):
            useBodyvec = False
        if le_likelihood >= LIKELIHOOD_THRESHOLD and re_likelihood >= LIKELIHOOD_THRESHOLD:
            useEyes = True


        # Step 1: Save coordinates as numpy vectors
        eye_left = np.array([le_x, le_y])
        eye_right = np.array([re_x, re_y])
        head = np.array([h_x, h_y])
        center = np.array([c_x, c_y])

        # Step 2: Translate gaze
        eye_left = eye_left - center
        eye_right = eye_right - center
        head = head - center
        center_zero = center - center

        # Create vectors and normalize (makes length of all vectors one, useful for comparisons) in our case,
        # It's for comparing to the vectors corresponding to the arena boundaries later.
        eye_vector = eye_left - eye_right
        body_vector = head
        gaze_vector = eye_left + eye_right

        eye_vector /= np.linalg.norm(eye_vector)
        body_vector /= np.linalg.norm(body_vector)
        gaze_vector /= np.linalg.norm(gaze_vector)

        # Step 3: Compute and normalize wall vectors (relative to animal center)
        left_wall_vector_top = np.array([LEFT_STIMWALL_BOUNDARY_X, TOP_WALL_BOUNDARY_Y]) - center
        left_wall_vector_bottom = np.array([LEFT_STIMWALL_BOUNDARY_X, BOTTOM_WALL_BOUNDARY_Y]) - center
        right_wall_vector_top = np.array([RIGHT_STIMWALL_BOUNDARY_X , TOP_WALL_BOUNDARY_Y]) - center
        right_wall_vector_bottom = np.array([RIGHT_STIMWALL_BOUNDARY_X, BOTTOM_WALL_BOUNDARY_Y]) - center

        # Convert to float before normalization to avoid integer division issues
        left_wall_vector_top = left_wall_vector_top.astype(float) / np.linalg.norm(left_wall_vector_top)
        left_wall_vector_bottom = left_wall_vector_bottom.astype(float) / np.linalg.norm(left_wall_vector_bottom)
        right_wall_vector_top = right_wall_vector_top.astype(float) / np.linalg.norm(right_wall_vector_top)
        right_wall_vector_bottom = right_wall_vector_bottom.astype(float) / np.linalg.norm(right_wall_vector_bottom)

        arena_boundary_vectors = {
            "left_wall_vector_top": left_wall_vector_top,
            "left_wall_vector_bottom": left_wall_vector_bottom,
            "right_wall_vector_top": left_wall_vector_top,
            "right_wall_vector_bottom": right_wall_vector_bottom
        }

        # Using same boolean as rest of the script for consistency
        x_head = row["Head", "x"]
        y_head = row["Head", "y"]
        x_center = row["Center", "x"]
        y_center = row["Center", "y"]

        # Create a reusable boolean for whether the animal is inside the arena but not touching the walls
        in_boundaries = (
            (LEFT_STIMWALL_BOUNDARY_X + BOUNDARY_BUFFER)< x_head < (RIGHT_STIMWALL_BOUNDARY_X - BOUNDARY_BUFFER) and
            (LEFT_STIMWALL_BOUNDARY_X + BOUNDARY_BUFFER) < x_center < (RIGHT_STIMWALL_BOUNDARY_X - BOUNDARY_BUFFER) and
            (BOTTOM_WALL_BOUNDARY_Y + BOUNDARY_BUFFER) < y_head < (TOP_WALL_BOUNDARY_Y - BOUNDARY_BUFFER) and
            (BOTTOM_WALL_BOUNDARY_Y + BOUNDARY_BUFFER) < y_center < (TOP_WALL_BOUNDARY_Y - BOUNDARY_BUFFER)
        )

        if in_boundaries:
            if useEyes:
                check_arenavector_bounds(gaze_vector, arena_boundary_vectors, gaze_data_total)
            elif useBodyvec:
                check_arenavector_bounds(body_vector, arena_boundary_vectors, gaze_data_total)
        else:
            gaze_data_total["onwalls_f"] +=1

    on_arena_floor = (FRAMES_EXPECTED - gaze_data_total["onwalls_f"] - gaze_data_total["unknown_gaze_f"])

    if DEBUG:
        plt.figure(figsize=(8, 5))
        sns.histplot(dotprods, bins=50, kde=True)
        plt.axvline(0, color='red', linestyle='dashed', label="Zero")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Near-Zero Numbers")
        plt.legend()
        plt.show()

    gaze_data_total["on_arena_floor_f"] = on_arena_floor

    # TODO: FIGURE OUT IF YOU WANT TO COUNT FRAMES OR PERCENTAGE TIME SPENT LOOKING AT WALL OR WHATEVER ELSE
    return gaze_data_total

def calculate_stats(df, trial_num):
    """
        Calculates movement, positional, and angle based statistics for a given locust video.
        Args:
            df (pd.DataFrame): The .HDF5 dataframe from deeplabcut.analyzevideos()
            with coordinates and likelihood for the various body parts
        Returns:
            dict: A dictionary with the calculated statistics.
    """
    df = df.copy()
    # Rename coords multiindex to stats to reflect all the additional data we are going to add to this level.
    df.columns = df.columns.rename("stats", level="coords")

    # Calculate tracking accuracy for head and center
    head_track_acc = df.loc[:, ("Head", "likelihood")].mean()
    center_track_acc = df.loc[:, ("Center", "likelihood")].mean()

    # Flagging frames with bad head or center tracking
    for bodypart in ["Head", "Center"]:
        df.loc[:, (bodypart, "valid_tracking")] = df.loc[:, (bodypart, "likelihood")] >= LIKELIHOOD_THRESHOLD

    # Interpolation of low-likelihood frames
    bodyparts_to_interpolate = ["Head", "Center"]
    for bodypart in bodyparts_to_interpolate:
        for coord in ["x", "y"]:
            original_coord_series = df.loc[:, (bodypart, coord)].copy()
            likelihood_series = df.loc[:, (bodypart, "likelihood")]

            coord_to_interpolate = original_coord_series.copy()
            coord_to_interpolate[likelihood_series < LIKELIHOOD_THRESHOLD] = np.nan

            indices_to_keep_as_nan = get_indices_of_long_nan_gaps(
                coord_to_interpolate, MAX_INVALID_FRAMES
            )

            interpolated_coord_series = coord_to_interpolate.interpolate(
                method='linear',
                limit=MAX_INVALID_FRAMES,
                limit_direction='both'
            )

            if not indices_to_keep_as_nan.empty:
                interpolated_coord_series.loc[indices_to_keep_as_nan] = np.nan

            df.loc[:, (bodypart, coord)] = interpolated_coord_series

    largest_nan_gap_center = _calculate_max_consecutive_nans(df.loc[:, ("Center", "x")])
    largest_nan_gap_head = _calculate_max_consecutive_nans(df.loc[:, ("Head", "x")])


    # Convert X and Y pixel coordinates to centimeter distance after interpolation
    for bodypart in df.columns.levels[0]:
        df.loc[:, (bodypart, "x_cm")] = df.loc[:, (bodypart, "x")] * X_PIXEL_RATIO
        df.loc[:, (bodypart, "y_cm")] = df.loc[:, (bodypart, "y")] * Y_PIXEL_RATIO


    # Calculate distance movements for the Head and Center
    for bodypart in ["Head", "Center"]:
        df.loc[:, (bodypart, "distance_x")] = df.loc[:, (bodypart, "x_cm")].diff()
        df.loc[:, (bodypart, "distance_y")] = df.loc[:, (bodypart, "y_cm")].diff()
        df.loc[:, (bodypart, "distance")] = np.sqrt(df[bodypart, 'distance_x'] ** 2 + df[bodypart, 'distance_y'] ** 2)

    # For the Center, calculate the mean velocity (excluding points where the animal is not moving)
    valid_distances = df.loc[df["Center", "distance"] >= MOVEMENT_THRESHOLD_CM, ("Center","distance")]
    distance_travelled = valid_distances.sum()  # .sum() skips NaNs by default
    valid_distances_count = len(valid_distances.dropna()) # Count non-NaN valid distances

    mean_velocity_perframe = distance_travelled / valid_distances_count if valid_distances_count > 0 else 0.0
    mean_velocity_persecond = mean_velocity_perframe * FRAMES_PER_SECOND

    # Cumulative movement duration
    movement_dur_s = valid_distances_count  / FRAMES_PER_SECOND
    nonmovement_dur_s = EXPECTED_VIDEO_LENGTH_S - movement_dur_s

    avg_stimwall_dis_cm = 0.0  # Default if 'Center'/'x' not found or all NaNs
    if ("Center", "x") in df.columns:
        center_x_pixels = df.loc[:, ('Center', 'x')]
        # Initialize wall_distances_pixels with NaNs to handle cases where conditions aren't met or x is NaN
        wall_distances_pixels = pd.Series(np.nan, index=df.index, dtype=float)

        if LEFT_STIMWALL_SIDE:
            # mask inbounds -> within the arena on right side of left wall
            mask_inbounds = center_x_pixels > LEFT_STIMWALL_BOUNDARY_X
            wall_distances_pixels.loc[mask_inbounds] = abs(center_x_pixels.loc[mask_inbounds] - LEFT_STIMWALL_BOUNDARY_X)
            # Points validly at or before the wall (distance is 0)
            mask_at_or_before = center_x_pixels <= LEFT_STIMWALL_BOUNDARY_X
            wall_distances_pixels.loc[mask_at_or_before] = 0
        else:  # RIGHT_STIMWALL_SIDE
            mask_inbounds = center_x_pixels < RIGHT_STIMWALL_BOUNDARY_X
            wall_distances_pixels.loc[mask_inbounds] = abs(center_x_pixels.loc[mask_inbounds] - RIGHT_STIMWALL_BOUNDARY_X)
            mask_at_or_before = center_x_pixels >= RIGHT_STIMWALL_BOUNDARY_X
            wall_distances_pixels.loc[mask_at_or_before] = 0

        # NaNs in center_x_pixels will result in NaNs in wall_distances_pixels at those positions
        # .mean() skips NaNs by default
        avg_stimwall_dis_pixels = wall_distances_pixels.mean()
        avg_stimwall_dis_cm = avg_stimwall_dis_pixels * X_PIXEL_RATIO if pd.notna(avg_stimwall_dis_pixels) else 0.0


    # Time Spent in Each of the 3 zones
    center_x_for_zones = df.loc[:, ('Center', 'x')].dropna() # Drop NAs.
    left_zone_time = center_x_for_zones[center_x_for_zones <= LEFT_THIRD_BOUNDARY_X].shape[0] / FRAMES_PER_SECOND
    center_zone_time = center_x_for_zones[(center_x_for_zones > LEFT_THIRD_BOUNDARY_X) & (center_x_for_zones <= RIGHT_THIRD_BOUNDARY_X)].shape[0] / FRAMES_PER_SECOND
    right_zone_time = center_x_for_zones[center_x_for_zones > RIGHT_THIRD_BOUNDARY_X].shape[0] / FRAMES_PER_SECOND
    left_half_time = center_x_for_zones[center_x_for_zones < HALFLINE_X].shape[0] / FRAMES_PER_SECOND
    right_half_time = center_x_for_zones[center_x_for_zones > HALFLINE_X].shape[0] / FRAMES_PER_SECOND

    # Number of frames not assigned to Left, Right, Or Center zones
    seconds_in_lcr_zones = left_zone_time + center_zone_time + right_zone_time
    unassigned_lcr_zone_seconds = EXPECTED_VIDEO_LENGTH_S - seconds_in_lcr_zones

    # ANGLE and turn frequencies
    # First, calculate the slope for the body (Head and centerpoint)
    df['slope'] = (df[('Head', 'y')] - df[('Center', 'y')]) / (df[('Head', 'x')] - df[('Center', 'x')])
    # Calculate the angle (turn degree) with signed direction
    current_slope = df['slope']
    prev_slope = df['slope'].shift(1)
    # Compute the angle using the arctan formula
    with np.errstate(divide='ignore', invalid='ignore'):
        angle_rad = np.arctan((current_slope - prev_slope) / (1 + current_slope * prev_slope))
    angle_deg = np.degrees(angle_rad)
    # Determine the sign of the angle based on the cross product
    vx_t = df[('Head', 'x')] - df[('Center', 'x')]
    vy_t = df[('Head', 'y')] - df[('Center', 'y')]
    vx_tplus1 = (df[('Head', 'x')].shift(-1) - df[('Center', 'x')].shift(-1))
    vy_tplus1 = (df[('Head', 'y')].shift(-1) - df[('Center', 'y')].shift(-1))
    cross_product = (vy_t * vx_tplus1) - (vx_t * vy_tplus1)
    # Positive angle = clockwise, negative = counterclockwise
    df['turn_degree'] = np.where(cross_product > 0, -angle_deg, angle_deg)
    # Handle the first row's turn_degree (same as the second)
    if not df.empty:
        df['turn_degree'].iloc[0] = df['turn_degree'].iloc[1] if len(df) > 1 else np.nan

    # CREATE 2 LISTS OF ALL THE TURNS THE ANIMAL PERFORMS DURING THE EXPERIMENTS
    turns = []
    nowallturns = []

    for idx, row in df.iterrows():
        distance_head = row[("Head", "distance")]
        distance_center = row[("Center", "distance")]
        turn_degree_val = row.get('turn_degree', np.nan).item()
        x_head = row[("Head", "x")]
        y_head = row[("Head", "y")]
        x_center = row[("Center", "x")]
        y_center = row[("Center", "y")]
        coords_are_valid = pd.notna(x_head) and pd.notna(y_head) and \
                           pd.notna(x_center) and pd.notna(y_center)
        # Create a reusable boolean for whether the animal is inside the arena but not touching the walls
        in_boundaries = False
        if coords_are_valid:
            in_boundaries = (
                    (LEFT_STIMWALL_BOUNDARY_X + BOUNDARY_BUFFER) < x_head < (RIGHT_STIMWALL_BOUNDARY_X - BOUNDARY_BUFFER) and
                    (LEFT_STIMWALL_BOUNDARY_X + BOUNDARY_BUFFER) < x_center < (RIGHT_STIMWALL_BOUNDARY_X - BOUNDARY_BUFFER) and
                    (BOTTOM_WALL_BOUNDARY_Y + BOUNDARY_BUFFER) < y_head < (TOP_WALL_BOUNDARY_Y - BOUNDARY_BUFFER) and
                    (BOTTOM_WALL_BOUNDARY_Y + BOUNDARY_BUFFER) < y_center < (TOP_WALL_BOUNDARY_Y - BOUNDARY_BUFFER)
            )

        # If either the head or the centerpoint pass the movement threshold, and the angle passes the angle threshold,
        is_significant_turn = ((pd.notna(distance_head) and distance_head >= MOVEMENT_THRESHOLD_CM) or
                                (pd.notna(distance_center) and distance_center >= MOVEMENT_THRESHOLD_CM)) \
                              and (pd.notna(turn_degree_val) and abs(turn_degree_val) >= ANGLE_THRESHOLD)

        if is_significant_turn:
            # Regardless of whether the animal was within the boundaries of the wall or not, we need to add it to the overall turns
            if turns and np.sign(turns[-1]) == np.sign(turn_degree_val) and turns[-1] != 0:
                # If the angle turn is a continuation of the one from the last frame, add it.
                turns[-1] += turn_degree_val  # add to last angle turn from previous frame
            else:
                # Otherwise, the animal started turning in the other direction, or there are no elements in the list yet, or previous turn was 0.
                turns.append(turn_degree_val)
            if in_boundaries:
                if not nowallturns:
                    # First significant turn in boundaries, or list was empty.
                    nowallturns.append(turn_degree_val)
                elif nowallturns[-1] is None:
                    # The animal has just returned inside the boundary limit
                    nowallturns.append(turn_degree_val)
                elif np.sign(nowallturns[-1]) == np.sign(turn_degree_val) and nowallturns[-1] != 0:
                    # If the angle turn is a continuation of the one from the last frame, add it.
                    nowallturns[-1] += turn_degree_val  # add to last angle turn from previous frame
                else:
                    # Otherwise, the animal started turning in the other direction, or previous turn was 0.
                    nowallturns.append(turn_degree_val)
        else:
            # If the thresholds are not reached, we determine that the values are just noise and the animal has not turned.
            turns.append(0)
            if in_boundaries:
                nowallturns.append(0)
            else:  # Not in boundaries or coords are NaN
                nowallturns.append(None)

    # Now, we can calculate how many clockwise vs counterclockwise turns, total absolute value of turns, turn_frequency, average
    # absolute turn size, and prep all the other statistics as well.

    clockwise_turns = len([turn for turn in turns if pd.notna(turn) and turn > 0])
    nowall_clockwise_turns = len([turn for turn in nowallturns if turn is not None and pd.notna(turn) and turn > 0])
    counterclockwise_turns = len([turn for turn in turns if pd.notna(turn) and turn < 0])
    nowall_counterclockwise_turns = len([turn for turn in nowallturns if turn is not None and pd.notna(turn) and turn < 0])
    total_turns = clockwise_turns + counterclockwise_turns
    nowall_total_turns = nowall_clockwise_turns + nowall_counterclockwise_turns
    total_degrees_abs = np.nansum([abs(angle) for angle in turns if pd.notna(angle)])
    nowall_total_degrees_abs = np.nansum([abs(angle) for angle in nowallturns if angle is not None and pd.notna(angle)])
    turn_freq_denominator_s = (len(turns) / FRAMES_PER_SECOND) if FRAMES_PER_SECOND > 0 else 0
    turn_frequency_m = (total_turns / turn_freq_denominator_s * 60) if turn_freq_denominator_s > 0 else 0.0
    nowall_valid_turn_entries = [turn for turn in nowallturns if turn is not None]
    nowall_freq_denominator_s = (len(nowall_valid_turn_entries) / FRAMES_PER_SECOND) if FRAMES_PER_SECOND > 0 else 0
    nowall_turn_frequency_m = (nowall_total_turns / nowall_freq_denominator_s * 60) if nowall_freq_denominator_s > 0 else 0.0
    only_significant_turns = [abs(turn) for turn in turns if pd.notna(turn) and turn != 0]
    avg_degree_size = np.sum(only_significant_turns) / len(only_significant_turns) if len(only_significant_turns) > 0 else 0.0
    nowall_only_significant_turns = [abs(turn) for turn in nowallturns if turn is not None and pd.notna(turn) and turn != 0]
    nowall_avg_degree_size = np.sum(nowall_only_significant_turns) / len(nowall_only_significant_turns) if len(nowall_only_significant_turns) > 0 else 0.0

    # Using eye positions to calculate where the animal is looking at
    gaze_dict = get_gaze_direction(df)

    if SAVE_HD5:
        try:
            df.to_hdf(f'Locust_stats_{trial_num}.h5', key='df', mode='w')
        except Exception as YIKES:
            print(f"Error saving HDF5 file for trial {trial_num}: {YIKES}")

    def finalize_stat(value, is_count=False, digits=4):
        if pd.isna(value):
            return 0 if is_count else 0.0
        return round(value, digits) if not is_count else int(value)

    ultimate_stats_dict = {
        'mean_velocity_cm_s': finalize_stat(mean_velocity_persecond),
        'distance_travelled_cm': finalize_stat(distance_travelled),
        'movement_duration': finalize_stat(movement_dur_s),
        "nonmovement_duration": finalize_stat(nonmovement_dur_s),
        "avg_distance_from_stimwall": finalize_stat(avg_stimwall_dis_cm),
        "left_zone_time": finalize_stat(left_zone_time, is_count=True),
        "center_zone_time": finalize_stat(center_zone_time, is_count=True),
        "right_zone_time": finalize_stat(right_zone_time, is_count=True),
        "left_half_time": finalize_stat(left_half_time, is_count=True),
        "right_half_time": finalize_stat(right_half_time, is_count=True),
        "clockwise_turns": finalize_stat(clockwise_turns, is_count=True),
        "counterclockwise_turns": finalize_stat(counterclockwise_turns, is_count=True),
        "total_turn_count": finalize_stat(total_turns, is_count=True),
        "total_degrees_turned_abs": finalize_stat(total_degrees_abs),
        "turn_frequency_m": finalize_stat(turn_frequency_m),
        "average_degrees_turned_abs": finalize_stat(avg_degree_size),
        "nowall_clockwise_turns": finalize_stat(nowall_clockwise_turns, is_count=True),
        "nowall_counterclockwise_turns": finalize_stat(nowall_counterclockwise_turns, is_count=True),
        "nowall_total_turn_count": finalize_stat(nowall_total_turns, is_count=True),
        "nowall_total_degrees_turned_abs": finalize_stat(nowall_total_degrees_abs),
        "nowall_turn_frequency_m": finalize_stat(nowall_turn_frequency_m),
        "nowall_average_degrees_turned_abs": finalize_stat(nowall_avg_degree_size),
        "average_head_tracking_accuracy": finalize_stat(head_track_acc),
        "average_center_tracking_accuracy": finalize_stat(center_track_acc),
        "unassigned_lcr_zone_seconds": finalize_stat(unassigned_lcr_zone_seconds, is_count=True),
        "largest_nan_gap_center": finalize_stat(largest_nan_gap_center, is_count=True),
        "largest_nan_gap_head": finalize_stat(largest_nan_gap_head, is_count=True),
    }
    ultimate_stats_dict.update(gaze_dict)
    if not TRACKSTATS:
        ultimate_stats_dict.pop("largest_nan_gap_center")
        ultimate_stats_dict.pop("largest_nan_gap_head")
        ultimate_stats_dict.pop("unassigned_lcr_zone_seconds")
        ultimate_stats_dict.pop("average_head_tracking_accuracy")
        ultimate_stats_dict.pop("average_center_tracking_accuracy")
    return ultimate_stats_dict

def plot_trial(df):
    """
    Plot the range of movement and angle turning close to 0 to help determine good thresholds
        Args:
            df (pd.DataFrame): The .HDF5 dataframe from deeplabcut.analyzevideos()
            with coordinates and likelihood for the various body parts
    """

    #NOTE: Currently reuses a lot of code form calculate_stats, a more professional coder would probably consolidate that
    #into methods.

    df = df.copy()
    # Rename coords multiindex to stats to reflect all the additional data we are going to add to this level.
    df.columns = df.columns.rename("stats", level="coords")

    # Convert X and Y pixel coordinates to centimeter distance immediately
    for bodypart in df.columns.levels[0]:
        df.loc[:, (bodypart, "x_cm")] = df.loc[:, (bodypart, "x")] * X_PIXEL_RATIO
        df.loc[:, (bodypart, "y_cm")] = df.loc[:, (bodypart, "y")] * Y_PIXEL_RATIO

    # Calculate distance movements for the Head and Center
    for bodypart in ["Head", "Center"]:
        df.loc[:, (bodypart, "distance_x")] = df.loc[:, (bodypart, "x_cm")].diff()
        df.loc[:, (bodypart, "distance_y")] = df.loc[:, (bodypart, "y_cm")].diff()
        df.loc[:, (bodypart, "distance")] = np.sqrt(df[bodypart, 'distance_x'] ** 2 + df[bodypart, 'distance_y'] ** 2)

    stuff = df[("Center", "distance")].dropna()
    plt.hist(stuff, bins=4000)
    plt.xlabel('Distance Travelled (centerpoint)', fontsize = 18)
    plt.ylabel('Frequency', fontsize = 18)
    plt.title('Distribution of Distances under 0.1 cm/frame', fontsize = 20)
    plt.xlim(0, 0.1)
    plt.xticks(np.arange(0, 0.1, 0.005))
    plt.show()
    plt.clf()

    # ANGLE and turn frequencies
    # First, calculate the slope for the body (Head and centerpoint)
    df['slope'] = (df[('Head', 'y')] - df[('Center', 'y')]) / (df[('Head', 'x')] - df[('Center', 'x')])
    # Calculate the angle (turn degree) with signed direction
    current_slope = df['slope']
    prev_slope = df['slope'].shift(1)
    # Compute the angle using the arctan formula
    angle = np.degrees(np.arctan((current_slope - prev_slope) / (1 + current_slope * prev_slope)))
    # Determine the sign of the angle based on the cross product
    # (y1-y0)*(x2-x1) - (x1-x0)*(y2-y1)
    cross_product = ((df[('Head', 'y')] - df[('Center', 'y')]) * (
            df[('Head', 'x')].shift(-1) - df[('Center', 'x')].shift(-1)) -
                     (df[('Head', 'x')] - df[('Center', 'x')]) * (
                             df[('Head', 'y')].shift(-1) - df[('Center', 'y')].shift(-1)))
    # Positive angle = clockwise, negative = counterclockwise
    df['turn_degree'] = np.where(cross_product > 0, -angle, angle)
    # Handle the first row's turn_degree (same as the second)
    df['turn_degree'].iloc[0] = df['turn_degree'].iloc[1]

    plt.hist(df['turn_degree'].dropna(), bins=2000)
    plt.xlabel('Angle turn', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Degrees turned distribution (-5 to 5)', fontsize=20)
    plt.xlim(-5, 5)
    plt.xticks(np.arange(-5, 5, 0.5))
    plt.show()
    plt.clf()

stats_df = pd.DataFrame()
trial_numbers = []

if args["plot"]:
    file_path = os.path.join(data_directory, args["plot"])
    plotdata = pd.read_hdf(file_path)
    plotdata.columns = plotdata.columns.droplevel('scorer')
    match = re.search(regexstring, args["plot"])
    trial_number = int(match.group(1)) if match else None
    plot_trial(plotdata)
    print(f"Showed plots for Trial {trial_number}. Use UI to adjust and save!")
    exit()

metadata_df = pd.DataFrame()
if args["metadata"]:
    metadata_df = load_metadata_spreadsheet(args["metadata"])

# ITERATE THROUGH ALL .H5 FILES IN A DIRECTORY
for dataset in os.listdir(data_directory):
    if dataset.endswith(".h5"):
        file_path = os.path.join(data_directory, dataset)
        # Extract the trial number using regex
        #match = re.search(r'Trial\s+(\d+)', dataset)
        match = re.search(regexstring, dataset)
        trial_number = int(match.group(1)) if match else None
        print(f"Processing Trial {trial_number}")
        locustdata = pd.read_hdf(file_path)
        # BEGIN BY CLEANING UP THE .H5 DATAFRAME
        # Drop the 'scorer' level from the columns MultiIndex, since it's useless
        locustdata.columns = locustdata.columns.droplevel('scorer')

        # Invert Y-Axis to match classical graphing conventions
        y_columns = locustdata.loc[:, (slice(None), 'y')]
        locustdata.loc[:, (slice(None), 'y')] = Y_LIM - y_columns
        try:
            finalDF = trim_video_frames(locustdata)
            if trial_number is not None:
                trial_numbers.append(trial_number)
            else:
                print("Error: Trial Number Not Found!! Check your regex string and/or video names")
                print("Aborting analysis")
                exit()
        except ValueError as e:
            print(e)
            continue
        # Grab metadata info if it exists
        dict_to_merge = extract_metadata_for_trial(metadata_df, trial_number)

        # Dynamically update stimulus side if the information is available
        if dict_to_merge:
            if "StimulusSide" in dict_to_merge.keys():
                if dict_to_merge["StimulusSide"].lower() == "left":
                    LEFT_STIMWALL_SIDE = True
                elif dict_to_merge["StimulusSide"].lower() == "right":
                    LEFT_STIMWALL_SIDE = False
                else:
                    print("ERROR: Unclear stimulus side: ", dict_to_merge["StimulusSide"])
                    print("Aborting analysis")
                    exit()


        #If debugging, use the non-trimmed video data so that the timestamps match the actual video
        if DEBUG:
            stats = calculate_stats(locustdata, trial_number)
            # Output tracking accuracy for each trial
            print(f"Head tracking accuracy: {stats['average_head_tracking_accuracy']:.4f}")
            print(f"Center tracking accuracy: {stats['average_center_tracking_accuracy']:.4f}")
            print(f"Distance travelled CM: {stats['distance_travelled_cm']:.4f}")
        else:
            stats = calculate_stats(finalDF, trial_number)

        # Reset stimulus side from potential dynamic update from metadata
        LEFT_STIMWALL_SIDE = False if args["stimulus"] == "r" or args["stimulus"] =="right" else True
        stats["trial"] = trial_number
        if dict_to_merge: stats.update(dict_to_merge)
        stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)


stats_df = stats_df.sort_values(by="trial", ascending=True)
if trial_numbers:
    lowest_trial = min(trial_numbers)
    highest_trial = max(trial_numbers)
    output_filename = f"trials_{lowest_trial}_to_{highest_trial}.csv"
else:
    output_filename = "trials.csv"

# Save the DataFrame to the results folder with the appropriate name
output_path = os.path.join(results_folder, output_filename)
stats_df.to_csv(output_path, index=False)

print(f"Processing complete. Results saved to {output_filename}")
