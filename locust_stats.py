import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re
import argparse
import seaborn as sns

warnings.filterwarnings('ignore')

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
                         " under 0.1 cm and -5 to 5 angles to help determine a good movement threshold. Plots will be display to the UI one at a time, to be edit and saved at the user's disgression")
parser.add_argument("-m", "--movethresh", type=float,
                    help="The threshold of movement in centimeters that is considered detected movement in the script. The default is 0.025 centimeters per frame, but may be adjusted here.",
                    default=0.025
                    )
parser.add_argument("-a", "--anglethresh", type=float,
                    help="The threshold of angular change in degrees that is considered a turn in the script. The default is 1 degree per frame, but may be adjusted here.",
                    default=1)
parser.add_argument("-f", "--fps",
                    help="How many frames per second your videos were at when analyzed by DeepLabCut. Default is 2. Increasing this number may increase analysis time, especially"
                         " when generating the DeepLabCut coordinates.",
                    default=2)
parser.add_argument("-h5", "--saveh5",  action="store_true",
                    help="Save updated .H5 data with calculated movement on each frame, distances, and angles.")
parser.add_argument('-d', "--debug", action="store_true",
                    help="Output debug messages, graphs, and calculations.")


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

os.makedirs(results_folder, exist_ok=True)  # Create the folder if it doesn't exist


MAX_VIDEO_LENGTH_S = EXPECTED_VIDEO_LENGTH_S + 30
FRAMES_EXPECTED = FRAMES_PER_SECOND * EXPECTED_VIDEO_LENGTH_S
MAX_FRAMES = FRAMES_PER_SECOND * MAX_VIDEO_LENGTH_S
FRONT_TRIM_BUFFER = 30
FRONT_TRIM_BUFFER_FRAMES = FRAMES_PER_SECOND * FRONT_TRIM_BUFFER

X_LIM = 1280
Y_LIM = 1024
LEFT_THIRD_BOUNDARY_X = 386 # the x-intercept for the left third of the cage (SUBJECT TO CHANGE)
RIGHT_THIRD_BOUNDARY_X = 755 # the x-intercept for the right third of the cage (SUBJECT TO CHANGE)
LEFT_STIMWALL_BOUNDARY_X = 97 # x-intercept coordinate for the left stimulus wall (intersecting with the floor)
RIGHT_STIMWALL_BOUNDARY_X = 1050 # x-intercept coordinate for the left stimulus wall (intersecting with the floor)
TOP_WALL_BOUNDARY_Y = Y_LIM - 110 # y-intercept coordinate for the top wall (intersecting with the floor)
BOTTOM_WALL_BOUNDARY_Y = Y_LIM - 890 # y-intercept coordinate for the bottom wall (intersecting with the floor)
TOP_THIRD_BOUNDARY_Y = Y_LIM - 347
BOTTOM_THIRD_BOUNDARY_Y = Y_LIM - 644
HALFLINE_X = 589
HALFLINE_Y = 496

#PIXELS_TO_MM calculations
ARENA_DIMENSION_CM_X = 35
ARENA_DIMENSION_CM_Y = 30
X_PIXEL_RATIO = ARENA_DIMENSION_CM_X / abs(RIGHT_STIMWALL_BOUNDARY_X - LEFT_STIMWALL_BOUNDARY_X)
Y_PIXEL_RATIO =  ARENA_DIMENSION_CM_Y / abs(TOP_WALL_BOUNDARY_Y - BOTTOM_WALL_BOUNDARY_Y)
pd.options.display.float_format = '{:.2f}'.format

#MOVEMENT_THRESHOLD = 1.5 #How many pixels of movement per frame should be counted as movement from the animal
#ANGLE_THRESHOLD = 1 # How many degrees of movement is valid detected movement
#MOVEMENT_THRESHOLD_CM = 0.025

test_file_name = "~/OneDrive/gabbiani/TOOLS/locust_tracker/trial_coordinates/david_165DLC_Resnet50_locustbehavior2Jan10shuffle5_snapshot_500.h5"

# REMOVE LOW LIKELIHOOD COORDINATES AND SMOOTH OUT THE REST.
def filter_and_smooth_predictions(videodata, likelihood_threshold=0.5, window_length=5, polyorder=2):
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
# calculate various statistics and add them to the dataframe

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

def point_slope_form(x1, y1, m, x):
  return m * (x - x1) + y1

def get_gaze_direction(df):
    """
    Calculates whether the animal is looking at the left or right wall
    Args:
       df: The dataframe containing all the information on the animal's coordinates.

    Returns: a dictionary with the number of seconds the animal spent looking at the stimulus wall (or not)
    """
    leftwall_gaze_f = 0
    rightwall_gaze_f = 0
    other_gaze_f = 0
    onwalls = 0
    dotprods = []
    for idx, row in df.iterrows():
        # Step 1: Save coordinates as numpy vectors
        eye_left = np.array([row["Left-eye", "x"], row["Left-eye", "y"]])
        eye_right = np.array([row["Right-eye", "x"], row["Right-eye", "y"]])
        head = np.array([row["Head", "x"], row["Head", "y"]])
        center = np.array([row["Center", "x"], row["Center", "y"]])

        # Step 2: Translate gaze
        eye_left = eye_left - center
        eye_right = eye_right - center
        head = head - center
        center = center - center

        # Create vectors and normalize
        eye_vector = eye_left - eye_right
        body_vector = head
        gaze_vector = (eye_left - center) + (eye_right - center)

        eye_vector /= np.linalg.norm(eye_vector)
        body_vector /= np.linalg.norm(body_vector)
        gaze_vector /= np.linalg.norm(gaze_vector)

        # Step 3: Compute and normalize wall vectors (relative to arena center)
        left_wall_vector_top = np.array([LEFT_STIMWALL_BOUNDARY_X, TOP_WALL_BOUNDARY_Y]) - center
        left_wall_vector_bottom = np.array([LEFT_STIMWALL_BOUNDARY_X, BOTTOM_WALL_BOUNDARY_Y]) - center
        right_wall_vector_top = np.array([RIGHT_STIMWALL_BOUNDARY_X , TOP_WALL_BOUNDARY_Y]) - center
        right_wall_vector_bottom = np.array([RIGHT_STIMWALL_BOUNDARY_X, BOTTOM_WALL_BOUNDARY_Y]) - center

        # Convert to float before normalization to avoid integer division issues
        left_wall_vector_top = left_wall_vector_top.astype(float) / np.linalg.norm(left_wall_vector_top)
        left_wall_vector_bottom = left_wall_vector_bottom.astype(float) / np.linalg.norm(left_wall_vector_bottom)
        right_wall_vector_top = right_wall_vector_top.astype(float) / np.linalg.norm(right_wall_vector_top)
        right_wall_vector_bottom = right_wall_vector_bottom.astype(float) / np.linalg.norm(right_wall_vector_bottom)

        # Using same boolean as rest of the script for consistency
        x_head = row["Head", "x"]
        y_head = row["Head", "y"]
        x_center = row["Center", "x"]
        y_center = row["Center", "y"]

        # Create a reusable boolean for whether the animal is inside the arena but not touching the walls
        in_boundaries = (
                LEFT_STIMWALL_BOUNDARY_X < x_head < RIGHT_STIMWALL_BOUNDARY_X and
                LEFT_STIMWALL_BOUNDARY_X < x_center < RIGHT_STIMWALL_BOUNDARY_X and
                BOTTOM_WALL_BOUNDARY_Y < y_head < TOP_WALL_BOUNDARY_Y and
                BOTTOM_WALL_BOUNDARY_Y < y_center < TOP_WALL_BOUNDARY_Y
        )

        if in_boundaries:
            dotprod = np.dot(eye_vector, body_vector)
            if DEBUG:
                angle_error =  90 - abs(round(np.degrees(np.arccos(np.clip(dotprod, -1.0, 1.0))), 2))
                if angle_error > 15:
                    print(f"Bad eye vs. body orientation vectors at min {int(idx/FRAMES_PER_SECOND//60)}:{int(idx/FRAMES_PER_SECOND%60)}. Angle deviation of {angle_error:.2f}°")
            dotprods.append(dotprod)
            if gaze_vector[0] > 0:
                #looking right
                if right_wall_vector_top[1] >= gaze_vector[1] >= right_wall_vector_bottom[1]:
                    rightwall_gaze_f+=1
                else:
                    other_gaze_f+=1
            elif gaze_vector[0] < 0:
                if left_wall_vector_top[1] >= gaze_vector[1] >= left_wall_vector_bottom[1]:
                    leftwall_gaze_f+=1
                else:
                    other_gaze_f+=1
            else:
                other_gaze_f+=1
        else: onwalls +=1
    on_arena_floor = FRAMES_EXPECTED - onwalls

    if DEBUG:
        plt.figure(figsize=(8, 5))
        sns.histplot(dotprods, bins=50, kde=True)
        plt.axvline(0, color='red', linestyle='dashed', label="Zero")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Near-Zero Numbers")
        plt.legend()
        plt.show()

    return {
        'leftwall_gaze': round(leftwall_gaze_f / on_arena_floor, 4) if on_arena_floor else 0,
        'rightwall_gaze': round(rightwall_gaze_f / on_arena_floor, 4) if on_arena_floor else 0,
        'other_gaze': round(other_gaze_f / on_arena_floor, 4) if on_arena_floor else 0,
        'on_arena_floor': round(on_arena_floor / FRAMES_PER_SECOND, 4)
    }

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

    # Convert all Y pixels to use bottom-left axis convention
    for bodypart in df.columns.levels[0]:
        df.loc[:, (bodypart, "y")] = Y_LIM - df.loc[:, (bodypart, "y")]

    # Convert X and Y pixel coordinates to centimeter distance immediately
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
    distance_travelled = valid_distances.sum()
    mean_velocity_perframe = distance_travelled / len(valid_distances) if len(valid_distances) > 0 else 0
    mean_velocity_persecond = mean_velocity_perframe * FRAMES_PER_SECOND

    # Cumulative movement duration
    movement_dur_s = len(valid_distances) / FRAMES_PER_SECOND
    nonmovement_dur_s = EXPECTED_VIDEO_LENGTH_S - movement_dur_s

    # Calculate average distance from stimwall
    if LEFT_STIMWALL_SIDE:
        wall_distances = df['Center', 'x'].apply(lambda x: abs(x - LEFT_STIMWALL_BOUNDARY_X) if x > LEFT_STIMWALL_BOUNDARY_X else 0)
    else:
        wall_distances = df['Center', 'x'].apply(lambda x: abs(x - RIGHT_STIMWALL_BOUNDARY_X) if x < RIGHT_STIMWALL_BOUNDARY_X else 0)
    avg_stimwall_dis_cm = wall_distances.mean() * X_PIXEL_RATIO

    # Time Spent in Each of the 3 zones
    left_zone_time = df[df['Center', 'x'] <= LEFT_THIRD_BOUNDARY_X].shape[0]
    center_zone_time = df[(df['Center', 'x'] > LEFT_THIRD_BOUNDARY_X) & (df['Center', 'x'] <= RIGHT_THIRD_BOUNDARY_X)].shape[0]
    right_zone_time = df[df['Center', 'x'] > RIGHT_THIRD_BOUNDARY_X].shape[0]
    left_half_time = df[df['Center', 'x'] < HALFLINE_X].shape[0]
    right_half_time = df[df['Center', 'x'] > HALFLINE_X].shape[0]

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

    # CREATE 2 LISTS OF ALL THE TURNS THE ANIMAL PERFORMS DURING THE EXPERIMENTS
    turns = []
    nowallturns = []

    # TODO: CHECK NA HANDLING
    for idx, row in df.iterrows():
        distance_head = row["Head","distance"]
        distance_center = row["Center","distance"]
        turn_degree = row['turn_degree'][0]
        x_head = row["Head", "x"]
        y_head = row["Head", "y"]
        x_center = row["Center", "x"]
        y_center = row["Center", "y"]
        # Create a reusable boolean for whether the animal is inside the arena but not touching the walls
        in_boundaries = (
                LEFT_STIMWALL_BOUNDARY_X < x_head < RIGHT_STIMWALL_BOUNDARY_X and
                LEFT_STIMWALL_BOUNDARY_X < x_center < RIGHT_STIMWALL_BOUNDARY_X and
                BOTTOM_WALL_BOUNDARY_Y < y_head < TOP_WALL_BOUNDARY_Y and
                BOTTOM_WALL_BOUNDARY_Y < y_center < TOP_WALL_BOUNDARY_Y
        )
        # If either the head or the centerpoint pass the movement threshold, and the angle passes the angle threshold,
        if (distance_head >= MOVEMENT_THRESHOLD_CM or distance_center >= MOVEMENT_THRESHOLD_CM) and abs(turn_degree) >= ANGLE_THRESHOLD:
            if in_boundaries:
                # If the angle turn is a continuation of the one from the last frame, add it.
                if nowallturns[-1] is None: # The animal has just returned inside the boundary limit
                    nowallturns.append(turn_degree)
                elif np.sign(nowallturns[-1]) == np.sign(turn_degree):
                    nowallturns[-1] += turn_degree #add to last angle turn from previous frame
                else : # Otherwise, the animal started turning in the other direction, there are no elements in the list yet,
                    nowallturns.append(turn_degree)
                # If the angle turn is a continuation of the one from the last frame, add it.
            # Regardless of whether the animal was within the boundaries of the wall or not, we need to add it to the overall turns
            if np.sign(turns[-1]) == np.sign(turn_degree):
                turns[-1] += turn_degree
            else:
                turns.append(turn_degree)
        else: # If the thresholds are not reached, we determine that the values are just noise and the animal has not turned.
            turns.append(0)
            if in_boundaries:
                nowallturns.append(0)
            else:
                nowallturns.append(None)

    # Now, we can calculate how many clockwise vs counterclockwise turns, total absolute value of turns, turn_frequency, average
    # absolute turn size.

    # Clockwise turns
    clockwise_turns = len([turn for turn in turns if turn > 0])
    nowall_clockwise_turns = len([turn for turn in nowallturns if turn is not None and turn > 0])

    # Counter-clockwise turns
    counterclockwise_turns = len([turn for turn in turns if turn < 0])
    nowall_counterclockwise_turns = len([turn for turn in nowallturns if turn is not None and turn < 0])

    # Total turns
    total_turns = clockwise_turns + counterclockwise_turns
    nowall_total_turns = nowall_clockwise_turns + nowall_counterclockwise_turns

    # Total degrees turned
    total_degrees_abs = sum(abs(angle) for angle in turns)
    nowall_total_degrees_abs = sum(abs(angle) for angle in nowallturns if angle is not None)

    # Average turn frequency per minute
    turn_frequency_m = total_turns / len(turns) / FRAMES_PER_SECOND * 60
    nowall_valid_turns = [turn for turn in nowallturns if turn is not None]
    nowall_valid_count = len(nowall_valid_turns)
    nowall_only_turns = [abs(turn) for turn in nowallturns if turn != 0 and turn is not None]
    if nowall_valid_count > 0:
        nowall_turn_frequency_m = nowall_total_turns / nowall_valid_count / FRAMES_PER_SECOND * 60
        nowall_avg_degree_size = sum(nowall_only_turns) / len(nowall_only_turns)
    else:
        nowall_turn_frequency_m = 0
        nowall_avg_degree_size = 0
    # Average turn size
    only_turns = [abs(turn) for turn in turns if turn != 0]
    avg_degree_size = sum(only_turns) / len(only_turns)

    # Using eye positions to calculate where the animal is looking at
    gaze_dict = get_gaze_direction(df)

    # If asked to save hd5, then do
    if SAVE_HD5:
        df.to_hdf(f'Locust_stats_{trial_num}.h5', key='df', mode='w')

    ultimate_stats_dict = {
        'mean_velocity_cm_s': round(mean_velocity_persecond, 4),
        'distance_travelled_cm': round(distance_travelled, 4),
        'movement_duration': movement_dur_s,
        "nonmovement_duration": nonmovement_dur_s,
        "avg_distance_from_stimwall": round(avg_stimwall_dis_cm, 4),
        "left_zone_time": left_zone_time,
        "center_zone_time": center_zone_time,
        "right_zone_time": right_zone_time,
        "left_half_time": left_half_time,
        "right_half_time": right_half_time,
        "clockwise_turns": clockwise_turns,
        "counterclockwise_turns": counterclockwise_turns,
        "total_turn_count": total_turns,
        "total_degrees_turned_abs": round(total_degrees_abs, 4),
        "turn_frequency_m": round(turn_frequency_m, 4),
        "average_degrees_turned_abs": round(avg_degree_size, 4),
        "nowall_clockwise_turns": nowall_clockwise_turns,
        "nowall_counterclockwise_turns": nowall_counterclockwise_turns,
        "nowall_total_turn_count": nowall_total_turns,
        "nowall_degrees_turned_abs": round(nowall_total_degrees_abs, 4),
        "nowall_turn_frequency_m": round(nowall_turn_frequency_m, 4),
        "nowall_average_degrees_turned_abs": round(nowall_avg_degree_size, 4)
    }
    ultimate_stats_dict.update(gaze_dict)
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
        except ValueError as e:
            print(e)
            continue
        #If debugging, use the non-trimmed video data so that the timestamps match the actual video
        if DEBUG:
            stats = calculate_stats(locustdata, trial_number)
        else:
            stats = calculate_stats(finalDF, trial_number)
        stats["trial"] = trial_number
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