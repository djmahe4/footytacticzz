import sys 
sys.path.append('../')
from utils import measure_distance 

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
    
    def update_df_with_speed_and_distance(self, tracks, df):
        # List of columns to initialize
        columns_to_initialize = ['Distance_covered', 'Avg_speed', 'Highest_speed']

        # Initialize the columns with zeros (whether they already exist or not)
        for column in columns_to_initialize:
            if column not in df.columns:
                df[column] = 0  # Add the column if it doesn't exist
            else:
                df[column] = df[column].fillna(0)  # Use direct assignment to avoid FutureWarning

        total_distance = {}
        speed_sum = {}
        highest_speed = {}

        # Iterate through each object (player, team, etc.)
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees" or object == "goalkeepers":
                continue

            number_of_frames = len(object_tracks)

            # Iterate through frames to compute distance and speed
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Check if 'position_transformed' exists
                    if 'position_transformed' not in object_tracks[frame_num][track_id] or 'position_transformed' not in object_tracks[last_frame][track_id]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    # Compute distance covered and speed
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Initialize player stats if not already in the dictionaries
                    if track_id not in total_distance:
                        total_distance[track_id] = 0
                        speed_sum[track_id] = 0
                        highest_speed[track_id] = 0

                    # Update total distance, speed sum, and highest speed for the player
                    total_distance[track_id] += distance_covered
                    speed_sum[track_id] += speed_km_per_hour
                    highest_speed[track_id] = max(highest_speed[track_id], speed_km_per_hour)

            # After processing all frames, calculate average speed for each player
            for track_id in total_distance.keys():
                avg_speed = speed_sum[track_id] / (number_of_frames / self.frame_window)

                # Update the total_distance_covered, avg_speed, and highest_speed in the DataFrame using track_id as the index
                df.at[track_id, 'Distance_covered'] = total_distance[track_id]
                df.at[track_id, 'Avg_speed'] = avg_speed
                df.at[track_id, 'Highest_speed'] = highest_speed[track_id]

        return df