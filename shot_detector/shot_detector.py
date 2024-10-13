import sys 
sys.path.append('../')
from utils import get_center_of_bbox

class ShotDetector:
    def __init__(self, tracks, df, team_df, annotations_data, batch_size=60):
        """
        Initialize the ShotDetector with tracking data, a DataFrame for stats, and annotations data.
        :param tracks: Dictionary containing the tracking data of players and the ball.
        :param df: A Pandas DataFrame to store player statistics.
        :param team_df: A Pandas DataFrame to store team statistics.
        :param annotations_data: Dictionary with goal line and goal area points per frame.
        :param batch_size: Number of frames to process in each batch.
        """
        self.tracks = tracks
        self.df = df
        self.team_df = team_df
        self.annotations_data = annotations_data
        self.corners_team_1 = 0
        self.corners_team_2 = 0
        self.player_with_ball_history = []
        self.skip_until_frame = 0  # Tracks when to start processing again after skipping frames
        self.batch_size = batch_size  # Store the batch size to calculate frame skips

    def point_in_goal_area(self, ball_position, frame_num):
        """
        Check if the ball is inside the rectangular goal area for the current frame.
        If no goal polygons are defined for the current frame, check adjacent frames (20 backward and 20 forward).
        """
        x, y = ball_position
        # Step 1: Check for goal polygons in the current frame
        goal_polygons = self.annotations_data.get(frame_num, {}).get('goal_points', [])
    
        if goal_polygons:
            # If goal polygons are available for the current frame, check normally
            for goal_area in goal_polygons:
                if len(goal_area) != 4:
                    continue  # Skip non-rectangular areas

                # Extract x and y coordinates
                x_coords = [point[0] for point in goal_area]
                y_coords = [point[1] for point in goal_area]

                # Get min and max bounds
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                print(f"{min_x} and {max_x} and {min_y} and {max_y}")

                # Check if the ball is within the bounds
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    return True
        
            # If no goal area contains the ball, return False for the current frame
            return False

        # Step 2: If no goal polygons for the current frame, check adjacent frames
        frame_range = range(frame_num - 60, frame_num + 60)

        for frame in frame_range:
            # Safely handle missing frames in the dataset
            goal_polygons = self.annotations_data.get(frame, {}).get('goal_points', [])
        
            if not goal_polygons:
                continue  # Skip frames that don't exist or have no goal points

            # Check the goal polygons in the same way as above
            for goal_area in goal_polygons:
                if len(goal_area) != 4:
                    continue  # Skip non-rectangular areas

                # Extract x and y coordinates
                x_coords = [point[0] for point in goal_area]
                y_coords = [point[1] for point in goal_area]

                # Get min and max bounds
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # Check if the ball is within the bounds
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    return True

        # Return False if no goal area contains the ball across all the checked frames
        return False

    def ball_crosses_goal_line(self, ball_position, frame_num):
        """
        Check if the ball has crossed the goal line (used to detect shots).
        """
        x_ball, y_ball = ball_position
        goal_line_points = self.annotations_data[frame_num]['goal_line_points']
        if not goal_line_points:
            return False

        nearest_line = self.get_nearest_goal_line(goal_line_points)
        (x1, y1), (x2, y2) = nearest_line
        cross_product = (x2 - x1) * (y_ball - y1) - (y2 - y1) * (x_ball - x1)
        return cross_product > 0

    def get_nearest_goal_line(self, goal_line_points):
        """
        Get the nearest goal line to the edge of the frame.
        """
        min_distance = float('inf')
        nearest_line = goal_line_points[0]
        for line in goal_line_points:
            x1, y1 = line[0]
            x2, y2 = line[1]
            distance_from_edge = min(x1, x2)
            if distance_from_edge < min_distance:
                min_distance = distance_from_edge
                nearest_line = line
        return nearest_line

    def detect_shots_and_outcomes(self, frame_range):
        """
        Detect shots, goals, assists, key passes, and corners from a range of frames.
        """
        for frame_num in frame_range:
            if frame_num < self.skip_until_frame:
                continue  # Skip processing frames until the skip period is over

            if frame_num not in self.annotations_data:
                continue

            ball_position = self.tracks['ball'][frame_num][1]['bbox']
            ball_position = get_center_of_bbox(ball_position)
            self.track_passes(frame_num)

            current_shooter = self.player_with_ball_history[-1] if self.player_with_ball_history else None

            if current_shooter:
                last_touch_team = self.tracks['players'][frame_num][current_shooter]['team']

                if self.ball_crosses_goal_line(ball_position, frame_num):
                    self.update_statistics(current_shooter, 'Total_shots')
                    if self.point_in_goal_area(ball_position, frame_num):
                        self.update_statistics(current_shooter, 'Goals')
                        self.update_statistics(current_shooter, 'Shots_on_Target')
                        assist_player = self.find_assist_player(current_shooter, frame_num)
                        if assist_player:
                            self.update_statistics(assist_player, 'Assists')
                    else:
                        if self.is_saved(current_shooter, frame_num):
                            self.update_statistics(current_shooter, 'Saved_shots')
                            self.update_statistics(current_shooter, 'Shots_on_Target')
                            if self.is_corner(ball_position, frame_num):
                                if last_touch_team == 1:
                                    self.corners_team_2 += 1
                                else:
                                    self.corners_team_1 += 1
                        elif self.is_blocked(current_shooter, frame_num):
                            self.update_statistics(current_shooter, 'Blocked_shots')
                            self.update_statistics(current_shooter, 'Shots_on_Target')
                            if self.is_corner(ball_position, frame_num):
                                if last_touch_team == 1:
                                    self.corners_team_2 += 1
                                else:
                                    self.corners_team_1 += 1
                        else:
                            self.update_statistics(current_shooter, 'Shots_off_Target')

                        key_pass_player = self.find_assist_player(current_shooter, frame_num)
                        if key_pass_player:
                            self.update_statistics(key_pass_player, 'Key_passes')

                    

                    # Skip the next 2 batches of frames after a shot is detected
                    self.skip_until_frame = frame_num + (7 * self.batch_size)

    def is_corner(self, ball_position, frame_num):
        """
        Determine if a corner should be awarded.
        """
        if self.ball_crosses_goal_line(ball_position, frame_num) and not self.point_in_goal_area(ball_position, frame_num):
            return True
        return False

    def track_passes(self, frame_num):
        """
        Track the passing sequence by detecting when the player who has the ball changes.
        """
        player_with_ball = None
        for player_id, player_data in self.tracks['players'][frame_num].items():
            if player_data.get('has_ball', False):
                player_with_ball = player_id
                break

        if self.player_with_ball_history and self.player_with_ball_history[-1] != player_with_ball:
            self.player_with_ball_history.append(player_with_ball)
            if len(self.player_with_ball_history) > 10:
                self.player_with_ball_history.pop(0)
        elif not self.player_with_ball_history:
            self.player_with_ball_history.append(player_with_ball)

    def find_assist_player(self, current_shooter, frame_num):
        """
        Find the player who made the pass to the current shooter.
        """
        shooter_team = self.tracks['players'][frame_num][current_shooter]['team']

        for player_id in reversed(self.player_with_ball_history[:-1]):
            if player_id is None or player_id not in self.tracks['players'][frame_num]:
                continue
            player_team = self.tracks['players'][frame_num][player_id]['team']
            if player_id == current_shooter:
                continue
            if player_team == shooter_team:
                return player_id
        return None

    def is_saved(self, current_shooter, frame_num):
        """
        Determine if the shot was saved by the goalkeeper.
        """
        shooter_team = self.tracks['players'][frame_num][current_shooter]['team']
        shooter_found = False

        for player_id in reversed(self.player_with_ball_history):
            if player_id is None or player_id not in self.tracks['players'][frame_num]:
                continue
            if player_id == current_shooter:
                shooter_found = True
                continue
            if shooter_found:
                player_team = self.tracks['players'][frame_num][player_id]['team']
                if player_team != shooter_team:
                    self.update_statistics(player_id, 'Clearances')
                    return True
        return False

    def is_blocked(self, current_shooter, frame_num):
        """
        Determine if the shot was blocked by a defender.
        """
        shooter_team = self.tracks['players'][frame_num][current_shooter]['team']
        shooter_found = False

        for player_id in reversed(self.player_with_ball_history):
            if player_id is None or player_id not in self.tracks['players'][frame_num]:
                continue
            if player_id == current_shooter:
                shooter_found = True
                continue
            if shooter_found:
                player_team = self.tracks['players'][frame_num][player_id]['team']
                if player_team != shooter_team:
                    self.update_statistics(player_id, 'Blocked_shots')
                    return True
        return False

    def update_statistics(self, player_id, column_name):
        """
        Update the specified column in the DataFrame for a given player.
        """
        initialized_columns = ['Goals', 'Assists', 'Key_passes', 'Shots_on_Target',
                               'Shots_off_Target', 'Saved_shots', 'Blocked_shots',
                               'Clearances',"Total_shots"]
        if player_id not in self.df.index or self.df.loc[player_id, initialized_columns].isnull().any():
            self.df.loc[player_id] = {col: 0 for col in initialized_columns}

        # Increment the specific column for the player
        self.df.at[player_id, column_name] += 1

    def process_frames_in_batches(self, batch_size=60):
        """
        Process the game data in batches of frames.
        :param batch_size: The number of frames to process in each batch.
        :return: None. The corners are assigned to the team DataFrame.
        """
        total_frames = len(self.tracks['ball'])

        for start_frame in range(0, total_frames, batch_size):
            end_frame = min(start_frame + batch_size, total_frames)
            frame_range = list(range(start_frame, end_frame))

            self.detect_shots_and_outcomes(frame_range)

        # Assign the number of corners to the DataFrame for both teams
        self.team_df.at[1, 'corners'] = self.corners_team_1
        self.team_df.at[2, 'corners'] = self.corners_team_2

        return self.df, self.team_df