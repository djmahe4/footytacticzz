import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox

class EventProcessor:
    def __init__(self, tracks, yolo_detections, df):
        """
        Initialize the EventProcessor with tracking data, YOLO detections, and a DataFrame for storing results.
        :param tracks: Dictionary containing the tracking data of players.
        :param yolo_detections: YOLO model results from the prediction (list of lists).
        :param df: A Pandas DataFrame to store player statistics (dribbles, tackles, aerial_duels, injuries).
        """
        self.tracks = tracks
        self.yolo_detections = yolo_detections
        self.df = df  # Reference to the DataFrame

    def find_closest_player(self, event_center, frame_num):
        distances = []
    
        # Iterate over player detections in the given frame
        for player_id, player_data in self.tracks['players'][frame_num].items():
            player_bbox = player_data['bbox']
            player_center = get_center_of_bbox(player_bbox)
            distance = np.linalg.norm(np.array(event_center) - np.array(player_center))
            distances.append((player_id, distance))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Check if distances list is not empty before accessing
        if len(distances) > 0:
            closest_player = distances[0][0]
            return closest_player
        else:
            return None  # Handle case where no players are detected

    def find_ball_possessor(self, frame_num):
        """
        Find the player who has possession of the ball in a given frame.
        :param frame_num: The frame number being processed.
        :return: The player ID of the ball possessor or None if no one has the ball.
        """
        if frame_num < 0 or frame_num >= len(self.tracks['players']):
            return None

        for player_id, data in self.tracks['players'][frame_num].items():
            if data.get('has_ball', False):  # Assuming 'has_ball' is True if the player has the ball
                return player_id
        return None

    def update_statistics(self, player_id, column_name):
        """
        Update the specified column in the DataFrame for a given player.
        :param player_id: The ID of the player.
        :param column_name: The column to update (e.g., 'dribble_success', 'injuries').
        """
        initialized_columns = ['dribble_attempt', 'dribble_success', 'dribble_failure', 'dribbled_past',
                               'offensive_success', 'offensive_failure', 'defensive_success', 'defensive_failure',
                               'tackling_success', 'tackling_failure', 'injuries']

        if player_id not in self.df.index or self.df.loc[player_id, initialized_columns].isnull().any():
            self.df.loc[player_id] = {'dribble_attempt': 0, 'dribble_success': 0, 'dribble_failure': 0, 'dribbled_past': 0,
                                      'offensive_success': 0, 'offensive_failure': 0, 'defensive_success': 0, 'defensive_failure': 0,
                                      'Tackles_attempted': 0, 'tackling_success': 0, 'tackling_failure': 0, 'injuries': 0}
        self.df.at[player_id, column_name] += 1

    def group_event_frames(self, event_name, min_gap=20):
        """
        Group consecutive frames for the same event into a list of events, allowing for a gap of up to `min_gap` frames.
    
        :param event_name: The name of the event (e.g., 'dribble', 'tackle', 'aerial_duel', 'injury').
        :param min_gap: Maximum number of frames to allow as a gap between two parts of the same event.
        :return: A list of grouped events, each containing the start and end frame of the event.
        """
        event_class_ids = {
            'dribble': 1,  # Example mapping of event names to class IDs
            'tackle': 6,
            'aerial_duel': 0,
            'injury': 4
        }

        event_class_id = event_class_ids.get(event_name)
        if event_class_id is None:
            return []  # Return empty if the event name is not valid

        grouped_events = []
        ongoing_events = {}  # Dictionary to track ongoing events: {track_id: (start_frame, last_frame)}
        gap_count = {}  # Dictionary to track gaps for each event
    
        # Iterate through the frames and detect the events
        for frame_num, detections in enumerate(self.yolo_detections):
            # Filter out only the detections relevant to the event
            relevant_detections = [detection for detection in detections if int(detection[5]) == event_class_id]

            # Process each event in the frame separately
            for detection in relevant_detections:
                track_id = int(detection[4])  # Assuming detection[4] is the unique identifier for the event (like a player ID)

                # If the event is already ongoing, update its end frame
                if track_id in ongoing_events:
                    ongoing_events[track_id] = (ongoing_events[track_id][0], frame_num)  # Update end frame
                    gap_count[track_id] = 0  # Reset the gap counter for this event
                else:
                    # Start a new event for this track_id with current frame as start_frame and end_frame
                    ongoing_events[track_id] = (frame_num, frame_num)
                    gap_count[track_id] = 0  # Initialize the gap counter for this event

            # Handle gaps for all ongoing events
            for track_id in list(ongoing_events.keys()):  # Use list() to avoid modifying the dictionary while iterating
                if track_id not in [int(detection[4]) for detection in relevant_detections]:
                    # No detection for this track_id in the current frame, increment the gap counter
                    gap_count[track_id] += 1
                    if gap_count[track_id] > min_gap:
                        # End the event if the gap exceeds the allowed size, and append start and end frames
                        start_frame, end_frame = ongoing_events[track_id]
                        grouped_events.append((start_frame, end_frame))
                        del ongoing_events[track_id]  # Remove the completed event
                        del gap_count[track_id]  # Remove the gap counter for this event

        # Append any remaining ongoing events after looping through frames
        for track_id, (start_frame, end_frame) in ongoing_events.items():
            grouped_events.append((start_frame, end_frame))

        return grouped_events


    def process_dribble(self):
        """
        Process grouped dribble events, update dribble attempts and success/failure stats.
        """
        grouped_dribbles = self.group_event_frames('dribble')
        print(len(grouped_dribbles))
        for start_frame, end_frame in grouped_dribbles:
            dribbler = self.find_ball_possessor(start_frame - 1 if start_frame > 0 else start_frame)

            if dribbler is not None:
                self.update_statistics(dribbler, 'dribble_attempt')

                if end_frame + 1 < len(self.tracks['players']):
                    player_with_ball_after = self.find_ball_possessor(end_frame + 1)
                else:
                    player_with_ball_after = None

                if player_with_ball_after == dribbler:
                    self.update_statistics(dribbler, 'dribble_success')
                    event_bbox = self.yolo_detections[end_frame][0][:4]
                    defender = self.find_closest_player(get_center_of_bbox(event_bbox), end_frame)
                    if defender is not None:
                        self.update_statistics(defender, 'dribbled_past')
                else:
                    self.update_statistics(dribbler, 'dribble_failure')

    def process_aerial_duel(self):
        """
        Process grouped aerial duel events, update offensive and defensive success/failure stats.
        """
        grouped_aerial_duels = self.group_event_frames('aerial_duel')

        for start_frame, end_frame in grouped_aerial_duels:
            attacker = self.find_ball_possessor(start_frame - 1 if start_frame > 0 else start_frame)

            if attacker is not None:
                attacker_team = self.tracks['players'][start_frame][attacker]['team']

                if end_frame + 1 < len(self.tracks['players']):
                    player_with_ball_after = self.find_ball_possessor(end_frame + 1)
                else:
                    player_with_ball_after = None

                event_bbox = self.yolo_detections[end_frame][0][:4]
                event_center = get_center_of_bbox(event_bbox)
                defender = self.find_second_closest_opponent(event_center, end_frame, attacker_team)

                if player_with_ball_after == attacker:
                    self.update_statistics(attacker, 'offensive_success')
                    if defender is not None:
                        self.update_statistics(defender, 'defensive_failure')
                else:
                    self.update_statistics(attacker, 'offensive_failure')
                    if defender is not None:
                        self.update_statistics(defender, 'defensive_success')

    def find_second_closest_opponent(self, event_center, frame_num, attacker_team):
        """
        Find the second closest player to the event who is on the opposing team (likely the defender).
        """
        distances = []

        for player_id, player_data in self.tracks['players'][frame_num].items():
            if player_data['team'] != attacker_team:
                player_bbox = player_data['bbox']
                player_center = get_center_of_bbox(player_bbox)
                distance = np.linalg.norm(np.array(event_center) - np.array(player_center))
                distances.append((player_id, distance))

        distances.sort(key=lambda x: x[1])
    
        return distances[1][0] if len(distances) > 1 else None
    
    def process_tackle(self):
        """
        Process grouped tackle events, update tackling success/failure stats.
        """
        grouped_tackles = self.group_event_frames('tackle')

        for start_frame, end_frame in grouped_tackles:
            tackled_player = self.find_ball_possessor(start_frame - 1 if start_frame > 0 else start_frame)

            if tackled_player is not None:
                tackled_team = self.tracks['players'][start_frame][tackled_player]['team']

                if end_frame + 1 < len(self.tracks['players']):
                    player_with_ball_after = self.find_ball_possessor(end_frame + 1)
                else:
                    player_with_ball_after = None

                event_bbox = self.yolo_detections[end_frame][0][:4]
                event_center = get_center_of_bbox(event_bbox)
                tackler = self.find_second_closest_opponent(event_center, end_frame, tackled_team)

                if player_with_ball_after != tackled_player:
                    # Tackle success
                    self.update_statistics(tackler, 'Tackles_attempted')
                    if tackler is not None:
                        self.update_statistics(tackler, 'tackling_success')
                else:
                    # Tackle failure for the tackler
                    if tackler is not None:
                        self.update_statistics(tackler, 'tackling_failure')

    def process_injury(self):
        """
        Process grouped injury events, update injury stats.
        """
        grouped_injuries = self.group_event_frames('injury')

        for start_frame, end_frame in grouped_injuries:
            event_bbox = self.yolo_detections[end_frame][0][:4]
            injured_player = self.find_closest_player(get_center_of_bbox(event_bbox), end_frame)

            if injured_player is not None:
                # Update injury statistics for the injured player
                self.update_statistics(injured_player, 'injuries')

    def process_frames_in_batches(self):
        """
        Process events in batches of frames.
        :param batch_size: The number of frames to process in each batch.
        """

        self.process_dribble()
        self.process_aerial_duel()
        self.process_tackle()
        self.process_injury()

        return self.df