class PassDetector:
    def __init__(self, tracks, df):
        """
        Initialize the PassDetector with tracking data of players and the reference to a DataFrame.
        :param tracks: Dictionary containing the tracking data of players and ball positions.
        :param df: A Pandas DataFrame to store pass results with player IDs as rows and columns:
                   'pass_success', 'pass_failure', 'total_passes', 'interceptions'
        """
        self.tracks = tracks
        self.df = df  # Reference to the DataFrame
        self.last_player_with_the_ball = None
        self.last_frame_num = None
        
    def update_pass_results(self, player_id, success, interceptor_id=None):
        """
        Update the pass results for a specific player in the DataFrame.
        :param player_id: The ID of the player making the pass.
        :param success: Boolean indicating if the pass was successful (True) or failed (False).
        :param interceptor_id: ID of the player who intercepted the ball (if pass failed).
        """
        # List of columns that are being initialized by this function
        initialized_columns = ['Pass_Success', 'pass_failure', 'pass_failure', 'Interceptions']

        # Check if player_id exists in df or if any of the initialized columns for the player contain NaN values
        if player_id not in self.df.index or self.df.loc[player_id, initialized_columns].isnull().any():
            self.df.loc[player_id] = {
                'Pass_Success': 0, 
                'pass_failure': 0, 
                'pass_failure': 0, 
                'Interceptions': 0
            }

        if success:
            self.df.at[player_id, 'Pass_Success'] += 1
        else:
            self.df.at[player_id, 'pass_failure'] += 1

            # If the pass failed, and an interceptor is identified, update their interception count
            if interceptor_id is not None:
                if interceptor_id not in self.df.index or self.df.loc[interceptor_id].isnull().any():
                    self.df.loc[interceptor_id] = {
                        'Pass_Success': 0, 
                        'pass_failure': 0, 
                        'Total_passes': 0, 
                        'Interceptions': 0
                    }
                self.df.at[interceptor_id, 'Interceptions'] += 1

        # Update total passes
        self.df.at[player_id, 'Total_passes'] = self.df.at[player_id, 'Pass_Success'] + self.df.at[player_id, 'pass_failure']

    def detect_pass(self, player_id, player_data, frame_num):
        """
        Detect passes frame by frame for a given player.
        :param player: The player object representing the player who currently has the ball.
        :param frame_num: The current frame number being analyzed.
        :return: None
        """
        if self.last_frame_num:
            current_team = self.tracks['players'][self.last_frame_num][self.last_player_with_the_ball]['team']
        
        # Handle cases where 'has_ball' is not set, assume False if missing
        has_ball = player_data.get('has_ball', False)
        
        # If the player has the ball in this frame
        if has_ball:
            # Check if the ball was passed to a player
            if self.last_player_with_the_ball and self.last_player_with_the_ball != player_id:
                next_team = self.tracks['players'][frame_num][player_id]['team']
                if self.last_frame_num and current_team == next_team:
                    # Successful pass
                    self.update_pass_results(self.last_player_with_the_ball, success=True)
                else:
                    # Failed pass, interception by the opposing player
                    self.update_pass_results(self.last_player_with_the_ball, success=False, interceptor_id=player_id)

            self.last_player_with_the_ball = player_id
            self.last_frame_num = frame_num
                
    def process_game_in_batches(self, batch_size=20):
        """
        Process the game frame by frame in batches to detect and classify passes.
        :param batch_size: Number of frames to process in each batch.
        :return: Updated DataFrame with pass results
        """
        total_frames = len(self.tracks['players'])
        
        for start_frame in range(0, total_frames, batch_size):
            end_frame = min(start_frame + batch_size, total_frames)  # Ensure we don't go out of bounds
            
            for frame_num in range(start_frame, end_frame):
                players_in_frame = self.tracks['players'][frame_num]
                for player_id, player_data in players_in_frame.items():
                    self.detect_pass(player_id, player_data, frame_num)

        # Return the updated DataFrame after processing
        return self.df