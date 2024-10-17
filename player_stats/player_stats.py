import pandas as pd
from collections import Counter

class PlayerStats: 
    def __init__(self, input_df):
        # Initialize the input DataFrame
        self.input_df = input_df
        
        # Define the desired column structure for the final output (for reference only)
        self.columns = [
            'shirtNumber', 'position', 'goals', 'total_shots', 'shots_on_target', 
            'shots_off_target', 'blocked_shots', 'saved_shots', 'total_passes', 
            'accurate_passes', '%_pass_success', 'key_passes', 'dribbles_attempted', 
            'dribbles_past', 'dribbles', '%_dribbles_success', 'aerial_duels', 
            '%_aerial_success', 'offensive_aerials', 'defensive_aerials', 
            'tackles_attempted', 'tackles_won', '%_tackles_success', 'clearances', 
            'interceptions', 'injuries', 'distance_covered', 'avg_speed', 'highest_speed', 
            'team', 'team_color'
        ]

    def process_data(self):
        # Step 1: Drop rows where shirt_number is 0
        self.input_df = self.input_df[self.input_df['shirt_number'] != 0]
        self.input_df = self.input_df[self.input_df['team'] != 0]
        
        # Step 2: Combine rows based on 'shirt_number' and 'team' by summing all columns
        grouped = self.input_df.groupby(['shirt_number', 'team']).agg({
            'Position': 'first',  # Assuming 'Position' does not change for the same 'shirt_number'
            'Goals': 'sum',
            'Assists': 'sum',
            'Total_shots': 'sum',
            'Shots_on_Target': 'sum',
            'Shots_off_Target': 'sum',
            'Blocked_shots': 'sum',
            'Saved_shots': 'sum',
            'Total_passes': 'sum',
            'pass_failure': 'sum',
            'Pass_Success': 'sum',
            'Key_passes': 'sum',
            'dribble_attempt': 'sum',
            'dribble_failure': 'sum',
            'dribble_success': 'sum',
            'offensive_failure': 'sum',
            'defensive_failure': 'sum',
            'offensive_success': 'sum',
            'defensive_success': 'sum',
            'Tackles_attempted': 'sum',
            'tackling_failure': 'sum',
            'tackling_success': 'sum',
            'Clearances': 'sum',
            'Interceptions': 'sum',
            'injuries': 'sum',
            'Distance_covered': 'sum',
            'Avg_speed': lambda x: x[x != 0].mean() if (x != 0).any() else 0,  # Mean of non-zero values or 0
            'Highest_speed': 'max',  # Max of highest_speed
            'dribbled_past': 'sum'
        }).reset_index()

        # Step 3: Assign the team_color column manually
        grouped['team_color'] = grouped.apply(lambda row: self.handle_team_color(self.input_df[(self.input_df['shirt_number'] == row['shirt_number']) & 
                                                                                               (self.input_df['team'] == row['team'])]['team_color'], grouped), axis=1)
    
        # Step 4: Create team_1 and team_2 DataFrames
        team_1, team_2 = self.create_teams(grouped)

        return team_1, team_2

    def handle_team_values(self, team_series):
        """
        Handle missing team values: If all team values are 0, assign to team 1.
        """
        # If all values are 0, return team 1
        if (team_series == 0).all():
            return 1
        else:
            # Otherwise, return the most frequent non-zero value
            return Counter([i for i in team_series if i != 0]).most_common(1)[0][0]

    def handle_team_color(self, team_color_series, grouped_df):
        """
        Handle missing team_color values: Save the first non-zero/non-null value, 
        or return 'Unknown' if no valid colors are found.
        """

        # Filter out empty or null values
        valid_colors = team_color_series[team_color_series.notnull() & (team_color_series != 0)]
        
        if len(valid_colors) > 0:
            # Return the first valid color
            return valid_colors.iloc[0]
        else:
            # Return a fallback value if no valid colors are found
            return "Unknown"
        
    def create_teams(self, grouped_df):
        # Split the combined DataFrame into two based on the 'team' column (values 1 and 2)
        team_1 = grouped_df[grouped_df['team'] == 1]
        team_2 = grouped_df[grouped_df['team'] == 2]
        # Map columns and return final DataFrames
        team_1_df = self.map_columns(team_1)
        team_2_df = self.map_columns(team_2)
        
        return team_1_df, team_2_df

    def map_columns(self, df):
        # Create a new DataFrame with mapped columns and calculations
        new_df = pd.DataFrame()

        # Direct mappings
        new_df['shirtNumber'] = df['shirt_number']
        new_df['position'] = df['Position']
        new_df['goals'] = df['Goals']
        new_df['total_shots'] = df['Total_shots']
        new_df['shots_on_target'] = df['Shots_on_Target']
        new_df['shots_off_target'] = df['Shots_off_Target']
        new_df['blocked_shots'] = df['Blocked_shots']
        new_df['saved_shots'] = df['Saved_shots']
        new_df['total_passes'] = df['Total_passes']
        new_df['accurate_passes'] = df['Pass_Success']
        new_df['key_passes'] = df['Key_passes']
        new_df['dribbles_attempted'] = df['dribble_attempt']
        new_df['dribbles_past'] = df['dribbled_past']
        new_df['dribbles'] = df['dribble_success']
        new_df['tackles_attempted'] = df['Tackles_attempted']
        new_df['tackles_won'] = df['tackling_success']
        new_df['clearances'] = df['Clearances']
        new_df['interceptions'] = df['Interceptions']
        new_df['injuries'] = df['injuries']
        new_df['distance_covered'] = df['Distance_covered']
        new_df['avg_speed'] = df['Avg_speed']
        new_df['highest_speed'] = df['Highest_speed']
        new_df['team'] = df['team']
        new_df['team_color'] = df['team_color']

        # Calculations
        new_df['%_pass_success'] = (df['Pass_Success'] / df['Total_passes']) * 100
        new_df['%_dribbles_success'] = (df['dribble_success'] / df['dribble_attempt']) * 100
        new_df['aerial_duels'] = (df['offensive_success'] + df['defensive_success'])
        new_df['%_aerial_success'] = new_df['aerial_duels'] / (df['offensive_failure'] + df['defensive_failure'] +
                                      df['offensive_success'] + df['defensive_success']) * 100
        new_df['offensive_aerials'] = df['offensive_success'] + df['offensive_failure']
        new_df['defensive_aerials'] = df['defensive_success'] + df['defensive_failure']
        new_df['%_tackles_success'] = df['tackling_success'] / df['Tackles_attempted'] * 100
        new_df = new_df.fillna(0)
        
        return new_df
