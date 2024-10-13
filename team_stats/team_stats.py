import pandas as pd

class SoccerMatchDataProcessorFullWithSubs:
    def __init__(self, df_team1, df_team2, df_match_info):
        # Store the dataframes for both teams and the match info
        self.df_team1 = df_team1
        self.df_team2 = df_team2
        self.df_match_info = df_match_info

        # Define the columns for the final DataFrame
        self.final_columns = [
            'id_match', 'formations', 'score', 'goals', 'goals_past', 
            'total_shots', 'shots_on_target', 'shots_off_target', 'blocked_shots', 
            'total_possession', 'total_passes', 'accurate_passes', 'pass_success', 
            'key_passes', 'dribbles_attempted', 'dribbles', 'dribbles_past', 'dribbles_success', 
            'aerials_won', 'aerial_success', 'offensive_aerials', 'defensive_aerials', 
            'tackles', 'tackle_success', 'tackles_attempted', 'clearances', 
            'interceptions', 'corners', 'win', 'draw', 'lose', 'team_color',
            'substitution_1', 'substitution_2', 'substitution_3', 'substitution_4', 'substitution_5'
        ]

        # Initialize the final DataFrame
        self.final_df = pd.DataFrame(columns=self.final_columns)

    def calculate_team_stats(self, team_df):
        # Calculate the necessary statistics for a team
        stats = {
            'goals': team_df['goals'].sum(),
            'total_shots': team_df['total_shots'].sum(),
            'shots_on_target': team_df['shots_on_target'].sum(),
            'shots_off_target': team_df['shots_off_target'].sum(),
            'blocked_shots': team_df['blocked_shots'].sum(),
            'total_passes': team_df['total_passes'].sum(),
            'accurate_passes': team_df['accurate_passes'].sum(),
            'pass_success': (team_df['accurate_passes'].sum() / team_df['total_passes'].sum()) * 100 if team_df['total_passes'].sum() > 0 else 0,
            'key_passes': team_df['key_passes'].sum(),
            'dribbles_attempted': team_df['dribbles_attempted'].sum(),
            'dribbles': team_df['dribbles'].sum(),
            'dribbles_past': team_df['dribbles_past'].sum(),
            'dribbles_success': (team_df['dribbles'].sum() / team_df['dribbles_attempted'].sum()) * 100 if team_df['dribbles_attempted'].sum() > 0 else 0,
            'aerials_won': team_df['aerial_duels'].sum(),
            'aerial_success': team_df['%_aerial_success'].mean(),
            'offensive_aerials': team_df['offensive_aerials'].sum(),
            'defensive_aerials': team_df['defensive_aerials'].sum(),
            'tackles': team_df['tackles_won'].sum(),
            'tackle_success': team_df['%_tackles_success'].mean(),
            'tackles_attempted': team_df['tackles_attempted'].sum(),
            'clearances': team_df['clearances'].sum(),
            'interceptions': team_df['interceptions'].sum(),
            'team_color': team_df.loc[team_df['team_color'] != 'Unknown', 'team_color'].iloc[0] if not team_df[team_df['team_color'] != 'Unknown'].empty else 'Unknown'
        }
        return stats

    def process_match_data(self):
        # Process match data for both teams
        
        # Extract team 1 and 2 data from match info and ensure proper casting of formations and substitutions
        team1_info = self.df_match_info.iloc[0]
        team2_info = self.df_match_info.iloc[1]

        # Handle string types for formations and substitutions by ensuring they are strings
        team1_info['formations'] = str(team1_info['formations'])
        team2_info['formations'] = str(team2_info['formations'])

        # Substitution columns may also need to be cast to strings
        substitution_columns = ['substitution_1', 'substitution_2', 'substitution_3', 'substitution_4', 'substitution_5']
        for col in substitution_columns:
            team1_info[col] = str(team1_info[col])
            team2_info[col] = str(team2_info[col])

        # Calculate stats for team 1
        team1_stats = self.calculate_team_stats(self.df_team1)
        team1_stats.update({
            'formations': team1_info['formations'],
            'corners': team1_info['corners'],
            'substitution_1': team1_info['substitution_1'],
            'substitution_2': team1_info['substitution_2'],
            'substitution_3': team1_info['substitution_3'],
            'substitution_4': team1_info['substitution_4'],
            'substitution_5': team1_info['substitution_5'],
            'id_match': 1,  # Assign the match ID
            'win': 0, 'draw': 0, 'lose': 0
        })

        # Calculate stats for team 2
        team2_stats = self.calculate_team_stats(self.df_team2)
        team2_stats.update({
            'formations': team2_info['formations'],
            'corners': team2_info['corners'],
            'substitution_1': team2_info['substitution_1'],
            'substitution_2': team2_info['substitution_2'],
            'substitution_3': team2_info['substitution_3'],
            'substitution_4': team2_info['substitution_4'],
            'substitution_5': team2_info['substitution_5'],
            'id_match': 1,  # Assign the match ID
            'win': 0, 'draw': 0, 'lose': 0
        })

        # Calculate total possession based on total passes
        total_passes = team1_stats['total_passes'] + team2_stats['total_passes']
        team1_stats['total_possession'] = (team1_stats['total_passes'] / total_passes) * 100 if total_passes > 0 else 0
        team2_stats['total_possession'] = (team2_stats['total_passes'] / total_passes) * 100 if total_passes > 0 else 0

        # Determine goals past for each team
        team1_stats['goals_past'] = team2_stats['goals']
        team2_stats['goals_past'] = team1_stats['goals']

        # Assign the score in the format "our goals - their goals"
        team1_stats['score'] = f"{team1_stats['goals']} - {team2_stats['goals']}"
        team2_stats['score'] = f"{team2_stats['goals']} - {team1_stats['goals']}"

        # Determine the result based on goals
        if team1_stats['goals'] > team2_stats['goals']:
            team1_stats['win'] = 1
            team2_stats['lose'] = 1
        elif team1_stats['goals'] < team2_stats['goals']:
            team2_stats['win'] = 1
            team1_stats['lose'] = 1
        else:
            team1_stats['draw'] = 1
            team2_stats['draw'] = 1

        # Use pd.concat instead of append to add the rows
        self.final_df = pd.concat([self.final_df, pd.DataFrame([team1_stats])], ignore_index=True)
        self.final_df = pd.concat([self.final_df, pd.DataFrame([team2_stats])], ignore_index=True)

        # Fill NaN values with 0 for missing data
        self.final_df = self.final_df.fillna(0)

        return self.final_df