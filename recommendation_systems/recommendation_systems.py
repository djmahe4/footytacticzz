import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from pulp import LpProblem, LpVariable, lpSum, LpMaximize
from collections import Counter
import numpy as np
import difflib
import sys
sys.path.append('../')
from utils import correct_formation_format

class MyPlayerStats:
    def __init__(self, input_df, correct_shirt_numbers, mobile_data_df):
        self.input_df = input_df
        self.correct_shirt_numbers = correct_shirt_numbers
        self.mobile_data_df = mobile_data_df  # The mobile data DataFrame
        self.columns = [
            'position', 'shirtNumber', 'goals', 'total_shots',
            'shots_on_target', 'shots_off_target', 'blocked_shots', 'saved_shots', 'total_passes', 'accurate_passes',
            '%_pass_success', 'key_passes', 'dribbles_attempted', 'dribbles', '%_dribbles_success', 'dribbles_past',
            'aerial_duels', '%_aerial_success', 'offensive_aerials', 'defensive_aerials', 'tackles_attempted',
            'tackles_won', '%_tackles_success', 'clearances', 'interceptions', 'injuries', 'distance_covered',
            'avg_speed', 'highest_speed'
        ]

    def correct_shirt_numbers_func(self):
        def find_closest_shirt_number(incorrect_number):
            correct_number = difflib.get_close_matches(str(incorrect_number), self.correct_shirt_numbers, n=1, cutoff=0.6)
            return correct_number[0] if correct_number else incorrect_number

        self.input_df['corrected_shirt_number'] = self.input_df['shirtNumber'].apply(find_closest_shirt_number)

    def process_data(self):
        self.correct_shirt_numbers_func()

        # Group the data as before
        grouped = self.input_df.groupby(['corrected_shirt_number']).agg({
            'position': 'first',
            'goals': 'sum',
            'total_shots': 'sum',
            'shots_on_target': 'sum',
            'shots_off_target': 'sum',
            'blocked_shots': 'sum',
            'saved_shots': 'sum',
            'total_passes': 'sum',
            'accurate_passes': 'sum',
            '%_pass_success': lambda x: x[x != 0].mean() if (x != 0).any() else 0,
            'key_passes': 'sum',
            'dribbles_attempted': 'sum',
            'dribbles': 'sum',
            '%_dribbles_success': lambda x: x[x != 0].mean() if (x != 0).any() else 0,
            'dribbles_past': 'sum',
            'aerial_duels': 'sum',
            '%_aerial_success': lambda x: x[x != 0].mean() if (x != 0).any() else 0,
            'offensive_aerials': 'sum',
            'defensive_aerials': 'sum',
            'tackles_attempted': 'sum',
            'tackles_won': 'sum',
            '%_tackles_success': lambda x: x[x != 0].mean() if (x != 0).any() else 0,
            'clearances': 'sum',
            'interceptions': 'sum',
            'injuries': 'sum',
            'distance_covered': 'sum',
            'avg_speed': lambda x: x[x != 0].mean() if (x != 0).any() else 0,
            'highest_speed': 'max',
        }).reset_index()

        # After processing, assign the 'position' from the mobile_data_df
        grouped = self.assign_position_from_mobile_data(grouped)

        return grouped

    def assign_position_from_mobile_data(self, grouped_df):
        # Convert both columns to string for a proper merge
        grouped_df['corrected_shirt_number'] = grouped_df['corrected_shirt_number'].astype(str)
        self.mobile_data_df['Shirt_Number'] = self.mobile_data_df['Shirt_Number'].astype(str)
        # Merge grouped_df with mobile_data_df on the shirt number to get the 'position' from mobile_data_df
        # Ensure 'shirt_number' and 'corrected_shirt_number' columns are used correctly
        grouped_df = grouped_df.merge(self.mobile_data_df[['Shirt_Number', 'Position']],
                                      left_on='corrected_shirt_number', right_on='Shirt_Number',
                                      how='left')

        # Assign the position from the mobile data and drop the temporary column
        grouped_df['position'] = grouped_df['Position']
        grouped_df.drop(columns=['Position'], inplace=True)

        return grouped_df

# Define FirstModel class
class FirstModel:
    def __init__(self, teams, data_cleaned):
        self.teams = teams
        self.data_cleaned = data_cleaned
        self.features_to_compare = [
            'goals', 'goals_past', 'total_shots', 'shots_on_target', 'shots_off_target',
            'blocked_shots', 'total_possession', 'total_passes', 'accurate_passes',
            'pass_success', 'key_passes', 'dribbles_attempted', 'dribbles',
            'dribbles_past', 'dribbles_success', 'aerials_won', 'aerial_success',
            'offensive_aerials', 'defensive_aerials', 'tackles', 'tackle_success',
            'tackles_attempted', 'clearances', 'interceptions', 'corners', 'defenders',
            'midfielders', 'attackers', 'advanced_midfielders'
        ]

    def split_formation(self, df):
        # Apply the function to the 'formations' column
        df['formations'] = df['formations'].apply(correct_formation_format)
        # Split the formations column and fill any null values with the default formation "4-3-3"
        formation_split = df['formations'].fillna('4-3-3').str.split('-')
        
        # Rename the columns in the DataFrame to the original names
        df.rename(columns={
            '%_pass_success': 'pass_success',
            '%_tackles_success': 'tackles_success',
            '%_aerial_success': 'aerial_success',
            '%_dribbles_success': 'dribbles_success'
        }, inplace=True)

        # Ensure missing or invalid values are handled by defaulting to "4-3-3"
        df['defenders'] = formation_split.str[0].fillna(4).astype(int)
        df['midfielders'] = formation_split.str[1].fillna(3).astype(int)
        df['attackers'] = formation_split.str[-1].fillna(3).astype(int)

        # Handle cases where there are advanced midfielders (third value in the split)
        # If the length of the split is less than 3, default to no advanced midfielders (0)
        df['advanced_midfielders'] = formation_split.apply(lambda x: int(x[2]) if len(x) > 3 and pd.notna(x[2]) else 0)

        return df

    def find_similar_rows(self):
        lost_games = self.data_cleaned[self.data_cleaned['lose'] == 1]
        lost_games = self.split_formation(lost_games)

        input_row = self.teams.iloc[0].to_dict()
        input_df = pd.DataFrame([input_row])
        input_df = self.split_formation(input_df)
        filtered_lost_games = lost_games[self.features_to_compare]
        distances = euclidean_distances(filtered_lost_games, input_df[self.features_to_compare])
        lost_games['distance'] = distances

        top_n = 100
        similar_rows = lost_games.sort_values(by='distance').head(top_n)
        similar_rows = similar_rows.drop(columns=['distance'])

        return similar_rows

    def find_winning_rows(self, similar_rows):
        lost_game_ids = similar_rows['id_match'].unique()
        winning_rows = self.data_cleaned[(self.data_cleaned['id_match'].isin(lost_game_ids)) & (self.data_cleaned['win'] == 1)]
        games = winning_rows

        games = self.split_formation(games)

        input_row = self.teams.iloc[1].to_dict()
        input_df = pd.DataFrame([input_row])
        input_df = self.split_formation(input_df)

        filtered_games = games[self.features_to_compare]
        distances = euclidean_distances(filtered_games, input_df[self.features_to_compare])
        games['distance'] = distances

        top_n = 10
        similar_rows = games.sort_values(by='distance').head(top_n)
        similar_rows = similar_rows.drop(columns=['distance'])

        return similar_rows


# Define SecondModel class
class SecondModel:
    def __init__(self, input_row, player_data):
        self.input_row = input_row
        self.player_data = player_data
        self.STATS_TO_MAXIMIZE = {
            'Goalkeeper': ['aerial_success', 'blocked_shots'],
            'Right-back': ['tackles_attempted', 'tackle_success', 'clearances', 'interceptions', 'dribbles_success', 'aerial_success', 'pass_success'],
            'Left-back': ['tackles_attempted', 'tackle_success', 'clearances', 'interceptions', 'dribbles_success', 'aerial_success', 'pass_success'],
            'Center-back': ['tackles_attempted', 'tackle_success', 'clearances', 'interceptions', 'aerial_success', 'pass_success', 'defensive_aerials'],
            'Defensive Midfielder': ['tackles_attempted', 'interceptions', 'pass_success', 'key_passes', 'defensive_aerials'],
            'Central Midfielder': ['total_passes', 'accurate_passes', 'key_passes', 'dribbles_attempted', 'dribbles_success', 'pass_success', 'dribbles'],
            'Attacking Midfielder': ['goals', 'key_passes', 'dribbles_attempted', 'dribbles_success', 'pass_success', 'shots_on_target', 'dribbles', 'offensive_aerials'],
            'Right Winger': ['goals', 'assists', 'key_passes', 'dribbles_attempted', 'dribbles_success', 'crossing_accuracy', 'shots_on_target', 'pass_success', 'dribbles'],
            'Left Winger': ['goals', 'assists', 'key_passes', 'dribbles_attempted', 'dribbles_success', 'crossing_accuracy', 'shots_on_target', 'pass_success', 'dribbles'],
            'Striker': ['goals', 'shots_on_target', 'total_shots', 'key_passes', 'aerial_success', 'dribbles_success', 'pass_success', 'offensive_aerials'],
            'Center Forward': ['goals', 'shots_on_target', 'total_shots', 'key_passes', 'aerial_success', 'dribbles_success', 'pass_success', 'offensive_aerials']
        }
        self.STATS_TO_MINIMIZE = {
            'Goalkeeper': ['goals_past', 'dribbles_past'],
            'Right-back': ['dribbles_past', 'injuries'],
            'Left-back': ['dribbles_past', 'injuries'],
            'Center-back': ['dribbles_past'],
            'Defensive Midfielder': ['dribbles_past', 'injuries'],
            'Central Midfielder': ['injuries'],
            'Attacking Midfielder': ['shots_off_target'],
            'Right Winger': ['shots_off_target', 'injuries'],
            'Left Winger': ['shots_off_target', 'injuries'],
            'Striker': ['shots_off_target', 'injuries'],
            'Center Forward': ['shots_off_target', 'injuries']
        }
        self.ABSENT_FEATURES_TO_MAXIMIZE = {
            'Goalkeeper': ['saved_shots', 'avg_speed', 'highest_speed', 'aerial_duels'],
            'Right-back': ['distance_covered', 'avg_speed', 'highest_speed', 'tackles_won'],
            'Left-back': ['distance_covered', 'avg_speed', 'highest_speed', 'tackles_won'],
            'Center-back': ['distance_covered', 'avg_speed', 'highest_speed', 'tackles_won'],
            'Defensive Midfielder': ['distance_covered', 'aerial_duels', 'tackles_won'],
            'Central Midfielder': ['distance_covered', 'aerial_duels'],
            'Attacking Midfielder': ['highest_speed'],
            'Right Winger': ['highest_speed', 'distance_covered'],
            'Left Winger': ['highest_speed', 'distance_covered'],
            'Striker': ['aerial_duels'],
            'Center Forward': ['aerial_duels']
        }
        self.FEATURE_INDEX_MAP = {
            'goals': 0,
            'total_shots': 1,
            'shots_on_target': 2,
            'shots_off_target': 3,
            'blocked_shots': 4,
            'total_passes': 5,
            'accurate_passes': 6,
            'pass_success': 7,
            'key_passes': 8,
            'dribbles_attempted': 9,
            'dribbles': 10,
            'dribbles_success': 11,
            'dribbles_past': 12,
            'aerial_success': 13,
            'tackles_attempted': 14,
            'tackles_success': 15,
            'clearances': 16,
            'interceptions': 17,
            'offensive_aerials': 18,
            'defensive_aerials': 19,
            'distance_covered': 20,
            'avg_speed': 21,
            'highest_speed': 22,
            'saved_shots': 23,
            'aerial_duels': 24,
            'tackles_won': 25
        }

    def count_players_per_position(self, formation):
        formation_numbers = list(map(int, formation.split('-')))
        position_groups = {
            'defender': ['Right-back', 'Left-back', 'Center-back'],
            'midfielder': ['Defensive Midfielder', 'Central Midfielder', 'Attacking Midfielder', 'Right Winger', 'Left Winger'],
            'forward': ['Striker', 'Center Forward']
        }
        position_counter = Counter()
        num_defenders = formation_numbers[0]
        num_midfielders = formation_numbers[1] if len(formation_numbers) > 1 else 0
        num_midfielders_advanced = formation_numbers[2] if len(formation_numbers) > 3 else 0
        num_forwards = formation_numbers[-1]

        if num_defenders == 4:
            position_counter['Center-back'] = 2
            position_counter['Right-back'] = 1
            position_counter['Left-back'] = 1
        elif num_defenders == 3:
            position_counter['Center-back'] = 3
        elif num_defenders == 5:
            position_counter['Center-back'] = 3
            position_counter['Right-back'] = 1
            position_counter['Left-back'] = 1

        if num_midfielders == 4:
            position_counter['Central Midfielder'] = 2
            position_counter['Right Winger'] = 1
            position_counter['Left Winger'] = 1
        elif num_midfielders == 3:
            position_counter['Central Midfielder'] = 2
            position_counter['Attacking Midfielder'] = 1
        elif num_midfielders == 5:
            position_counter['Central Midfielder'] = 3
            position_counter['Right Winger'] = 1
            position_counter['Left Winger'] = 1
        elif num_midfielders == 2:
            position_counter['Central Midfielder'] = 2
        elif num_midfielders == 1:
            position_counter['Central Midfielder'] = 1

        if num_midfielders_advanced > 0:
            position_counter['Attacking Midfielder'] += num_midfielders_advanced

        if num_forwards == 2:
            position_counter['Striker'] = 1
            position_counter['Center Forward'] = 1
        elif num_forwards == 1:
            position_counter['Striker'] = 1
        elif num_forwards >= 3:
            position_counter['Striker'] = 2
            position_counter['Center Forward'] = 1

        return position_counter

    def get_relevant_feature_indices(self, position):
        max_features = [self.FEATURE_INDEX_MAP[feat] for feat in self.STATS_TO_MAXIMIZE[position] if feat in self.FEATURE_INDEX_MAP]
        min_features = [self.FEATURE_INDEX_MAP[feat] for feat in self.STATS_TO_MINIMIZE[position] if feat in self.FEATURE_INDEX_MAP]
        return max_features, min_features

    def objective(self, player_vars, player_stats, target_stats, max_features, min_features, absent_features):
        team_max_stats = [lpSum(player_vars[i] * player_stats[i][j] for i in range(len(player_vars))) for j in max_features]
        team_min_stats = [lpSum(player_vars[i] * player_stats[i][j] for i in range(len(player_vars))) for j in min_features]
        absent_stats = [lpSum(player_vars[i] * player_stats[i][j] for i in range(len(player_vars))) for j in absent_features]

        maximize_term = lpSum(lpSum(team_max_stats[j] - target_stats[max_features[j]] for j in range(len(max_features))))
        maximize_absent_term = lpSum(absent_stats)
        minimize_term = lpSum(lpSum(target_stats[min_features[j]] - team_min_stats[j] for j in range(len(min_features))))

        return maximize_term + maximize_absent_term - minimize_term

    def recommend_team(self):
        formation = self.input_row['formations']
        position_counts = self.count_players_per_position(formation)
        self.player_data = self.player_data.reset_index(drop=True)

        num_goalkeepers = 1
        num_right_backs = position_counts['Right-back']
        num_left_backs = position_counts['Left-back']
        num_center_backs = position_counts['Center-back']
        num_defensive_midfielders = position_counts['Defensive Midfielder']
        num_central_midfielders = position_counts['Central Midfielder']
        num_attacking_midfielders = position_counts['Attacking Midfielder']
        num_right_wingers = position_counts['Right Winger']
        num_left_wingers = position_counts['Left Winger']
        num_strikers = position_counts['Striker']
        num_center_forwards = position_counts['Center Forward']
        self.player_data.columns = self.player_data.columns.str.lower()

        goalkeepers = self.player_data[self.player_data['position'].str.contains('Goalkeeper')].index
        right_backs = self.player_data[self.player_data['position'].str.contains('Right-back')].index
        left_backs = self.player_data[self.player_data['position'].str.contains('Left-back')].index
        center_backs = self.player_data[self.player_data['position'].str.contains('Center-back')].index
        defensive_midfielders = self.player_data[self.player_data['position'].str.contains('Defensive Midfielder')].index
        central_midfielders = self.player_data[self.player_data['position'].str.contains('Central Midfielder')].index
        attacking_midfielders = self.player_data[self.player_data['position'].str.contains('Attacking Midfielder')].index
        right_wingers = self.player_data[self.player_data['position'].str.contains('Right Winger')].index
        left_wingers = self.player_data[self.player_data['position'].str.contains('Left Winger')].index
        strikers = self.player_data[self.player_data['position'].str.contains('Striker')].index
        center_forwards = self.player_data[self.player_data['position'].str.contains('Center Forward')].index

        player_stats = self.player_data[['goals', 'total_shots', 'shots_on_target', 'shots_off_target', 'blocked_shots',
                                          'total_passes', 'accurate_passes', 'pass_success', 'key_passes', 'dribbles_attempted',
                                          'dribbles', 'dribbles_success', 'dribbles_past', 'aerial_success',
                                          'tackles_attempted', 'tackles_success',
                                          'clearances', 'interceptions', 'offensive_aerials', 'defensive_aerials',
                                          'injuries', 'distance_covered', 'avg_speed', 'highest_speed', 'saved_shots', 'aerial_duels',
                                          'tackles_won']].values

        target_stats = np.array([self.input_row['goals'], self.input_row['total_shots'], self.input_row['shots_on_target'],
                                 self.input_row['shots_off_target'], self.input_row['blocked_shots'], self.input_row['total_passes'],
                                 self.input_row['accurate_passes'], self.input_row['pass_success'], self.input_row['key_passes'], self.input_row['dribbles_attempted'],
                                 self.input_row['dribbles'], self.input_row['dribbles_success'], self.input_row['dribbles_past'],
                                 self.input_row['aerial_success'], self.input_row['tackles_attempted'], self.input_row['tackles_success'],
                                 self.input_row['clearances'], self.input_row['interceptions'], self.input_row['goals_past'],
                                 self.input_row['aerials_won'], self.input_row['offensive_aerials'], self.input_row['defensive_aerials'],
                                 self.input_row['tackles']])

        problem = LpProblem("Team_Selection", LpMaximize)
        player_vars = [LpVariable(f'player_{i}', lowBound=0, cat='Binary') for i in range(len(self.player_data))]

        problem += lpSum([player_vars[i] for i in goalkeepers]) == num_goalkeepers, "Goalkeeper_Constraint"
        problem += lpSum([player_vars[i] for i in right_backs]) == num_right_backs, "Right_Back_Constraint"
        problem += lpSum([player_vars[i] for i in left_backs]) == num_left_backs, "Left_Back_Constraint"
        problem += lpSum([player_vars[i] for i in center_backs]) == num_center_backs, "Center_Back_Constraint"
        problem += lpSum([player_vars[i] for i in defensive_midfielders]) == num_defensive_midfielders, "Defensive_Midfielder_Constraint"
        problem += lpSum([player_vars[i] for i in central_midfielders]) == num_central_midfielders, "Central_Midfielder_Constraint"
        problem += lpSum([player_vars[i] for i in attacking_midfielders]) == num_attacking_midfielders, "Attacking_Midfielder_Constraint"
        problem += lpSum([player_vars[i] for i in right_wingers]) == num_right_wingers, "Right_Winger_Constraint"
        problem += lpSum([player_vars[i] for i in left_wingers]) == num_left_wingers, "Left_Winger_Constraint"
        problem += lpSum([player_vars[i] for i in strikers]) == num_strikers, "Striker_Constraint"
        problem += lpSum([player_vars[i] for i in center_forwards]) == num_center_forwards, "Center_Forward_Constraint"

        injury_limit = 3
        for i in range(len(player_vars)):
            problem += player_vars[i] * self.player_data['injuries'].iloc[i] <= injury_limit, f'Injury_Constraint_{i}'

        positions = {
            'Goalkeeper': goalkeepers,
            'Right-back': right_backs,
            'Left-back': left_backs,
            'Center-back': center_backs,
            'Defensive Midfielder': defensive_midfielders,
            'Central Midfielder': central_midfielders,
            'Attacking Midfielder': attacking_midfielders,
            'Right Winger': right_wingers,
            'Left Winger': left_wingers,
            'Striker': strikers,
            'Center Forward': center_forwards
        }

        for position, indices in positions.items():
            max_features, min_features = self.get_relevant_feature_indices(position)
            absent_features = [self.FEATURE_INDEX_MAP[feat] for feat in self.ABSENT_FEATURES_TO_MAXIMIZE[position] if feat in self.FEATURE_INDEX_MAP]

            if len(indices) > 0:
                problem += self.objective(player_vars, player_stats, target_stats, max_features, min_features, absent_features)

        problem.solve()

        
        selected_players = self.player_data[[player_vars[i].varValue > 0 for i in range(len(self.player_data))]]
        team_stats = selected_players[['goals', 'total_shots', 'shots_on_target', 'shots_off_target', 'blocked_shots', 'saved_shots',
                                           'total_passes', 'accurate_passes', 'pass_success', 'key_passes', 'dribbles_attempted', 'dribbles',
                                           'dribbles_success', 'aerial_duels', 'aerial_success', 'offensive_aerials', 'defensive_aerials',
                                           'tackles_attempted', 'tackles_won', 'tackles_success', 'clearances', 'interceptions', 'injuries',
                                           'distance_covered', 'avg_speed', 'highest_speed']].sum()
        return selected_players, team_stats
       