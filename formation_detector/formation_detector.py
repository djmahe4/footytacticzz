import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class FormationDetector:
    def __init__(self, tracks, possible_formations, team_df):
        """
        Initialize the FormationDetector with tracking data, a list of possible formations, and a DataFrame.
        :param tracks: Dictionary containing the tracking data of players.
        :param possible_formations: List of possible formations (e.g., ['4-3-3', '4-4-2', '4-2-3-1']).
        :param team_df: DataFrame containing team information where formations will be assigned.
        """
        self.tracks = tracks
        self.possible_formations = possible_formations
        self.team_df = team_df  # DataFrame where formation will be assigned

    def average_positions(self, frame_range, team):
        """
        Compute the average positions of players over a range of frames for a specific team.
        :param frame_range: List of frame numbers to process (e.g., 20 frames).
        :param team: The team for which to compute the average positions.
        :return: A dictionary with player IDs as keys and their average positions (x, y) as values.
        """
        positions = {}

        for frame_num in frame_range:
            for player_id, player_data in self.tracks['players'][frame_num].items():
                if player_data['team'] == team:
                    if player_id not in positions:
                        positions[player_id] = {'x': [], 'y': []}
                    positions[player_id]['x'].append(player_data['position_adjusted'][0])  # Player's x-coordinate
                    positions[player_id]['y'].append(player_data['position_adjusted'][1])  # Player's y-coordinate

        # Compute the average position for each player
        average_positions = {
            player_id: (
                np.mean(positions[player_id]['x']),
                np.mean(positions[player_id]['y'])
            )
            for player_id in positions
        }

        return average_positions

    def cluster_positions(self, average_positions, num_clusters):
        """
        Cluster the players' average positions based on the number of clusters.
        :param average_positions: Dictionary of player IDs and their average positions (x, y).
        :param num_clusters: Number of clusters (defense, midfield, attack, etc.) based on the formation.
        :return: A dictionary with cluster labels (0, 1, 2, ...) as keys and lists of player IDs as values.
        """
        player_ids = list(average_positions.keys())
        positions = np.array(list(average_positions.values()))

        # Use K-Means to cluster players into groups based on the number of clusters required by the formation
        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init='auto')
        labels = kmeans.fit_predict(positions)

        clusters = {i: [] for i in range(num_clusters)}
        for i, player_id in enumerate(player_ids):
            clusters[labels[i]].append(player_id)

        return clusters

    def parse_formation(self, formation):
        """
        Parse a formation string (e.g., '4-2-3-1') into a list of numbers representing players in each group.
        :param formation: A formation string (e.g., '4-2-3-1').
        :return: A list of integers representing the formation (e.g., [4, 2, 3, 1]).
        """
        return [int(num) for num in formation.split('-')]

    def determine_formation(self, clusters, formation):
        """
        Determine if the clusters match the given formation.
        :param clusters: A dictionary with cluster labels (0, 1, 2, ...) and lists of player IDs.
        :param formation: A list of integers representing the expected formation (e.g., [4, 2, 3, 1]).
        :return: Boolean indicating if the detected cluster sizes match the expected formation.
        """
        cluster_sizes = sorted([len(clusters[i]) for i in clusters])

        # Sort the formation as well to avoid ordering issues
        formation_sorted = sorted(formation)

        return cluster_sizes == formation_sorted

    def find_best_formation(self, clusters):
        """
        Find the best matching formation from the list of possible formations.
        :param clusters: A dictionary with cluster labels (0, 1, 2, ...) and lists of player IDs.
        :return: The best matching formation string (e.g., '4-3-3') or 'Unknown' if no match is found.
        """
        for formation in self.possible_formations:
            parsed_formation = self.parse_formation(formation)
            if self.determine_formation(clusters, parsed_formation):
                return formation
        return 'Unknown'

    def find_optimal_cluster_and_formation(self, average_positions):
        """
        Try clustering with 3, 4, and 5 clusters, and find the best matching formation.
        :param average_positions: A dictionary of player IDs and their average positions (x, y).
        :return: The best matching formation and the number of clusters used.
        """
        best_formation = 'Unknown'
        best_num_clusters = 3

        for num_clusters in range(3, 6):  # Try clustering with 3, 4, and 5 clusters
            if len(average_positions) >= num_clusters:
                clusters = self.cluster_positions(average_positions, num_clusters)
                formation = self.find_best_formation(clusters)
                if formation != 'Unknown':
                    best_formation = formation
                    best_num_clusters = num_clusters
                    break  # If a match is found, break out of the loop

        return best_formation, best_num_clusters

    def process_frames_in_batches(self, batch_size=20):
        """
        Process the player formations for each team in batches of frames.
        :param batch_size: Number of frames to process in each batch (default is 20).
        :return: team_df updated with the most common formation for each team.
        """
        total_frames = len(self.tracks['players'])

        # Initialize dictionaries to store formations for each team
        team_formations = {1: [], 2: []}

        for start_frame in range(0, total_frames, batch_size):
            end_frame = min(start_frame + batch_size, total_frames)
            frame_range = list(range(start_frame, end_frame))

            for team in [1, 2]:  
                average_positions = self.average_positions(frame_range, team)

                # Try clustering with 3, 4, and 5 clusters and find the best formation
                best_formation, best_num_clusters = self.find_optimal_cluster_and_formation(average_positions)

                # Collect formations that aren't 'Unknown' for each team
                if best_formation != 'Unknown':
                    team_formations[team].append(best_formation)

        # For each team, find the most common formation and assign it to the team_df
        for team_id in self.team_df.index:
            if team_formations[team_id]:
                most_common_formation = Counter(team_formations[team_id]).most_common(1)[0][0]
            else:
                most_common_formation = 'Unknown'

            print(most_common_formation)
            # Assign the most common formation to the team_df for the respective team (using row index)
            if most_common_formation != 'Unknown':
                self.team_df.at[team_id, 'formations'] = str(most_common_formation)
            
        self.team_df['formations'] = self.team_df['formations'].astype(str)
        
        return self.team_df