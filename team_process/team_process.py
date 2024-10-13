class TeamProcess:
    def __init__(self, df):
        """
        Initialize the WinLoseProcess class with a DataFrame that contains
        'goals' and 'goals_conceded' columns.
        :param df: Pandas DataFrame containing the match data.
        """
        self.df = df

    def process_win_lose_draw(self):
        """
        Process the DataFrame to add three new columns: 'win', 'lose', 'draw'.
        Based on the comparison of 'goals' and 'goals_conceded' columns, the appropriate
        column (win/lose/draw) will be set to 1.
        """
        # Initialize the columns to 0
        self.df['win'] = 0
        self.df['lose'] = 0
        self.df['draw'] = 0

        # Apply the logic to set the appropriate column to 1 based on goals and goals conceded
        self.df.loc[self.df['goals'] > self.df['goals_conceded'], 'win'] = 1
        self.df.loc[self.df['goals'] < self.df['goals_conceded'], 'lose'] = 1
        self.df.loc[self.df['goals'] == self.df['goals_conceded'], 'draw'] = 1

        return self.df

    def calculate_possession(self):
        """
        Calculate the possession for each team based on the 'completed_passes' column.
        Possession is calculated as the percentage of total completed passes in a match (two rows).
        Adds a new column 'possession' to the DataFrame.
        """
        # Initialize the possession column to 0
        self.df['possession'] = 0

        # Process two rows at a time (assuming each pair of rows is one match)
        for i in range(0, len(self.df), 2):
            # Get the completed passes for both teams in the match
            team_a_passes = self.df.loc[i, 'completed_passes']
            team_b_passes = self.df.loc[i + 1, 'completed_passes']
            
            # Total passes for both teams in the match
            total_passes = team_a_passes + team_b_passes
            
            # Calculate possession for both teams
            self.df.loc[i, 'possession'] = (team_a_passes / total_passes) * 100
            self.df.loc[i + 1, 'possession'] = (team_b_passes / total_passes) * 100

        return self.df