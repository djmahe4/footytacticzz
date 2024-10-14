import numpy as np

# Helper functions
def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((color1 - color2) ** 2))

def clean_color_string(color_string):
    color_list = color_string.strip("[]").split()  # Split on spaces
    return np.array(color_list, dtype=float)

def find_closest_player_dataset(player_data, target_color, dataset_key):
    non_goalkeeper_data = player_data[player_data['position'] != 'goalkeeper']  # Exclude goalkeepers
    player_colors = non_goalkeeper_data['team_color'].apply(clean_team_color)
    distances = np.array([euclidean_distance(target_color, player_color) for player_color in player_colors])

    # Find the index of the minimum distance
    if len(distances) > 0:
        closest_distance_idx = np.argmin(distances)
        return dataset_key  # Return the dataset key if a valid match is found

def color_distance(row_color, target_color):
    return np.linalg.norm(row_color - target_color)

def find_closest_match(df, target_color):
    df['team_color_array'] = df['team_color'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))

    df['distance'] = df['team_color_array'].apply(lambda x: color_distance(x, target_color))

    closest_row = df.loc[df['distance'].idxmin()]

    df.drop(columns=['team_color_array', 'distance'], inplace=True)

    return closest_row

# Function to clean and convert the 'Team_Color' string to a NumPy array
def clean_team_color(color_string):
    # Remove any unwanted characters (e.g., commas) and split the string into numeric parts
    cleaned_string = color_string.replace(",", " ").strip("[]")  # Remove commas and brackets
    return np.array(cleaned_string.split(), dtype=float)

def correct_formation_format(value):
    if isinstance(value, str) and '/' in value:  # Check if the value contains '/'
        # Split by '/' assuming it's a wrongly formatted date-like string (e.g., 4/3/2003)
        parts = value.split('/')
        if len(parts) == 3:  # Ensure it's a date-like format
            # Extract the day, month, and last digit of the year to form the correct formation
            defenders = parts[0]  # First part represents the number of defenders
            midfielders = parts[1]  # Second part represents the number of midfielders
            attackers = parts[2][-1]  # Use the last digit of the year as the number of attackers
            return f'{defenders}-{midfielders}-{attackers}'
    return value  # Return the original value if no correction is needed