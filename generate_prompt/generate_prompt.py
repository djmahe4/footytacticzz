import json

def generate_match_summary_prompt(my_team_info_str, opponent_info_str):
    """
    Generates a match summary prompt based on team and opponent data provided as strings.
    
    Args:
        my_team_info_str (str): String containing my team performance data.
        opponent_info_str (str): String containing opponent performance data.
        
    Returns:
        str: The structured match summary prompt formatted as JSON.
    """

    prompt = f"""
    ### Match Summary Request

    You are tasked with providing a comprehensive analysis of the match between my team and the opponent team based on the provided statistics. Use the following information to derive insights and training suggestions.

    **My Team Performance:**
    {my_team_info_str}

    **Opponent Performance:**
    {opponent_info_str}

    ### Output Format:
    Return your response as a JSON object with the following structure:

    {{
      "match_summary": {{
        "my_team_performance": {{
            "formation": "<Extract the formation from my team>",
            "score": "<Extract the score from my team>",
            "goals": <Extract the number of goals from my team>,
            "total_shots": <Extract total shots from my team>,
            "shots_on_target": <Extract shots on target from my team>,
            "total_possession": "<Extract total possession percentage from my team>",
            "accurate_passes": <Extract the number of accurate passes from my team>,
            "pass_success_rate": "<Extract the pass success percentage from my team>",
            "key_passes": <Extract the number of key passes from my team>,
            "dribbles_success_rate": "<Extract successful dribbles>/<attempted dribbles from my team>",
            "tackle_success_rate": "<Extract the tackle success percentage from my team>",
            "corners": <Extract the number of corners from my team>
        }},
        "opponent_performance": {{
            "formation": "<Extract the formation from opponent>",
            "score": "<Extract the score from opponent>",
            "goals": <Extract the number of goals from opponent>,
            "total_shots": <Extract total shots from opponent>,
            "shots_on_target": <Extract shots on target from opponent>,
            "total_possession": "<Extract total possession percentage from opponent>",
            "accurate_passes": <Extract the number of accurate passes from opponent>,
            "pass_success_rate": "<Extract the pass success percentage from opponent>",
            "key_passes": <Extract the number of key passes from opponent>,
            "dribbles_success_rate": "<Extract successful dribbles>/<attempted dribbles from opponent>",
            "tackle_success_rate": "<Extract the tackle success percentage from opponent>",
            "corners": <Extract the number of corners from opponent>
        }}
      }}
    }}

    Please summarize the key statistics for both teams based on the above data, highlight the strengths and weaknesses observed during the match, and provide insights into how each team performed relative to their strategies and tactics. Your response should be structured as specified in the output format.
    """

    return prompt




def generate_player_suggestions_prompt(best_formations_str, match_players_recommendations_str):
    """
    Generates a player suggestions prompt based on recommendations data provided as strings.
    
    Args:
        best_formations_str (str): String containing my team performance data.
        match_players_recommendations_str (str): String containing opponent performance data.
        
    Returns:
        str: The structured player_suggestions_prompt formatted as JSON.
    """

    prompt = f"""
    ### Player suggestions Request

    You are tasked with providing the best formation to play with, which is the first one in the provided data, and also player suggestions
    for the match based on the provided statistics. Use the following information to derive insights.

    **Best Formations:**
    {best_formations_str}

    **Player Suggestions:**
    {match_players_recommendations_str}

    ### Output Format:
    Return your response as a JSON object with the following structure:

    {{
      "recommendations_output": {{
        "best_formation": {{
            "formation": "<Extract the first formation from best_formation_str>"
        }},
        "players_recommendations": [
            {{
                "number": "<Extract the player shirt number from match_players_recommendations_str>",
                "position": "<Extract the position from match_players_recommendations_str>",
                "status": "<Extract the status from match_players_recommendations_str>"
            }},
            {{
                "number": "<Extract the next player shirt number>",
                "position": "<Extract the next player's position>",
                "status": "<Extract the next player's status>"
            }}
            // Repeat for all players in match_players_recommendations_str
        ]
      }}
    }}

    Your response should be structured as specified in the output format.
    """

    return prompt




def generate_opponent_analysis_prompt(opponent_info_str, opponent_players_str):
    """
    Generates a prompt for analyzing the opponent's strengths, weaknesses, and counter-strategies.
    
    Parameters:
        opponent_data (dict): Dictionary containing opponent match statistics.
        opponent_players_str (str): String containing individual data for opponent players.
        
    Returns:
        str: Generated prompt for opponent analysis.
    """
    prompt = f"""
    You are an expert football analyst. Based on the following data about the opponent's recent performances, provide a detailed analysis covering their strengths, weaknesses, and suggest counter-strategies.

    Opponent Data (for the match information):
    {opponent_info_str}


    Opponent Players Data (for each individual player):
    {opponent_players_str}

    Based on this data, analyze:
    1. The strengths of the opponent that can be exploited.
    2. The weaknesses that can be targeted.
    3. Recommended counter-strategies to implement against their play style.

    Provide a comprehensive and actionable analysis.
    """
    return prompt




def generate_training_suggestions_prompt(my_team_players_str, my_team_info_str, opponent_analysis_json):
    """
    Generates a prompt for providing training suggestions based on team players' data, team information,
    and the opponent's analysis (strengths, weaknesses, counter-strategies) as a single string.

    Parameters:
        my_team_players_str (str): String containing the team players' data.
        my_team_info_str (str): String containing the team information.
        opponent_analysis_json (dict): JSON object containing the opponent's analysis.

    Returns:
        str: Generated prompt for training suggestions.
    """
    
    # Convert opponent_analysis_json back to a string for embedding in the prompt
    opponent_analysis_str = json.dumps(opponent_analysis_json, indent=4)

    # Create a structured and detailed prompt using the opponent analysis as a string
    prompt = (
        "You are an expert football coach tasked with improving your team's performance through targeted training sessions. "
        "Please analyze the following data and provide detailed training suggestions in JSON format.\n\n"
        
        "### Team Players Data:\n"
        f"{my_team_players_str}\n\n"
        
        "### Team Information:\n"
        f"{my_team_info_str}\n\n"
        
        "### Opponent Analysis:\n"
        f"{opponent_analysis_str}\n\n"
        
        "### Task:\n"
        "1. Based on the weaknesses identified in the opponent's analysis, suggest a concise training session for the entire team. "
        "This session should specifically address these weaknesses to enhance your team's ability to exploit them during matches.\n"
        "2. For each player, provide a specific actionable drill tailored to their individual strengths and weaknesses, "
        "focusing on how these drills can help exploit the opponent's weaknesses.\n"
        
        "### Output Format:\n"
        "Return your suggestions as a JSON object with the following structure:\n"
        "{\n"
        "  \"team_training_session\": \"<training session suggestion>\",\n"
        "  \"individual_sessions\": [\n"
        "    {\n"
        "      \"player\": \"<player name>\",\n"
        "      \"shirt_number\": <shirt number>,\n"
        "      \"drill\": \"<specific drill for player>\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        
        "### Notes:\n"
        "Be creative and ensure each suggestion is clear, actionable, and easy to implement.\n"
        "Focus on improving players' skills, tactical awareness, and overall team performance by specifically addressing the opponent's weaknesses."
    )
    
    return prompt