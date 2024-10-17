import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from io import StringIO
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Now access the GEMINI_API_KEY from environment variables
my_api_key = os.getenv("GEMINI_API_KEY")


# Function to fetch CSV data from an API URL and load it into a DataFrame
def fetch_csv_from_url(csv_url):
    response = requests.get(csv_url)
    if response.status_code == 200:
        # Convert the CSV data from the API to a pandas dataframe
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    else:
        raise Exception(
            f"Failed to fetch data from {csv_url}. Status Code: {response.status_code}"
        )


# Function to process the chatbot query and fetch CSV data
def process_chatbot_request(data):
    # Extract the URLs from the request body
    file_urls = data.get("files", {})
    best_formations_url = file_urls.get("best_formations_url", "")
    generated_player_data_url = file_urls.get("generated_player_data_url", "")
    my_team_url = file_urls.get("my_team_url", "")
    opponent_team_url = file_urls.get("opponent_team_url", "")
    selected_players_url = file_urls.get("selected_players_url", "")

    # Fetch CSV data from the provided URLs
    best_formations = fetch_csv_from_url(best_formations_url)
    generated_player_data = fetch_csv_from_url(generated_player_data_url)
    my_team = fetch_csv_from_url(my_team_url)
    opponent_team = fetch_csv_from_url(opponent_team_url)
    selected_players = fetch_csv_from_url(selected_players_url)

    # Convert dataframes to strings
    best_formations_str = best_formations.to_string(index=False)
    generated_player_data_str = generated_player_data.to_string(index=False)
    my_team_str = my_team.to_string(index=False)
    opponent_team_str = opponent_team.to_string(index=False)
    selected_players_str = selected_players.to_string(index=False)

    # Create the system instruction
    system_instruction = f"""
    You are an intelligent football assistant, designed to help the coach of a football team make data-driven decisions. 
    You will assist the coach by answering questions about team performance, player statistics, match formations, and opponent analysis, 
    using the data provided below:

    Best Formations:
    {best_formations_str}

    Generated Player Data:
    {generated_player_data_str}

    My Team:
    {my_team_str}

    Opponent Team:
    {opponent_team_str}

    Selected Players:
    {selected_players_str}

    The coach will ask you questions about the team’s strategies, the performance of specific players, comparisons with the opponent, 
    and recommendations for upcoming matches. You will respond using detailed information from these data to provide accurate and insightful answers.
    """

    return system_instruction


# Configure the API key
genai.configure(api_key=my_api_key)

# Create the model with generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 192,
    "response_mime_type": "text/plain",
}


# Function to send a message and retrieve response from the chatbot
def ask_chatbot(question, system_instruction):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    # Start chat session
    chat_session = model.start_chat(history=[])
    # Send question to the chatbot and return response
    response = chat_session.send_message(question)
    return response.text


# Define the route for asking the chatbot
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        # Process the chatbot request and fetch the CSV data
        system_instruction = process_chatbot_request(data)

        # Ask the chatbot the question and get the response
        response = ask_chatbot(question, system_instruction)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=5000)
