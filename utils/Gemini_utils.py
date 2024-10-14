import pandas as pd
import json
import time
import google.generativeai as genai
import sys
sys.path.append('../')

# Load the team data from CSV files
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

# Read the system instructions from generate_prompt.py
def read_system_instructions(path):
    with open(path, 'r') as file:
        return file.read()

def send_to_gemini_api_with_retry(prompt, max_retries=5, delay=2):
    """
    Tries to send the prompt to the Gemini API multiple times until it receives valid JSON.
    """
    # Create the model
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
     "max_output_tokens": 792,
     "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
        system_instruction=read_system_instructions('generate_prompt/generate_prompt.py')
    )
    attempt = 0
    while attempt < max_retries:
        try:
            # Use the correct method to generate the response
            response = model.generate_content(prompt)
            if response:
                # Try to parse the response as JSON
                response_text = response._result.candidates[0].content.parts[0].text
                return json.loads(response_text)  # Attempt to return parsed JSON
            else:
                print("Error: No response from the model.")
        except json.JSONDecodeError:
            print(f"Attempt {attempt+1}/{max_retries}: Response not valid JSON. Retrying...")
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {str(e)}. Retrying...")

        attempt += 1
        time.sleep(delay)

    return None  # If all retries fail, return None