from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import cv2
import gc
import json
import time

app = FastAPI()

# Define your data models
class VideoRequest(BaseModel):
    video_paths: list

# Initialize FastAPI app
@app.post("/process_videos/")
async def process_videos(request: VideoRequest):
    try:
        video_paths = request.video_paths
        main(video_paths)
        return {"message": "Videos processed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_reports/")
async def generate_reports():
    try:
        main_report()
        return {"message": "Reports generated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_match_summary/")
async def generate_match_summary():
    try:
        match_summary_json = generate_match_summary()
        if match_summary_json:
            return {"match_summary": match_summary_json}
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve valid JSON for match summary.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_opponent_analysis/")
async def generate_opponent_analysis():
    try:
        opponent_analysis_json = generate_opponent_analysis()
        if opponent_analysis_json:
            return {"opponent_analysis": opponent_analysis_json}
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve valid JSON for opponent analysis.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_training_suggestions/")
async def generate_training_suggestions():
    try:
        training_suggestions_json = generate_training_suggestions()
        if training_suggestions_json:
            return {"training_suggestions": training_suggestions_json}
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve valid JSON for training suggestions.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Your existing main function
def main(video_paths):
    # List of video file paths
    # video_paths = [
    #     '/kaggle/input/testvideos/Untitled design.mp4',
    #     '/kaggle/input/videos/input_vid.mp4'
    # ]

    # Loop through each video
    for video_index, video_path in enumerate(video_paths):
        # Initialize DataFrames and persistent objects for each video
        df = initialize_dataframe()  # Initialize empty DataFrame for players
        team_df = initialize_team_df()  # Initialize empty DataFrame for teams
        tracker = Tracker('/kaggle/input/old_data.pt/pytorch/default/1/old_data.pt')  # Initialize tracker
        team_assigner = TeamAssigner()

        # Set batch size
        batch_size = 200
        video_reader = cv2.VideoCapture(video_path)

        # Get total number of frames in the video
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define possible formations
        possible_formations = ['4-3-3', '4-2-3-1', '4-3-2-1', '4-1-4-1', '3-5-2', '3-4-1-2',
                               '4-4-2', '4-4-1-1', '5-4-1', '3-4-3', '4-1-3-2', '3-1-4-2',
                               '3-4-2-1', '4-5-1', '4-3-1-2', '4-2-2-2', '3-5-1-1', '4-1-3-2',
                               '5-3-2', '3-3-3-1', '4-2-4']
        i = 0
        # Loop through the video, processing batch_size frames at a time
        for start_frame in range(0, total_frames, batch_size):
            i += 1
            # Read a batch of frames
            video_frames = read_video_in_batches(video_reader, start_frame, batch_size)

            # If no frames were read, break the loop
            if len(video_frames) == 0:
                break

            # Process batch: Initialize per-batch objects and perform operations
            tracks = tracker.get_object_tracks(video_frames)  # Get object tracks for batch
            tracker.add_position_to_tracks(tracks)  # Add position to tracks

            # Camera movement estimator
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

            # Interpolate Ball Positions
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

            # View Transformer
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)

            # Speed and Distance Estimation
            speed_and_distance_estimator = SpeedAndDistance_Estimator()
            df = speed_and_distance_estimator.update_df_with_speed_and_distance(tracks, df)

            # Team Assignment
            if i == 1:
                team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

            for frame_num, player_track in enumerate(tracks['players']):
                for player_id, track in player_track.items():
                    team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

                    # Update DataFrame with team and team color
                    if player_id in df.index:
                        df.at[player_id, 'team'] = team
                        df.at[player_id, 'team_color'] = str(team_assigner.team_colors[team])
                    else:
                        df.loc[player_id] = {'team': team, 'team_color': str(team_assigner.team_colors[team])}

            # Ball Assignment
            player_assigner = PlayerBallAssigner()
            for frame_num, player_track in enumerate(tracks['players']):
                if frame_num < len(tracks['ball']):
                    ball_data_for_frame = tracks['ball'][frame_num]
                    if len(ball_data_for_frame) > 0 and 1 in ball_data_for_frame:
                        ball_bbox = ball_data_for_frame[1]['bbox']
                        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                        if assigned_player != -1:
                            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            # Pass Detection
            pass_detector = PassDetector(tracks, df)
            df = pass_detector.process_game_in_batches(batch_size=20)

            # YOLO Processor and Event Processing
            class_thresholds = {0: 0.8, 1: 0.7, 2: 0.3, 3: 0.1, 4: 0.7, 5: 0.6, 6: 0.85}
            yolo_processor = YOLOVideoProcessor('/kaggle/input/new_data.pt/pytorch/default/1/new_data.pt', class_thresholds)
            filtered_detections, detections_classes_2_and_3 = yolo_processor.process_frames_combined(video_frames)

            # Detect other events
            event_processor = EventProcessor(tracks, filtered_detections, df)
            df = event_processor.process_frames_in_batches()

            # Process Goal and Line Points
            processor = GoalAndLineProcessor()
            goals_and_lines_annotations = processor.get_goal_and_line_data(video_frames, detections_classes_2_and_3)

            # Detect shots, corners, saves, goals
            shot_detector = ShotDetector(tracks, df, team_df, goals_and_lines_annotations)
            df, team_df = shot_detector.process_frames_in_batches()

            # Initialize OCR
            player_number_tracker = PlayerShirtNumberTracker(video_frames, tracks, df,
                                           '/kaggle/input/ocr/pytorch/default/1/best.pt')

            # OCR: Detect player shirt numbers and update DataFrame
            df = player_number_tracker.run()

            # Initialize FormationDetector for each batch
            formation_detector = FormationDetector(tracks, possible_formations, team_df)

            # Formation Detection
            team_df = formation_detector.process_frames_in_batches()

            # Initialize SubstitutionDetector
            detector = SubstitutionDetector(class_thresholds, '/kaggle/input/substitution_ocr/pytorch/default/1/best.pt', team_df)
            # Run the extraction process
            ocr_results, team_df = detector.extract_annotation(video_frames, filtered_detections, tracks)

            # Delete batch-specific objects and free up memory
            del video_frames, camera_movement_estimator, view_transformer, speed_and_distance_estimator
            del player_assigner, pass_detector, yolo_processor, event_processor, processor, shot_detector
            del filtered_detections, player_number_tracker, formation_detector, detector, ocr_results

            # Force garbage collection
            gc.collect()

            # After processing all batches, fill in any missing data in DataFrames
            df = df.fillna(0)

            # Final statistics processing for teams and players
            player_stats = PlayerStats(df)
            team_1_df, team_2_df = player_stats.process_data()

            processor = SoccerMatchDataProcessorFullWithSubs(team_1_df, team_2_df, team_df)
            final_df = processor.process_match_data()

            # Save tracks and DataFrames to CSV files with unique names for each video
            output_suffix = f"_video_{video_index+1}"
            save_tracks_to_csv(tracks, csv_path=f'/kaggle/working/tracks_csv{output_suffix}.csv')
            df.to_csv(f'/kaggle/working/player_statistics{output_suffix}.csv', index=True)
            team_df.to_csv(f'/kaggle/working/team_statistics{output_suffix}.csv', index=True)
            team_1_df.to_csv(f'/kaggle/working/team_1_player_statistics{output_suffix}.csv', index=True)
            team_2_df.to_csv(f'/kaggle/working/team_2_player_statistics{output_suffix}.csv', index=True)
            final_df.to_csv(f'/kaggle/working/teams_final_statistics{output_suffix}.csv', index=True)
            print(f"BATCH DONE {i} for video {video_index+1}!")

        print(f"Processing completed for video {video_index+1} - {video_path}")

    print("All videos processed successfully!")

# Your existing main_report function
def main_report():
    # Load the teams data (you may need to adjust file paths)
    teams1 = pd.read_csv('/kaggle/working/teams_final_statistics_video_1.csv')
    teams2 = pd.read_csv('/kaggle/working/teams_final_statistics_video_2.csv')

    # Combine teams1 and teams2 into a single DataFrame
    combined_teams = pd.concat([teams1, teams2], ignore_index=True)

    # Load the data
    mobile_data1 = pd.read_csv('/kaggle/input/mobdatat/mobile_data.csv')
    mobile_data2 = pd.read_csv('/kaggle/input/mobdatat/mobile_data_2.csv')

    correct_shirt_numbers = [str(num) for num in mobile_data1['Shirt_Number']]
    correct_shirt_numbers2 = [str(num) for num in mobile_data2['Shirt_Number']]

    player_data1 = pd.read_csv('/kaggle/working/team_1_player_statistics_video_1.csv')
    player_data2 = pd.read_csv('/kaggle/working/team_2_player_statistics_video_1.csv')
    player_data3 = pd.read_csv('/kaggle/working/team_1_player_statistics_video_2.csv')
    player_data4 = pd.read_csv('/kaggle/working/team_2_player_statistics_video_2.csv')

    player_data_dict = {
        'player_data1': player_data1,
        'player_data2': player_data2,
        'player_data3': player_data3,
        'player_data4': player_data4,
    }

    # Extract the first team's color from mobile_data1
    first_team_color_mobile1 = clean_team_color(mobile_data1.iloc[0]['Team_Color '])

    # Extract the opponent's team color from mobile_data2
    first_team_color_mobile2 = clean_team_color(mobile_data2.iloc[0]['Team_Color '])

    # Handle player data 1 and 2 based on mobile_data1
    closest_player_data_1_2 = []
    for player_data_key in ['player_data1', 'player_data2']:
        closest_player_dataset_key = find_closest_player_dataset(player_data_dict[player_data_key], first_team_color_mobile1, player_data_key)
        if closest_player_dataset_key:
            closest_player_data_1_2.append(player_data_dict[closest_player_dataset_key])

    # Combine the closest data for player 1 and 2 based on mobile_data1
    closest_player_data_mobile1 = pd.concat(closest_player_data_1_2, ignore_index=True)

    # Process the player stats
    player_stats = MyPlayerStats(closest_player_data_mobile1, correct_shirt_numbers, mobile_data2)
    closest_player_data_mobile1 = player_stats.process_data()

    # Correct the shirt numbers and drop the temporary column
    closest_player_data_mobile1['shirt_number'] = closest_player_data_mobile1['corrected_shirt_number']
    closest_player_data_mobile1.drop(columns=['corrected_shirt_number'], inplace=True)

    closest_player_data_mobile1.to_csv('/kaggle/working/closest_player_data_mobile1.csv', index=False)

    # Handle player data 3 and 4 based on mobile_data2
    closest_player_data_3_4 = []
    for player_data_key in ['player_data3', 'player_data4']:
        closest_player_dataset_key = find_closest_player_dataset(player_data_dict[player_data_key], first_team_color_mobile2, player_data_key)
        if closest_player_dataset_key:
            closest_player_data_3_4.append(player_data_dict[closest_player_dataset_key])

    # Combine the closest data for player 3 and 4 based on mobile_data2
    closest_player_data_mobile2 = pd.concat(closest_player_data_3_4, ignore_index=True)

    # Process the player stats
    player_stats = MyPlayerStats(closest_player_data_mobile2, correct_shirt_numbers2, mobile_data2)
    closest_player_data_mobile2 = player_stats.process_data()

    # Correct the shirt numbers and drop the temporary column
    closest_player_data_mobile2['shirt_number'] = closest_player_data_mobile2['corrected_shirt_number']
    closest_player_data_mobile2.drop(columns=['corrected_shirt_number'], inplace=True)

    closest_player_data_mobile2.to_csv('/kaggle/working/closest_player_data_mobile2.csv', index=False)

    # From here on, use only closest_player_data_mobile1 in the rest of the code

    # Find the closest matching rows in teams
    closest_row_team1 = find_closest_match(combined_teams, first_team_color_mobile1)
    opponent_team_color = clean_team_color(mobile_data2.iloc[0]['Team_Color '])
    closest_row_team2 = find_closest_match(combined_teams, opponent_team_color)

    combined_closest_rows = pd.DataFrame([closest_row_team1, closest_row_team2])

    my_team = pd.DataFrame([closest_row_team1])
    opponent_team = pd.DataFrame([closest_row_team2])
    my_team.to_csv('/kaggle/working/my_team.csv', index=False)
    opponent_team.to_csv('/kaggle/working/opponent_team.csv', index=False)

    # Run the models
    teams = combined_closest_rows
    data_cleaned = pd.read_csv('/kaggle/input/cleanedw2/data_cleaned.csv')

    # Use the closest player data based on mobile data 1
    player_data = closest_player_data_mobile1

    player_data['shirt_number'] = player_data.pop('Shirt_Number')
    player_data['pass_success'] = player_data.pop('%_pass_success')
    player_data['dribbles_success'] = player_data.pop('%_dribbles_success')
    player_data['aerial_success'] = player_data.pop('%_aerial_success')
    player_data['tackles_success'] = player_data.pop('%_tackles_success')

    # Initialize the first model
    model1 = FirstModel(teams, data_cleaned)

    # Find similar rows based on the first model
    similar_rows = model1.find_similar_rows()

    # Save recommended formations to CSV
    recommended_formations = model1.find_winning_rows(similar_rows)
    display(similar_rows)
    recommended_formations.to_csv('/kaggle/working/recommended_formations.csv', index=False)

    # Select the first match data row
    match_data = recommended_formations
    i = 0
    solution = False
    while i < 10 and not solution:
        input_row = match_data.iloc[i].to_dict()
        input_row['tackles_success'] = input_row.pop('tackle_success')

        # Convert input row to DataFrame for team selection
        row = pd.DataFrame([input_row])

        # Initialize the second model to recommend a team based on input row and processed player data
        team_recommender = SecondModel(input_row, player_data)
        selected_players, team_stats = team_recommender.recommend_team()

        # If a team was successfully selected, display the results
        if selected_players is not None:
            solution = True
            print("\nRecommended Team:")
            display(selected_players)
            selected_players['status'] = 'Starting 11'  # Add a column to indicate starting players

            print("\nTeam Stats:")
            display(team_stats)

        # Remove the selected players and recommend substitutes
            player_shirt_number_to_remove = selected_players['shirt_number'].tolist()
            player_data_updated = player_data[~player_data['shirt_number'].isin(player_shirt_number_to_remove)]

            substitute_recommender = SecondModel(input_row, player_data_updated)
            selected_substitutes, team_stats = substitute_recommender.recommend_team()

            # If substitutes were successfully selected, display the results
            if selected_substitutes is not None:
                print("\nRecommended Substitutes:")
                display(selected_substitutes)
                selected_substitutes['status'] = 'Substitute'  # Add a column to indicate substitutes

                # Combine both selected players and substitutes into a single file
                combined_team = pd.concat([selected_players, selected_substitutes], ignore_index=True)
                combined_team.to_csv('/kaggle/working/combined_team.csv', index=False)

                print("\nSubstitute Team Stats:")
                display(team_stats)

            else:
                selected_players.to_csv('/kaggle/working/selected_players.csv', index=False)
        i += 1
# Helper functions for Gemini API integration
def read_system_instructions(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_match_summary_prompt(my_team_info, opponent_info):
    return f"Generate a match summary based on the following data:\nMy Team Info:\n{my_team_info}\nOpponent Info:\n{opponent_info}"

def generate_opponent_analysis_prompt(opponent_info, opponent_players):
    return f"Analyze the opponent based on the following data:\nOpponent Info:\n{opponent_info}\nOpponent Players:\n{opponent_players}"

def generate_training_suggestions_prompt(my_team_players, my_team_info, opponent_analysis_json):
    return f"Generate training suggestions based on the following data:\nMy Team Players:\n{my_team_players}\nMy Team Info:\n{my_team_info}\nOpponent Analysis:\n{opponent_analysis_json}"

def send_to_gemini_api_with_retry(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = model.generate(prompt)
            if response.get("mime_type") == "application/json":
                return json.loads(response.get("content"))
            else:
                return None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return None

# Gemini API integration functions
def generate_match_summary():
    opponent_info = pd.read_csv(r'/kaggle/working/opponent_team.csv')
    opponent_info_str = opponent_info.to_string(index=False)

    my_team_info = pd.read_csv(r'/kaggle/working/my_team.csv')
    my_team_info_str = my_team_info.to_string(index=False)

    match_summary_prompt = generate_match_summary_prompt(my_team_info_str, opponent_info_str)
    match_summary_json = send_to_gemini_api_with_retry(match_summary_prompt)

    return match_summary_json

def generate_opponent_analysis():
    opponent_info = pd.read_csv(r'/kaggle/working/opponent_team.csv')
    opponent_info_str = opponent_info.to_string(index=False)

    opponent_players = pd.read_csv(r'/kaggle/working/closest_player_data_mobile2.csv')
    opponent_players_str = opponent_players.to_string(index=False)

    opponent_analysis_prompt = generate_opponent_analysis_prompt(opponent_info_str, opponent_players_str)
    opponent_analysis_json = send_to_gemini_api_with_retry(opponent_analysis_prompt)

    return opponent_analysis_json

def generate_training_suggestions():
    my_team_players = pd.read_csv(r'/kaggle/working/closest_player_data_mobile1.csv')
    my_team_players_str = my_team_players.to_string(index=False)

    my_team_info = pd.read_csv(r'/kaggle/working/my_team.csv')
    my_team_info_str = my_team_info.to_string(index=False)

    opponent_analysis_json = generate_opponent_analysis()

    training_suggestions_prompt = generate_training_suggestions_prompt(my_team_players_str, my_team_info_str, opponent_analysis_json)
    training_suggestions_json = send_to_gemini_api_with_retry(training_suggestions_prompt)

    if training_suggestions_json:
        output = {
            "team_training_session": training_suggestions_json.get("team_training_session", ""),
            "worst_5_players_individual_sessions": training_suggestions_json.get("individual_sessions", {})[:4]
        }
        return output
    else:
        return None

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
