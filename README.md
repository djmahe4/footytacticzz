# Tactic Zone Project

## Introduction
The goal of this project is to create a Match analysis & Coach assistant app to help in preparing for every match based on last matches' videos for your team and for the opponent team, the app consists of 3 main parts:
- Models & modules: this part holds the computer vision, feature extraction and match analysis part.
- Recommendation systems: this part holds the formation, play style, starting 11 and bench players recommendation systems.
- LLM integration: this part holds the integration with Gemini LLM model to create a human readable opponent weakness analysis, training plan and match summary based on the data from the previous parts in addition to a chatbot to add the ability of interaction and modifying to coaches.

## Contributors ✨

This project is available thanks to these amazing people who contributed to it:

| [<img src="https://avatars.githubusercontent.com/u/109768834?v=4" width="100px;"/><br /><sub><b>[Abdelrahman Atef](https://github.com/AbdelrahmanAtef01)</b></sub><br />**Computer Vision & Modules**] | [<img src="https://avatars.githubusercontent.com/u/79551355?v=4" width="100px;"/><br /><sub><b>[Ziad Hassan](https://github.com/ziad640)</b></sub><br />**Computer Vision & Modules**] | [<img src="https://avatars.githubusercontent.com/u/96237323?v=4" width="100px;"/><br /><sub><b>[Ahmed Ibrahim](https://github.com/AhmedIbrahemAhmed)</b></sub><br />**Recommendation Systems**] |
| :---: | :---: | :---: |
| [<img src="https://avatars.githubusercontent.com/u/119015507?v=4" width="100px;"/><br /><sub><b>[Abdelrahman Alaa](https://github.com/NA70X)</b></sub><br />**Recommendation Systems**] | [<img src="https://avatars.githubusercontent.com/u/136976977?v=4" width="100px;"/><br /><sub><b>[Abdelrahman Mohamed](https://github.com/AbdooMohamedd)</b></sub><br />**LLM Integration**] | [<img src="https://avatars.githubusercontent.com/u/98721410?v=4" width="100px;"/><br /><sub><b>[Ziad Sameh](https://github.com/Ziad-Sameh3)</b></sub><br />**LLM Integration**] |


## Table of Contents
- [Installation](#Installation)
- [Modules Used](#ModulesUsed)
- [Our Modules](#OurModules)
- [Trained Models](#TrainedModels)
- [Usage](#Usage)
- [Outputs](#Outputs)
- [Datasets](#Datasets)
- [AssociatedProjects](#AssociatedProjects)
- [References](#References)

## Installation
to clone and use this repo you can either access it directly using the end points file, or you can use the main file but don't forget to install the requirements and it doesn't contain the chatbot

## ModulesUsed
The following modules are used in this project:
- ultraLytics Yolo v8x: computer vision object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color, assign teams
- pulp: linear programming and players selection
- flask: end points handling
- google-generativeai: Integrating with Gemini LLM model for chatbot and generating some responses 
- supervision: for Tracking objects
- opencv-python: handling videos and frames
- gc: to delete objects and reduce resource usage after every batch
- numpy: to handle arrays and frames
- pandas: to handle dataframes through the different modules
- pytesseract: performing OCR
- collections: to adjust dataframes and perform operations on them
- csv: to output the extracted data into csv files
- json: to handle gemini json response
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## OurModules
- camera movement estimator: estimates the movement of the camera to take it into consederation when detecting the positions of the players on the field.
- chatbot: this module holds the chatbot files that creates a chat with gemini fed with all our extraxted data for the coach.
- event process: processes every event related detection (e.g. dribbling, duels, tackling, injuries).
- formation detector: detects the formation of the team based on player positions through different frames and different states of the match.
- generate prompt: generates the needed prompts to gemini that creates the human readable static page based on our extracted data.
- goal and line processor: processes the goal and goal line annotations and applies segmenation, edge detection and some other techniques to return the goal and goal line points.
- new data handler: handles the event related object detection models (e.g. dribble, tackle, substitution board, etc).
- pass detector: detects passes whether successed or failed or intercepted.
- player ball assigner: assign the ball within each frame to a player and a team based on the position of both.
- player number detector: uses object detection to detect the players' shirt numbers.
- player stats: converts the output of all computer vision & feature extraction modules to the needed format for the recommendation systems, also processes and combines different tracks with the same shirt number.
- recommendation systems: handles mapping and preprocessing after computer vision based on a user input players positions and numbers, recommend the perfect states and formation to win the match against your opponent based on the last 5 seasons 900 match data from different leagues, recommend the best 11 players from your team to start the match based on the needed stats by the first recommender system output and also the game changing needed bench players.
- shot detector: detects shots, goals, key passes, assists, whether the shot is on target or off target, whether the shot is saved or blocked, etc.
- speed and distance estimator: calculates the players' highest speed, avg speed and distance covered.
- substitution detector: detects substitution boards and which player is out and which is in.
- team assigner: assigns each player to a team based on segmented cropped player shirts photos. 
- team process: determines whether the team won, lost or got a draw.
- team stats: processes the output of all computer vision & feature extraction modules and combines them in addition to other team related feaures like formation to create the input of the first recommender system.
- trackers: tracks every object during different frames (e.g. players, referees, ball, goal keepers).
- view transformer: transforms the view based on the prespective of the field to make the calculations more accurate.

## TrainedModels
- [Trained Yolo v8x on the most occuring objects like players, goalkeepers and goals](https://github.com/AbdelrahmanAtef01/Tactic_Zone/blob/main/models/old_data.pt)
- [Trained Yolo v8x on the less occuring events like dribling, tackling , goal and goal line, etc](https://github.com/AbdelrahmanAtef01/Tactic_Zone/blob/main/models/new_data.pt)
- [Trained Yolo v8x on player shirt numbers detection](https://github.com/AbdelrahmanAtef01/Tactic_Zone/blob/main/models/playershirt.pt)
- [Trained Yolo v8x on digital numbers detection(substitution board numbers)](https://github.com/AbdelrahmanAtef01/Tactic_Zone/blob/main/models/Substitution.pt)


## Usage
You can use this project by:
- Feeding it the paths of a video for your last match &  the opponent's last match.
- get the json outputs or the csv analytics.
- use the chatbot to ask and modify.
or: 
you can just useit through end points or through our app directly.

Note: the project should only be used with eagle eye 1080p or higher match videos either than that it will work in a very wrong way and its output isn't reliable.

## Outputs
- Analyis csv files of your team and/or opponent teams matches.
- recommendations for the best formation to play with, the best starting 11 for this match, the best bench players.
- Human readable json file containing opponent strengths, weaknesses and the best counter stratigies to play on him, training schedule based on these weaknesses, specific training schedule for your weakest players, it will also contain the first two outputs to use all of them statically in an app if needed.
- Chatbot for the coach fed with all these data to chat with and modify in the outputs.

## DataSets
- [11 labels (players, referees, ball, goalkeeper, goal, goalline, substitution board, dribble, tackling, Auriel_duel, injurie) manually annotated 2000 photos data that covers most of the project object detection features](https://universe.roboflow.com/abdelrahmanatef-kd8yt/tactic_zone-c0oag)
Note: this dataset is associated with this project

- [player shirt number object detection data(7000 photos)]( https://universe.roboflow.com/volleyai-actions/jersey-number-detection-s01j4 )
- [digital number object detection data(1750 photos)](https://universe.roboflow.com/project-kaqwa/numbers-ssusk/dataset/1)
- [past 5 easons 900 matches data from 5 different leagues](https://www.kaggle.com/datasets/leslliesayrus/soccer-matches)

## AssociatedProjects
- [Tactic Zone App UI design](https://www.figma.com/design/Ue1XtbxTyZa05I62G22bm7/Tactic-zone?node-id=1-107&t=7sEtAa1aTVYJeRTg-1)
- [RoboFlow Project for the first dataset](https://universe.roboflow.com/abdelrahmanatef-kd8yt/tactic_zone-c0oag)

## References

credit should be given for [Abdullah Tarek](https://github.com/abdullahtarek) amazing repository for providing all the informations needed to start as we used some of is modules to start our project.

- [youtube link explaining every thing in Abdullah Tarek's repo](https://youtu.be/neBZ6huolkg?si=YxXXMiTmlvRzAoC-) 
- [the repository](https://github.com/abdullahtarek/football_analysis)