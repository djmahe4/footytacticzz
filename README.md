# Football Analysis Project

## Introduction
The goal of this project is to create a Match analysis & Coach assistant app to help in preparing for every match based on last matches' videos for your team and for the opponent team, the app consists of 3 main parts:
- Models & modules: this part holds the computer vision, feature extraction and match analysis part.
- Recommendation systems: this part holds the formation, play style, starting 11 and bench players recommendation systems.
- LLM integration: this part holds the integration with Gemini LLM model to create a human readable opponent weakness analysis, training plan and match summary based on the data from the previous parts in addition to a chatbot to add the ability of interaction and modifying to coaches.

## Contributors ✨

This project is available thanks to these amazing people who contributed to it:

| [<img src="https://avatars.githubusercontent.com/u/109768834?v=4" width="100px;"/><br /><sub><b>Abdelrahman Atef</b></sub>](https://github.com/AbdelrahmanAtef01) | [<https://avatars.githubusercontent.com/u/79551355?v=4" width="100px;"/><br /><sub><b>Ziad Hassan</b></sub>](https://github.com/ziad640) | [<img src="https://avatars.githubusercontent.com/u/96237323?v=4" width="100px;"/><br /><sub><b>Ahmed Ibrahim</b></sub>](https://github.com/AhmedIbrahemAhmed) | [<img src="https://avatars.githubusercontent.com/u/119015507?v=4" width="100px;"/><br /><sub><b>Abdelrahman Alaa</b></sub>](https://github.com/NA70X) | [<img src="https://avatars.githubusercontent.com/u/136976977?v=4" width="100px;"/><br /><sub><b>Abdelrahman Mohamed</b></sub>](https://github.com/AbdooMohamedd) | [<img src="https://avatars.githubusercontent.com/u/98721410?v=4" width="100px;"/><br /><sub><b>Zid Sameh</b></sub>](https://github.com/Ziad-Sameh3) |
| :---: | :---: | :---: |
| Computer vision & modules | Computer vision & modules | Recommendation Systems | Recommendation Systems | LLM Integration | LLM Integration |

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas