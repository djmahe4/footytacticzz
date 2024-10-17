from ultralytics import YOLO
import supervision as sv
import pandas as pd
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
                    print("position")
                    print(tracks[object][frame_num][track_id]['position'])

    def interpolate_ball_positions(self, ball_positions):
        # Ensure 'bbox' always contains 4 values, even if missing or incomplete
        ball_positions = [
            x.get(1, {}).get('bbox', [None, None, None, None]) 
            for x in ball_positions
        ]

        # Skip rows where all bbox values are None
        ball_positions = [bbox for bbox in ball_positions if any(v is not None for v in bbox)]

        # Check if there's any data left after skipping all-None rows
        if not ball_positions:
            return ball_positions  # or handle empty case as needed

        # Create DataFrame with bounding box coordinates
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    
        # Handle cases where 'bbox' contains missing values: interpolate and backfill
        df_ball_positions = df_ball_positions.interpolate().bfill()

        # Convert the DataFrame back to the original format
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
            "goalkeepers": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to Supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeepers"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == cls_names_inv['goalkeeper']:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        return tracks
