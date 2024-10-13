import cv2
from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self, number_model_path):
        self.number_model = YOLO(number_model_path)
        self.number_classes = [str(i) for i in range(10)]
    
    def enhance_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_image = cv2.LUT(equalized, table)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        return enhanced_image

    def group_digits(self, number_boxes, number_confidences, number_class_ids, threshold_distance=15):
        # Sort number_boxes and corresponding confidences/class_ids by x1 coordinate to ensure left-to-right reading
        sorted_digits = sorted(zip(number_boxes, number_confidences, number_class_ids), key=lambda x: x[0][0])

        grouped_digits = []
        current_group = ''
        last_x2 = 0
        
        for number_box, number_confidence, number_class_id in sorted_digits:
            nx1, ny1, nx2, ny2 = int(number_box[0]), int(number_box[1]), int(number_box[2]), int(number_box[3])
            
            # Group digits based on proximity in the x-axis
            if nx1 - last_x2 < threshold_distance:
                current_group += str(int(number_class_id))
            else:
                if current_group:
                    grouped_digits.append(current_group)
                current_group = str(int(number_class_id))
            
            last_x2 = nx2
        
        if current_group:
            grouped_digits.append(current_group)
        
        return grouped_digits

    def detect_numbers(self, player_image):
        number_results = self.number_model(player_image)
        number_boxes = number_results[0].boxes.xyxy.cpu().numpy()
        number_confidences = number_results[0].boxes.conf.cpu().numpy()
        number_class_ids = number_results[0].boxes.cls.cpu().numpy()

        # Group nearby digits to form multi-digit numbers, ensuring left-to-right order
        shirt_numbers = self.group_digits(number_boxes, number_confidences, number_class_ids)
        
        return shirt_numbers


class PlayerShirtNumberTracker:
    def __init__(self, video_frames, tracks, df, number_model_path):
        self.video_frames = video_frames
        self.tracks = tracks
        self.df = df
        self.player_detector = PlayerDetector(number_model_path)

    def run(self):
        # Process every 5th frame
        for frame_num, frame in enumerate(self.video_frames):
            if frame_num % 5 != 0:  # Skip frames not divisible by 5
                continue
            
            tracked_players = self.tracks['players'][frame_num]
            
            for player_id, player_data in tracked_players.items():
                bbox = player_data['bbox']  # Use the tracked bounding box
                
                # Convert bounding box coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)

                # Extract the player image using the bounding box
                player_image = frame[y1:y2, x1:x2]
                
                # Run the number detection model on the cropped player image
                shirt_numbers = self.player_detector.detect_numbers(player_image)
                
                if player_id is not None and shirt_numbers:
                    # Assign the detected shirt number(s) to the player in the DataFrame
                    self.df.at[player_id, 'shirt_number'] = shirt_numbers[0] if shirt_numbers else None

        return self.df