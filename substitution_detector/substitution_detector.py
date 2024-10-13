import cv2
import numpy as np
from ultralytics import YOLO

class SubstitutionDetector:
    def __init__(self, class_thresholds, model_path, team_df):
        self.class_thresholds = class_thresholds
        self.team_df = team_df.copy()  # Create a copy of the team DataFrame to avoid SettingWithCopyWarning

        # Load the model using the provided model path
        self.number_model = YOLO(model_path)

    def process_annotations_class_5_only(self, annotations):
        # Flatten annotations (annotations per frame to a single list)
        flattened_annotations = [
            (frame_num, detection) for frame_num, frame_annotations in enumerate(annotations)
            for detection in frame_annotations
        ]

        filtered_detections = []
        for frame_num, detection in flattened_annotations:
            class_id = int(detection[5])
            confidence = detection[4]
            # Only include class 5 with the specified confidence threshold
            if class_id == 5 and confidence >= self.class_thresholds.get(class_id, 0):
                # Append the frame number along with the detection to preserve context
                filtered_detections.append((frame_num, detection))

        return filtered_detections

    def detect_numbers(self, cropped_image):
        # Use the YOLO model to detect numbers from the cropped image directly
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        # Detect numbers using the YOLO model
        results = self.number_model(image_rgb)
        df_results = results[0].boxes.data.cpu().numpy()

        detected_numbers = []
        for detection in df_results:
            class_id = int(detection[5])
            confidence = detection[4]
            if confidence >= 0.25:  # Adjusted confidence threshold
                detected_numbers.append(class_id)

        return detected_numbers

    def crop_image(self, image, bbox):
        xmin, ymin, xmax, ymax = map(int, bbox[:4])
        return image[ymin:ymax, xmin:xmax]

    def detect_dominant_color(self, image):
        """Detect whether the cropped image is predominantly red or green."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define red and green color ranges in HSV
        red_lower1, red_upper1 = np.array([0, 90, 90]), np.array([10, 255, 255])
        red_lower2, red_upper2 = np.array([160, 90, 90]), np.array([180, 255, 255])
        green_lower, green_upper = np.array([40, 50, 50]), np.array([80, 255, 255])

        # Create masks for red and green
        red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
        red_mask = red_mask1 | red_mask2  # Combine red masks
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

        # Count the number of red and green pixels
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        # Determine the dominant color based on pixel count
        if red_pixels > green_pixels:
            return "red"
        elif green_pixels > red_pixels:
            return "green"
        else:
            return "unknown"

    def update_substitution(self, team, red_number, green_number):
        """Update the substitution count for the team in the format 'red_number-green_number'."""
        substitution_value = f"{red_number}-{green_number}"
        # Find the first substitution slot that is zero and update it with "red_number-green_number"
        for i in range(1, 6):
            substitution_column = f"substitution_{i}"
            if self.team_df.loc[team, substitution_column] == 0:
                self.team_df.loc[team, substitution_column] = substitution_value
                break

    def extract_annotation(self, frames, annotations, tracks):
        ocr_results = {'green': [], 'red': []}  # Separate results for green and red numbers
        red_number = None
        green_number = None

        # Process annotations and handle frame numbers correctly
        detections = self.process_annotations_class_5_only(annotations)

        for frame_num, bbox in detections:
            # Process each detection within the frame
            cropped_image = self.crop_image(frames[frame_num], bbox)
            dominant_color = self.detect_dominant_color(cropped_image)

            if dominant_color in ["green", "red"]:
                detected_numbers = self.detect_numbers(cropped_image)
                if detected_numbers:
                    closest_player_id = self.find_closest_player(bbox, tracks, frame_num)

                    if closest_player_id is None:
                        team = 1
                    else :
                        team = tracks['players'][frame_num][closest_player_id]['team']

                    # Assign detected numbers to red or green
                    if dominant_color == "red" and detected_numbers:
                        red_number = detected_numbers[0]  # Take the first detected red number
                    elif dominant_color == "green" and detected_numbers:
                        green_number = detected_numbers[0]  # Take the first detected green number

                    # If both red and green numbers are detected, update substitution
                    if red_number is not None and green_number is not None:
                        self.update_substitution(team, red_number, green_number)

                    # Store results based on color
                    ocr_results[dominant_color].append(detected_numbers)

        return ocr_results, self.team_df

    def find_closest_player(self, bbox, tracks, frame_num):
        """Find the closest player to the class 5 annotation based on bounding box distance."""
        xmin, ymin, xmax, ymax = map(int, bbox[:4])
        detection_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

        min_distance = float('inf')
        closest_player_id = None

        for player_id, player_info in tracks['players'][frame_num].items():
            player_bbox = player_info['bbox']
            player_center = np.array([(player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2])
            distance = np.linalg.norm(detection_center - player_center)

            if distance < min_distance:
                min_distance = distance
                closest_player_id = player_id

        return closest_player_id