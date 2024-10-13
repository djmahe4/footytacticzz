from ultralytics import YOLO

class YOLOVideoProcessor:
    def __init__(self, model_path, class_thresholds):
        """
        Initialize the YOLOVideoProcessor with a model and class-specific thresholds.
        :param model_path: Path to the YOLO weights file.
        :param class_thresholds: Dictionary mapping class IDs to confidence thresholds.
        """
        # Load the YOLO model
        self.model = YOLO(model_path)
        # Set the class-specific confidence thresholds
        self.class_thresholds = class_thresholds

    def process_frames_combined(self, video_frames):
        """
        Process the video frames and return two sets of YOLO detections:
        1. Detections for classes 0 to 6 (excluding 2 and 3).
        2. Detections only for class 2 and class 3.
        
        :param video_frames: List of video frames (as images).
        :return: A tuple of two lists:
                 - The first list contains filtered detections for classes 0 to 6 (excluding classes 2 and 3).
                 - The second list contains detections for classes 2 and 3.
                 Each detection is a list with full detection info [xmin, ymin, xmax, ymax, confidence, class_id, ...].
        """
        filtered_detections = []           # List for classes 0 to 6 (excluding 2 and 3)
        detections_for_classes_2_and_3 = []  # List for class 2 and 3

        for frame in video_frames:
            # Run YOLO model on the frame
            results = self.model(frame)

            # Extract bounding box data from the results, move to CPU, then convert to NumPy
            df_results = results[0].boxes.data.cpu().numpy()  # Use .cpu() to move to host memory

            # Filter detections
            frame_filtered_detections = []
            frame_detections_2_and_3 = []
            for detection in df_results:
                class_id = int(detection[5])  # Class ID is the 6th column
                confidence = detection[4]     # Confidence score is the 5th column
                
                # Apply threshold for valid classes (skip if confidence is below the threshold)
                if class_id in self.class_thresholds and confidence >= self.class_thresholds[class_id]:
                    if class_id not in [2, 3]:  # Filter for classes 0 to 6 (excluding 2 and 3)
                        frame_filtered_detections.append(detection.tolist())
                    elif class_id in [2, 3]:    # Filter for class 2 and class 3
                        frame_detections_2_and_3.append(detection.tolist())

            # Append the filtered detections for this frame
            filtered_detections.append(frame_filtered_detections)
            detections_for_classes_2_and_3.append(frame_detections_2_and_3)

        return filtered_detections, detections_for_classes_2_and_3