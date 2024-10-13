import cv2
import numpy as np

class GoalAndLineProcessor:
    def __init__(self):
        self.video_writer = None  # Video writer for saving the output video

    def process_annotations(self, video_frames, annotations, output_video_path, fps=60.0):
        frame_height, frame_width = video_frames[0].shape[:2]
        
        # Initialize the video writer to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for i, frame in enumerate(video_frames):
            frame_annotations = annotations[i]  # Get annotations for the current frame
            goal_points, goal_line_points = self.extract_goal_and_goal_line(frame, frame_annotations)
            annotated_frame = self.draw_goal_and_goal_line(frame, goal_points, goal_line_points)
            
            # Write the annotated frame to the video
            self.video_writer.write(annotated_frame)

        # Release the video writer after writing all frames
        self.video_writer.release()

    def extract_goal_and_goal_line(self, frame, detections):
        goal_points = []
        goal_line_points = []

        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]

            # Convert coordinates to integers
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            # If the detection class is the goal (assumed class_id == 2)
            if class_id == 2:  
                # Define the goal polygon using the bounding box corners
                goal_points.append([
                    (xmin, ymin),  # Top-left
                    (xmax, ymin),  # Top-right
                    (xmax, ymax),  # Bottom-right
                    (xmin, ymax)   # Bottom-left
                ])
            elif class_id == 3:  # Class 3 is the goal line
                goal_line_points = self.detect_goal_line_points(frame, xmin, ymin, xmax, ymax)

        return goal_points, goal_line_points

    def detect_goal_line_points(self, frame, xmin, ymin, xmax, ymax):
        # Convert ROI to grayscale and apply Gaussian Blur
        gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred_roi, 50, 150)

        # Use Hough Transform to find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                return [(x1, y1), (x2, y2)]  # Return the start and end points of the line

        return []  # Return empty if no points are found

    def draw_goal_and_goal_line(self, frame, goal_points, goal_line_points):
        # Draw all detected goals as filled polygons
        for points in goal_points:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], color=(0, 255, 0))  # Fill polygon in green
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Draw outline in blue

        # Draw the goal line
        if len(goal_line_points) == 2:
            pt1, pt2 = goal_line_points
            cv2.line(frame, pt1, pt2, (255, 0, 0), 3)  # Draw line in blue

        return frame  # Return the annotated frame

    # New function to return the frame number, goal points, and goal line points
    def get_goal_and_line_data(self, video_frames, annotations):
        result = {}

        for i, frame in enumerate(video_frames):
            frame_annotations = annotations[i]  # Get annotations for the current frame
            goal_points, goal_line_points = self.extract_goal_and_goal_line(frame, frame_annotations)

            # Ensure goal_line_points is always a list of lists (even if it's empty)
            if goal_line_points:
                goal_line_points = [goal_line_points] 
            # Store the data for each frame
            result[i] = {
                'goal_points': goal_points,
                'goal_line_points': goal_line_points
            }

        return result