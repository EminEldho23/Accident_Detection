import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys

# Constants for detection logic
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7   # For overlap
FREEZE_THRESHOLD = 5.0 # Seconds
SPEED_HISTORY = 30    # Frames to track speed
MOTION_WEIGHT = 0.1
YOLO_WEIGHT = 0.7
TRACK_WEIGHT = 0.2

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0
    return intersection_area / union_area

def confirm_accident(frame, text="ACCIDENT DETECTED"):
    """
    Trigger function for confirmed accident.
    """
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 0, 255), 3, cv2.LINE_AA)
    # Placeholder for audio_peak logic or external alert
    print(f"ALERT: {text}")

def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Load Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")
    if not os.path.exists(model_path):
        print("Model best.pt not found. Using yolov8n.pt as fallback/test.")
        model = YOLO("yolov8n.pt") 
    else:
        model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # Background Subtractor for Motion Anomaly
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Tracking Data
    # {track_id: [ (x,y), (x,y), ... ]}
    track_history = {}
    # {track_id: frames_since_stop}
    stop_tracker = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. Motion Analysis (MOG2)
        fgMask = backSub.apply(frame)
        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        
        # Check for large motion anomalies (smoke/debris)
        motion_score = 0
        white_pixels = np.count_nonzero(fgMask)
        total_pixels = fgMask.size
        motion_ratio = white_pixels / total_pixels
        if motion_ratio > 0.05: # >5% of screen moving might be chaotic
            motion_score = 1.0

        # 2. YOLO + Tracking
        results = model.track(frame, persist=True, verbose=False)
        
        current_boxes = []
        current_ids = []
        current_confs = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Store for overlap check
            for box, track_id, conf in zip(boxes, track_ids, confs):
                current_boxes.append(box)
                current_ids.append(track_id)
                current_confs.append(conf)

                # Update history for speed check
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                if track_id not in track_history:
                    track_history[track_id] = []
                    stop_tracker[track_id] = 0
                
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > SPEED_HISTORY:
                    track_history[track_id].pop(0)

            # 3. Overlap Logic
            overlap_detected = False
            for i in range(len(current_boxes)):
                for j in range(i + 1, len(current_boxes)):
                    iou = calculate_iou(current_boxes[i], current_boxes[j])
                    if iou > IOU_THRESHOLD:
                        overlap_detected = True
                        # Draw overlapping boxes in Red
                        cv2.rectangle(frame, 
                                    (int(current_boxes[i][0]), int(current_boxes[i][1])),
                                    (int(current_boxes[i][2]), int(current_boxes[i][3])),
                                    (0, 0, 255), 2)
                        cv2.rectangle(frame, 
                                    (int(current_boxes[j][0]), int(current_boxes[j][1])),
                                    (int(current_boxes[j][2]), int(current_boxes[j][3])),
                                    (0, 0, 255), 2)

            # 4. Speed/Freeze Logic
            freeze_detected = False
            for track_id in current_ids:
                hist = track_history[track_id]
                if len(hist) > 5:
                    # Calculate displacement over last few frames
                    # Simple distance between start and end of history window
                    dist = np.sqrt((hist[-1][0] - hist[0][0])**2 + (hist[-1][1] - hist[0][1])**2)
                    
                    # Heuristic: High speed would have large dist. 
                    # If dist is sudden low after being high -> STOP.
                    # For now simplistically: if dist < threshold, it is "stopped"
                    if dist < 5.0: 
                        stop_tracker[track_id] += 1
                    else:
                        stop_tracker[track_id] = 0
                    
                    # If stopped for > FREEZE_THRESHOLD seconds
                    if stop_tracker[track_id] > (FREEZE_THRESHOLD * fps):
                        freeze_detected = True
                        # Visual marker for freeze
                        centroid = hist[-1]
                        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 10, (0, 255, 255), -1)

            # 5. Fusion Logic
            # "if yolo_conf >0.5 and deepsort_overlap >0.7 and audio_peak: confirm_accident()"
            # We treat 'motion_score' and 'freeze' as supporting signals
            
            # Simplified trigger based on user request:
            # We'll use the max confidence of overlapping boxes
            
            max_conf = max(current_confs) if len(current_confs) > 0 else 0
            
            # Logic A: User's specific condition
            # Overlap > 0.7 AND Conf > 0.5 (Audio ignored/simulated)
            if max_conf > CONF_THRESHOLD and overlap_detected:
                 confirm_accident(frame, "CRASH: OVERLAP DETECTED")

            # Logic B: Weighted Voting (Fuse signals)
            # Normalize signals
            yolo_signal = 1.0 if max_conf > CONF_THRESHOLD else 0.0
            motion_signal = motion_score
            track_signal = 1.0 if (overlap_detected or freeze_detected) else 0.0
            
            fusion_score = (yolo_signal * YOLO_WEIGHT) + \
                           (motion_signal * MOTION_WEIGHT) + \
                           (track_signal * TRACK_WEIGHT)
            
            # If fusion score is high enough (e.g., > 0.6)
            if fusion_score > 0.6:
                 cv2.putText(frame, f"Fusion Risk: {fusion_score:.2f}", (50, 100), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)


        cv2.imshow("Advanced Accident Detection", frame)
        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_src = sys.argv[1]
    else:
        # Default placeholder if run without args
        print("Please provide video path as argument.")
        sys.exit(1)
        
    process_video(video_src)
