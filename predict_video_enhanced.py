from ultralytics import YOLO
import sys
import os
import cv2
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_overlap_ratio(box1, box2):
    """Calculate how much box1 overlaps with box2 (intersection / box1_area)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    return inter_area / box1_area if box1_area > 0 else 0.0

def calculate_distance(box1, box2):
    """Calculate center-to-center distance between two boxes"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def calculate_box_area(box):
    """Calculate area of a bounding box"""
    return (box[2] - box[0]) * (box[3] - box[1])

def check_size_compatibility(box1, box2, max_ratio=5.0):
    """Check if two boxes have compatible sizes (not too different)"""
    area1 = calculate_box_area(box1)
    area2 = calculate_box_area(box2)
    
    if area1 == 0 or area2 == 0:
        return False
    
    ratio = max(area1, area2) / min(area1, area2)
    return ratio <= max_ratio

def predict_video_enhanced(video_path, conf_threshold=0.25, save=True, show=False, speed_factor=0.5):
    """Enhanced prediction that labels specific vehicles involved in accidents
    
    Args:
        video_path: Path to input video
        conf_threshold: Confidence threshold for detections
        save: Whether to save output video
        show: Whether to display video live
        speed_factor: Video speed multiplier (0.5 = half speed, 0.25 = quarter speed, 1.0 = normal)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Running enhanced prediction on {video_path}...")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Speed factor: {speed_factor}x (slower = easier to see)")
    print(f"Save output: {save}")
    print(f"Show live: {show}")
    print()

    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust FPS for slower playback
    output_fps = max(1, int(fps * speed_factor))
    print(f"Original FPS: {fps}, Output FPS: {output_fps} ({speed_factor}x speed)")
    
    # Setup output video writer if saving
    out = None
    if save:
        output_dir = os.path.join(script_dir, "runs", "detect")
        os.makedirs(output_dir, exist_ok=True)
        
        # Find next available predict folder
        predict_num = 1
        while os.path.exists(os.path.join(output_dir, f"predict{predict_num}")):
            predict_num += 1
        
        output_folder = os.path.join(output_dir, f"predict{predict_num}")
        os.makedirs(output_folder, exist_ok=True)
        
        output_path = os.path.join(output_folder, os.path.basename(video_path).replace('.mp4', '.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        print(f"Output will be saved to: {output_path}")
        print()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.45,
            max_det=50,
            verbose=False
        )[0]
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        class_names = results.names  # Class name mapping
        
        # Separate vehicle and accident detections
        vehicles = []  # (box, conf, class_id, class_name)
        accidents = []  # (box, conf, class_id, class_name)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cls_name = class_names[cls_id]
            
            if 'accident' in cls_name.lower():
                accidents.append((box, conf, cls_id, cls_name))
            elif cls_name.lower() in ['bike', 'car', 'person']:
                vehicles.append((box, conf, cls_id, cls_name))
        
        # Match accidents to vehicles with improved precision
        vehicle_accident_map = {}  # vehicle_idx -> (accident_type, confidence, score)
        
        for acc_box, acc_conf, acc_cls_id, acc_cls_name in accidents:
            best_vehicle_idx = None
            best_score = 0
            best_match_info = None
            
            for v_idx, (v_box, v_conf, v_cls_id, v_cls_name) in enumerate(vehicles):
                # Check if accident type matches vehicle type
                accident_type_lower = acc_cls_name.lower()
                vehicle_type_lower = v_cls_name.lower()
                
                # Type compatibility check
                type_compatible = False
                if 'bike' in accident_type_lower and vehicle_type_lower in ['bike', 'person']:
                    type_compatible = True
                elif 'car' in accident_type_lower and vehicle_type_lower in ['car']:
                    type_compatible = True
                elif 'person' in accident_type_lower and vehicle_type_lower in ['person', 'bike']:
                    type_compatible = True
                
                # Skip if types don't match
                if not type_compatible:
                    continue
                
                # Check size compatibility (boxes shouldn't be drastically different)
                if not check_size_compatibility(acc_box, v_box, max_ratio=8.0):
                    continue
                
                # Calculate multiple metrics for precise matching
                iou = calculate_iou(acc_box, v_box)
                overlap_ratio = calculate_overlap_ratio(v_box, acc_box)  # How much vehicle overlaps accident
                distance = calculate_distance(acc_box, v_box)
                
                # Calculate box diagonal for distance normalization
                box_diagonal = np.sqrt((v_box[2] - v_box[0])**2 + (v_box[3] - v_box[1])**2)
                
                # Distance score (prefer very close boxes)
                if box_diagonal > 0:
                    distance_score = max(0, 1 - (distance / (box_diagonal * 2)))
                else:
                    distance_score = 0
                
                # Combined scoring with strict requirements
                # Prioritize: IoU > Overlap > Distance
                score = (iou * 0.5 +           # High IoU means strong overlap
                        overlap_ratio * 0.3 +   # Vehicle should be inside/overlap accident area
                        distance_score * 0.2)   # Close proximity bonus
                
                # Extremely strict threshold - require very high overlap (>93%)
                # Check if either IoU or overlap_ratio is above 93%
                if score > best_score and (iou > 0.93 or overlap_ratio > 0.93):
                    best_score = score
                    best_vehicle_idx = v_idx
                    best_match_info = (acc_cls_name, acc_conf, score)
            
            # Assign accident to vehicle only if very high confidence match
            if best_vehicle_idx is not None and best_score > 0.85:
                if best_vehicle_idx not in vehicle_accident_map:
                    vehicle_accident_map[best_vehicle_idx] = []
                vehicle_accident_map[best_vehicle_idx].append(best_match_info)
        
        # Draw annotated frame
        annotated_frame = frame.copy()
        
        # Draw vehicles with accident labels
        for v_idx, (v_box, v_conf, v_cls_id, v_cls_name) in enumerate(vehicles):
            x1, y1, x2, y2 = map(int, v_box)
            
            # Determine label
            if v_idx in vehicle_accident_map:
                # Vehicle involved in accident
                accident_infos = vehicle_accident_map[v_idx]
                accident_types = [info[0] for info in accident_infos]
                accident_label = ", ".join(set([a.replace('_', ' ').title() for a in accident_types]))
                
                # Get highest confidence accident match
                highest_acc_conf = max([info[1] for info in accident_infos])
                match_score = max([info[2] for info in accident_infos])
                
                label = f"{v_cls_name.upper()} - ACCIDENT"
                detail_label = f"{accident_label} (Match: {match_score:.2f})"
                color = (0, 0, 255)  # Red for accident
                thickness = 3
            else:
                # Normal vehicle
                label = f"{v_cls_name.upper()}"
                detail_label = None
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw main label background and text
            label_text = f"{label} {v_conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw detailed accident info if present
            if detail_label:
                y_offset = y2 + 20
                (detail_w, detail_h), _ = cv2.getTextSize(detail_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y_offset - detail_h - 5), (x1 + detail_w, y_offset), (255, 100, 0), -1)
                cv2.putText(annotated_frame, detail_label, (x1, y_offset - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw unmatched accidents (if any)
        for acc_box, acc_conf, acc_cls_id, acc_cls_name in accidents:
            x1, y1, x2, y2 = map(int, acc_box)
            label = f"{acc_cls_name.replace('_', ' ').upper()} {acc_conf:.2f}"
            color = (0, 165, 255)  # Orange for unmatched accidents
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display progress
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
        
        # Save or show frame
        if save and out is not None:
            out.write(annotated_frame)
        
        if show:
            cv2.imshow('Accident Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"\n\nPrediction complete!")
    if save:
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_video_enhanced.py <video_path> [confidence_threshold] [speed_factor]")
        print("Example: python predict_video_enhanced.py accident1.mp4 0.6 0.5")
        print("  speed_factor: 0.5 = half speed (slower), 0.25 = quarter speed, 1.0 = normal speed")
        sys.exit(1)
    
    video_source = sys.argv[1]
    
    # Get confidence threshold from command line (default 0.25)
    conf_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    
    # Get speed factor from command line (default 0.5 for half speed)
    speed_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    # Resolve absolute path for the video source
    video_source = os.path.abspath(video_source)
    predict_video_enhanced(video_source, conf_threshold=conf_threshold, speed_factor=speed_factor)
