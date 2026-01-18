"""
Verify collision frame detection by extracting key frames
This helps confirm if Frame 21 is the actual collision moment
"""

import cv2
import sys
import os

def extract_key_frames(video_path):
    """Extract frames around the detected collision to verify timing"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total} frames @ {fps:.1f}fps")
    print(f"Collision detected at: Frame 21 (t=0.45s)")
    print()
    
    # Extract frames around collision
    frames_to_extract = [1, 10, 15, 18, 20, 21, 22, 23, 25, 30, 40, 50]
    
    output_dir = os.path.dirname(video_path)
    
    print("Extracting key frames for visual verification...")
    
    for frame_num in frames_to_extract:
        if frame_num >= total:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        
        if ret:
            time_sec = frame_num / fps
            label = ""
            if frame_num == 21:
                label = "_COLLISION_DETECTED"
            elif frame_num < 15:
                label = "_baseline"
            elif frame_num < 21:
                label = "_pre_collision"
            else:
                label = "_post_collision"
            
            filename = f"frame_{frame_num:03d}{label}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Add text overlay
            cv2.putText(frame, f"Frame {frame_num} (t={time_sec:.2f}s)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if frame_num == 21:
                cv2.putText(frame, ">>> COLLISION DETECTED <<<", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imwrite(filepath, frame)
            print(f"  Saved: {filename}")
    
    cap.release()
    print()
    print("Check these frames to verify if Frame 21 is the actual collision moment.")
    print("If collision happens LATER, we can adjust the detection thresholds.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = r"C:\trafcon\4_basics\accident1.mp4"
    
    extract_key_frames(video_path)
