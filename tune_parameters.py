"""
Parameter Tuning Script for Accident Detection
Test different confidence thresholds and parameters to optimize detection
"""
import os
import sys
from ultralytics import YOLO

def test_parameters(video_path, conf_thresholds=[0.1, 0.25, 0.5, 0.7]):
    """Test different confidence thresholds on the same video"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Testing parameters on: {video_path}")
    print("="*60)

    model = YOLO(model_path)

    for conf in conf_thresholds:
        print(f"\n--- Testing Confidence Threshold: {conf} ---")

        # Run prediction with current threshold
        results = model.predict(
            source=video_path,
            conf=conf,
            save=True,
            verbose=False,
            stream=True  # Use streaming to avoid memory issues
        )

        # Count detections
        total_frames = 0
        accident_frames = 0
        total_accidents = 0

        for result in results:
            total_frames += 1

            # Count accident detections in this frame
            accident_count = 0
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                if 'accident' in class_name.lower():
                    accident_count += 1
                    total_accidents += 1

            if accident_count > 0:
                accident_frames += 1

        print(f"  Frames processed: {total_frames}")
        print(f"  Frames with accidents: {accident_frames}")
        print(f"  Total accident detections: {total_accidents}")
        print(f"  Accident detection rate: {(accident_frames/total_frames*100):.1f}%")

        if accident_frames > 0:
            print(f"  Average accidents per frame: {(total_accidents/accident_frames):.2f}")
        else:
            print("  ⚠️  No accidents detected with this threshold")

def optimize_for_collision_detection(video_path):
    """Optimize parameters specifically for collision detection"""

    print("\n" + "="*60)
    print("COLLISION DETECTION OPTIMIZATION")
    print("="*60)
    print("Testing parameters optimized for detecting actual collisions")
    print("(not just accident scenes)")
    print()

    # Test higher confidence thresholds to reduce false positives
    high_precision_thresholds = [0.6, 0.7, 0.8, 0.9]

    test_parameters(video_path, conf_thresholds=high_precision_thresholds)

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("For collision detection (not accident scenes):")
    print("• Use confidence threshold 0.7-0.8 to reduce false positives")
    print("• Look for 'car_car_accident' detections specifically")
    print("• Consider temporal analysis (accidents should be brief events)")
    print("• Combine with motion analysis for better accuracy")
    print()
    print("Current model detects accident SCENES, not collision EVENTS.")
    print("For true collision detection, consider:")
    print("• Using optical flow to detect sudden motion changes")
    print("• Tracking vehicle overlap/bbox intersection")
    print("• Detecting sudden stops or direction changes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tune_parameters.py <video_path>")
        print("Example: python tune_parameters.py accident1.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    video_path = os.path.abspath(video_path)

    # Test different confidence thresholds
    test_parameters(video_path)

    # Provide optimization recommendations
    optimize_for_collision_detection(video_path)