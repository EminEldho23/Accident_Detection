from ultralytics import YOLO
import sys
import os

def predict_video(video_path, conf_threshold=0.25, save=True, show=False):
    # Resolve script directory to find the model reliably
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Running prediction on {video_path}...")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Save output: {save}")
    print(f"Show live: {show}")
    print()

    model = YOLO(model_path)
    
    # Run prediction with adjustable parameters
    # save=True will save the video to runs/detect/predict/
    # show=True will display the video live
    results = model.predict(
        source=video_path, 
        conf=conf_threshold,  # Adjustable confidence threshold
        save=save, 
        show=show,
        # Additional parameters for better detection
        iou=0.45,  # IoU threshold for NMS
        max_det=50,  # Maximum detections per frame
        agnostic_nms=False  # Class-specific NMS
    )
    
    print("\nPrediction complete.")
    print("Check the 'runs/detect/predict' folder for the saved output video.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_video.py <video_path> [confidence_threshold]")
        print("Example: python predict_video.py accident1.mp4 0.5")
        sys.exit(1)
    
    video_source = sys.argv[1]
    
    # Get confidence threshold from command line (default 0.25)
    conf_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    
    # Resolve absolute path for the video source
    video_source = os.path.abspath(video_source)
    predict_video(video_source, conf_threshold=conf_threshold)
