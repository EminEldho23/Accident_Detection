from ultralytics import YOLO
import sys
import os

def predict_video(video_path):
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
    model = YOLO(model_path)
    
    # Run prediction
    # save=True will save the video to runs/detect/predict/
    # show=True will display the video live
    results = model.predict(source=video_path, conf=0.25, save=True, show=True)
    
    print("\nPrediction complete.")
    print("Check the 'runs/detect/predict' folder for the saved output video.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        video_source = input("Enter the path to your video file: ").strip("'\"")
    
    # Resolve absolute path for the video source
    video_source = os.path.abspath(video_source)
    predict_video(video_source)
