"""
Run accident detection on a video file
"""
import sys
from enhanced_accident_detection import EnhancedAccidentDetector

# Video path
video_path = sys.argv[1] if len(sys.argv) > 1 else 'C:/trafcon/4_basics/accident1.mp4'
output_path = video_path.replace('.mp4', '_detected.mp4')

print("="*60)
print("ENHANCED ACCIDENT DETECTION")
print("="*60)
print(f"\nInput: {video_path}")
print(f"Output: {output_path}")
print()

# Initialize detector
print("Initializing detector...")
detector = EnhancedAccidentDetector('best.pt', conf_threshold=0.5)

# Process video
print("Processing video...\n")
accidents = detector.process_video(video_path, output_path)

# Summary
print("\n" + "="*60)
print("DETECTION SUMMARY")
print("="*60)
print(f"\nTotal accidents detected: {len(accidents)}")

for i, acc in enumerate(accidents, 1):
    print(f"\nAccident #{i}")
    print(f"  Confidence: {acc['confidence']:.2f}")
    print(f"  Motion Score: {acc['motion_score']:.3f}")
    print("  Reasons:")
    for r in acc['reasons']:
        print(f"    - {r}")

print(f"\nOutput saved to: {output_path}")
print("="*60)
