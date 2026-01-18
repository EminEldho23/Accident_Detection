"""
Configuration file for Enhanced Accident Detection System
Adjust these parameters to tune the detection system for your use case
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# YOLO Model Settings
YOLO_MODEL_PATH = 'best.pt'          # Path to trained YOLO weights
YOLO_CONFIDENCE_THRESHOLD = 0.5      # Min confidence for detections (0.0-1.0)
YOLO_IOU_THRESHOLD = 0.45            # NMS IoU threshold
YOLO_WEIGHT = 0.7                    # Weight in fusion (higher = trust YOLO more)

# ============================================================================
# DEEPSORT TRACKING CONFIGURATION
# ============================================================================

# DeepSORT Parameters
DEEPSORT_MAX_AGE = 30                # Frames to keep track alive without detection
DEEPSORT_N_INIT = 3                  # Frames required to initialize track
DEEPSORT_MAX_IOU_DISTANCE = 0.7      # Max IoU distance for matching
DEEPSORT_MAX_COSINE_DISTANCE = 0.3   # Max appearance similarity distance
DEEPSORT_NN_BUDGET = 100             # Max samples per class for appearance
TRACKING_WEIGHT = 0.2                # Weight in fusion

# Collision Detection
OVERLAP_THRESHOLD = 0.7              # Min IoU for collision (0.7 = 70% overlap)

# ============================================================================
# MOTION ANOMALY DETECTION
# ============================================================================

# MOG2 Background Subtraction
MOG2_HISTORY = 500                   # Frames for background learning
MOG2_VAR_THRESHOLD = 16              # Variance threshold for detection
MOG2_DETECT_SHADOWS = True           # Enable shadow detection
MOTION_WEIGHT = 0.1                  # Weight in fusion

# Motion Thresholds
MOTION_THRESHOLD = 0.1               # Min motion score to flag anomaly
DEBRIS_SMOKE_THRESHOLD = 0.15        # Higher motion = likely debris/smoke

# ============================================================================
# TRACKING ANALYTICS
# ============================================================================

# Speed and Movement
SPEED_THRESHOLD = 50                 # Pixels/frame for "high speed"
                                     # Adjust based on video resolution and FPS
                                     # Example: 1080p @ 30fps: 50-100
                                     #          720p @ 30fps: 30-60
                                     #          4K @ 30fps: 100-200

SUDDEN_STOP_RATIO = 0.3              # Speed ratio for sudden stop
                                     # (current < previous × 0.3 = sudden stop)

FREEZE_DURATION = 5.0                # Seconds of low speed = "frozen"
FREEZE_SPEED_THRESHOLD = 5           # Max pixels/frame to be "frozen"

# History Tracking
POSITION_HISTORY_LENGTH = 30         # Frames of position history
SPEED_HISTORY_LENGTH = 10            # Frames of speed history

# ============================================================================
# ACCIDENT CONFIRMATION
# ============================================================================

# Multi-Criteria Fusion
ACCIDENT_THRESHOLD = 0.6             # Min weighted score for accident
AUDIO_BONUS_WEIGHT = 0.1             # Additional weight if audio peak detected

# Required Conditions (ALL must be true)
REQUIRE_YOLO_CONFIDENCE = True       # Must have YOLO detection > threshold
REQUIRE_OVERLAP = True               # Must have bbox overlap > threshold

# Supporting Evidence (ANY can trigger with above)
ENABLE_MOTION_ANOMALY = True         # Use motion detection as evidence
ENABLE_SUDDEN_STOP = True            # Use sudden stop as evidence
ENABLE_FREEZE_DETECTION = True       # Use freeze detection as evidence
ENABLE_AUDIO_DETECTION = False       # Use audio peak as evidence (not implemented)

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

# Input/Output
DEFAULT_OUTPUT_PATH = 'enhanced_output.mp4'
VIDEO_CODEC = 'mp4v'                 # Output codec (mp4v, avc1, XVID)

# Performance
SKIP_FRAMES = 1                      # Process every Nth frame (1 = all frames)
RESIZE_INPUT = None                  # Resize input (width, height) or None
USE_GPU = True                       # Enable GPU if available

# ============================================================================
# VISUALIZATION
# ============================================================================

# Display Settings
SHOW_TRACK_IDS = True                # Display tracking IDs
SHOW_CONFIDENCE = True               # Display confidence scores
SHOW_MOTION_MASK = True              # Show motion detection in corner
SHOW_STATISTICS = True               # Show stats panel

# Colors (BGR format)
COLOR_NORMAL = (0, 255, 0)           # Green for normal detections
COLOR_ACCIDENT = (0, 0, 255)         # Red for accidents
COLOR_TRACK_ID = (255, 255, 0)       # Yellow for track IDs
COLOR_OVERLAP = (0, 165, 255)        # Orange for overlaps
COLOR_TEXT = (255, 255, 255)         # White for text

# Alert Display
SHOW_ACCIDENT_ALERT = True           # Show large accident alert
ALERT_BACKGROUND_COLOR = (0, 0, 255) # Red background for alert
ALERT_TEXT_COLOR = (255, 255, 255)   # White text for alert
MAX_REASONS_DISPLAY = 2              # Max reasons to show in alert

# ============================================================================
# PRESETS
# ============================================================================

class DetectionPreset:
    """Predefined configuration presets for different scenarios"""
    
    @staticmethod
    def highway():
        """High-speed highway scenario"""
        return {
            'SPEED_THRESHOLD': 100,
            'SUDDEN_STOP_RATIO': 0.2,
            'OVERLAP_THRESHOLD': 0.75,
            'YOLO_CONFIDENCE_THRESHOLD': 0.6,
            'MOTION_THRESHOLD': 0.15
        }
    
    @staticmethod
    def urban():
        """Urban traffic scenario"""
        return {
            'SPEED_THRESHOLD': 40,
            'SUDDEN_STOP_RATIO': 0.4,
            'OVERLAP_THRESHOLD': 0.7,
            'YOLO_CONFIDENCE_THRESHOLD': 0.5,
            'MOTION_THRESHOLD': 0.1
        }
    
    @staticmethod
    def parking_lot():
        """Low-speed parking lot scenario"""
        return {
            'SPEED_THRESHOLD': 15,
            'SUDDEN_STOP_RATIO': 0.5,
            'OVERLAP_THRESHOLD': 0.65,
            'YOLO_CONFIDENCE_THRESHOLD': 0.45,
            'FREEZE_DURATION': 3.0
        }
    
    @staticmethod
    def high_precision():
        """High precision, low false positives"""
        return {
            'YOLO_CONFIDENCE_THRESHOLD': 0.7,
            'OVERLAP_THRESHOLD': 0.8,
            'ACCIDENT_THRESHOLD': 0.75,
            'MOTION_THRESHOLD': 0.2
        }
    
    @staticmethod
    def high_recall():
        """High recall, catch all potential accidents"""
        return {
            'YOLO_CONFIDENCE_THRESHOLD': 0.3,
            'OVERLAP_THRESHOLD': 0.6,
            'ACCIDENT_THRESHOLD': 0.5,
            'MOTION_THRESHOLD': 0.05
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Use default configuration
from enhanced_accident_detection import EnhancedAccidentDetector
detector = EnhancedAccidentDetector()

# Example 2: Custom configuration
import config
detector = EnhancedAccidentDetector(
    model_path=config.YOLO_MODEL_PATH,
    conf_threshold=config.YOLO_CONFIDENCE_THRESHOLD
)
detector.OVERLAP_THRESHOLD = config.OVERLAP_THRESHOLD
detector.SPEED_THRESHOLD = config.SPEED_THRESHOLD

# Example 3: Use preset
from config import DetectionPreset
preset = DetectionPreset.highway()
detector = EnhancedAccidentDetector()
for key, value in preset.items():
    if hasattr(detector, key):
        setattr(detector, key, value)

# Example 4: Fine-tune for your video
detector = EnhancedAccidentDetector()
detector.SPEED_THRESHOLD = 80        # Adjust for your video resolution
detector.OVERLAP_THRESHOLD = 0.75    # Stricter collision detection
detector.YOLO_CONFIDENCE_THRESHOLD = 0.6  # Higher confidence
"""

# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================

"""
PROBLEM: Too many false positives
SOLUTION: Increase thresholds
    YOLO_CONFIDENCE_THRESHOLD = 0.7
    OVERLAP_THRESHOLD = 0.8
    ACCIDENT_THRESHOLD = 0.75

PROBLEM: Missing real accidents
SOLUTION: Decrease thresholds
    YOLO_CONFIDENCE_THRESHOLD = 0.3
    OVERLAP_THRESHOLD = 0.6
    ACCIDENT_THRESHOLD = 0.5

PROBLEM: Slow processing
SOLUTION: Optimize performance
    SKIP_FRAMES = 2  # Process every 2nd frame
    RESIZE_INPUT = (640, 480)  # Smaller input
    Use YOLO nano model (yolov8n.pt)

PROBLEM: Missing collisions
SOLUTION: Lower overlap threshold
    OVERLAP_THRESHOLD = 0.6
    
PROBLEM: Too sensitive to motion
SOLUTION: Increase motion threshold
    MOTION_THRESHOLD = 0.2
    MOG2_VAR_THRESHOLD = 25

PROBLEM: Not detecting sudden stops
SOLUTION: Adjust speed parameters
    SPEED_THRESHOLD = 40  # Lower for slower videos
    SUDDEN_STOP_RATIO = 0.4  # More lenient
"""
