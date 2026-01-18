# Accident Detection System: How It Works

This document explains the files in the project and the step-by-step logic used to detect accidents in `advanced_predict.py`.

## 1. Key Project Files

| File Name | Description |
| :--- | :--- |
| **`best.pt`** | The "Brain". A standard YOLOv8 model trained to recognize cars and crashes in single images. |
| **`advanced_predict.py`** | The "Logic". The main script that runs the video, tracks cars over time, and makes the final decision. |
| **`requirements.txt`** | The "Ingredients". A list of Python libraries needed to run the code (e.g., `ultralytics`, `opencv`, `lapx`). |
| **`runs/`** | The "Output". Where saved videos and detection results are stored. |

---

## 2. The Detection Logic (Step-by-Step)

The system processes the video **frame by frame**. For every single image in the video, it follows this pipeline:

### Step 1: Object Tracking (Who goes where?)
*   **Action**: The system assigns a unique ID to every car (e.g., "Car #1", "Car #2").
*   **Why?** Unlike simple detection, "tracking" lets us know it's the *same* car moving across the screen, which allows us to calculate speed.
*   **Code**: `model.track(frame, persist=True)`

### Step 2: Overlap Detection (The Collision)
*   **Action**: The system draws boxes around every car. It calculates the **IoU (Intersection over Union)** between every pair of boxes.
*   **Trigger**: If two boxes overlap by more than **70%** (`IoU > 0.7`), it flags a potential crash.
*   **Visual**: You see **Red Boxes** around the cars.

### Step 3: Speed & "Freeze" Analysis (The Aftermath)
*   **Action**: The system remembers the position of every car for the last 30 frames.
*   **Logic**: It calculates how far a car moved.
    *   **High Speed**: Large distance between frames.
    *   **Stopped**: Almost zero distance.
*   **Trigger**: If a car goes from moving to **Stopped ("Frozen")** for > 5 seconds, it flags a "Freeze".
*   **Visual**: A **Yellow Dot** appears on the car.

### Step 4: Motion Anomaly (The Chaos)
*   **Action**: Uses a technique called **MOG2 (Background Subtraction)** to ignore the road and see what's moving.
*   **Logic**: Accidents often create chaotic movement that isn't a defined "car" shape—like smoke, flying debris, or pedestrians running.
*   **Trigger**: If a large percentage of the screen has chaotic motion, it increases the "Risk Score".

### Step 5: Fusion Decision (The Verdict)
The system combines all the evidence to make a final decision. It doesn't rely on just one clue.

**The Formula:**
$$ \text{Risk} = (0.7 \times \text{YOLO Conf}) + (0.2 \times \text{Tracking/Freeze}) + (0.1 \times \text{Motion}) $$

**The Trigger:**
If **(YOLO sees a crash) AND (Overlap > 70%)**, the system confirms: **"ACCIDENT DETECTED"**.

---

## 3. Visual Guide

When you run the script, look for these indicators on the screen:

*   **Green Boxes**: Normal cars driving safely.
*   **Red Boxes**: Cars that are overlapping/crashing.
*   **Yellow Dot**: A car that has suddenly stopped (frozen).
*   **"CRASH: OVERLAP DETECTED"**: The final confirmed alert.
