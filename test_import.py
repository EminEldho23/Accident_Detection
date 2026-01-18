import sys
import time
print("Starting import test...")
start = time.time()
try:
    import torch
    print(f"Torch imported in {time.time() - start:.2f}s")
    
    import ultralytics
    print(f"Ultralytics imported in {time.time() - start:.2f}s")
    
    from ultralytics import YOLO
    print("YOLO class imported.")
    
    print("Import test passed!")
except KeyboardInterrupt:
    print("Caught KeyboardInterrupt during test!")
except Exception as e:
    print(f"Caught exception: {e}")
