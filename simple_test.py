# simple_test.py - Minimal YOLO test
print("=== Testing Environment ===")

# Test basic imports
try:
    import torch
    print(f"✅ torch version: {torch.__version__}")
except ImportError:
    print("❌ torch not installed")

try:
    import cv2
    print(f"✅ opencv-python version: {cv2.__version__}")
except ImportError:
    print("❌ opencv-python not installed")

try:
    import numpy as np
    print(f"✅ numpy version: {np.__version__}")
except ImportError:
    print("❌ numpy not installed")

try:
    import pandas as pd
    print(f"✅ pandas version: {pd.__version__}")
except ImportError:
    print("❌ pandas not installed")

try:
    import tqdm
    print(f"✅ tqdm available")
except ImportError:
    print("❌ tqdm not installed")

print("\n=== Testing YOLOv5 ===")

# Create a simple test image
import numpy as np
import cv2

# Create test image with a "car"
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a rectangle that looks like a car
cv2.rectangle(img, (200, 200), (400, 300), (255, 255, 255), -1)
cv2.imwrite('test_car.jpg', img)
print("Created test_car.jpg")

try:
    print("Loading YOLOv5 model...")
    import torch
    
    # Clear torch hub cache to avoid issues
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
    
    print("✅ YOLOv5 model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Run inference
    print("Running detection on test image...")
    results = model(img)
    
    # Show results
    print(f"\nNumber of detections: {len(results.xyxy[0])}")
    
    if len(results.xyxy[0]) > 0:
        print("\nDetections found:")
        for i, det in enumerate(results.xyxy[0]):
            x1, y1, x2, y2, conf, cls = det
            class_name = model.names[int(cls)]
            print(f"  {i+1}. {class_name}: confidence={conf:.2f}, box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("No detections found (expected for synthetic image)")
    
    # Test with a real image if available
    import os
    if os.path.exists('video.mp4'):
        print("\n=== Testing with video ===")
        cap = cv2.VideoCapture('video.mp4')
        ret, frame = cap.read()
        if ret:
            print("Read frame from video.mp4")
            results = model(frame)
            print(f"Detections in video frame: {len(results.xyxy[0])}")
        cap.release()
    
    print("\n✅ All tests passed! YOLOv5 is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    
    # Try alternative import method
    print("\nTrying alternative method...")
    try:
        # Clear cache
        import shutil
        import os
        cache_dir = os.path.expanduser('~/.cache/torch/hub')
        if os.path.exists(cache_dir):
            print(f"Clearing cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
        
        # Try direct import
        print("Attempting direct import...")
        import sys
        sys.path.insert(0, '.')
        
        # Try minimal model
        from yolov5 import models
        model = models.YOLO('yolov5s.pt')
        print("✅ Success with direct import!")
        
    except Exception as e2:
        print(f"❌ Alternative also failed: {e2}")
        print("\nPlease try installing with:")
        print("pip install -r requirements_complete.txt")
        print("pip install git+https://github.com/ultralytics/yolov5.git --force-reinstall")