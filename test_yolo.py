# test_yolo.py - Simple test to verify YOLO works
import torch
import cv2
import numpy as np

print("Testing YOLOv5 installation...")

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
cv2.rectangle(img, (300, 150), (400, 250), (255, 255, 255), -1)

# Save test image
cv2.imwrite("test_image.jpg", img)
print("Created test_image.jpg")

try:
    # Load model
    print("Loading YOLOv5...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("✅ Model loaded successfully!")
    
    # Test detection
    print("Running detection...")
    results = model(img)
    
    print(f"\nDetections found: {len(results.xyxy[0])}")
    
    if len(results.xyxy[0]) > 0:
        print("\nSample detection:")
        print(f"  Bounding box: {results.xyxy[0][0][:4]}")
        print(f"  Confidence: {results.xyxy[0][0][4]:.2f}")
        print(f"  Class: {results.names[int(results.xyxy[0][0][5])]}")
    
    print("\n✅ YOLOv5 is working correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTry installing missing dependencies:")
    print("pip install pandas matplotlib seaborn pyyaml requests")