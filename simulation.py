import csv
import json
import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from gst import calculate_gst

# ===================== CONFIGURATION =====================
# For quick test, all four paths point to the same video.
# When you're ready for real operation, replace with your four distinct files.
VIDEO_PATHS = [
    "video1.mp4",      # North
    "video1.mp4",      # East
    "video1.mp4",      # South
    "video1.mp4"       # West
]
OUTPUT_DIR = "outputs"
CACHE_FILE = os.path.join(OUTPUT_DIR, "gst_cache.json")
CSV_FILE   = os.path.join(OUTPUT_DIR, "traffic_log.csv")

YOLO_MODEL = 'yolov8s.pt'
CONFIDENCE = 0.35
CLASSES    = [1, 2, 3, 5, 7]   # bicycle, car, motorcycle, bus, truck

# Direction names (cycling order)
DIR_NAMES = ["North", "East", "South", "West"]

# Drawing colors (same as your main.py)
TYPE_COLORS = {
    'car':        (0, 255, 0),
    'truck':      (0, 255, 255),
    'bus':        (255, 0, 0),
    'motorcycle': (255, 0, 255),
    'bicycle':    (0, 165, 255)
}

# ========== 1. Merging functions (identical to your main.py) ==========
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def merge_all_detections(detections, iou_threshold=0.4):
    items = [{'c_x': d[0], 'c_y': d[1], 'vtype': d[2], 'bbox': d[3]} for d in detections]
    merged = []
    items.sort(
        key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
        reverse=True
    )
    while items:
        best = items.pop(0)
        to_merge = [best]
        i = 0
        while i < len(items):
            if compute_iou(best['bbox'], items[i]['bbox']) > iou_threshold:
                to_merge.append(items.pop(i))
            else:
                i += 1
        if len(to_merge) > 1:
            all_boxes = [m['bbox'] for m in to_merge]
            x1 = min(b[0] for b in all_boxes)
            y1 = min(b[1] for b in all_boxes)
            x2 = max(b[2] for b in all_boxes)
            y2 = max(b[3] for b in all_boxes)
            merged.append(((x1 + x2) / 2, (y1 + y2) / 2,
                           best['vtype'],
                           np.array([x1, y1, x2, y2])))
        else:
            merged.append((best['c_x'], best['c_y'],
                           best['vtype'], best['bbox']))
    return merged

# ========== 2. Process a single video → return counts & GST ==========
def process_video(video_path, model, time_limit_sec=200):
    print(f"Processing {video_path} (max {time_limit_sec}s) ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(time_limit_sec * fps) if fps and fps > 0 else float('inf')

    x_pt      = (int(height / 3.5), int(width / 2.5))
    line1_pt1 = (0, 0)
    line1_pt2 = x_pt
    line2_pt1 = (width, height)
    line2_pt2 = x_pt

    tracker = CentroidTracker(
        max_disappeared=40,
        max_distance=120,
        line1_pt1=line1_pt1, line1_pt2=line1_pt2,
        line2_pt1=line2_pt1, line2_pt2=line2_pt2,
        min_movement=40,
        history_len=20
    )

    CLASS_TO_TYPE_MAP = {
        2: 'car',
        7: 'truck',
        5: 'bus',
        3: 'motorcycle',
        1: 'bicycle'
    }

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count > max_frames:
            break

        results = model(frame, imgsz=416, verbose=False)[0]
        raw_detections = []
        for det in results.boxes:
            cls_id = int(det.cls[0])
            if cls_id not in CLASS_TO_TYPE_MAP:
                continue
            vtype = CLASS_TO_TYPE_MAP[cls_id]
            xyxy  = det.xyxy[0].cpu().numpy()
            c_x   = (xyxy[0] + xyxy[2]) / 2
            c_y   = (xyxy[1] + xyxy[3]) / 2
            raw_detections.append((c_x, c_y, vtype, xyxy))

        detections = merge_all_detections(raw_detections, iou_threshold=0.4)
        tracker.update(detections)

    cap.release()
    counts = tracker.vehicle_counts
    gst = calculate_gst(counts, num_lanes=1)
    return counts, gst

# ========== 3. Load or compute GST values – process once, use for all ==========
def load_or_compute_gst():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        print("Loaded GST from cache.")
        return data['gst_values'], data['counts_list']

    print("Loading YOLOv8 model (first time) ...")
    model = YOLO(YOLO_MODEL)
    model.conf = CONFIDENCE
    model.classes = CLASSES
    try:
        model.half()
        print("FP16 mode enabled (GPU).")
    except:
        pass

    # Process first video once, then duplicate for the other three directions
    path = VIDEO_PATHS[0]
    counts, gst = process_video(path, model)
    print(f"  {path}: GST = {gst:.1f}s, counts = {counts}")

    gst_values = [gst] * 4
    counts_list = [counts] * 4

    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'gst_values': gst_values,
            'counts_list': [dict(c) for c in counts_list]
        }, f)
    print("GST values cached.")
    return gst_values, counts_list

# ========== 4. Real‑time multi‑video player + traffic light ==========
def run_realtime_simulation(gst_values, counts_list):
    # Open all four videos
    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            sys.exit(f"Error opening {VIDEO_PATHS[i]}")

    # Get video dimensions (assume all same size, or resize later)
    widths, heights = [], []
    for cap in caps:
        widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Resize each video to a uniform size for display (e.g., 640x360)
    DISPLAY_W, DISPLAY_H = 640, 360
    GRID_W, GRID_H = 2, 2

    canvas_w = DISPLAY_W * GRID_W
    canvas_h = DISPLAY_H * GRID_H
    cv2.namedWindow("Traffic Video Grid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Video Grid", canvas_w, canvas_h)

    cv2.namedWindow("Traffic Light", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Light", 400, 400)

    # Initialize trackers per video (each gets its own tracker)
    trackers = []
    for w, h in zip(widths, heights):
        x_pt      = (int(h / 3.5), int(w / 2.5))
        line1_pt1 = (0, 0)
        line1_pt2 = x_pt
        line2_pt1 = (w, h)
        line2_pt2 = x_pt
        tracker = CentroidTracker(
            max_disappeared=40,
            max_distance=120,
            line1_pt1=line1_pt1, line1_pt2=line1_pt2,
            line2_pt1=line2_pt1, line2_pt2=line2_pt2,
            min_movement=40,
            history_len=20
        )
        trackers.append(tracker)

    # Load YOLO model once
    model = YOLO(YOLO_MODEL)
    model.conf = CONFIDENCE
    model.classes = CLASSES
    try:
        model.half()
    except:
        pass

    # Traffic simulation state
    current_green = 0
    gst_remaining = gst_values[current_green]
    last_time = time.time()

    # CSV setup
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "direction", "green_start", "green_end",
                "gst_seconds", "car_count", "truck_count", "bus_count",
                "motorcycle_count", "bicycle_count"
            ])

    print("\n=== Real‑time simulation started (press 'q' in video grid window to quit) ===")
    CLASS_TO_TYPE_MAP = {2: 'car', 7: 'truck', 5: 'bus', 3: 'motorcycle', 1: 'bicycle'}

    while True:
        frames = []
        rets = []
        # Read next frame from each video, loop if necessary
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                # Rewind and retry
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            rets.append(ret)
            frames.append(frame)

        # Build 2x2 grid
        grid = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i, (frame, tracker) in enumerate(zip(frames, trackers)):
            resized = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            results = model(resized, imgsz=320, verbose=False)[0]

            raw_detections = []
            for det in results.boxes:
                cls_id = int(det.cls[0])
                if cls_id not in CLASS_TO_TYPE_MAP:
                    continue
                vtype = CLASS_TO_TYPE_MAP[cls_id]
                xyxy  = det.xyxy[0].cpu().numpy()
                c_x   = (xyxy[0] + xyxy[2]) / 2
                c_y   = (xyxy[1] + xyxy[3]) / 2
                raw_detections.append((c_x, c_y, vtype, xyxy))

            detections = merge_all_detections(raw_detections, iou_threshold=0.4)
            tracks = tracker.update(detections)

            draw_frame = resized.copy()
            # Draw counting lines (scaled to display size)
            w_orig, h_orig = widths[i], heights[i]
            scale_x = DISPLAY_W / w_orig
            scale_y = DISPLAY_H / h_orig
            x_pt_scaled = (int(h_orig/3.5 * scale_y), int(w_orig/2.5 * scale_x))
            cv2.line(draw_frame, (0,0), x_pt_scaled, (0,0,255), 2)
            cv2.line(draw_frame, (DISPLAY_W, DISPLAY_H), x_pt_scaled, (0,0,255), 2)

            for track in tracks:
                track_id, c_x, c_y, vtype, bbox, _ = track
                color = TYPE_COLORS.get(vtype, (255,255,255))
                if bbox is not None:
                    x1,y1,x2,y2 = map(int, bbox)
                    cv2.rectangle(draw_frame, (x1,y1), (x2,y2), color, 2)
                    label = f"ID:{track_id} {vtype}"
                    cv2.putText(draw_frame, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.circle(draw_frame, (int(c_x), int(c_y)), 4, color, -1)
                trail = tracker.get_trail(track_id)
                if len(trail) > 1:
                    pts = np.array(trail, np.int32).reshape((-1,1,2))
                    cv2.polylines(draw_frame, [pts], False, color, 1)

            # Place in grid
            row = i // GRID_W
            col = i % GRID_W
            y_start = row * DISPLAY_H
            x_start = col * DISPLAY_W
            grid[y_start:y_start+DISPLAY_H, x_start:x_start+DISPLAY_W] = draw_frame

        # Update traffic light state
        now = time.time()
        dt = now - last_time
        last_time = now
        gst_remaining -= dt

        if gst_remaining <= 0:
            # Log the completed green phase
            direction = DIR_NAMES[current_green]
            start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - dt))
            end_str   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            counts    = counts_list[current_green]
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), direction, start_str, end_str,
                    gst_values[current_green],
                    counts.get('car',0), counts.get('truck',0),
                    counts.get('bus',0), counts.get('motorcycle',0),
                    counts.get('bicycle',0)
                ])
            # Switch to next direction
            current_green = (current_green + 1) % 4
            gst_remaining = gst_values[current_green]

        # Draw traffic light window
        light_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        positions = [(200, 80), (320, 200), (200, 320), (80, 200)]  # N,E,S,W
        for idx, pos in enumerate(positions):
            color = (0, 255, 0) if idx == current_green else (0, 0, 255)
            cv2.circle(light_canvas, pos, 30, color, -1)
            cv2.putText(light_canvas, DIR_NAMES[idx], (pos[0]-25, pos[1]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(light_canvas, f"Green: {DIR_NAMES[current_green]} {gst_remaining:.1f}s",
                    (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Traffic Light", light_canvas)
        cv2.imshow("Traffic Video Grid", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print("Simulation ended. CSV log updated.")

if __name__ == "__main__":
    gst_vals, counts = load_or_compute_gst()
    run_realtime_simulation(gst_vals, counts)