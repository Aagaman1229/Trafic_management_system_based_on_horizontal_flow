import json
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker

ROAD_CONFIGS = [
    {"path": "china_pull_main.mp4", "name": "A", "full_name": "China Pull Road", "num_lanes": 6, "min_gst": 15, "max_gst": 44, "straight_ratio": 0.72},
    {"path": "airport_main.mp4", "name": "B", "full_name": "Airport Road", "num_lanes": 4, "min_gst": 15, "max_gst": 28, "straight_ratio": 0.72},
    {"path": "sabha_main.mp4", "name": "C", "full_name": "Sabhagriha Chowk Road", "num_lanes": 4, "min_gst": 12, "max_gst": 44, "straight_ratio": 0.78},
    {"path": "nayabazar_road.mp4", "name": "D", "full_name": "Naya Bazar Road", "num_lanes": 4, "min_gst": 20, "max_gst": 27, "straight_ratio": 0.6},
]

OUTPUT_DIR = "outputs"
TIMELINE_FILE = os.path.join(OUTPUT_DIR, "traffic_timeline.json")

CLASS_TO_TYPE = {2: 'car', 7: 'truck', 5: 'bus', 3: 'motorcycle', 1: 'bicycle'}
CLASSES = [1, 2, 3, 5, 7]
CONFIDENCE = 0.35
INFERENCE_SIZE = 320
FRAME_SKIP = 2


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def merge_all_detections(detections, iou_threshold=0.4):
    items = [{'c_x': d[0], 'c_y': d[1], 'vtype': d[2], 'bbox': d[3]} for d in detections]
    items.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
    merged = []
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
            merged.append(((x1 + x2) / 2, (y1 + y2) / 2, best['vtype'], np.array([x1, y1, x2, y2])))
        else:
            merged.append((best['c_x'], best['c_y'], best['vtype'], best['bbox']))
    return merged


def process_video_timeline(cfg, model):
    cap = cv2.VideoCapture(cfg["path"])
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {cfg['path']}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x_pt = (int(height / 3.5), int(width / 2.5))
    line1_pt1 = (0, 0)
    line1_pt2 = x_pt
    line2_pt1 = (width, height)
    line2_pt2 = x_pt

    tracker = CentroidTracker(
        max_disappeared=40, max_distance=120,
        line1_pt1=line1_pt1, line1_pt2=line1_pt2,
        line2_pt1=line2_pt1, line2_pt2=line2_pt2,
        min_movement=40, history_len=20
    )

    timeline = []
    frame_count = 0
    last_log = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        video_time = frame_count / fps

        if frame_count - last_log >= 500:
            print(f"    Frame {frame_count}/{total_frames} (video time {video_time:.1f}s)")
            last_log = frame_count

        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False, conf=CONFIDENCE, iou=0.45)[0]
        raw_detections = []
        for det in results.boxes:
            cls_id = int(det.cls[0])
            if cls_id not in CLASS_TO_TYPE:
                continue
            vtype = CLASS_TO_TYPE[cls_id]
            xyxy = det.xyxy[0].cpu().numpy()
            c_x = (xyxy[0] + xyxy[2]) / 2
            c_y = (xyxy[1] + xyxy[3]) / 2
            raw_detections.append((c_x, c_y, vtype, xyxy))

        detections = merge_all_detections(raw_detections, iou_threshold=0.4)
        tracker.update(detections)

        timeline.append({
            "t": round(video_time, 3),
            "counts": dict(tracker.vehicle_counts)
        })

    cap.release()
    return timeline


def main():
    print("=" * 60)
    print("  TRAFFIC TIMELINE EXTRACTOR")
    print("  Extracts time-stamped vehicle counts from all 4 videos")
    print("  Output: outputs/traffic_timeline.json")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading YOLOv8s model...")
    model = YOLO("yolov8s.pt")
    model.conf = CONFIDENCE
    model.classes = CLASSES
    try:
        model.to("cuda")
        try:
            model.half()
            print("  Using CUDA with FP16")
        except Exception:
            print("  Using CUDA (FP32)")
    except Exception:
        try:
            model.to("mps")
            print("  Using MPS (Apple Silicon)")
        except Exception:
            print("  Using CPU")

    all_timelines = {}
    for cfg in ROAD_CONFIGS:
        if not os.path.exists(cfg["path"]):
            print(f"\n  SKIP: {cfg['path']} not found.")
            all_timelines[cfg["name"]] = []
            continue

        print(f"\n  Processing Road {cfg['name']}: {cfg['full_name']} ({cfg['path']})")
        tl = process_video_timeline(cfg, model)
        all_timelines[cfg["name"]] = tl
        if tl:
            last = tl[-1]
            total = sum(last["counts"].values())
            print(f"    Done — {len(tl)} entries, final counts: {last['counts']} (total={total})")

    output = {"timeline": all_timelines}
    with open(TIMELINE_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nTimeline saved to {TIMELINE_FILE}")

    print("\nSummary:")
    for name, tl in all_timelines.items():
        if tl:
            duration = tl[-1]["t"]
            total = sum(tl[-1]["counts"].values())
            print(f"  Road {name}: {duration:.1f}s timeline, {total} total vehicles")
        else:
            print(f"  Road {name}: no data")


if __name__ == "__main__":
    main()
