import csv
import hashlib
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from gst import calculate_gst

VIDEO_CONFIGS = [
    {"path": "china_pull_main.mp4", "name": "A", "full_name": "China Pull Road", "num_lanes": 6, "min_gst": 15, "max_gst": 44, "straight_ratio": 0.72},
    {"path": "airport_main.mp4", "name": "B", "full_name": "Airport Road", "num_lanes": 4, "min_gst": 15, "max_gst": 28, "straight_ratio": 0.72},
    {"path": "sabha_main.mp4", "name": "C", "full_name": "Sabhagriha Chowk Road", "num_lanes": 4, "min_gst": 12, "max_gst": 44, "straight_ratio": 0.78},
    {"path": "nayabazar_road.mp4", "name": "D", "full_name": "Naya Bazar Road", "num_lanes": 4, "min_gst": 20, "max_gst": 27, "straight_ratio": 0.6},
]

OUTPUT_DIR = "outputs"
CACHE_FILE = os.path.join(OUTPUT_DIR, "gst_cache.json")
CSV_FILE = os.path.join(OUTPUT_DIR, "full_traffic_data.csv")
TXT_FILE = os.path.join(OUTPUT_DIR, "gst_summary.txt")
TRAFFIC_LOG_CSV = os.path.join(OUTPUT_DIR, "traffic_log.csv")

YOLO_MODEL_PATH = "yolov8s.pt"
CONFIDENCE = 0.35
CLASSES = [1, 2, 3, 5, 7]
INFERENCE_SIZE = 416
FRAME_SKIP = 2
TIME_LIMIT_SEC = 180

CLASS_TO_TYPE = {
    2: 'car',
    7: 'truck',
    5: 'bus',
    3: 'motorcycle',
    1: 'bicycle'
}


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def merge_all_detections(detections, iou_threshold=0.4):
    items = [{'c_x': d[0], 'c_y': d[1], 'vtype': d[2], 'bbox': d[3]}
             for d in detections]
    items.sort(
        key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
        reverse=True
    )
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
            merged.append(((x1 + x2) / 2, (y1 + y2) / 2,
                           best['vtype'],
                           np.array([x1, y1, x2, y2])))
        else:
            merged.append((best['c_x'], best['c_y'],
                           best['vtype'], best['bbox']))
    return merged


def video_cache_path(video_path):
    hashed = hashlib.md5(video_path.encode()).hexdigest()[:12]
    return os.path.join(OUTPUT_DIR, f".cache_{hashed}.json")


def load_video_cache(video_path):
    path = video_cache_path(video_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_video_cache(video_path, counts, gst):
    path = video_cache_path(video_path)
    with open(path, 'w') as f:
        json.dump({"counts": counts, "gst": gst}, f)


def process_single_video(video_path, model, num_lanes, min_gst, max_gst, straight_ratio=0.75):
    cached = load_video_cache(video_path)
    if cached is not None:
        print(f"  [CACHED] {video_path} — counts={cached['counts']}, GST={cached['gst']:.2f}s")
        return cached["counts"], cached["gst"]

    print(f"  Processing {video_path} (lanes={num_lanes}, GST=[{min_gst},{max_gst}]s) ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(int(TIME_LIMIT_SEC * fps), total_frames) if fps and fps > 0 else total_frames

    x_pt = (int(height / 3.5), int(width / 2.5))
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

    frame_count = 0
    last_log = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue
        if frame_count > max_frames:
            break

        if frame_count - last_log >= 500:
            print(f"    Frame {frame_count}/{max_frames} ...")
            last_log = frame_count

        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)[0]
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

    cap.release()
    counts = dict(tracker.vehicle_counts)
    gst = calculate_gst(counts, num_lanes=num_lanes, min_gst=min_gst, max_gst=max_gst, straight_ratio=straight_ratio)
    print(f"    Done — counts={counts}, GST={gst:.2f}s")

    save_video_cache(video_path, counts, gst)
    return counts, gst


def load_aggregated_results():
    results = []
    for cfg in VIDEO_CONFIGS:
        vp = cfg["path"]
        cached = load_video_cache(vp)
        if cached is None:
            print(f"  WARNING: No cached data for {vp}. Run processing first.")
            counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'bicycle': 0}
            gst = cfg["min_gst"]
        else:
            counts = cached["counts"]
            gst = cached["gst"]
        results.append({
            "video_name": vp,
            "path_name": cfg["name"],
            "full_name": cfg.get("full_name", f"Road {cfg['name']}"),
            "num_lanes": cfg["num_lanes"],
            "min_gst": cfg["min_gst"],
            "max_gst": cfg["max_gst"],
            "counts": counts,
            "gst": gst
        })
    return results


def write_outputs(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "timestamp", "car", "truck", "bus",
            "motorcycle", "bicycle", "total_vehicles", "num_lanes", "gst_seconds"
        ])
        for r in results:
            counts = r["counts"]
            total = sum(counts.values())
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                r["video_name"], ts,
                counts.get('car', 0), counts.get('truck', 0),
                counts.get('bus', 0), counts.get('motorcycle', 0),
                counts.get('bicycle', 0),
                total, r["num_lanes"], round(r["gst"], 2)
            ])
    print(f"CSV: {CSV_FILE}")

    with open(TXT_FILE, 'w') as f:
        f.write("=== GST Summary Report — Prithivi Chowk, Pokhara ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in results:
            counts = r["counts"]
            total = sum(counts.values())
            rn = r.get("full_name", f"Road {r['path_name']}")
            f.write(f"{rn} ({r['video_name']}):\n")
            f.write(f"  Lane count: {r['num_lanes']}\n")
            f.write(f"  GST range: [{r['min_gst']}, {r['max_gst']}] s\n")
            f.write(f"  Vehicle counts: {dict(counts)} (total={total})\n")
            f.write(f"  Computed GST: {r['gst']:.2f} s\n\n")
    print(f"TXT: {TXT_FILE}")

    gst_values = [r["gst"] for r in results]
    counts_list = [r["counts"] for r in results]
    lanes_list = [r["num_lanes"] for r in results]
    with open(CACHE_FILE, 'w') as f:
        json.dump({
            "gst_values": gst_values,
            "total_counts_list": counts_list,
            "lanes_list": lanes_list
        }, f)
    print(f"Cache: {CACHE_FILE}")

    dir_names = ["East", "South", "West", "North"]
    file_exists = os.path.exists(TRAFFIC_LOG_CSV)
    with open(TRAFFIC_LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "direction", "green_start", "green_end",
                "gst_seconds", "car_count", "truck_count", "bus_count",
                "motorcycle_count", "bicycle_count"
            ])
        now = time.time()
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        for i, r in enumerate(results):
            counts = r["counts"]
            writer.writerow([
                now, dir_names[i], ts_str, ts_str,
                round(r["gst"], 2),
                counts.get('car', 0), counts.get('truck', 0),
                counts.get('bus', 0), counts.get('motorcycle', 0),
                counts.get('bicycle', 0)
            ])
    print(f"Traffic log: {TRAFFIC_LOG_CSV}")


def main():
    print("=" * 60)
    print("Traffic Management System - Multi-Video Processor")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = None
    for cfg in VIDEO_CONFIGS:
        vp = cfg["path"]
        if not os.path.exists(vp):
            print(f"\n  SKIP: {vp} not found.")
            continue

        cached = load_video_cache(vp)
        if cached is not None:
            print(f"\n  [{cfg['name']}] {vp} — already cached, skipping.")
            continue

        if model is None:
            print(f"\nLoading YOLOv8s (first use, GPU if available)...")
            model = YOLO(YOLO_MODEL_PATH)
            model.conf = CONFIDENCE
            model.classes = CLASSES
            device = "cuda"
            try:
                model.to(device)
                model.half()
                print(f"  Using device: {device} (FP16)")
            except Exception:
                try:
                    model.to("mps")
                    print("  Using device: MPS (Apple Silicon)")
                except Exception:
                    print("  Using device: CPU")
                try:
                    model.half()
                except Exception:
                    pass

        print(f"\n--- Path {cfg['name']}: {vp} ---")
        process_single_video(
            vp, model,
            num_lanes=cfg["num_lanes"],
            min_gst=cfg["min_gst"],
            max_gst=cfg["max_gst"],
            straight_ratio=cfg.get("straight_ratio", 0.75)
        )

    print("\n" + "=" * 60)
    print("Aggregating results and writing output files...")
    results = load_aggregated_results()
    write_outputs(results)

    print("\n" + "=" * 60)
    print("Complete!")
    for r in results:
        rn = r.get("full_name", f"Path {r['path_name']}")
        print(f"  {rn}: GST={r['gst']:.2f}s  counts={dict(r['counts'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
