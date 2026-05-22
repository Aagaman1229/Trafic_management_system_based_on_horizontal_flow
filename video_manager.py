import os
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from shared_state import SharedTrafficState

ROAD_CONFIGS = [
    {"path": "v1.MOV", "name": "A", "full_name": "China Pull Road", "num_lanes": 4, "min_gst": 12, "max_gst": 44},
    {"path": "v2.mp4", "name": "B", "full_name": "Airport Road", "num_lanes": 6, "min_gst": 20, "max_gst": 27},
    {"path": "v3.MOV", "name": "C", "full_name": "Sabhagriha Chowk Road", "num_lanes": 4, "min_gst": 15, "max_gst": 44},
    {"path": "v4.MOV", "name": "D", "full_name": "Naya Bazar Road", "num_lanes": 4, "min_gst": 15, "max_gst": 28},
]

CLASS_TO_TYPE = {2: 'car', 7: 'truck', 5: 'bus', 3: 'motorcycle', 1: 'bicycle'}
CLASSES = [1, 2, 3, 5, 7]
CONFIDENCE = 0.35
INFERENCE_SIZE = 416
FRAME_SKIP = 2


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
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


class VideoManager:
    """Runs 4 video-processing threads in parallel, each continuously
    detecting and tracking vehicles on its road.  Updates are written
    to a SharedTrafficState that the signal controller and simulation
    read from."""

    def __init__(self, shared_state, model_path="yolov8s.pt"):
        self.shared_state = shared_state
        self.model_path = model_path
        self.model = None
        self.model_lock = threading.Lock()
        self.threads = []
        self.running = False

    def start(self):
        self.running = True
        self.model = YOLO(self.model_path)
        self.model.conf = CONFIDENCE
        self.model.classes = CLASSES
        try:
            self.model.to("cuda")
            try:
                self.model.half()
            except Exception:
                pass
        except Exception:
            try:
                self.model.to("mps")
            except Exception:
                pass

        for cfg in ROAD_CONFIGS:
            if not os.path.exists(cfg["path"]):
                print(f"  SKIP: {cfg['path']} not found.")
                continue
            t = threading.Thread(target=self._process_road, args=(cfg,), daemon=True)
            t.start()
            self.threads.append(t)
            print(f"  Thread started for Road {cfg['name']} ({cfg['path']})")
        print(f"VideoManager: {len(self.threads)} road thread(s) running.")

    def stop(self):
        self.running = False
        print("VideoManager: Stopping...")

    def _process_road(self, cfg):
        cap = cv2.VideoCapture(cfg["path"])
        if not cap.isOpened():
            print(f"  ERROR: Cannot open {cfg['path']}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        TYPE_COLORS = {
            'car': (0, 255, 0),
            'truck': (0, 255, 255),
            'bus': (255, 0, 0),
            'motorcycle': (255, 0, 255),
            'bicycle': (0, 165, 255)
        }

        display_w, display_h = 480, 360

        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            with self.model_lock:
                results = self.model(frame, imgsz=INFERENCE_SIZE, verbose=False)[0]

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
            tracks = tracker.update(detections)

            self.shared_state.update_counts(cfg["name"], tracker.vehicle_counts)

            # --- Annotate frame for verification display ---
            draw = frame.copy()
            scale_x = display_w / width
            scale_y = display_h / height

            # Count lines
            cv2.line(draw, line1_pt1, line1_pt2, (0, 0, 255), 3)
            cv2.line(draw, line2_pt1, line2_pt2, (0, 0, 255), 3)
            cv2.circle(draw, x_pt, 8, (0, 0, 255), -1)

            # Tracked vehicles
            for trk in tracks:
                track_id, c_x, c_y, vtype, bbox, _ = trk
                color = TYPE_COLORS.get(vtype, (255, 255, 255))
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)
                    label = f"ID:{track_id} {vtype}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(draw, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(draw, label, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    cv2.circle(draw, (int(c_x), int(c_y)), 6, color, -1)
                trail = tracker.get_trail(track_id)
                if len(trail) > 1:
                    pts = np.array(trail, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(draw, [pts], False, color, 2)

            # Overlay counted vehicles — large, high-contrast, with dark background
            # Road title bar
            rn = cfg.get('full_name', f"Road {cfg['name']}")
            title = f"  {rn}  "
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.rectangle(draw, (8, 8), (8 + tw + 12, 8 + th + 12), (20, 20, 20), -1)
            cv2.putText(draw, title, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Vehicle counts with colored type badges
            y0 = 70
            # Header
            cv2.putText(draw, "COUNTED VEHICLES:", (15, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            y0 += 35

            total_vehicles = sum(tracker.vehicle_counts.values())
            for vtype, count in tracker.vehicle_counts.items():
                color = TYPE_COLORS.get(vtype, (255, 255, 255))
                line = f"{vtype}: {count}"
                (lw, lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                # Dark pill background per row
                cv2.rectangle(draw, (15, y0 - lh + 2), (15 + lw + 20, y0 + 6), (30, 30, 30), -1)
                cv2.putText(draw, line, (22, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                y0 += 32

            # Total badge
            total_line = f"TOTAL: {total_vehicles}"
            (tlw, tlh), _ = cv2.getTextSize(total_line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(draw, (15, y0 - tlh + 2), (15 + tlw + 20, y0 + 6), (0, 80, 0), -1)
            cv2.putText(draw, total_line, (22, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Resize for uniform grid display
            display_frame = cv2.resize(draw, (display_w, display_h))
            self.shared_state.update_frame(cfg["name"], display_frame)

        cap.release()
