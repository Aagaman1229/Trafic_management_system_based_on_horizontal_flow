import csv
import json
import os
import time
import random
import numpy as np
import cv2
from ultralytics import YOLO
from tracking import CentroidTracker
from gst import calculate_gst

# ===================== CONFIGURATION =====================
VIDEO_PATH = "video1.mp4"          # single video for GST calculation
OUTPUT_DIR = "outputs"
CACHE_FILE = os.path.join(OUTPUT_DIR, "gst_cache.json")
CSV_FILE   = os.path.join(OUTPUT_DIR, "traffic_log.csv")

YOLO_MODEL = 'yolov8s.pt'
CONFIDENCE = 0.35
CLASSES    = [1, 2, 3, 5, 7]      # bicycle, car, motorcycle, bus, truck

# Direction names (cycling order)
DIR_NAMES = ["North", "East", "South", "West"]

# Vehicle colors
TYPE_COLORS = {
    'car':        (0, 255, 0),
    'truck':      (0, 255, 255),
    'bus':        (255, 0, 0),
    'motorcycle': (255, 0, 255),
    'bicycle':    (0, 165, 255)
}

# ========== 1. Merging functions (unchanged) ==========
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

# ========== 2. Process video → get counts & GST (using straight traffic only) ==========
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
    total_counts = tracker.vehicle_counts
    # Use only 2/3 of vehicles for straight traffic (1/3 are turning)
    straight_counts = {k: int(v * 2/3) for k, v in total_counts.items()}
    gst = calculate_gst(straight_counts, num_lanes=1)
    return total_counts, gst

# ========== 3. Load or compute GST ==========
def load_or_compute_gst():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Delete old cache if key error
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        if 'gst_values' in data and 'total_counts_list' in data:
            print("Loaded GST from cache.")
            return data['gst_values'], data['total_counts_list']

    print("Loading YOLOv8 model (first time) ...")
    model = YOLO(YOLO_MODEL)
    model.conf = CONFIDENCE
    model.classes = CLASSES
    try:
        model.half()
        print("FP16 mode enabled (GPU).")
    except:
        pass

    total_counts, gst = process_video(VIDEO_PATH, model)
    print(f"  {VIDEO_PATH}: GST (straight) = {gst:.1f}s, total counts = {total_counts}")

    # Use same GST for all four directions (straight traffic timing)
    gst_values = [gst] * 4
    total_counts_list = [total_counts] * 4

    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'gst_values': gst_values,
            'total_counts_list': [dict(c) for c in total_counts_list]
        }, f)
    print("GST values cached.")
    return gst_values, total_counts_list

# ========== 4. Vehicle class with straight/left-turn support ==========
class Vehicle:
    def __init__(self, x, y, vtype, direction, movement_type='straight', speed=2.0):
        self.x = float(x)
        self.y = float(y)
        self.vtype = vtype
        self.direction = direction      # 0=N,1=E,2=S,3=W (initial approach)
        self.movement_type = movement_type  # 'straight' or 'left'
        self.speed = speed
        self.stopped = False
        self.passed_intersection = False
        self.turning = False           # True when executing left turn
        self.turn_progress = 0.0       # 0..1 interpolation for turn arc

    def update(self, light_green, stop_line_positions, vehicles_ahead):
        # If already passed intersection, keep moving
        if self.passed_intersection:
            self._move()
            return

        # Turning vehicles ignore red light
        effective_green = light_green if self.movement_type == 'straight' else True

        # Check if need to stop
        stop_pos = stop_line_positions[self.direction]
        should_stop = False
        if not effective_green:
            # Red light – stop at stop line
            if self._distance_to_stop(stop_pos) < 5:
                self.stopped = True
            else:
                should_stop = True
        else:
            # Green light – move, but respect vehicles ahead
            if vehicles_ahead:
                if self._distance_to_vehicle_ahead(vehicles_ahead[0]) < 20:
                    should_stop = True
                    self.stopped = True
                else:
                    self.stopped = False

        if self.stopped and effective_green and not vehicles_ahead:
            self.stopped = False

        if self.stopped:
            return

        # Execute left turn if at intersection and movement_type is 'left'
        if self.movement_type == 'left' and not self.turning and self._near_intersection(stop_pos):
            self.turning = True
            # Change direction to the left-turn destination
            new_dir = {0: 3, 1: 0, 2: 1, 3: 2}[self.direction]
            self.direction = new_dir
            self.passed_intersection = True   # free from signal after turning

        self._move()

    def _move(self):
        if self.direction == 0:   # North → moving down on screen
            self.y += self.speed
        elif self.direction == 1: # East → moving left
            self.x -= self.speed
        elif self.direction == 2: # South → moving up
            self.y -= self.speed
        elif self.direction == 3: # West → moving right
            self.x += self.speed

    def _distance_to_stop(self, stop_pos):
        return abs(self.y - stop_pos) if self.direction in (0,2) else abs(self.x - stop_pos)

    def _distance_to_vehicle_ahead(self, other):
        return abs(self.y - other.y) if self.direction in (0,2) else abs(self.x - other.x)

    def _near_intersection(self, stop_pos):
        return self._distance_to_stop(stop_pos) < 30

# ========== 5. Main simulation ==========
def run_simulation(gst_values, total_counts_list):
    WINDOW_W, WINDOW_H = 800, 800
    INTERSECTION_CENTER = (WINDOW_W//2, WINDOW_H//2)
    ROAD_WIDTH = 100
    LANE_WIDTH = ROAD_WIDTH // 2   # 50 px per lane

    # Stop line positions for each direction
    STOP_LINES = {
        0: INTERSECTION_CENTER[1] - ROAD_WIDTH//2 - 10,  # North
        1: INTERSECTION_CENTER[0] + ROAD_WIDTH//2 + 10,  # East
        2: INTERSECTION_CENTER[1] + ROAD_WIDTH//2 + 10,  # South
        3: INTERSECTION_CENTER[0] - ROAD_WIDTH//2 - 10   # West
    }

    # Light positions
    LIGHT_POSITIONS = [
        (INTERSECTION_CENTER[0], INTERSECTION_CENTER[1] - 150),  # N
        (INTERSECTION_CENTER[0] + 150, INTERSECTION_CENTER[1]),  # E
        (INTERSECTION_CENTER[0], INTERSECTION_CENTER[1] + 150),  # S
        (INTERSECTION_CENTER[0] - 150, INTERSECTION_CENTER[1])   # W
    ]

    # Vehicles per direction, separate by lane type
    vehicles_straight = {d: [] for d in range(4)}
    vehicles_left = {d: [] for d in range(4)}

    def initialize_vehicles():
        """Place all vehicles based on total_counts_list onto the map."""
        for d in range(4):
            counts = total_counts_list[d]  # dict of vehicle counts
            # Build a list of vehicle types
            vtype_list = []
            for vtype, cnt in counts.items():
                vtype_list.extend([vtype] * cnt)
            # Shuffle for variety
            random.shuffle(vtype_list)

            # Split into straight (2/3) and left (1/3) – already accounted in counts? No, counts are total, we split again
            # But note: the GST was computed using 2/3 straight counts. The total_counts_list contains the original total.
            # We want to show all vehicles, with 1/3 turning left. So we split the list.
            n_total = len(vtype_list)
            n_left = n_total // 3
            n_straight = n_total - n_left

            straight_types = vtype_list[:n_straight]
            left_types = vtype_list[n_straight:]

            # Spacing between vehicles
            SPACING = 60

            # Direction-specific offsets
            if d == 0:  # North -> vehicles move DOWN (y increases)
                straight_lane_x = INTERSECTION_CENTER[0] + LANE_WIDTH//2
                left_lane_x     = INTERSECTION_CENTER[0] - LANE_WIDTH//2
                stop_y = STOP_LINES[0]
                for i, vtype in enumerate(straight_types):
                    y = stop_y - (i+1)*SPACING
                    vehicles_straight[d].append(Vehicle(straight_lane_x, y, vtype, d, 'straight', speed=random.uniform(1.5, 3.0)))
                for i, vtype in enumerate(left_types):
                    y = stop_y - (i+1)*SPACING
                    vehicles_left[d].append(Vehicle(left_lane_x, y, vtype, d, 'left', speed=random.uniform(1.5, 3.0)))

            elif d == 1:  # East -> vehicles move LEFT (x decreases)
                straight_lane_y = INTERSECTION_CENTER[1] - LANE_WIDTH//2
                left_lane_y     = INTERSECTION_CENTER[1] + LANE_WIDTH//2
                stop_x = STOP_LINES[1]
                for i, vtype in enumerate(straight_types):
                    x = stop_x + (i+1)*SPACING
                    vehicles_straight[d].append(Vehicle(x, straight_lane_y, vtype, d, 'straight', speed=random.uniform(1.5, 3.0)))
                for i, vtype in enumerate(left_types):
                    x = stop_x + (i+1)*SPACING
                    vehicles_left[d].append(Vehicle(x, left_lane_y, vtype, d, 'left', speed=random.uniform(1.5, 3.0)))

            elif d == 2:  # South -> vehicles move UP (y decreases)
                straight_lane_x = INTERSECTION_CENTER[0] - LANE_WIDTH//2
                left_lane_x     = INTERSECTION_CENTER[0] + LANE_WIDTH//2
                stop_y = STOP_LINES[2]
                for i, vtype in enumerate(straight_types):
                    y = stop_y + (i+1)*SPACING
                    vehicles_straight[d].append(Vehicle(straight_lane_x, y, vtype, d, 'straight', speed=random.uniform(1.5, 3.0)))
                for i, vtype in enumerate(left_types):
                    y = stop_y + (i+1)*SPACING
                    vehicles_left[d].append(Vehicle(left_lane_x, y, vtype, d, 'left', speed=random.uniform(1.5, 3.0)))

            elif d == 3:  # West -> vehicles move RIGHT (x increases)
                straight_lane_y = INTERSECTION_CENTER[1] + LANE_WIDTH//2
                left_lane_y     = INTERSECTION_CENTER[1] - LANE_WIDTH//2
                stop_x = STOP_LINES[3]
                for i, vtype in enumerate(straight_types):
                    x = stop_x - (i+1)*SPACING
                    vehicles_straight[d].append(Vehicle(x, straight_lane_y, vtype, d, 'straight', speed=random.uniform(1.5, 3.0)))
                for i, vtype in enumerate(left_types):
                    x = stop_x - (i+1)*SPACING
                    vehicles_left[d].append(Vehicle(x, left_lane_y, vtype, d, 'left', speed=random.uniform(1.5, 3.0)))

    initialize_vehicles()

    # Traffic state
    current_green = 0
    gst_remaining = gst_values[current_green]
    last_time = time.time()

    # CSV logging
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "direction", "green_start", "green_end",
                "gst_seconds", "car_count", "truck_count", "bus_count",
                "motorcycle_count", "bicycle_count"
            ])

    print("\n=== Traffic simulation with real vehicle counts started (press 'q' to quit) ===")
    cv2.namedWindow("Traffic Simulation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Simulation", WINDOW_W, WINDOW_H)

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        # Update timer
        gst_remaining -= dt
        if gst_remaining <= 0:
            direction = DIR_NAMES[current_green]
            start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - dt))
            end_str   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            counts    = total_counts_list[current_green]
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), direction, start_str, end_str,
                    gst_values[current_green],
                    counts.get('car',0), counts.get('truck',0),
                    counts.get('bus',0), counts.get('motorcycle',0),
                    counts.get('bicycle',0)
                ])
            current_green = (current_green + 1) % 4
            gst_remaining = gst_values[current_green]

        # Update straight vehicles
        for d in range(4):
            light_green = (d == current_green)
            vehicles_straight[d].sort(key=lambda v: -v._distance_to_stop(STOP_LINES[d]))
            for i, v in enumerate(vehicles_straight[d]):
                ahead = vehicles_straight[d][:i]
                v.update(light_green, STOP_LINES, ahead)

        # Update left-turn vehicles (always green effectively)
        for d in range(4):
            vehicles_left[d].sort(key=lambda v: -v._distance_to_stop(STOP_LINES[d]))
            for i, v in enumerate(vehicles_left[d]):
                ahead = vehicles_left[d][:i]
                v.update(True, STOP_LINES, ahead)

        # Remove vehicles that left the screen
        all_empty = True
        for d in range(4):
            vehicles_straight[d] = [v for v in vehicles_straight[d] if 0 < v.x < WINDOW_W and 0 < v.y < WINDOW_H]
            vehicles_left[d] = [v for v in vehicles_left[d] if 0 < v.x < WINDOW_W and 0 < v.y < WINDOW_H]
            if vehicles_straight[d] or vehicles_left[d]:
                all_empty = False

        # If all vehicles gone, reset
        if all_empty:
            initialize_vehicles()

        # ---- Drawing ----
        canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

        # Draw roads
        cv2.rectangle(canvas,
                      (INTERSECTION_CENTER[0] - ROAD_WIDTH//2, 0),
                      (INTERSECTION_CENTER[0] + ROAD_WIDTH//2, WINDOW_H),
                      (80,80,80), -1)
        cv2.rectangle(canvas,
                      (0, INTERSECTION_CENTER[1] - ROAD_WIDTH//2),
                      (WINDOW_W, INTERSECTION_CENTER[1] + ROAD_WIDTH//2),
                      (80,80,80), -1)
        # Intersection box
        cv2.rectangle(canvas,
                      (INTERSECTION_CENTER[0] - ROAD_WIDTH//2, INTERSECTION_CENTER[1] - ROAD_WIDTH//2),
                      (INTERSECTION_CENTER[0] + ROAD_WIDTH//2, INTERSECTION_CENTER[1] + ROAD_WIDTH//2),
                      (120,120,120), -1)

        # Lane markings (dashed center line)
        for x in range(0, WINDOW_W, 20):
            cv2.line(canvas, (x, INTERSECTION_CENTER[1]), (x+10, INTERSECTION_CENTER[1]), (255,255,255), 1)
        for y in range(0, WINDOW_H, 20):
            cv2.line(canvas, (INTERSECTION_CENTER[0], y), (INTERSECTION_CENTER[0], y+10), (255,255,255), 1)

        # Draw stop lines
        for d, stop in STOP_LINES.items():
            if d in (0,2):
                pt1 = (INTERSECTION_CENTER[0] - ROAD_WIDTH//2, stop)
                pt2 = (INTERSECTION_CENTER[0] + ROAD_WIDTH//2, stop)
            else:
                pt1 = (stop, INTERSECTION_CENTER[1] - ROAD_WIDTH//2)
                pt2 = (stop, INTERSECTION_CENTER[1] + ROAD_WIDTH//2)
            cv2.line(canvas, pt1, pt2, (255,255,0), 2)

        # Draw vehicles
        for d in range(4):
            for v in vehicles_straight[d] + vehicles_left[d]:
                color = TYPE_COLORS.get(v.vtype, (255,255,255))
                w, h = 15, 25
                if v.vtype == 'truck':
                    w, h = 20, 35
                elif v.vtype == 'bus':
                    w, h = 20, 40
                elif v.vtype == 'motorcycle':
                    w, h = 10, 20
                elif v.vtype == 'bicycle':
                    w, h = 8, 18
                cx, cy = int(v.x), int(v.y)
                if v.direction in (0,2):  # vertical
                    cv2.rectangle(canvas, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color, -1)
                else:  # horizontal
                    cv2.rectangle(canvas, (cx-h//2, cy-w//2), (cx+h//2, cy+w//2), color, -1)

        # Draw traffic lights
        for idx, pos in enumerate(LIGHT_POSITIONS):
            color = (0, 255, 0) if idx == current_green else (0, 0, 255)
            cv2.circle(canvas, pos, 20, color, -1)
            cv2.putText(canvas, DIR_NAMES[idx], (pos[0]-20, pos[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Timer and info
        cv2.putText(canvas, f"Green: {DIR_NAMES[current_green]} {gst_remaining:.1f}s",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(canvas, "Left lane: always moving", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Traffic Simulation", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Simulation ended. CSV log updated.")

if __name__ == "__main__":
    gst_vals, total_counts = load_or_compute_gst()
    run_simulation(gst_vals, total_counts)