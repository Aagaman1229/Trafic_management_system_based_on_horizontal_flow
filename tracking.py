import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    """
    Tracker that works only inside a defined ROI (left‑moving lane).
    Tracks vehicles, verifies right‑to‑left direction, and counts them
    when they cross a virtual counting line.
    """
    def __init__(self, max_disappeared=15, max_distance=70, line_x=0.4,
                 roi_polygon=None, direction_frames=10, min_movement=30):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.roi_polygon = roi_polygon
        self.history = {}
        self.direction_verified = {}
        self.direction_frames = direction_frames
        self.min_movement = min_movement
        self.line_x = line_x
        self.counted = {}
        self.vehicle_types = {}

        self.vehicle_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0
        }

    def _point_in_roi(self, x, y):
        if self.roi_polygon is None:
            return True
        return cv2.pointPolygonTest(
            np.array(self.roi_polygon, np.int32), (x, y), False) >= 0

    def register(self, centroid, bbox, vehicle_type):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = [centroid]
        self.direction_verified[self.next_object_id] = False
        self.counted[self.next_object_id] = False
        self.vehicle_types[self.next_object_id] = vehicle_type
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]
        del self.direction_verified[object_id]
        del self.counted[object_id]
        del self.vehicle_types[object_id]

    def update(self, detections):
        valid_detections = []
        for d in detections:
            c_x, c_y = d[0], d[1]
            if self._point_in_roi(c_x, c_y):
                valid_detections.append(d)

        if len(valid_detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._build_return_list()

        input_centroids = np.array([d[:2] for d in valid_detections])
        input_types = [d[2] for d in valid_detections]
        input_bboxes = [d[3] for d in valid_detections]

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_types[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                self.vehicle_types[object_id] = input_types[col]

                self.history[object_id].append(input_centroids[col])
                if len(self.history[object_id]) > self.direction_frames:
                    self.history[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(valid_detections))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col], input_types[col])

        return self._build_return_list()

    def _build_return_list(self):
        active_tracks = []
        if not hasattr(self, 'frame_width'):
            self.frame_width = 1280

        line_pos = int(self.line_x * self.frame_width)

        for object_id, centroid in self.objects.items():
            # Direction verification (same as before)
            if not self.direction_verified[object_id]:
                hist = self.history[object_id]
                if len(hist) >= self.direction_frames:
                    xs = [p[0] for p in hist]
                    start_x = xs[0] - xs[-1]   # positive = leftwards
                    if start_x > self.min_movement:
                        self.direction_verified[object_id] = True
                    else:
                        self.direction_verified[object_id] = False
                else:
                    continue

            if not self.direction_verified[object_id]:
                continue

            # ---- COUNTING (ONCE, with dead zone) ----
            if not self.counted[object_id]:
                if centroid[0] < line_pos:
                    vtype = self.vehicle_types[object_id]
                    if vtype in self.vehicle_counts:
                        self.vehicle_counts[vtype] += 1
                    self.counted[object_id] = True
            # (if already counted, do nothing – no resetting)

            bbox = self.bboxes.get(object_id, None)
            active_tracks.append((
                object_id,
                centroid[0], centroid[1],
                self.vehicle_types[object_id],
                bbox,
                True
            ))

        return active_tracks

    def set_frame_width(self, width):
        self.frame_width = width