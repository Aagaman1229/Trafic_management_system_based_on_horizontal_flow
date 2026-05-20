import cv2
import numpy as np
from collections import OrderedDict, Counter
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=120,
                 line_pt1=(100, 200), line_pt2=(400, 300),
                 min_movement=40, history_len=20):
        self.next_object_id = 0
        self.objects = OrderedDict()      # id -> centroid
        self.bboxes = OrderedDict()       # id -> bbox
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.line_pt1 = np.array(line_pt1, dtype=np.float32)
        self.line_pt2 = np.array(line_pt2, dtype=np.float32)

        self.history = {}                 # id -> list of centroids
        self.class_history = {}           # id -> list of class names (capped)
        self.history_len = history_len
        self.min_movement = min_movement

        self.counted = {}                 # id -> bool
        self.final_class = {}             # id -> final majority class (None until voted)

        self.vehicle_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0
        }

    def register(self, centroid, bbox, vehicle_type):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = [centroid]
        self.class_history[self.next_object_id] = [vehicle_type]
        self.counted[self.next_object_id] = False
        self.final_class[self.next_object_id] = None
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]
        del self.class_history[object_id]
        del self.counted[object_id]
        del self.final_class[object_id]

    def update(self, detections):
        # detections: list of (c_x, c_y, vtype, bbox)
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._build_return_list()

        input_centroids = np.array([d[:2] for d in detections])
        input_types = [d[2] for d in detections]
        input_bboxes = [d[3] for d in detections]

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

                # Append current class to history (capped)
                self.class_history[object_id].append(input_types[col])
                if len(self.class_history[object_id]) > self.history_len:
                    self.class_history[object_id].pop(0)

                self.history[object_id].append(input_centroids[col])
                if len(self.history[object_id]) > self.history_len:
                    self.history[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col], input_types[col])

        return self._build_return_list()

    def _crossed_line(self, prev_centroid, curr_centroid):
        line_vec = self.line_pt2 - self.line_pt1
        perp = np.array([-line_vec[1], line_vec[0]])
        prev_vec = np.array(prev_centroid) - self.line_pt1
        curr_vec = np.array(curr_centroid) - self.line_pt1
        prev_sign = np.sign(np.dot(prev_vec, perp))
        curr_sign = np.sign(np.dot(curr_vec, perp))
        return prev_sign > 0 and curr_sign < 0

    def _get_final_class(self, object_id):
        """Return the most frequent class seen for this object."""
        if self.final_class[object_id] is not None:
            return self.final_class[object_id]
        if len(self.class_history[object_id]) == 0:
            return 'unknown'
        # majority vote
        cnt = Counter(self.class_history[object_id])
        return cnt.most_common(1)[0][0]

    def _build_return_list(self):
        active_tracks = []
        for object_id, centroid in self.objects.items():
            # Count if not already done and movement conditions met
            if not self.counted[object_id]:
                hist = self.history[object_id]
                if len(hist) >= 2:
                    total_dx = hist[0][0] - hist[-1][0]
                    if total_dx >= self.min_movement:
                        if self._crossed_line(hist[-2], hist[-1]):
                            # Use majority voted class
                            final_vtype = self._get_final_class(object_id)
                            if final_vtype in self.vehicle_counts:
                                self.vehicle_counts[final_vtype] += 1
                                self.counted[object_id] = True

            # For drawing, still use the current detected class (or final)
            draw_class = self._get_final_class(object_id)
            bbox = self.bboxes.get(object_id, None)
            active_tracks.append((
                object_id,
                centroid[0], centroid[1],
                draw_class,
                bbox,
                True
            ))
        return active_tracks

    def get_trail(self, object_id):
        return self.history.get(object_id, [])