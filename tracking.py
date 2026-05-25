import numpy as np
from collections import OrderedDict, Counter
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=25, max_distance=80,
                 line1_pt1=(0, 0), line1_pt2=(200, 200),
                 line2_pt1=(640, 480), line2_pt2=(200, 200),
                 min_movement=25, history_len=20,
                 min_frames=5,
                 min_x_displacement=30,   # net pixels moved LEFT before counting
                 min_direction_ratio=0.5): # 50%+ of steps must be leftward

        self.next_object_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.line1_pt1 = np.array(line1_pt1, dtype=np.float64)
        self.line1_pt2 = np.array(line1_pt2, dtype=np.float64)
        self.line2_pt1 = np.array(line2_pt1, dtype=np.float64)
        self.line2_pt2 = np.array(line2_pt2, dtype=np.float64)

        self.history = {}
        self.class_history = {}
        self.history_len = history_len
        self.min_movement = min_movement
        self.min_frames = min_frames
        self.min_x_displacement = min_x_displacement
        self.min_direction_ratio = min_direction_ratio

        self.frame_count = {}
        self.counted = {}
        self.final_class = {}

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
        self.frame_count[self.next_object_id] = 1
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]
        del self.class_history[object_id]
        del self.counted[object_id]
        del self.final_class[object_id]
        del self.frame_count[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._build_return_list()

        input_centroids = np.array([d[:2] for d in detections], dtype=np.float64)
        input_types  = [d[2] for d in detections]
        input_bboxes = [d[3] for d in detections]

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_types[i])
        else:
            object_ids       = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D    = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id]     = input_centroids[col]
                self.bboxes[object_id]      = input_bboxes[col]
                self.disappeared[object_id] = 0
                self.frame_count[object_id] += 1

                self.class_history[object_id].append(input_types[col])
                if len(self.class_history[object_id]) > self.history_len:
                    self.class_history[object_id].pop(0)

                self.history[object_id].append(input_centroids[col])
                if len(self.history[object_id]) > self.history_len:
                    self.history[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(object_ids))) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in set(range(len(input_centroids))) - used_cols:
                self.register(input_centroids[col], input_bboxes[col], input_types[col])

        return self._build_return_list()

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        cross = float(d1[0]) * float(d2[1]) - float(d1[1]) * float(d2[0])
        if abs(cross) < 1e-10:
            return False
        diff = p3 - p1
        t = (float(diff[0]) * float(d2[1]) - float(diff[1]) * float(d2[0])) / cross
        u = (float(diff[0]) * float(d1[1]) - float(diff[1]) * float(d1[0])) / cross
        return (0.0 <= t <= 1.0) and (0.0 <= u <= 1.0)

    def _crossed_line(self, hist):
        if len(hist) < 2:
            return False
        for i in range(1, len(hist)):
            p = np.array(hist[i-1], dtype=np.float64)
            c = np.array(hist[i], dtype=np.float64)
            if (
                self._segments_intersect(p, c, self.line1_pt1, self.line1_pt2) or
                self._segments_intersect(p, c, self.line2_pt1, self.line2_pt2)
            ):
                return True
        return False

    def _get_final_class(self, object_id):
        if self.final_class[object_id] is not None:
            return self.final_class[object_id]
        history = self.class_history[object_id]
        if not history:
            return 'unknown'
        return Counter(history).most_common(1)[0][0]

    def _is_moving_right_to_left(self, object_id):
        """
        Two checks combined:

        1. NET x displacement: first position minus last must be >= min_x_displacement
           (vehicle genuinely moved left overall, not just wobbled)

        2. DIRECTIONAL CONSISTENCY: count frame steps where x decreased (moved left).
           Must be >= min_direction_ratio of all steps.
           Camera shake causes ~50/50 left/right oscillation → fails this check.
           A real left-moving vehicle has 70-90% leftward steps → passes.
        """
        hist = self.history[object_id]
        if len(hist) < 2:
            return False

        # Check 1 — net leftward displacement
        net_x = hist[0][0] - hist[-1][0]   # positive = moved left
        if net_x < self.min_x_displacement:
            return False

        # Check 2 — directional consistency
        leftward_steps = sum(
            1 for i in range(1, len(hist))
            if hist[i][0] < hist[i-1][0]   # x decreased → moved left
        )
        total_steps = len(hist) - 1
        ratio = leftward_steps / total_steps
        if ratio < self.min_direction_ratio:
            return False

        return True

    def _build_return_list(self):
        active_tracks = []
        for object_id, centroid in self.objects.items():

            if not self.counted[object_id]:
                hist = self.history[object_id]

                # Gate 1: tracked long enough (avoids ghost detections)
                if self.frame_count[object_id] >= self.min_frames and len(hist) >= 2:

                    # Gate 2: total 2D displacement (avoids near-zero movement)
                    displacement = np.linalg.norm(
                        np.array(hist[-1]) - np.array(hist[0])
                    )

                    # Gate 3: must be moving right-to-left consistently
                    # (blocks static vehicles + camera shake + left-to-right vehicles)
                    moving_rtl = self._is_moving_right_to_left(object_id)

                    if displacement >= self.min_movement and moving_rtl:

                        # Gate 4: centroid path actually crossed the line
                        if self._crossed_line(hist):
                            final_vtype = self._get_final_class(object_id)
                            if final_vtype in self.vehicle_counts:
                                self.vehicle_counts[final_vtype] += 1
                                self.final_class[object_id] = final_vtype
                                self.counted[object_id] = True

            draw_class = self._get_final_class(object_id)
            
            # --- Dynamic visual cleanup ---
            # 1. Instantly remove bounding box if vehicle disappeared (not in current frame)
            if self.disappeared[object_id] > 0:
                bbox = None
            else:
                bbox = self.bboxes.get(object_id)
                
            # 2. Check if vehicle is moving left-to-right (rightward). If so, we do not draw/trace it.
            hist = self.history[object_id]
            is_moving_right = False
            if len(hist) >= 4:
                net_x_right = hist[-1][0] - hist[0][0]  # positive = moved right
                if net_x_right > 15:
                    is_moving_right = True
            
            if is_moving_right:
                bbox = None

            active_tracks.append((
                object_id,
                centroid[0], centroid[1],
                draw_class,
                bbox,
                not is_moving_right
            ))
        return active_tracks

    def get_trail(self, object_id):
        return self.history.get(object_id, [])