from gst import calculate_gst

ROAD_CONFIGS = {
    'A': {"num_lanes": 6, "min_gst": 15, "max_gst": 44, "straight_ratio": 0.72},
    'B': {"num_lanes": 4, "min_gst": 15, "max_gst": 28, "straight_ratio": 0.72},
    'C': {"num_lanes": 4, "min_gst": 12, "max_gst": 44, "straight_ratio": 0.78},
    'D': {"num_lanes": 4, "min_gst": 20, "max_gst": 27, "straight_ratio": 0.6},
}

ORANGE_PREVIEW_SECONDS = 5.0


class TrafficSignalController:
    """Cycles A → B → C → D → loop, driven by update(dt) from the simulation.

    GST for the next road is computed *immediately before* that road
    turns green, using the cumulative counts from the TimelineReplay.

    Phase sequence per road: GREEN (full GST) → ORANGE (5 s) → RED.
    """

    def __init__(self, timeline_replay):
        self.timeline = timeline_replay
        self.road_order = ['A', 'B', 'C', 'D']
        self._current_idx = 0
        self._timer = 12.0
        self._orange_timer = 0.0
        self._cycle_count = 0
        self._signal_states = {r: 'RED' for r in self.road_order}
        self._gst_values = {r: ROAD_CONFIGS[r]["min_gst"] for r in self.road_order}
        self.switch_callbacks = []
        self._switch_to('A')

    def _compute_gst(self, road):
        cfg = ROAD_CONFIGS[road]
        counts = self.timeline.get_counts(road)
        return calculate_gst(counts, cfg["num_lanes"], cfg["min_gst"], cfg["max_gst"], cfg["straight_ratio"])

    def _switch_to(self, road):
        was_idx = self._current_idx
        self._current_idx = self.road_order.index(road)
        gst = self._compute_gst(road)
        self._timer = gst
        self._orange_timer = 0.0
        self._gst_values[road] = gst
        for r in self.road_order:
            self._signal_states[r] = 'GREEN' if r == road else 'RED'
        for cb in self.switch_callbacks:
            cb(road, gst)

    def update(self, dt):
        if self._orange_timer > 0:
            self._orange_timer -= dt
            if self._orange_timer <= 0:
                self._orange_timer = 0.0
                next_road = self.road_order[(self._current_idx + 1) % 4]
                if next_road == 'A':
                    self._cycle_count += 1
                self._switch_to(next_road)
        else:
            self._timer -= dt
            if self._timer <= 0:
                self._timer = 0.0
                self._orange_timer = ORANGE_PREVIEW_SECONDS
                road = self.road_order[self._current_idx]
                for r in self.road_order:
                    self._signal_states[r] = 'ORANGE' if r == road else 'RED'

    def get_active_road(self):
        return self.road_order[self._current_idx]

    def get_remaining_time(self):
        if self._orange_timer > 0:
            return self._orange_timer
        return max(0.0, self._timer)

    def get_signal_state(self, road):
        if self._orange_timer > 0:
            return 'ORANGE' if road == self.road_order[self._current_idx] else 'RED'
        if road == self.road_order[self._current_idx] and self._timer > 0:
            return 'GREEN'
        return 'RED'

    def get_gst(self, road):
        return self._gst_values.get(road, ROAD_CONFIGS[road]["min_gst"])

    def get_gst_values(self):
        return [self._gst_values[r] for r in self.road_order]

    def get_cycle_count(self):
        return self._cycle_count

    def on_switch(self, callback):
        self.switch_callbacks.append(callback)
