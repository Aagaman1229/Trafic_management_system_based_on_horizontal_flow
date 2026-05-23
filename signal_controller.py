import time
import threading
from shared_state import SharedTrafficState
from gst import calculate_gst

ROAD_CONFIGS = {
    'A': {"num_lanes": 6, "min_gst": 15, "max_gst": 44, "straight_ratio": 0.72},
    'B': {"num_lanes": 4, "min_gst": 15, "max_gst": 28, "straight_ratio": 0.72},
    'C': {"num_lanes": 4, "min_gst": 12, "max_gst": 44, "straight_ratio": 0.78},
    'D': {"num_lanes": 4, "min_gst": 20, "max_gst": 27, "straight_ratio": 0.6},
}

ORANGE_PREVIEW_SECONDS = 5.0


class TrafficSignalController:
    """Cycles A → B → C → D → loop.
    GST for the next road is computed *immediately before* that road turns green,
    using the live accumulated counts from the SharedTrafficState."""

    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.road_order = ['A', 'B', 'C', 'D']
        self._current_idx = 0
        self._timer = 12.0
        self._cycle_count = 0
        self._lock = threading.Lock()
        self.running = False
        self.thread = None
        self.switch_callbacks = []

    def _compute_gst(self, road):
        cfg = ROAD_CONFIGS[road]
        counts = self.shared_state.get_counts(road)
        return calculate_gst(counts, cfg["num_lanes"], cfg["min_gst"], cfg["max_gst"], cfg["straight_ratio"])

    def _switch_to(self, road):
        if self._current_idx == 3 and road == 'A':
            self._cycle_count += 1
            self.shared_state.set_cycle_count(self._cycle_count)
        gst = self._compute_gst(road)
        with self._lock:
            self._timer = gst
            self._current_idx = self.road_order.index(road)
        self.shared_state.set_active_road(road)
        self.shared_state.set_gst(road, gst)
        for r in self.road_order:
            self.shared_state.set_signal_state(r, 'GREEN' if r == road else 'RED')
        for cb in self.switch_callbacks:
            cb(road, gst)

    def get_active_road(self):
        with self._lock:
            return self.road_order[self._current_idx]

    def get_remaining_time(self):
        with self._lock:
            return max(0.0, self._timer)

    def get_signal_state(self, road):
        active = self.get_active_road()
        if road == active:
            return 'GREEN'
        next_idx = (self._current_idx + 1) % 4
        next_road = self.road_order[next_idx]
        remaining = self.get_remaining_time()
        if road == next_road and remaining <= ORANGE_PREVIEW_SECONDS:
            return 'ORANGE'
        return 'RED'

    def start(self):
        self.running = True
        self._switch_to('A')
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def on_switch(self, callback):
        self.switch_callbacks.append(callback)

    def _run_loop(self):
        TICK = 0.05
        while self.running:
            time.sleep(TICK)
            if not self.running:
                break
            with self._lock:
                self._timer -= TICK
                if self._timer <= 0:
                    next_road = self.road_order[(self._current_idx + 1) % 4]
                else:
                    next_road = None
            if next_road is not None:
                self._switch_to(next_road)
