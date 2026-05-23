import threading


class SharedTrafficState:
    """Thread-safe container for live traffic data shared between
    video processing threads, signal controller, and simulation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}}
        self._total_vehicles = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self._active_road = 'A'
        self._signal_states = {'A': 'GREEN', 'B': 'RED', 'C': 'RED', 'D': 'RED'}
        self._gst_values = {'A': 12.0, 'B': 20.0, 'C': 15.0, 'D': 15.0}
        self._frames = {'A': None, 'B': None, 'C': None, 'D': None}
        self._cycle_count = 0
        self._running = True

    # --- Counts ---

    def update_counts(self, road, counts):
        with self._lock:
            self._counts[road] = dict(counts)
            self._total_vehicles[road] = sum(counts.values())

    def get_counts(self, road):
        with self._lock:
            return dict(self._counts.get(road, {}))

    def get_all_counts(self):
        with self._lock:
            return {r: dict(c) for r, c in self._counts.items()}

    def get_total_vehicles(self, road):
        with self._lock:
            return self._total_vehicles.get(road, 0)

    # --- Active road ---

    def set_active_road(self, road):
        with self._lock:
            self._active_road = road

    def get_active_road(self):
        with self._lock:
            return self._active_road

    # --- Signal states ---

    def set_signal_state(self, road, state):
        with self._lock:
            self._signal_states[road] = state

    def get_signal_state(self, road):
        with self._lock:
            return self._signal_states.get(road, 'RED')

    def get_all_signal_states(self):
        with self._lock:
            return dict(self._signal_states)

    # --- GST values ---

    def set_gst(self, road, value):
        with self._lock:
            self._gst_values[road] = value

    def get_gst(self, road):
        with self._lock:
            return self._gst_values.get(road, 0)

    def get_all_gst(self):
        with self._lock:
            return dict(self._gst_values)

    # --- Annotated frames (for verification window) ---

    def update_frame(self, road, frame):
        with self._lock:
            self._frames[road] = frame

    def get_frame(self, road):
        with self._lock:
            return self._frames.get(road)

    def get_all_frames(self):
        with self._lock:
            return {r: self._frames.get(r) for r in ['A', 'B', 'C', 'D']}

    # --- Cycle count ---

    def set_cycle_count(self, count):
        with self._lock:
            self._cycle_count = count

    def get_cycle_count(self):
        with self._lock:
            return self._cycle_count

    # --- Lifecycle ---

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self):
        with self._lock:
            return self._running
