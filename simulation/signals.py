from .settings import DIRECTIONS

ORANGE_PREVIEW_SECONDS = 5.0

ROAD_ORDER = ['D', 'A', 'B', 'C']


class SignalController:
    """Visual signal controller.

    When *traffic_controller* is provided (the root TrafficSignalController),
    signal states and timers are delegated to it (single source of truth).

    When *gst_values* are provided instead, runs standalone with its own
    internal timer in the correct GREEN → ORANGE → RED sequence.
    """

    def __init__(self, traffic_controller_or_gst):
        self.direction_names = ROAD_ORDER
        self.road_order = ROAD_ORDER

        if isinstance(traffic_controller_or_gst, (list, tuple)):
            self.tc = None
            self.gst = list(traffic_controller_or_gst)
            self._current_green = 0
            self._timer = self.gst[0] if self.gst else 15.0
            self._orange_timer = 0.0
        else:
            self.tc = traffic_controller_or_gst
            self._current_green = 0
            self._timer = 0.0
            self._orange_timer = 0.0

    @property
    def current_green(self):
        if self.tc is not None:
            road = self.tc.get_active_road()
            return self.road_order.index(road)
        return self._current_green

    @current_green.setter
    def current_green(self, val):
        self._current_green = val

    @property
    def timer(self):
        if self.tc is not None:
            return self.tc.get_remaining_time()
        if self._orange_timer > 0:
            return self._orange_timer
        return max(0.0, self._timer)

    def update(self, dt):
        if self.tc is not None:
            return
        if self._orange_timer > 0:
            self._orange_timer -= dt
            if self._orange_timer <= 0:
                self._orange_timer = 0.0
                self._current_green = (self._current_green + 1) % 4
                self._timer = self.gst[self._current_green]
        else:
            self._timer -= dt
            if self._timer <= 0:
                self._timer = 0.0
                self._orange_timer = ORANGE_PREVIEW_SECONDS

    def get_green_direction(self):
        return self.current_green

    def get_signal_state(self, direction_index):
        if self.tc is not None:
            road_code = self.road_order[direction_index]
            return self.tc.get_signal_state(road_code)
        if direction_index == self._current_green:
            if self._orange_timer > 0:
                return 'ORANGE'
            return 'GREEN'
        return 'RED'

    def is_green(self, direction_index):
        return self.get_signal_state(direction_index) == 'GREEN'
