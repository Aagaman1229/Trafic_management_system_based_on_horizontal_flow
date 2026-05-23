from settings import DIRECTIONS

# How many seconds before the current green expires to show orange on the NEXT approach
ORANGE_PREVIEW_SECONDS = 5.0

ROAD_ORDER = ['D', 'A', 'B', 'C']


class SignalController:
    """
    Manages the traffic signals for the 4-way intersection.
    Only one signal is GREEN at a time, cycling sequentially: A -> B -> C -> D.

    When an external_controller and shared_state are provided (from the root
    TrafficSignalController), the visual signal state is read from the live
    controller rather than computed from static GST values.

    Orange preview behavior (NO extra time added):
      - The current approach runs GREEN for its full GST duration.
      - When the current GREEN has <= 5 s remaining, the NEXT approach
        shows ORANGE as a heads-up to drivers ("prepare to move").
        Vehicles still STOP on orange -- it is not a go signal.
      - When timer hits 0, the next approach immediately becomes GREEN.
      - The 5-second orange window is carved out of the current approach's
        existing GST, not added on top.
    """

    def __init__(self, gst_values, external_controller=None, shared_state=None):
        self.external = external_controller
        self.shared_state = shared_state
        self.direction_names = ROAD_ORDER
        self.road_order = ROAD_ORDER

        if self.external is not None:
            self.current_green = 0
            self.timer = external_controller.get_remaining_time()
            self.gst = gst_values
        else:
            self.gst = gst_values
            self.current_green = 0
            self.timer = self.gst[0] if self.gst else 15.0

    def update(self, dt):
        if self.external is not None:
            road = self.external.get_active_road()
            self.current_green = self.road_order.index(road)
            self.timer = self.external.get_remaining_time()
        else:
            self.timer -= dt
            if self.timer <= 0:
                self.current_green = (self.current_green + 1) % 4
                self.timer = self.gst[self.current_green]

    def get_green_direction(self):
        return self.current_green

    def get_phase_state(self):
        return 'GREEN'

    def get_signal_state(self, direction_index):
        if direction_index == self.current_green:
            return 'GREEN'

        next_green = (self.current_green + 1) % 4
        if direction_index == next_green and self.timer <= ORANGE_PREVIEW_SECONDS:
            return 'ORANGE'

        return 'RED'

    def is_green(self, direction_index):
        return direction_index == self.current_green