import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import Simulation, SignalController
from timeline_replay import TimelineReplay

TIMELINE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "outputs", "traffic_timeline.json")
DIR_NAMES = ["A", "B", "C", "D"]


def main():
    timeline = None
    gst_values = None
    traffic_controller = None

    if os.path.exists(TIMELINE_PATH):
        print(f"Loading timeline from {TIMELINE_PATH}")
        timeline = TimelineReplay(TIMELINE_PATH)
        from signal_controller import TrafficSignalController
        traffic_controller = TrafficSignalController(timeline)
        gst_values = [traffic_controller.get_gst(r) for r in DIR_NAMES]
        print(f"Timeline loaded: {timeline.get_duration():.1f}s of data")
        print(f"GST values: {[f'{g:.2f}s' for g in gst_values]}")
    else:
        print("No timeline found. Using fallback example data.")
        timeline_data = {
            "A": {"car": 4, "motorcycle": 5},
            "B": {"car": 6, "motorcycle": 10, "truck": 2},
            "C": {"bus": 3, "car": 5},
            "D": {"car": 2, "truck": 1, "motorcycle": 7}
        }

        class FakeTimeline:
            def __init__(self, data):
                self.data = data
                self.roads = list(data.keys())
            def get_counts(self, road):
                return self.data.get(road, {})
            def get_all_counts(self):
                return self.data
            def advance(self, dt):
                pass
            @property
            def time(self): return 0.0
            @time.setter
            def time(self, v): pass
            def get_duration(self): return 60.0

        timeline = FakeTimeline(timeline_data)
        gst_values = [15.0, 12.0, 20.0, 18.0]

    visual_sc = SignalController(traffic_controller if traffic_controller else gst_values)

    sim = Simulation(
        timeline_replay=timeline,
        gst_values=gst_values,
        debug=False,
        signal_controller=visual_sc,
        traffic_controller=traffic_controller
    )
    sim.run()


if __name__ == "__main__":
    main()
