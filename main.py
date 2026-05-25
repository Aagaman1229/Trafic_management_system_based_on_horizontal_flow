import os
import sys

from signal_controller import TrafficSignalController
from timeline_replay import TimelineReplay
from result_exporter import ResultExporter
from simulation import Simulation, SignalController

TIMELINE_PATH = os.path.join("outputs", "traffic_timeline.json")

DIR_NAMES = ['A', 'B', 'C', 'D']


def main():
    print("=" * 60)
    print("  REAL-TIME TRAFFIC SIMULATION REPLAY")
    print("  Prithivi Chowk, Pokhara - Highest Traffic Intersection")
    print("  Reads pre-extracted timeline -> runs at true 60 FPS")
    print("  Roads: China Pull | Airport | Sabhagriha Chowk | Naya Bazar")
    print("=" * 60)

    if not os.path.exists(TIMELINE_PATH):
        print(f"\nERROR: Timeline file not found at {TIMELINE_PATH}")
        print("Run 'python extract_timeline.py' first to process the videos.")
        sys.exit(1)

    print("\n[1/3] Loading pre-extracted traffic timeline...")
    timeline = TimelineReplay(TIMELINE_PATH)
    print(f"  Timeline loaded: {timeline.get_duration():.1f}s of data across {len(timeline.roads)} roads")

    print("\n[2/3] Initializing TrafficSignalController...")
    sc = TrafficSignalController(timeline)

    gst_values = [sc.get_gst(r) for r in DIR_NAMES]
    print(f"  Initial GST values: {[f'{g:.2f}s' for g in gst_values]}")

    visual_sc = SignalController(sc)

    print("\n[3/3] Starting PyGame simulation (real-time replay)...")
    exporter = ResultExporter(sc, timeline, sample_interval=5.0)
    sim = Simulation(
        timeline_replay=timeline,
        gst_values=gst_values,
        signal_controller=visual_sc,
        traffic_controller=sc,
        data_callback=lambda t: exporter.sample(t)
    )

    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("\n=== Simulation ended. ===")
        exporter.export()


if __name__ == "__main__":
    main()
