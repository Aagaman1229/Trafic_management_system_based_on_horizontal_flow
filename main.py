import os
import sys
import csv
import json
import time
import threading
from datetime import datetime

from shared_state import SharedTrafficState
from video_manager import VideoManager
from signal_controller import TrafficSignalController
from video_display import start_display_thread

OUTPUT_DIR = "outputs"
FULL_CSV = os.path.join(OUTPUT_DIR, "full_traffic_data.csv")
GST_SUMMARY = os.path.join(OUTPUT_DIR, "gst_summary.txt")
GST_CACHE = os.path.join(OUTPUT_DIR, "gst_cache.json")
TRAFFIC_LOG = os.path.join(OUTPUT_DIR, "traffic_log.csv")

DIR_NAMES = ['A', 'B', 'C', 'D']

ROAD_FULL_NAMES = {
    'A': 'China Pull Road',
    'B': 'Airport Road',
    'C': 'Sabhagriha Chowk Road',
    'D': 'Naya Bazar Road',
}

ROAD_CONFIG_MAP = {
    'A': {"num_lanes": 6, "min_gst": 15, "max_gst": 44, "straight_ratio": 0.72},
    'B': {"num_lanes": 4, "min_gst": 15, "max_gst": 28, "straight_ratio": 0.72},
    'C': {"num_lanes": 4, "min_gst": 12, "max_gst": 44, "straight_ratio": 0.78},
    'D': {"num_lanes": 4, "min_gst": 20, "max_gst": 27, "straight_ratio": 0.6},
}

switch_queue = []


def write_full_csv(shared_state):
    with open(FULL_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "road", "road_name", "car", "truck", "bus",
            "motorcycle", "bicycle", "total_vehicles",
            "num_lanes", "gst_seconds", "active_signal"
        ])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for road in DIR_NAMES:
            counts = shared_state.get_counts(road)
            total = sum(counts.values())
            cfg = ROAD_CONFIG_MAP[road]
            gst = shared_state.get_gst(road)
            sig = shared_state.get_signal_state(road)
            rn = ROAD_FULL_NAMES.get(road, road)
            writer.writerow([
                ts, road, rn,
                counts.get('car', 0), counts.get('truck', 0),
                counts.get('bus', 0), counts.get('motorcycle', 0),
                counts.get('bicycle', 0),
                total, cfg["num_lanes"], round(gst, 2), sig
            ])


def write_gst_summary(shared_state):
    with open(GST_SUMMARY, 'w') as f:
        f.write("=== GST Summary Report — Prithivi Chowk, Pokhara ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for road in DIR_NAMES:
            counts = shared_state.get_counts(road)
            total = sum(counts.values())
            cfg = ROAD_CONFIG_MAP[road]
            gst = shared_state.get_gst(road)
            rn = ROAD_FULL_NAMES.get(road, f"Road {road}")
            f.write(f"{rn} ({road}):\n")
            f.write(f"  Lane count: {cfg['num_lanes']}\n")
            f.write(f"  GST range: [{cfg['min_gst']}, {cfg['max_gst']}] s\n")
            f.write(f"  Vehicle counts: {dict(counts)} (total={total})\n")
            f.write(f"  Latest GST: {gst:.2f} s\n\n")


def write_gst_cache(shared_state):
    gst_values = [shared_state.get_gst(r) for r in DIR_NAMES]
    counts_list = [shared_state.get_counts(r) for r in DIR_NAMES]
    data = {
        "gst_values": gst_values,
        "total_counts_list": counts_list
    }
    with open(GST_CACHE, 'w') as f:
        json.dump(data, f, indent=2)


def append_traffic_log(road, gst_val, ts_float, ts_str, shared_state):
    file_exists = os.path.exists(TRAFFIC_LOG)
    with open(TRAFFIC_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "direction", "green_start", "green_end",
                "gst_seconds", "car_count", "truck_count", "bus_count",
                "motorcycle_count", "bicycle_count"
            ])
        counts = shared_state.get_counts(road)
        writer.writerow([
            ts_float, road, ts_str, ts_str,
            round(gst_val, 2),
            counts.get('car', 0), counts.get('truck', 0),
            counts.get('bus', 0), counts.get('motorcycle', 0),
            counts.get('bicycle', 0)
        ])


def on_signal_switch(road, gst_val):
    global switch_queue
    now = time.time()
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    switch_queue.append((road, gst_val, now, ts_str))


def flush_outputs(shared_state):
    global switch_queue
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        write_full_csv(shared_state)
        write_gst_summary(shared_state)
        write_gst_cache(shared_state)
        for item in switch_queue:
            append_traffic_log(*item, shared_state)
        switch_queue.clear()
    except Exception as e:
        print(f"Output write error: {e}")


def main():
    print("=" * 60)
    print("  TRUE REAL-TIME MULTI-ROAD ADAPTIVE TRAFFIC CONTROLLER")
    print("  Prithivi Chowk, Pokhara — Highest Traffic Intersection")
    print("  All 4 videos run continuously in parallel.")
    print("  GST computed dynamically before each signal switch.")
    print("  Roads: China Pull | Airport | Sabhagriha Chowk | Naya Bazar")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Shared thread-safe state
    shared_state = SharedTrafficState()

    # 2. VideoManager — 4 parallel processing threads
    print("\n[1/5] Starting VideoManager (parallel YOLO inference on 4 videos)...")
    vm = VideoManager(shared_state)
    vm.start()

    # 3. TrafficSignalController — dynamic GST cycling A→B→C→D
    print("\n[2/5] Starting TrafficSignalController...")
    sc = TrafficSignalController(shared_state)
    sc.on_switch(lambda r, g: on_signal_switch(r, g))
    sc.start()

    # 4. Let some initial detections accumulate
    print("\n[3/5] Waiting for initial vehicle detections (3s)...")
    time.sleep(3)

    # 5. Launch verification display (2x2 grid showing live counting)
    print("\n[4/5] Opening verification window (2x2 grid with bounding boxes)...")
    start_display_thread(shared_state)

    # 6. Launch PyGame simulation
    print("\n[5/5] Starting PyGame simulation with LIVE data...")
    sim_path = os.path.join(os.path.dirname(__file__), "simulation")
    sys.path.insert(0, sim_path)

    import simulation as sim_mod
    from signals import SignalController as VisualSignalController

    # Read actual live GST from shared_state (already computed by TrafficSignalController)
    live_gst = [shared_state.get_gst(r) for r in DIR_NAMES]

    visual_sc = VisualSignalController(
        live_gst,
        external_controller=sc,
        shared_state=shared_state
    )

    sim = sim_mod.Simulation(
        detected_data=None,
        debug=False,
        gst_values=live_gst,
        shared_state=shared_state,
        signal_controller=visual_sc,
        traffic_controller=sc
    )

    # Output writer thread — flushes on signal switches
    out_lock = threading.Lock()
    out_running = True

    def output_loop():
        last_flush = 0
        while out_running:
            now = time.time()
            triggered = len(switch_queue) > 0
            if triggered and now - last_flush >= 1.0:
                with out_lock:
                    flush_outputs(shared_state)
                last_flush = now
            time.sleep(0.5)
        with out_lock:
            flush_outputs(shared_state)

    out_thread = threading.Thread(target=output_loop, daemon=True)
    out_thread.start()

    try:
        # Simulation blocks until window is closed
        sim.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        out_running = False
        sc.stop()
        vm.stop()
        time.sleep(1)
        with out_lock:
            flush_outputs(shared_state)
        print("\n=== Final outputs written. System halted. ===")


if __name__ == "__main__":
    main()
