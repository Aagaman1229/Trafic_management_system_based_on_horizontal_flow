import csv
import os
from simulation.settings import DIRECTIONS, ROAD_NAMES


class ResultExporter:
    """Captures simulation data (signal transitions, GST, vehicle counts)
    and exports to CSVs after the simulation ends."""

    def __init__(self, traffic_controller, timeline, sample_interval=5.0):
        self.tc = traffic_controller
        self.timeline = timeline
        self.sample_interval = sample_interval
        self.sim_time = 0.0

        # Data stores
        self.signal_log = []
        self.gst_snapshots = []
        self.cycle_log = []

        # Hook into signal controller for switch events
        self.tc.on_switch(self._on_signal_switch)

        self._last_sample = -1000.0

    def _on_signal_switch(self, road, gst):
        timestamp = self.timeline.time if self.timeline else 0
        counts = self.timeline.get_counts(road) if self.timeline else {}

        entry = {
            'timestamp': f'{timestamp:.2f}',
            'road': road,
            'road_name': ROAD_NAMES.get(road, road),
            'gst': f'{gst:.2f}',
            'cycle': self.tc.get_cycle_count(),
            'active_road': self.tc.get_active_road(),
        }
        for vt in ['car', 'motorcycle', 'truck', 'bus', 'bicycle']:
            entry[f'{vt}_count'] = counts.get(vt, 0)
        self.signal_log.append(entry)

        self.cycle_log.append({
            'cycle': self.tc.get_cycle_count(),
            'road': road,
            'road_name': ROAD_NAMES.get(road, road),
            'gst': f'{gst:.2f}',
            'timestamp': f'{timestamp:.2f}',
        })

    def sample(self, sim_time):
        """Periodic sampling - call from the simulation run loop."""
        self.sim_time = sim_time
        if sim_time - self._last_sample < self.sample_interval:
            return
        self._last_sample = sim_time

        gst_vals = self.tc.get_gst_values()
        entry = {'timestamp': f'{sim_time:.2f}'}
        for i, r in enumerate(['A', 'B', 'C', 'D']):
            entry[f'gst_{r}'] = f'{gst_vals[i]:.2f}' if i < len(gst_vals) else 'N/A'
        self.gst_snapshots.append(entry)

    def export(self, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        files_written = []

        if self.signal_log:
            path = os.path.join(output_dir, 'signal_transitions.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(self.signal_log[0].keys()))
                w.writeheader()
                w.writerows(self.signal_log)
            files_written.append(path)

        if self.gst_snapshots:
            path = os.path.join(output_dir, 'gst_snapshots.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(self.gst_snapshots[0].keys()))
                w.writeheader()
                w.writerows(self.gst_snapshots)
            files_written.append(path)

        if self.cycle_log:
            path = os.path.join(output_dir, 'gst_per_cycle.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(self.cycle_log[0].keys()))
                w.writeheader()
                w.writerows(self.cycle_log)
            files_written.append(path)

        summary_path = os.path.join(output_dir, 'simulation_summary.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Metric', 'Value'])
            w.writerow(['total_simulation_time_s', f'{self.sim_time:.2f}'])
            w.writerow(['total_cycles', self.tc.get_cycle_count()])
            w.writerow(['total_signal_switches', len(self.signal_log)])
            for r in ['A', 'B', 'C', 'D']:
                w.writerow([f'final_gst_{r}_s', f'{self.tc.get_gst(r):.2f}'])
        files_written.append(summary_path)

        print("\n" + "=" * 60)
        print("  SIMULATION RESULTS EXPORTED")
        print("=" * 60)
        for p in files_written:
            size = os.path.getsize(p)
            print(f"  [OK] {p}  ({size:,} bytes)")
        print("=" * 60)
