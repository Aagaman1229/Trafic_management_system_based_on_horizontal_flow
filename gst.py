CROSSING_TIMES = {
    'motorcycle': 4,
    'car': 5,
    'bus': 8,
    'truck': 9,
    'bicycle': 12
}

def calculate_gst(vehicle_counts, num_lanes=1, min_gst=15.0, max_gst=60.0, straight_ratio=0.75):
    total_weighted_time = 0
    for vtype, count in vehicle_counts.items():
        if vtype in CROSSING_TIMES:
            total_weighted_time += count * CROSSING_TIMES[vtype]

    if num_lanes + 1 == 0:
        return min_gst

    gst = straight_ratio * total_weighted_time / (num_lanes + 1)
    gst = max(min_gst, min(max_gst, gst))
    return gst