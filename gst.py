def calculate_gst(vehicle_counts, num_lanes=1):
    """
    Calculate Green Signal Time (GST) based on vehicle counts.
    Clamped between 15 and 60 seconds.
    
    vehicle_counts: dict with keys 'car_truck', 'bus', 'motorcycle', 'bicycle'
    """
    CROSSING_TIMES = {
        'car_truck': 5.0,    # average of car(4) and truck(6)
        'bus': 6.0,
        'motorcycle': 2.0,
        'bicycle': 3.0
    }

    total_weighted_time = 0
    for vtype, count in vehicle_counts.items():
        if vtype in CROSSING_TIMES:
            total_weighted_time += count * CROSSING_TIMES[vtype]

    if num_lanes + 1 == 0:
        return 15.0

    gst = (2 / 3) * total_weighted_time / (num_lanes + 1)
    gst = max(15.0, min(60.0, gst))
    return gst