def calculate_gst(vehicle_counts, num_lanes=1):
    """
    Calculate Green Signal Time (GST) based on vehicle counts.

    Formula:
        gst = (2/3) * ( Σ (vehicle_count_i * crossing_time_i) ) / (num_lanes + 1)

    crossing_time_i (seconds) for each vehicle type:
        - car:        4
        - truck:      6
        - bus:        6
        - motorcycle: 2
        - bicycle:    3

    GST is clamped between 15 and 60 seconds.

    Args:
        vehicle_counts (dict): {'car': int, 'truck': int, 'bus': int, 'motorcycle': int, 'bicycle': int}
        num_lanes (int): number of lanes (default 1)

    Returns:
        float: GST in seconds (clamped)
    """
    CROSSING_TIMES = {
        'car': 4,
        'truck': 6,
        'bus': 6,
        'motorcycle': 2,
        'bicycle': 3
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