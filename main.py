import cv2
import os
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from gst import calculate_gst

CLASS_TO_TYPE = {
    2: 'car',
    5: 'bus',
    7: 'truck',
    3: 'motorcycle',
    1: 'bicycle'
}

TYPE_COLORS = {
    'car': (0, 255, 0),
    'truck': (0, 255, 255),
    'bus': (255, 0, 0),
    'motorcycle': (255, 0, 255),
    'bicycle': (0, 165, 255)
}

def main():
    video_path = "video1.mov"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "tracked_video.mp4")
    output_gst = os.path.join(output_dir, "gst_result.txt")

    print("Loading YOLOv8 nano (optimised for speed)...")
    model = YOLO('yolov8n.pt')
    model.conf = 0.35
    model.classes = [1,2,3,5,7]

    # ❌ DO NOT use model.half() – it crashes on some GPUs during layer fusion

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {width}×{height} @ {fps:.1f} FPS")

    # ---- ROI (adjust to your lane – keep the original) ----
    roi_polygon = np.array([
        [0, height],
        [0, int(height * 0.4)],
        [int(width * 0.6), int(height * 0.4)],
        [int(width * 0.6), height]
    ], np.int32)

    # ---- Display window: downscale huge frames (max 1280 pixels on largest side) ----
    MAX_DISPLAY = 1280
    scale = min(MAX_DISPLAY / width, MAX_DISPLAY / height, 1.0)
    display_w = int(width * scale)
    display_h = int(height * scale)
    WIN_NAME = "Left-moving vehicle tracker"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, display_w, display_h)

    counting_line_x = int(width * 0.3)

    # ---- Tracker settings ----
    tracker = CentroidTracker(
        max_disappeared=25,
        max_distance=90,
        line_x=0.3,
        roi_polygon=roi_polygon,
        direction_frames=10,
        min_movement=30
    )
    tracker.set_frame_width(width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # ---- Speed settings ----
    INFERENCE_SIZE = 256   # lower = faster; try 320 if you miss small vehicles
    SKIP_FRAMES = 0        # set to 1 if still laggy (detection every 2nd frame)

    frame_counter = 0
    print("Processing... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        run_detection = (SKIP_FRAMES == 0) or (frame_counter % (SKIP_FRAMES + 1) == 1)

        # ------------------- DETECTION -------------------
        if run_detection:
            results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)[0]
            detections = []
            for det in results.boxes:
                cls_id = int(det.cls[0])
                if cls_id not in CLASS_TO_TYPE:
                    continue
                vtype = CLASS_TO_TYPE[cls_id]
                xyxy = det.xyxy[0].cpu().numpy()
                c_x = (xyxy[0] + xyxy[2]) / 2
                c_y = (xyxy[1] + xyxy[3]) / 2
                detections.append((c_x, c_y, vtype, xyxy))
        else:
            detections = []

        # ------------------- TRACKING -------------------
        tracks = tracker.update(detections)

        # ------------------- DRAWING -------------------
        # Draw on a copy of the original frame (for saving)
        draw_frame = frame.copy()

        cv2.polylines(draw_frame, [roi_polygon], True, (0, 255, 0), 2)
        cv2.line(draw_frame, (counting_line_x, 0), (counting_line_x, height), (0, 0, 255), 3)
        cv2.putText(draw_frame, "COUNT LINE", (counting_line_x+5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        for track in tracks:
            track_id, c_x, c_y, vtype, bbox, moving_left = track
            color = TYPE_COLORS.get(vtype, (255,255,255))
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
                label = f"ID:{track_id} {vtype}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(draw_frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                cv2.putText(draw_frame, label, (x1+5, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            else:
                cv2.circle(draw_frame, (int(c_x), int(c_y)), 5, color, -1)

        # Live counts + GST (position adjusted for large frame)
        y0 = int(height * 0.05)  # 5% from top
        cv2.putText(draw_frame, "Left-moving counts:", (20, y0-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        for vtype, count in tracker.vehicle_counts.items():
            color = TYPE_COLORS.get(vtype, (255,255,255))
            cv2.putText(draw_frame, f"{vtype}: {count}", (30, y0+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            y0 += 40

        live_gst = calculate_gst(tracker.vehicle_counts, num_lanes=1)
        cv2.putText(draw_frame, f"GST: {live_gst:.1f} s", (20, y0+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 3)

        # ------------------- DISPLAY (downscaled) -------------------
        display_frame = cv2.resize(draw_frame, (display_w, display_h))
        cv2.imshow(WIN_NAME, display_frame)

        # ------------------- SAVE (original resolution) -------------------
        out.write(draw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    counts = tracker.vehicle_counts
    final_gst = calculate_gst(counts, num_lanes=1)

    with open(output_gst, 'w') as f:
        f.write("Final Vehicle Counts (left-moving only):\n")
        for vtype, count in counts.items():
            f.write(f"  {vtype}: {count}\n")
        f.write(f"\nGreen Signal Time (GST): {final_gst:.2f} seconds\n")

    print("\n=== Processing finished ===")
    print("Counts:", counts)
    print(f"Green Signal Time: {final_gst:.2f} sec")
    print(f"Output saved in '{output_dir}/'")

if __name__ == "__main__":
    main()