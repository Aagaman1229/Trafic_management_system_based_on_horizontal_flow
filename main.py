import cv2
import os
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from gst import calculate_gst

CLASS_TO_TYPE = {
    2: 'car',
    7: 'truck',
    5: 'bus',
    3: 'motorcycle',
    1: 'bicycle'
}

TYPE_COLORS = {
    'car':        (0,   255,   0),
    'truck':      (0,   255, 255),
    'bus':        (255,   0,   0),
    'motorcycle': (255,   0, 255),
    'bicycle':    (0,   165, 255)
}


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def merge_all_detections(detections, iou_threshold=0.4):
    """Merge overlapping boxes across all classes — prevents double-detection."""
    items = [{'c_x': d[0], 'c_y': d[1], 'vtype': d[2], 'bbox': d[3]}
             for d in detections]
    items.sort(
        key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]),
        reverse=True
    )
    merged = []
    while items:
        best = items.pop(0)
        to_merge = [best]
        i = 0
        while i < len(items):
            if compute_iou(best['bbox'], items[i]['bbox']) > iou_threshold:
                to_merge.append(items.pop(i))
            else:
                i += 1
        if len(to_merge) > 1:
            all_boxes = [m['bbox'] for m in to_merge]
            x1 = min(b[0] for b in all_boxes)
            y1 = min(b[1] for b in all_boxes)
            x2 = max(b[2] for b in all_boxes)
            y2 = max(b[3] for b in all_boxes)
            merged.append(((x1+x2)/2, (y1+y2)/2,
                           best['vtype'],
                           np.array([x1, y1, x2, y2])))
        else:
            merged.append((best['c_x'], best['c_y'],
                           best['vtype'], best['bbox']))
    return merged


def main():
    video_path   = "video1.MOV"
    output_dir   = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "tracked_video.mp4")
    output_gst   = os.path.join(output_dir, "gst_result.txt")

    print("Loading YOLOv8s...")
    model = YOLO('yolov8s.pt')
    model.conf    = 0.35
    model.iou     = 0.45
    model.classes = [1, 2, 3, 5, 7]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")

    # ── YOUR EXACT LINE COORDINATES ─────────────────────────────────────
    x_pt      = (int(height / 3.5), int(width / 2.5))   # meeting point
    line1_pt1 = (0, 0)                                   # top-left corner
    line1_pt2 = x_pt
    line2_pt1 = (width, height)                          # bottom-right corner
    line2_pt2 = x_pt
    # ────────────────────────────────────────────────────────────────────

    tracker = CentroidTracker(
        max_disappeared=40,
        max_distance=120,
        line1_pt1=line1_pt1, line1_pt2=line1_pt2,
        line2_pt1=line2_pt1, line2_pt2=line2_pt2,
        min_movement=40,
        history_len=20
    )

    MAX_DISPLAY = 1280
    scale = min(MAX_DISPLAY / width, MAX_DISPLAY / height, 1.0)
    display_w, display_h = int(width * scale), int(height * scale)
    WIN_NAME = "Vehicle Counter"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, display_w, display_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    INFERENCE_SIZE = 640
    print("Processing... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)[0]

        raw_detections = []
        for det in results.boxes:
            cls_id = int(det.cls[0])
            if cls_id not in CLASS_TO_TYPE:
                continue
            vtype = CLASS_TO_TYPE[cls_id]
            xyxy  = det.xyxy[0].cpu().numpy()
            c_x   = (xyxy[0] + xyxy[2]) / 2
            c_y   = (xyxy[1] + xyxy[3]) / 2
            raw_detections.append((c_x, c_y, vtype, xyxy))

        detections = merge_all_detections(raw_detections, iou_threshold=0.4)
        tracks     = tracker.update(detections)

        # ── DRAWING ─────────────────────────────────────────────────────
        draw_frame = frame.copy()

        cv2.line(draw_frame, line1_pt1, line1_pt2, (0, 0, 255), 3)
        cv2.line(draw_frame, line2_pt1, line2_pt2, (0, 0, 255), 3)
        cv2.circle(draw_frame, x_pt, 8, (0, 0, 255), -1)
        cv2.putText(draw_frame, "COUNT LINE",
                    (x_pt[0] + 10, x_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for track in tracks:
            track_id, c_x, c_y, vtype, bbox, _ = track
            color = TYPE_COLORS.get(vtype, (255, 255, 255))

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
                label = f"ID:{track_id} {vtype}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(draw_frame,
                              (x1, y1 - th - 10), (x1 + tw + 10, y1),
                              color, -1)
                cv2.putText(draw_frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                cv2.circle(draw_frame, (int(c_x), int(c_y)), 6, color, -1)

            trail = tracker.get_trail(track_id)
            if len(trail) > 1:
                pts = np.array(trail, np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw_frame, [pts], False, color, 2)

        y0 = int(height * 0.05)
        cv2.putText(draw_frame, "Counts:", (20, y0 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        for vtype, count in tracker.vehicle_counts.items():
            color = TYPE_COLORS.get(vtype, (255, 255, 255))
            cv2.putText(draw_frame, f"{vtype}: {count}", (30, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            y0 += 40

        live_gst = calculate_gst(tracker.vehicle_counts, num_lanes=1)
        cv2.putText(draw_frame, f"GST: {live_gst:.1f} s", (20, y0 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

        display_frame = cv2.resize(draw_frame, (display_w, display_h))
        cv2.imshow(WIN_NAME, display_frame)
        out.write(draw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    counts    = tracker.vehicle_counts
    final_gst = calculate_gst(counts, num_lanes=1)

    with open(output_gst, 'w') as f:
        f.write("Final Vehicle Counts:\n")
        for vtype, count in counts.items():
            f.write(f"  {vtype}: {count}\n")
        f.write(f"\nGreen Signal Time (GST): {final_gst:.2f} seconds\n")

    print("\n=== Processing finished ===")
    print("Counts:", counts)
    print(f"Green Signal Time: {final_gst:.2f} sec")
    print(f"Output saved in '{output_dir}/'")


if __name__ == "__main__":
    main()