import threading
import cv2
import numpy as np
from shared_state import SharedTrafficState

ROAD_ORDER = ['A', 'B', 'C', 'D']
ROAD_FULL_NAMES = {
    'A': 'China Pull Road',
    'B': 'Airport Road',
    'C': 'Sabhagriha Chowk Road',
    'D': 'Naya Bazar Road',
}
GRID_W, GRID_H = 2, 2
DISPLAY_W, DISPLAY_H = 480, 360
CANVAS_W = DISPLAY_W * GRID_W
CANVAS_H = DISPLAY_H * GRID_H
WIN_NAME = "Live Vehicle Counting Verification (2x2 Grid) — Prithivi Chowk, Pokhara"
FPS_TARGET = 20


def run_display_loop(shared_state):
    """
    Opens a single OpenCV window showing a 2x2 grid of the four road feeds.
    Each quadrant shows the annotated video (bounding boxes, count lines,
    vehicle counts) with the real road name overlay.
    Press 'q' to close the window.
    """
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, CANVAS_W, CANVAS_H)

    while shared_state.is_running():
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        for i, road in enumerate(ROAD_ORDER):
            frame = shared_state.get_frame(road)
            if frame is None:
                placeholder = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                rn = ROAD_FULL_NAMES.get(road, f"Road {road}")
                cv2.putText(placeholder, f"{rn} [waiting...]",
                            (30, DISPLAY_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (100, 100, 100), 2)
                frame = placeholder

            row = i // GRID_W
            col = i % GRID_W
            y_start = row * DISPLAY_H
            x_start = col * DISPLAY_W
            canvas[y_start:y_start + DISPLAY_H, x_start:x_start + DISPLAY_W] = frame

            # Overlay road name label at the bottom of each quadrant
            rn = ROAD_FULL_NAMES.get(road, f"Road {road}")
            label = f"  {rn}  "
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx = x_start + 10
            ly = y_start + DISPLAY_H - 15
            cv2.rectangle(canvas, (lx - 5, ly - th - 8), (lx + tw + 5, ly + 5), (30, 30, 30, 180), -1)
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add a thin grid divider
        cv2.line(canvas, (DISPLAY_W, 0), (DISPLAY_W, CANVAS_H), (50, 50, 50), 2)
        cv2.line(canvas, (0, DISPLAY_H), (CANVAS_W, DISPLAY_H), (50, 50, 50), 2)

        cv2.imshow(WIN_NAME, canvas)
        key = cv2.waitKey(1000 // FPS_TARGET) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(WIN_NAME)


def start_display_thread(shared_state):
    """
    Launches the verification window in a daemon thread so it runs
    alongside the PyGame simulation and video processing.
    """
    t = threading.Thread(target=run_display_loop, args=(shared_state,), daemon=True)
    t.start()
    return t
