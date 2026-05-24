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
DISPLAY_W, DISPLAY_H = 800, 600
CANVAS_W = DISPLAY_W * GRID_W
CANVAS_H = DISPLAY_H * GRID_H
WIN_NAME = "Live Vehicle Counting Verification (2x2 Grid) — Prithivi Chowk, Pokhara"
FPS_TARGET = 20


def run_display_loop(shared_state):
    """
    Opens a single OpenCV window showing a 2x2 grid of the four road feeds.
    Each quadrant shows the annotated video (bounding boxes, count lines,
    vehicle counts) with the real road name overlay.
    Press 'q' to close the window. Press 'f' to toggle fullscreen.
    """
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    # Initial size that fits most screens; content scales to fill window
    cv2.resizeWindow(WIN_NAME, min(CANVAS_W, 1280), min(CANVAS_H, 960))

    fullscreen = False

    while shared_state.is_running():
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        for i, road in enumerate(ROAD_ORDER):
            frame = shared_state.get_frame(road)
            if frame is None:
                placeholder = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                rn = ROAD_FULL_NAMES.get(road, f"Road {road}")
                cv2.putText(placeholder, f"{rn} [waiting...]",
                            (30, DISPLAY_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (100, 100, 100), 3)
                frame = placeholder

            row = i // GRID_W
            col = i % GRID_W
            y_start = row * DISPLAY_H
            x_start = col * DISPLAY_W
            canvas[y_start:y_start + DISPLAY_H, x_start:x_start + DISPLAY_W] = frame

            # Overlay road name label at the bottom of each quadrant
            rn = ROAD_FULL_NAMES.get(road, f"Road {road}")
            label = f"  {rn}  "
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            lx = x_start + 10
            ly = y_start + DISPLAY_H - 20
            cv2.rectangle(canvas, (lx - 5, ly - th - 10), (lx + tw + 10, ly + 8), (30, 30, 30, 180), -1)
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Add grid divider
        cv2.line(canvas, (DISPLAY_W, 0), (DISPLAY_W, CANVAS_H), (50, 50, 50), 3)
        cv2.line(canvas, (0, DISPLAY_H), (CANVAS_W, DISPLAY_H), (50, 50, 50), 3)

        # Scale canvas to fit the current window size (maintain aspect ratio)
        try:
            rect = cv2.getWindowImageRect(WIN_NAME)
            if rect is not None and rect[2] > 0 and rect[3] > 0:
                win_w, win_h = rect[2], rect[3]
            else:
                win_w, win_h = CANVAS_W, CANVAS_H
        except Exception:
            win_w, win_h = CANVAS_W, CANVAS_H

        if win_w != CANVAS_W or win_h != CANVAS_H:
            scale = min(win_w / CANVAS_W, win_h / CANVAS_H)
            new_w = int(CANVAS_W * scale)
            new_h = int(CANVAS_H * scale)
            scaled = cv2.resize(canvas, (new_w, new_h))
            display = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            y_off = max(0, (win_h - new_h) // 2)
            x_off = max(0, (win_w - new_w) // 2)
            display[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
            cv2.imshow(WIN_NAME, display)
        else:
            cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1000 // FPS_TARGET) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cv2.destroyWindow(WIN_NAME)


def start_display_thread(shared_state):
    """
    Launches the verification window in a daemon thread so it runs
    alongside the PyGame simulation and video processing.
    """
    t = threading.Thread(target=run_display_loop, args=(shared_state,), daemon=True)
    t.start()
    return t
