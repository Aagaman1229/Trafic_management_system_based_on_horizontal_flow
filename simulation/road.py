import pygame
from settings import *

def _inbound_lanes(d):
    return APPROACH_LANES.get(DIRECTIONS[d], 2)

def _outbound_lanes(d):
    return _inbound_lanes((d + 2) % 4)

def draw_road(surface, signal_controller):
    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    isect_half = INTERSECTION_SIZE // 2

    n_in = [_inbound_lanes(d) for d in range(4)]
    vert_half = max(n_in[0], n_in[2]) * LANE_WIDTH
    horiz_half = max(n_in[1], n_in[3]) * LANE_WIDTH

    # 1. Asphalt surfaces
    pygame.draw.rect(surface, COLOR_ROAD, (cx - vert_half, 0, vert_half * 2, SCREEN_HEIGHT))
    pygame.draw.rect(surface, COLOR_ROAD, (0, cy - horiz_half, SCREEN_WIDTH, horiz_half * 2))

    # 2. Intersection box
    pygame.draw.rect(surface, COLOR_ISLAND, (cx - isect_half, cy - isect_half, INTERSECTION_SIZE, INTERSECTION_SIZE))
    pygame.draw.rect(surface, COLOR_ROAD, (cx - isect_half, cy - isect_half, INTERSECTION_SIZE, INTERSECTION_SIZE), 3)

    # 3. Center lines (yellow dashed)
    dash_len = 15
    dash_gap = 15
    ccl = (234, 179, 8)
    for start_y in range(0, cy - isect_half, dash_len + dash_gap):
        pygame.draw.line(surface, ccl, (cx, start_y), (cx, min(cy - isect_half, start_y + dash_len)), 3)
    for start_y in range(cy + isect_half, SCREEN_HEIGHT, dash_len + dash_gap):
        pygame.draw.line(surface, ccl, (cx, start_y), (cx, min(SCREEN_HEIGHT, start_y + dash_len)), 3)
    for start_x in range(0, cx - isect_half, dash_len + dash_gap):
        pygame.draw.line(surface, ccl, (start_x, cy), (min(cx - isect_half, start_x + dash_len), cy), 3)
    for start_x in range(cx + isect_half, SCREEN_WIDTH, dash_len + dash_gap):
        pygame.draw.line(surface, ccl, (start_x, cy), (min(SCREEN_WIDTH, start_x + dash_len), cy), 3)

    # 4. Lane divider markings per arm
    # Helper: draw dashed line segment
    def _dash_horiz(y, x1, x2):
        if x1 >= x2:
            x1, x2 = x2, x1
        for sx in range(int(x1), int(x2) - dash_len, dash_len + dash_gap):
            ex = min(int(x2), sx + dash_len)
            pygame.draw.line(surface, COLOR_MARKING, (sx, y), (ex, y), 1)

    def _dash_vert(x, y1, y2):
        if y1 >= y2:
            y1, y2 = y2, y1
        for sy in range(int(y1), int(y2) - dash_len, dash_len + dash_gap):
            ey = min(int(y2), sy + dash_len)
            pygame.draw.line(surface, COLOR_MARKING, (x, sy), (x, ey), 1)

    # North arm (y: 0 to cy - isect_half)
    n_y1, n_y2 = 0, cy - isect_half
    for i in range(1, n_in[0]):       # Southbound (east side, x > cx)
        _dash_vert(cx + i * LANE_WIDTH, n_y1, n_y2)
    for i in range(1, _outbound_lanes(0)):  # Northbound (west side, x < cx)
        _dash_vert(cx - i * LANE_WIDTH, n_y1, n_y2)

    # South arm (y: cy + isect_half to SCREEN_HEIGHT)
    s_y1, s_y2 = cy + isect_half, SCREEN_HEIGHT
    for i in range(1, _outbound_lanes(2)):  # Southbound (east side)
        _dash_vert(cx + i * LANE_WIDTH, s_y1, s_y2)
    for i in range(1, n_in[2]):       # Northbound (west side)
        _dash_vert(cx - i * LANE_WIDTH, s_y1, s_y2)

    # West arm (x: 0 to cx - isect_half)
    w_x1, w_x2 = 0, cx - isect_half
    for i in range(1, n_in[3]):       # Eastbound (north side, y < cy)
        _dash_horiz(cy - i * LANE_WIDTH, w_x1, w_x2)
    for i in range(1, _outbound_lanes(3)):  # Westbound (south side, y > cy)
        _dash_horiz(cy + i * LANE_WIDTH, w_x1, w_x2)

    # East arm (x: cx + isect_half to SCREEN_WIDTH)
    e_x1, e_x2 = cx + isect_half, SCREEN_WIDTH
    for i in range(1, _outbound_lanes(1)):  # Eastbound (north side)
        _dash_horiz(cy - i * LANE_WIDTH, e_x1, e_x2)
    for i in range(1, n_in[1]):       # Westbound (south side)
        _dash_horiz(cy + i * LANE_WIDTH, e_x1, e_x2)

    # 5. Zebra crossings
    stripe_w = 6
    stripe_h = 20
    stripe_spacing = 12

    # North (inbound on east side: cx to cx + n_in[0]*LANE_WIDTH)
    z_n = int(n_in[0] * LANE_WIDTH / stripe_spacing)
    for i in range(z_n):
        x = int(cx + i * stripe_spacing + 3)
        pygame.draw.rect(surface, COLOR_ZEBRA, (x, cy - isect_half - 20, stripe_w, stripe_h))

    # South (inbound on west side: cx - n_in[2]*LANE_WIDTH to cx)
    z_s = int(n_in[2] * LANE_WIDTH / stripe_spacing)
    for i in range(z_s):
        x = int(cx - n_in[2] * LANE_WIDTH + i * stripe_spacing + 3)
        pygame.draw.rect(surface, COLOR_ZEBRA, (x, cy + isect_half, stripe_w, stripe_h))

    # East (inbound on south side: cy to cy + n_in[1]*LANE_WIDTH)
    z_e = int(n_in[1] * LANE_WIDTH / stripe_spacing)
    for i in range(z_e):
        y = int(cy + i * stripe_spacing + 3)
        pygame.draw.rect(surface, COLOR_ZEBRA, (cx + isect_half, y, stripe_h, stripe_w))

    # West (inbound on north side: cy - n_in[3]*LANE_WIDTH to cy)
    z_w = int(n_in[3] * LANE_WIDTH / stripe_spacing)
    for i in range(z_w):
        y = int(cy - n_in[3] * LANE_WIDTH + i * stripe_spacing + 3)
        pygame.draw.rect(surface, COLOR_ZEBRA, (cx - isect_half - 20, y, stripe_h, stripe_w))

    # 6. Stop lines (span all waiting lanes = all inbound except leftmost free-turn lane)
    stop_w = 4
    if n_in[0] > 1:  # North: east side, from cx + LANE_WIDTH to cx + n_in[0]*LANE_WIDTH
        x1 = cx + LANE_WIDTH
        x2 = cx + n_in[0] * LANE_WIDTH
        pygame.draw.rect(surface, COLOR_STOP_LINE, (x1, cy - isect_half - 25, x2 - x1, stop_w))
    if n_in[2] > 1:  # South: west side, from cx - n_in[2]*LANE_WIDTH to cx - LANE_WIDTH
        x1 = cx - n_in[2] * LANE_WIDTH
        x2 = cx - LANE_WIDTH
        pygame.draw.rect(surface, COLOR_STOP_LINE, (x1, cy + isect_half + 21, x2 - x1, stop_w))
    if n_in[1] > 1:  # East: south side, from cy + LANE_WIDTH to cy + n_in[1]*LANE_WIDTH
        y1 = cy + LANE_WIDTH
        y2 = cy + n_in[1] * LANE_WIDTH
        pygame.draw.rect(surface, COLOR_STOP_LINE, (cx + isect_half + 21, y1, stop_w, y2 - y1))
    if n_in[3] > 1:  # West: north side, from cy - n_in[3]*LANE_WIDTH to cy - LANE_WIDTH
        y1 = cy - n_in[3] * LANE_WIDTH
        y2 = cy - LANE_WIDTH
        pygame.draw.rect(surface, COLOR_STOP_LINE, (cx - isect_half - 25, y1, stop_w, y2 - y1))

    # 7. Traffic lights
    def _light_color(state):
        if state == 'GREEN':
            return COLOR_GREEN
        elif state == 'ORANGE':
            return COLOR_YELLOW
        return COLOR_RED

    margin = 20
    lr = isect_half + margin

    for d in range(4):
        state = signal_controller.get_signal_state(d)
        color = _light_color(state)

        if d == 0:
            pos = (cx + lr, cy - lr)
        elif d == 1:
            pos = (cx + lr, cy + lr)
        elif d == 2:
            pos = (cx - lr, cy + lr)
        else:
            pos = (cx - lr, cy - lr)

        pygame.draw.circle(surface, (0, 0, 0), pos, 12)
        pygame.draw.circle(surface, color, pos, 8)
