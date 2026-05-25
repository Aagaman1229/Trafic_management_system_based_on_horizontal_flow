import pygame
import sys
import random
import math
from .settings import *
from .vehicle import Vehicle
from .road import draw_road
from .signals import SignalController
from .hud import draw_hud


class Simulation:
    """
    Main Simulation Orchestrator for the 6-lane 2-way intersection.

    Reads pre-extracted vehicle counts from a TimelineReplay that
    advances in real time (1 sim second = 1 wall-clock second).
    Vehicles spawn dynamically to match the recorded counts.

    The signal controller is driven by update(dt) from this loop,
    and computes GST based on the cumulative counts in the timeline.
    """

    def __init__(self, timeline_replay, gst_values=None, debug=False,
                 signal_controller=None, traffic_controller=None):
        pygame.init()
        self.debug = debug
        self.clock = pygame.time.Clock()
        self.running = True

        self.timeline = timeline_replay
        self.traffic_controller = traffic_controller

        if gst_values is not None:
            self.gst_values = gst_values
        else:
            self.gst_values = [15.0, 12.0, 20.0, 15.0]

        if signal_controller is not None:
            self.signal_controller = signal_controller
        else:
            self.signal_controller = SignalController(traffic_controller)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Traffic Management System Simulation (6-Lane Nepal-Style)")

        self.vehicles = []
        self.spawn_timers = [0.0, 0.0, 0.0, 0.0]

        self.spawned_counts = {
            r: {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0, 'bicycle': 0}
            for r in DIRECTIONS
        }
        self.spawn_queues = {r: [] for r in DIRECTIONS}

        self.class_spawn_counts = [
            {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0}
            for _ in range(4)
        ]

        for d in range(4):
            self.spawn_timers[d] = (d + 1) * 0.5

    def _get_counts(self, direction_char):
        return self.timeline.get_counts(direction_char)

    def _get_curve(self, cx_arc, cy_arc, radius, start_angle_deg, end_angle_deg, num_pts=6):
        points = []
        for i in range(num_pts):
            t = i / (num_pts - 1)
            angle = math.radians(start_angle_deg + t * (end_angle_deg - start_angle_deg))
            px = cx_arc + radius * math.cos(angle)
            py = cy_arc + radius * math.sin(angle)
            points.append((int(px), int(py)))
        return points

    def _generate_path(self, direction, movement, lane_off=None):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        isect = max_lane_offset(direction)

        if lane_off is not None:
            if movement == 'left':
                l_off = lane_off
                s_off = get_lane_offset(direction, 'straight')
                r_off = get_lane_offset(direction, 'right')
            elif movement == 'straight':
                s_off = lane_off
                l_off = get_lane_offset(direction, 'left')
                r_off = get_lane_offset(direction, 'right')
            else:
                r_off = lane_off
                l_off = get_lane_offset(direction, 'left')
                s_off = get_lane_offset(direction, 'straight')
        else:
            l_off = get_lane_offset(direction, 'left')
            s_off = get_lane_offset(direction, 'straight')
            r_off = get_lane_offset(direction, 'right')

        path = []

        if direction == 0:
            if movement == 'left':
                arc_r = isect - l_off
                path.append((cx + l_off, -100))
                path.append((cx + l_off, cy - isect))
                path.extend(self._get_curve(cx + isect, cy - isect, arc_r, 180, 90))
                path.append((1800, cy - l_off))
            elif movement == 'straight':
                path.append((cx + s_off, -100))
                path.append((cx + s_off, 1100))
            else:
                path.append((cx + r_off, -100))
                path.append((cx + r_off, cy - r_off))
                path.extend(self._get_curve(cx - r_off, cy - r_off, 2 * r_off, 0, 90))
                path.append((-200, cy + r_off))

        elif direction == 1:
            if movement == 'left':
                arc_r = isect - l_off
                path.append((1800, cy + l_off))
                path.append((cx + isect, cy + l_off))
                path.extend(self._get_curve(cx + isect, cy + isect, arc_r, 270, 180))
                path.append((cx + l_off, 1100))
            elif movement == 'straight':
                path.append((1800, cy + s_off))
                path.append((-200, cy + s_off))
            else:
                path.append((1800, cy + r_off))
                path.append((cx + r_off, cy + r_off))
                path.extend(self._get_curve(cx + r_off, cy - r_off, 2 * r_off, 90, 180))
                path.append((cx - r_off, -100))

        elif direction == 2:
            if movement == 'left':
                arc_r = isect - l_off
                path.append((cx - l_off, 1100))
                path.append((cx - l_off, cy + isect))
                path.extend(self._get_curve(cx - isect, cy + isect, arc_r, 360, 270))
                path.append((-200, cy + l_off))
            elif movement == 'straight':
                path.append((cx - s_off, 1100))
                path.append((cx - s_off, -100))
            else:
                path.append((cx - r_off, 1100))
                path.append((cx - r_off, cy + r_off))
                path.extend(self._get_curve(cx + r_off, cy + r_off, 2 * r_off, 180, 270))
                path.append((1800, cy - r_off))

        elif direction == 3:
            if movement == 'left':
                arc_r = isect - l_off
                path.append((-200, cy - l_off))
                path.append((cx - isect, cy - l_off))
                path.extend(self._get_curve(cx - isect, cy - isect, arc_r, 90, 0))
                path.append((cx - l_off, -100))
            elif movement == 'straight':
                path.append((-200, cy - s_off))
                path.append((1800, cy - s_off))
            else:
                path.append((-200, cy - r_off))
                path.append((cx - r_off, cy - r_off))
                path.extend(self._get_curve(cx - r_off, cy + r_off, 2 * r_off, 270, 360))
                path.append((cx + r_off, 1100))

        return path

    def _choose_movement(self, dir_char):
        sr = STRAIGHT_RATIOS.get(dir_char, 0.75)
        r = random.random()
        if r < sr:
            return 'straight'
        remaining = 1.0 - sr
        return 'left' if r < sr + remaining / 2 else 'right'

    def _get_balanced_lane(self, direction, movement):
        if movement == 'left':
            return 0
        elif movement == 'right':
            return 2
        lane1_count = 0
        lane2_count = 0
        for v in self.vehicles:
            if v.direction == direction and v.movement == 'straight':
                if v.lane_idx == 1:
                    lane1_count += 1
                else:
                    lane2_count += 1
        return 1 if lane1_count <= lane2_count else 2

    def _is_spawn_clear(self, start_x, start_y, direction):
        for other in self.vehicles:
            if other.direction == direction:
                d = math.hypot(other.x - start_x, other.y - start_y)
                if d < (other.height + 25.0):
                    return False
        return True

    def _spawn_from_queue(self, d, vtype):
        dir_char = DIRECTIONS[d]
        movement = self._choose_movement(dir_char)
        lane_idx = self._get_balanced_lane(d, movement)
        lane_off = get_lane_offset(d, movement, lane_idx)

        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        if d == 0:
            start_x = cx + lane_off
            start_y = -150
        elif d == 1:
            start_x = 1750
            start_y = cy + lane_off
        elif d == 2:
            start_x = cx - lane_off
            start_y = 1050
        else:
            start_x = -150
            start_y = cy - lane_off

        if self._is_spawn_clear(start_x, start_y, d):
            path = self._generate_path(d, movement, lane_off)
            vehicle = Vehicle(start_x, start_y, vtype, d, movement, path, lane_idx)
            self.vehicles.append(vehicle)
            return True
        return False

    def _update_spawner(self, dt):
        for d, r in enumerate(DIRECTIONS):
            counts = self._get_counts(r)
            for vtype, count in counts.items():
                if vtype not in self.spawned_counts[r]:
                    self.spawned_counts[r][vtype] = 0
                if count > self.spawned_counts[r][vtype]:
                    diff = count - self.spawned_counts[r][vtype]
                    for _ in range(diff):
                        self.spawn_queues[r].append(vtype)
                    self.spawned_counts[r][vtype] = count

            if self.spawn_queues[r]:
                vtype = self.spawn_queues[r].pop(0)
                spawned = self._spawn_from_queue(d, vtype)
                if not spawned:
                    self.spawn_queues[r].insert(0, vtype)

    def run(self):
        elapsed = 0.0
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        self.debug = not self.debug
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

            elapsed += dt
            self.timeline.time = elapsed

            self.signal_controller.update(dt)
            if self.traffic_controller is not None:
                self.traffic_controller.update(dt)

            self._update_spawner(dt)

            for vehicle in self.vehicles[:]:
                vehicle.update(dt, self.vehicles, self.signal_controller)
                if (vehicle.x < -300 or vehicle.x > SCREEN_WIDTH + 300 or
                    vehicle.y < -300 or vehicle.y > SCREEN_HEIGHT + 300):
                    self.vehicles.remove(vehicle)

            self.screen.fill(COLOR_BG)
            draw_road(self.screen, self.signal_controller)

            for vehicle in self.vehicles:
                vehicle.draw(self.screen, self.signal_controller, debug=self.debug)

            hud_counts = self.timeline.get_all_counts()
            if self.traffic_controller is not None:
                hud_gst = {r: self.traffic_controller.get_gst(r) for r in DIRECTIONS}
                cycle_count = self.traffic_controller.get_cycle_count()
            else:
                hud_gst = {DIRECTIONS[i]: self.gst_values[i] for i in range(len(DIRECTIONS))}
                cycle_count = 0
            draw_hud(self.screen, self.signal_controller, hud_counts, hud_gst, self.vehicles, cycle_count)

            pygame.display.flip()

        pygame.quit()
        sys.exit()
