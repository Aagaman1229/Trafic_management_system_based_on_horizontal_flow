import pygame
import sys
import random
import math
from settings import *
from vehicle import Vehicle
from road import draw_road
from signals import SignalController
from hud import draw_hud
from gst import calculate_gst

class Simulation:
    """
    Main Simulation Orchestrator for the 6-lane 2-way intersection.
    Handles GST calculation, initial queue spawning, continuous spawning, path generation,
    frame locked game-loop execution, and cleanup of departed vehicles.

    When `shared_state`, `signal_controller`, and `traffic_controller` are provided,
    the simulation reads live vehicle counts and signal states from the real-time
    TrafficSignalController and VideoManager running in the root process.
    """

    def __init__(self, detected_data, debug=False, gst_values=None,
                 shared_state=None, signal_controller=None, traffic_controller=None):
        pygame.init()
        self.debug = debug
        self.clock = pygame.time.Clock()
        self.running = True

        self.shared_state = shared_state
        self.traffic_controller = traffic_controller

        # Store initial detections dictionary (may be None when shared_state in use)
        self.detected_data = detected_data

        # 1. Use pre-computed GST values if provided, otherwise compute from data
        if gst_values is not None:
            self.gst_values = gst_values
        elif detected_data is not None:
            self.gst_values = self._compute_initial_gsts()
        else:
            self.gst_values = [15.0, 12.0, 20.0, 15.0]

        # 2. Initialize Signal Controller (use external if provided)
        if signal_controller is not None:
            self.signal_controller = signal_controller
        else:
            self.signal_controller = SignalController(self.gst_values)

        # 3. Create Pygame Window
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Traffic Management System Simulation (6-Lane Nepal-Style)")

        # 4. Spawning State
        self.vehicles = []
        self.spawn_timers = [0.0, 0.0, 0.0, 0.0]  # Timers for A, B, C, D

        # Initialize continuous spawner class counts per approach for deterministic class-basis routing
        self.class_spawn_counts = [
            {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0}
            for _ in range(4)
        ]

        # 5. Decide spawning strategy
        if self.shared_state is not None:
            # LIVE MODE: no initial queue — vehicles appear gradually as
            # detected by the parallel video processing threads.
            # Set short staggered spawn timers so vehicles start arriving quickly.
            for d in range(4):
                self.spawn_timers[d] = (d + 1) * 0.5
        else:
            # STANDALONE MODE (cached/fallback data): seed an initial queue
            # so the road doesn't look empty at startup.
            self._reset_spawn_timers()
            self._spawn_initial_queues()

    def _get_counts(self, direction_char):
        """Return counts for a road: from shared_state (live) or detected_data (static)."""
        if self.shared_state is not None:
            return self.shared_state.get_counts(direction_char)
        if self.detected_data is not None:
            return self.detected_data.get(direction_char, {})
        return {}

    def _compute_initial_gsts(self):
        gsts = []
        for d, direction_char in enumerate(DIRECTIONS):
            counts = self._get_counts(direction_char)
            num_lanes = APPROACH_LANES.get(direction_char, 1)
            straight_ratio = STRAIGHT_RATIOS.get(direction_char, 0.75)
            gst = calculate_gst(counts, num_lanes=num_lanes, straight_ratio=straight_ratio)
            gsts.append(gst)
        return gsts

    def _reset_spawn_timers(self):
        for d in range(4):
            dir_char = DIRECTIONS[d]
            counts = self._get_counts(dir_char)
            total_detected = sum(counts.values())
            interval = max(2.0, 30.0 / (total_detected + 2.0))
            self.spawn_timers[d] = interval * random.uniform(0.1, 0.8)

    def _get_curve(self, cx_arc, cy_arc, radius, start_angle_deg, end_angle_deg, num_pts=6):
        """Generates a list of waypoint coordinates along a smooth circular arc."""
        points = []
        for i in range(num_pts):
            t = i / (num_pts - 1)
            angle = math.radians(start_angle_deg + t * (end_angle_deg - start_angle_deg))
            px = cx_arc + radius * math.cos(angle)
            py = cy_arc + radius * math.sin(angle)
            points.append((int(px), int(py)))
        return points

    def _generate_path(self, direction, movement):
        """
        Generates waypoint trajectories with lane positions based on per-approach lane count.
        Uses get_lane_offset() and max_lane_offset() from settings for dynamic lane positioning.
        """
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        isect = max_lane_offset(direction)  # intersection edge from center for this approach

        l_off = get_lane_offset(direction, 'left')
        s_off = get_lane_offset(direction, 'straight')
        r_off = get_lane_offset(direction, 'right')

        path = []

        if direction == 0:  # North approach (moving South)
            if movement == 'left':  # Turn East
                arc_r = isect - l_off
                path.append((cx + l_off, -100))
                path.append((cx + l_off, cy - isect))
                path.extend(self._get_curve(cx + isect, cy - isect, arc_r, 180, 90))
                path.append((1800, cy - l_off))
            elif movement == 'straight':  # South
                path.append((cx + s_off, -100))
                path.append((cx + s_off, 1100))
            else:  # Right turn West
                path.append((cx + r_off, -100))
                path.append((cx + r_off, cy - r_off))
                path.extend(self._get_curve(cx - r_off, cy - r_off, 2 * r_off, 0, 90))
                path.append((-200, cy + r_off))

        elif direction == 1:  # East approach (moving West)
            if movement == 'left':  # Turn South
                arc_r = isect - l_off
                path.append((1800, cy + l_off))
                path.append((cx + isect, cy + l_off))
                path.extend(self._get_curve(cx + isect, cy + isect, arc_r, 270, 180))
                path.append((cx + l_off, 1100))
            elif movement == 'straight':  # West
                path.append((1800, cy + s_off))
                path.append((-200, cy + s_off))
            else:  # Right turn North
                path.append((1800, cy + r_off))
                path.append((cx + r_off, cy + r_off))
                path.extend(self._get_curve(cx + r_off, cy - r_off, 2 * r_off, 90, 180))
                path.append((cx - r_off, -100))

        elif direction == 2:  # South approach (moving North)
            if movement == 'left':  # Turn West
                arc_r = isect - l_off
                path.append((cx - l_off, 1100))
                path.append((cx - l_off, cy + isect))
                path.extend(self._get_curve(cx - isect, cy + isect, arc_r, 360, 270))
                path.append((-200, cy + l_off))
            elif movement == 'straight':  # North
                path.append((cx - s_off, 1100))
                path.append((cx - s_off, -100))
            else:  # Right turn East
                path.append((cx - r_off, 1100))
                path.append((cx - r_off, cy + r_off))
                path.extend(self._get_curve(cx + r_off, cy + r_off, 2 * r_off, 180, 270))
                path.append((1800, cy - r_off))

        elif direction == 3:  # West approach (moving East)
            if movement == 'left':  # Turn North
                arc_r = isect - l_off
                path.append((-200, cy - l_off))
                path.append((cx - isect, cy - l_off))
                path.extend(self._get_curve(cx - isect, cy - isect, arc_r, 90, 0))
                path.append((cx - l_off, -100))
            elif movement == 'straight':  # East
                path.append((-200, cy - s_off))
                path.append((1800, cy - s_off))
            else:  # Right turn South
                path.append((-200, cy - r_off))
                path.append((cx - r_off, cy - r_off))
                path.extend(self._get_curve(cx - r_off, cy + r_off, 2 * r_off, 270, 360))
                path.append((cx + r_off, 1100))

        return path

    def _choose_movement(self, dir_char):
        """Pick movement weighted by per-road straight ratio."""
        sr = STRAIGHT_RATIOS.get(dir_char, 0.75)
        r = random.random()
        if r < sr:
            return 'straight'
        remaining = 1.0 - sr
        return 'left' if r < sr + remaining / 2 else 'right'

    def _spawn_initial_queues(self):
        """Spawns a visible queue of exactly 5 vehicles per approach at startup, using dynamic lane offsets."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        gap = 12.0
        
        for d in range(4):
            dir_char = DIRECTIONS[d]
            counts = self._get_counts(dir_char)

            vtype_pool = []
            for vt, count in counts.items():
                vtype_pool.extend([vt] * count)
            if not vtype_pool:
                vtype_pool = ['car']
            while len(vtype_pool) < 5:
                vtype_pool.extend(vtype_pool)

            rand_off = random.uniform(0, VEHICLE_SIZES['car'][1])
            left_offset = 200.0  # left-turn queue starts behind straight/right
            offsets = {'left': left_offset + rand_off, 'straight': 115.0 + rand_off, 'right': 115.0 + rand_off}

            random.shuffle(vtype_pool)
            approach_spawn_counts = {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0}

            for idx in range(5):
                vtype = vtype_pool[idx]
                c_idx = approach_spawn_counts[vtype]
                movement = self._choose_movement(dir_char)
                approach_spawn_counts[vtype] += 1

                lane_off = get_lane_offset(d, movement)
                curr_offset = offsets[movement]
                v_width, v_height = VEHICLE_SIZES[vtype]

                if d == 0:    # North (moving South)
                    start_x = cx + lane_off
                    start_y = cy - (curr_offset + v_height / 2.0)
                elif d == 1:  # East (moving West)
                    start_x = cx + (curr_offset + v_height / 2.0)
                    start_y = cy + lane_off
                elif d == 2:  # South (moving North)
                    start_x = cx - lane_off
                    start_y = cy + (curr_offset + v_height / 2.0)
                else:         # West (moving East)
                    start_x = cx - (curr_offset + v_height / 2.0)
                    start_y = cy - lane_off

                full_path = self._generate_path(d, movement)
                filtered_path = self._filter_path(full_path, start_x, start_y, d)
                vehicle = Vehicle(start_x, start_y, vtype, d, movement, filtered_path)
                self.vehicles.append(vehicle)
                offsets[movement] += v_height + gap

    def _filter_path(self, path, x, y, direction):
        """Removes waypoints that are behind the starting position to prevent driving backwards."""
        if direction == 0:    # Southbound
            return [pt for pt in path if pt[1] >= y - 5]
        elif direction == 1:  # Westbound
            return [pt for pt in path if pt[0] <= x + 5]
        elif direction == 2:  # Northbound
            return [pt for pt in path if pt[1] <= y + 5]
        else:                 # Eastbound
            return [pt for pt in path if pt[0] >= x - 5]

    def _is_spawn_clear(self, start_x, start_y, direction):
        """
        Safety Check: Returns True if the entry area is clear of other vehicles in that lane.
        Prevents overlapping spawns.
        """
        for other in self.vehicles:
            # Only check vehicles in the same direction
            if other.direction == direction:
                d = math.hypot(other.x - start_x, other.y - start_y)
                # If distance is too close (within a vehicle length + margin), lane is blocked
                if d < (other.height + 25.0):
                    return False
        return True

    def _spawn_continuous_vehicle(self, d):
        """Spawns a new vehicle at the edge of the screen if the entry area is clear, on a class-basis."""
        dir_char = DIRECTIONS[d]
        counts = self._get_counts(dir_char)

        vtype_pool = []
        for vt, count in counts.items():
            vtype_pool.extend([vt] * count)
        if not vtype_pool:
            return

        vtype = random.choice(vtype_pool)

        movement = self._choose_movement(dir_char)

        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        lane_off = get_lane_offset(d, movement)

        if d == 0:    # North
            start_x = cx + lane_off
            start_y = -150
        elif d == 1:  # East
            start_x = 1750
            start_y = cy + lane_off
        elif d == 2:  # South
            start_x = cx - lane_off
            start_y = 1050
        else:         # West
            start_x = -150
            start_y = cy - lane_off

        if self._is_spawn_clear(start_x, start_y, d):
            path = self._generate_path(d, movement)
            vehicle = Vehicle(start_x, start_y, vtype, d, movement, path)
            self.vehicles.append(vehicle)

    def run(self):
        """Executes the Pygame event and updates loops at 60 FPS."""
        while self.running:
            # Tick locked at 60 FPS and calculate frame delta time in seconds
            dt = self.clock.tick(FPS) / 1000.0
            
            # Event polling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:  # toggle debug mode
                        self.debug = not self.debug
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

            # 1. Update Signals
            self.signal_controller.update(dt)

            # 2. Continuous Spawning
            for d in range(4):
                self.spawn_timers[d] -= dt
                if self.spawn_timers[d] <= 0:
                    self._spawn_continuous_vehicle(d)
                    
                    # Reset spawn timer using live counts when available
                    dir_char = DIRECTIONS[d]
                    counts = self._get_counts(dir_char)
                    total_detected = sum(counts.values())
                    interval = max(2.0, 30.0 / (total_detected + 2.0))
                    # Add organic randomness
                    self.spawn_timers[d] = interval * random.uniform(0.7, 1.3)

            # 3. Update all active vehicles
            for vehicle in self.vehicles[:]:
                vehicle.update(dt, self.vehicles, self.signal_controller)
                
                # 4. Clean up vehicles that have driven fully off-screen
                if (vehicle.x < -300 or vehicle.x > SCREEN_WIDTH + 300 or
                    vehicle.y < -300 or vehicle.y > SCREEN_HEIGHT + 300):
                    self.vehicles.remove(vehicle)

            # 5. Render Scene
            self.screen.fill(COLOR_BG)
            
            # Draw Road Intersection and Lights
            draw_road(self.screen, self.signal_controller)
            
            # Draw Vehicles (renders image or premium vector fallback)
            for vehicle in self.vehicles:
                vehicle.draw(self.screen, self.signal_controller, debug=self.debug)
                
            # Live display data for HUD
            if self.shared_state is not None:
                hud_counts = self.shared_state.get_all_counts()
                hud_gst = [self.shared_state.get_gst(r) for r in DIRECTIONS]
                cycle_count = self.shared_state.get_cycle_count()
            else:
                hud_counts = self.detected_data if self.detected_data else {}
                hud_gst = self.gst_values if self.gst_values else [15.0, 12.0, 20.0, 15.0]
                cycle_count = 0
            draw_hud(self.screen, self.signal_controller, hud_counts, hud_gst, self.vehicles, cycle_count)
            
            pygame.display.flip()

        pygame.quit()
        sys.exit()