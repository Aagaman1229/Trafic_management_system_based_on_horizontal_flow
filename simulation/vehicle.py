import pygame
import math
import random
import os
from settings import *
from utils import draw_rotated_rect

class Vehicle:
    # Class-level cache for scaled vehicle images to optimize memory and performance
    images = {}

    variant_lists = {
        'car': ['car.png', 'car2.png', 'car3.png'],
        'motorcycle': ['bike.png', 'bike2.png'],
        'truck': ['truck.png', 'truck2.png'],
        'bus': ['bus.png'],
    }

    @classmethod
    def _load_variants(cls, vtype):
        """Load and cache all image variants for a vehicle type."""
        if vtype in cls.images:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        loaded = []
        for fname in cls.variant_lists.get(vtype, []):
            path = os.path.join(base_dir, 'assets', fname)
            try:
                img = pygame.image.load(path).convert_alpha()
                if img.get_width() > img.get_height():
                    img = pygame.transform.rotate(img, 90)
                w, h = VEHICLE_SIZES[vtype]
                img = pygame.transform.smoothscale(img, (w, h))
                loaded.append(img)
            except Exception:
                pass
        cls.images[vtype] = loaded

    @classmethod
    def load_image(cls, vtype):
        """Returns a random vehicle image variant for visual dynamism."""
        cls._load_variants(vtype)
        variants = cls.images.get(vtype, [])
        return random.choice(variants) if variants else None

    def __init__(self, x, y, vtype, direction, movement, path):
        """
        x, y: starting float positions
        vtype: 'car', 'motorcycle', 'truck', 'bus'
        direction: index (0=North, 1=East, 2=South, 3=West)
        movement: 'left', 'straight', 'right'
        path: list of waypoints [(x1,y1), (x2,y2), ...]
        """
        self.vtype = vtype
        self.direction = direction
        self.movement = movement
        self.path = path
        self.cur_path_idx = 0
        
        self.x = float(x)
        self.y = float(y)
        self.speed = 0.0
        
        # Add slight variation in driving profiles for realistic organic flow
        self.max_speed = MAX_SPEED * random.uniform(0.85, 1.15)
        self.acc = ACCELERATION * random.uniform(0.9, 1.1)
        self.dec = DECELERATION * random.uniform(0.9, 1.1)
        
        self.width, self.height = VEHICLE_SIZES[vtype]
        
        # Load asset image (None if missing, which falls back gracefully)
        self.image = Vehicle.load_image(vtype)
        self.use_image = (self.image is not None)
        self.fallback_color = VEHICLE_COLORS.get(vtype, (255, 255, 255))
        
        # Set initial heading angle facing the first waypoint
        if len(path) > 0:
            dx = path[0][0] - self.x
            dy = path[0][1] - self.y
            self.angle = math.degrees(math.atan2(-dy, dx)) - 90
        else:
            self.angle = 0.0

    def update(self, dt, vehicles, signal_controller):
        """
        Updates position, speed, and rotation using frame-rate independent physics.
        Implements target tracking and bumper-to-bumper queue buffer calculations.
        """
        if self.cur_path_idx < len(self.path):
            target = self.path[self.cur_path_idx]
            dx = target[0] - self.x
            dy = target[1] - self.y
            dist = math.hypot(dx, dy)
            
            # If reached current waypoint, advance to next
            if dist < 6.0:
                self.cur_path_idx += 1
                if self.cur_path_idx >= len(self.path):
                    # Off-screen or finished. Keep driving in straight line to exit
                    pass
            
            # Calculate heading unit vectors
            if dist > 0:
                vx = dx / dist
                vy = dy / dist
            else:
                vx, vy = 0.0, -1.0  # default up

            # --- Collision Avoidance & Stop Line Detection ---
            # We track absolute bumper-to-bumper physical gap to completely eliminate overlapping
            lead_gap = float('inf')
            
            # 1. Check other vehicles
            for other in vehicles:
                if other is self:
                    continue
                
                # Check if in same lane (same initial direction and lane type)
                if other.direction == self.direction and other.movement == self.movement:
                    # Flawless same-lane queue tracking: check if other is ahead along movement axis
                    is_ahead = False
                    if self.direction == 0:    # Southbound (y increases)
                        if other.y > self.y:
                            is_ahead = True
                    elif self.direction == 1:  # Westbound (x decreases)
                        if other.x < self.x:
                            is_ahead = True
                    elif self.direction == 2:  # Northbound (y decreases)
                        if other.y < self.y:
                            is_ahead = True
                    elif self.direction == 3:  # Eastbound (x increases)
                        if other.x > self.x:
                            is_ahead = True
                    
                    if is_ahead:
                        d = math.hypot(other.x - self.x, other.y - self.y)
                        gap = d - (self.height / 2.0 + other.height / 2.0)
                        lead_gap = min(lead_gap, gap)
                else:
                    # Cross-lane safety corridor checking (in case paths intersect inside junction)
                    ux = other.x - self.x
                    uy = other.y - self.y
                    d = math.hypot(ux, uy)
                    
                    if d < 180.0:
                        d_long = ux * vx + uy * vy
                        d_lat = abs(ux * (-vy) + uy * vx)
                        
                        # If vehicle is in front and in the same 26px wide lane buffer
                        if d_long > 0 and d_lat < 26.0:
                            gap = d_long - (self.height / 2.0 + other.height / 2.0)
                            lead_gap = min(lead_gap, gap)

            # 2. Check signal light (RED and ORANGE are treated as STOP commands)
            # Free left turn (Nepal left-hand traffic) ALWAYS bypasses signal lines.
            if self.movement != 'left' and signal_controller.get_signal_state(self.direction) in ['RED', 'ORANGE']:
                cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
                stop_line_dist = float('inf')
                
                # Check if we are approaching and before the intersection box entrance (90px from center)
                if self.direction == 0:    # North approach (moving South)
                    if self.y < cy - 90:
                        stop_line_dist = (cy - 115) - self.y
                elif self.direction == 1:  # East approach (moving West)
                    if self.x > cx + 90:
                        stop_line_dist = self.x - (cx + 115)
                elif self.direction == 2:  # South approach (moving North)
                    if self.y > cy + 90:
                        stop_line_dist = self.y - (cy + 115)
                elif self.direction == 3:  # West approach (moving East)
                    if self.x < cx - 90:
                        stop_line_dist = (cx - 115) - self.x
                
                # If within the detection range (including slight overshoots down to -25px past the stop line)
                if -25.0 < stop_line_dist < 150.0:
                    # Bumper gap to stop line is stop_line_dist minus half our length
                    gap = stop_line_dist - (self.height / 2.0)
                    lead_gap = min(lead_gap, gap)

            # --- Speed Controller ---
            target_speed = self.max_speed
            
            # SAFE_DISTANCE defines the deceleration zone. Stopped safety gap is 12px.
            if lead_gap < SAFE_DISTANCE:
                if lead_gap <= 12.0:
                    # Stop fully to maintain small safety buffer
                    target_speed = 0.0
                else:
                    # Smoothly decelerate as we get closer to the obstacle
                    ratio = (lead_gap - 12.0) / (SAFE_DISTANCE - 12.0)
                    target_speed = self.max_speed * ratio

            # Accelerate or brake towards target speed
            if self.speed < target_speed:
                self.speed = min(self.speed + self.acc * dt, target_speed)
            elif self.speed > target_speed:
                self.speed = max(self.speed - self.dec * dt, target_speed)

            # --- Update coordinates ---
            self.x += vx * self.speed * dt
            self.y += vy * self.speed * dt
            
            # Smoothly update rotation to match trajectory
            desired_angle = math.degrees(math.atan2(-vy, vx)) - 90
            
            # Prevent angle jumping when passing 180 degrees
            diff = (desired_angle - self.angle + 180) % 360 - 180
            self.angle += diff * 0.15  # Smooth interpolation
            
        else:
            # No waypoints remaining: keep driving straight at current heading
            rad = math.radians(self.angle + 90)
            vx = math.cos(rad)
            vy = -math.sin(rad)
            self.x += vx * self.speed * dt
            self.y += vy * self.speed * dt

    def draw(self, surface, signal_controller, debug=False):
        """
        Renders the vehicle. If assets are available, blits the rotated PNG image.
        Otherwise, draws an extremely premium vector fallback.
        Introduces a realistic engine vibration jitter when stopped during the ORANGE phase.
        """
        # Determine engine idle vibration offset
        draw_x, draw_y = self.x, self.y
        if self.speed < 1.0 and signal_controller.get_signal_state(self.direction) == 'ORANGE':
            # Add premium visual jitter/vibration (simulating engine start & gear engagement)
            draw_x += random.uniform(-0.8, 0.8)
            draw_y += random.uniform(-0.8, 0.8)

        if self.use_image and not debug:
            # Draw PNG image
            rot_img = pygame.transform.rotate(self.image, self.angle)
            rot_rect = rot_img.get_rect(center=(int(draw_x), int(draw_y)))
            surface.blit(rot_img, rot_rect)
        else:
            # Draw premium vector fallback with front/rear lights and windshield
            shape_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # 1. Main chassis (rounded corners)
            pygame.draw.rect(shape_surf, self.fallback_color, (0, 0, self.width, self.height), border_radius=4)
            
            # 2. Windshield glass (near the front)
            windshield_h = max(3, int(self.height * 0.18))
            windshield_y = int(self.height * 0.22)
            pygame.draw.rect(shape_surf, (30, 41, 59), (2, windshield_y, self.width - 4, windshield_h), border_radius=2)
            
            # 3. Headlights (glowing yellow at the front top corner)
            pygame.draw.circle(shape_surf, (253, 224, 71), (3, 3), 2)
            pygame.draw.circle(shape_surf, (253, 224, 71), (self.width - 3, 3), 2)
            
            # 4. Taillights (glowing red at the rear bottom corner)
            pygame.draw.circle(shape_surf, (239, 68, 68), (3, self.height - 3), 2)
            pygame.draw.circle(shape_surf, (239, 68, 68), (self.width - 3, self.height - 3), 2)
            
            # Rotate custom vector drawing
            rot_surf = pygame.transform.rotate(shape_surf, self.angle)
            rot_rect = rot_surf.get_rect(center=(int(draw_x), int(draw_y)))
            surface.blit(rot_surf, rot_rect)