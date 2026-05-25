import pygame
import math
import random
import os
from .settings import *
from .utils import draw_rotated_rect

class Vehicle:
    # Class-level cache for scaled vehicle images to optimize memory and performance
    images = {}
    variant_lists = {}

    @classmethod
    def _discover_assets(cls):
        """Dynamically discover all PNG assets from the assets folder and map to vehicle types."""
        if cls.variant_lists:
            return
        cls.variant_lists = {'car': [], 'motorcycle': [], 'truck': [], 'bus': [], 'bicycle': []}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_dir, 'assets')
        if not os.path.isdir(assets_dir):
            return
        for fname in os.listdir(assets_dir):
            if not fname.lower().endswith('.png'):
                continue
            name_lower = fname.lower()
            if name_lower.startswith('car'):
                cls.variant_lists['car'].append(fname)
            elif name_lower.startswith('bike') or name_lower.startswith('motorcycle'):
                cls.variant_lists['motorcycle'].append(fname)
            elif name_lower.startswith('truck'):
                cls.variant_lists['truck'].append(fname)
            elif name_lower.startswith('bus'):
                cls.variant_lists['bus'].append(fname)
            elif name_lower.startswith('bicycle') or name_lower.startswith('cycle'):
                cls.variant_lists['bicycle'].append(fname)

    @classmethod
    def _load_variants(cls, vtype):
        """Load and cache all image variants for a vehicle type."""
        if vtype in cls.images:
            return
        cls._discover_assets()
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

    def __init__(self, x, y, vtype, direction, movement, path, lane_idx=0):
        """
        x, y: starting float positions
        vtype: 'car', 'motorcycle', 'truck', 'bus'
        direction: index (0=North, 1=East, 2=South, 3=West)
        movement: 'left', 'straight', 'right'
        path: list of waypoints [(x1,y1), (x2,y2), ...]
        lane_idx: which lane this vehicle is in (0=left, 1=straight, 2=straight+right)
        """
        self.vtype = vtype
        self.direction = direction
        self.movement = movement
        self.path = path
        self.cur_path_idx = 0
        self.lane_idx = lane_idx
        
        self.x = float(x)
        self.y = float(y)
        self.speed = 0.0
        
        self.max_speed = MAX_SPEED * random.uniform(0.85, 1.15)
        self.acc = ACCELERATION * random.uniform(0.9, 1.1)
        self.dec = DECELERATION * random.uniform(0.9, 1.1)
        
        self.width, self.height = VEHICLE_SIZES[vtype]
        
        self.image = Vehicle.load_image(vtype)
        self.use_image = (self.image is not None)
        self.fallback_color = VEHICLE_COLORS.get(vtype, (255, 255, 255))
        
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
            
            if dist < 6.0:
                self.cur_path_idx += 1
                if self.cur_path_idx >= len(self.path):
                    pass
            
            if dist > 0:
                vx = dx / dist
                vy = dy / dist
            else:
                vx, vy = 0.0, -1.0

            # --- Collision Avoidance (same-lane only) ---
            lead_gap = float('inf')
            
            for other in vehicles:
                if other is self:
                    continue
                if not (other.direction == self.direction and
                        other.lane_idx == self.lane_idx):
                    continue
                
                is_ahead = False
                if self.direction == 0:
                    if other.y > self.y:
                        is_ahead = True
                elif self.direction == 1:
                    if other.x < self.x:
                        is_ahead = True
                elif self.direction == 2:
                    if other.y < self.y:
                        is_ahead = True
                elif self.direction == 3:
                    if other.x > self.x:
                        is_ahead = True
                
                if is_ahead:
                    d = math.hypot(other.x - self.x, other.y - self.y)
                    gap = d - (self.height / 2.0 + other.height / 2.0)
                    lead_gap = min(lead_gap, gap)

            # --- Stop-line check (only for waiting lanes: straight/right at RED/ORANGE) ---
            # Left-turn (Lane 0) NEVER stops — free-flowing by Nepal traffic rules
            if self.movement != 'left' and self.movement != 'left_free' and \
               signal_controller.get_signal_state(self.direction) in ['RED', 'ORANGE']:
                cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
                stop_line_dist = float('inf')
                
                if self.direction == 0:
                    if self.y < cy - 90:
                        stop_line_dist = (cy - 115) - self.y
                elif self.direction == 1:
                    if self.x > cx + 90:
                        stop_line_dist = self.x - (cx + 115)
                elif self.direction == 2:
                    if self.y > cy + 90:
                        stop_line_dist = self.y - (cy + 115)
                elif self.direction == 3:
                    if self.x < cx - 90:
                        stop_line_dist = (cx - 115) - self.x
                
                if 0.0 <= stop_line_dist < 150.0:
                    gap = stop_line_dist - (self.height / 2.0)
                    lead_gap = min(lead_gap, gap)

            # --- Speed Controller ---
            target_speed = self.max_speed
            
            if lead_gap < SAFE_DISTANCE:
                if lead_gap <= 4.0:
                    target_speed = 0.0
                elif lead_gap < 12.0:
                    target_speed = max(8.0, self.speed * 0.5)  # gentle creep
                else:
                    stopping_dist = (self.speed * self.speed) / (2.0 * self.dec)
                    available_dist = lead_gap - 4.0
                    if stopping_dist >= available_dist * 0.85:
                        target_speed = 0.0
                    else:
                        ratio = (lead_gap - 4.0) / (SAFE_DISTANCE - 4.0)
                        target_speed = self.max_speed * (ratio * ratio)

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
        """
        draw_x, draw_y = self.x, self.y

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